#!/usr/bin/env python3
"""7c_steering.py - Steering hooks and forward pass logic.

This module contains functions for:
1. Hook creation for on-the-fly steering
2. Batched and single forward passes with steering
3. Heterogeneous steering (cross-prefix batching)
4. Baseline computation
"""

import os
from collections import defaultdict
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Import shared utilities
import importlib.util
from pathlib import Path as _Path
import sys
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from utils.memory_utils import maybe_clear_memory

def _import_7c_utils():
    spec = importlib.util.spec_from_file_location("7c_utils", _Path(__file__).parent / "7c_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

utils = _import_7c_utils()

# Optional profiler (enabled via PROFILE_7C=1 env var)
# Use sys.modules to share profiler instance across all 7c modules
_PROFILING_ENABLED = os.environ.get("PROFILE_7C", "0") == "1"

if _PROFILING_ENABLED:
    if "7c_profiler" in sys.modules:
        profiler = sys.modules["7c_profiler"]
    else:
        spec = importlib.util.spec_from_file_location("7c_profiler", _Path(__file__).parent / "profiler.py")
        profiler = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profiler)
        sys.modules["7c_profiler"] = profiler
else:
    # Dummy profiler that does nothing
    class _DummyProfiler:
        @staticmethod
        def timed(name, sync_cuda=True):
            from contextlib import nullcontext
            return nullcontext()
    profiler = _DummyProfiler()


# =============================================================================
# Hook Creation Functions
# =============================================================================

def get_feature_activation_caching_hook(
    model,
    layer: int,
    encoder_cache: Dict[int, Dict[str, torch.Tensor]],
    activation_cache: Dict[int, Dict[int, torch.Tensor]]
):
    """Create a hook that caches ONLY the needed feature activations (column-wise).

    This hook runs at feature_input_hook (before MLP) and computes activations
    only for the features we want to intervene on, not the full 131k features.

    Args:
        model: ReplacementModel
        layer: Layer index
        encoder_cache: Pre-loaded encoder weights {layer: {'W_enc': ..., 'b_enc': ..., 'feat_ids': ...}}
        activation_cache: Dict to store {layer: {feat_id: [seq_len] activations}}

    Returns:
        Hook function
    """
    if layer not in encoder_cache:
        def noop_hook(activations, hook=None):
            return activations
        return noop_hook

    W_enc_subset = encoder_cache[layer]['W_enc']  # [n_feats, d_model]
    b_enc_subset = encoder_cache[layer]['b_enc']  # [n_feats]
    feat_ids = encoder_cache[layer]['feat_ids']   # [feat_id, ...]

    def hook_fn(activations, hook=None):
        # activations: [batch, seq_len, d_model]
        residual = activations.squeeze(0) if activations.dim() == 3 and activations.shape[0] == 1 else activations

        # Handle batched case
        if residual.dim() == 3:
            # Batched: [batch, seq_len, d_model]
            pre_acts = torch.einsum('bsd,fd->bsf', residual, W_enc_subset) + b_enc_subset
        else:
            # Single: [seq_len, d_model]
            pre_acts = residual @ W_enc_subset.T + b_enc_subset  # [seq_len, n_feats]

        # Apply activation function (JumpReLU or ReLU)
        acts = model.transcoders.apply_activation_function(layer, pre_acts)

        # Store as dict mapping feat_id -> activation values
        if residual.dim() == 3:
            # Batched case: store full tensor
            activation_cache[layer] = {
                'batched': True,
                'acts': acts.detach(),  # [batch, seq_len, n_feats]
                'feat_id_to_idx': encoder_cache[layer]['feat_id_to_idx']
            }
        else:
            # Single case: store per-feature
            activation_cache[layer] = {
                feat_id: acts[:, i].detach() for i, feat_id in enumerate(feat_ids)
            }

        return activations

    return hook_fn


def make_on_the_fly_steering_hook(
    model,
    layer: int,
    feature_interventions: List[Tuple[int, int, float]],
    activation_cache: Dict[int, Dict],
    decoder_cache: Dict[int, Tuple[List[int], List[int], torch.Tensor, torch.Tensor]],
    steering_method: str,
    epsilon: float,
    h_c_norm: float,
):
    """Create a hook that computes steering delta on-the-fly using current activations.

    This hook runs at feature_output_hook (after MLP) and computes deltas based
    on the CURRENT activation values (not pre-computed baselines).

    Args:
        model: ReplacementModel
        layer: Layer index
        feature_interventions: List of (pos, feat_id, h_c_val) for this layer
        activation_cache: Dict[layer] -> {feat_id: [seq_len] activations} or batched format
        decoder_cache: {layer: (positions, feat_ids, decoder_vecs, h_c_vals)} from precompute
        steering_method: "additive", "multiplicative", "absolute", "sign", or "scaling"
        epsilon: Steering strength
        h_c_norm: ||H_c|| for normalization

    Returns:
        Hook function that modifies activations
    """
    if layer not in decoder_cache:
        def noop_hook(activations, hook=None):
            return activations
        return noop_hook

    # Get pre-computed decoder vectors for this layer
    positions_list, feat_ids_list, decoder_vecs, h_c_vals = decoder_cache[layer]

    # Build lookup for quick access
    pos_feat_to_idx = {}
    for i, (pos, feat_id) in enumerate(zip(positions_list, feat_ids_list)):
        pos_feat_to_idx[(pos, feat_id)] = i

    def hook_fn(activations, hook=None):
        if layer not in activation_cache:
            return activations

        layer_acts = activation_cache[layer]
        deltas = torch.zeros_like(activations)

        # Check if batched
        is_batched = isinstance(layer_acts, dict) and layer_acts.get('batched', False)

        for pos, feat_id, h_c_val in feature_interventions:
            # Skip BOS and out-of-bounds positions
            if pos >= activations.shape[1] or pos <= 0:
                continue

            # Get decoder vector index
            key = (pos, feat_id)
            if key not in pos_feat_to_idx:
                continue
            idx = pos_feat_to_idx[key]

            # Get current activation value
            if is_batched:
                # Batched case
                feat_idx = layer_acts['feat_id_to_idx'].get(feat_id)
                if feat_idx is None:
                    continue
                # Bounds check for position
                if pos >= layer_acts['acts'].shape[1] or pos <= 0:
                    continue
                current_vals = layer_acts['acts'][:, pos, feat_idx]  # [batch]
            else:
                # Single case
                if feat_id not in layer_acts:
                    continue
                # Bounds check for position
                if pos >= len(layer_acts[feat_id]) or pos <= 0:
                    continue
                current_val = layer_acts[feat_id][pos].item()
                # Use consistent naming - current_vals for both cases
                current_vals = current_val

            # Compute delta based on steering method (using shared utility)
            delta_scalar = utils.compute_steering_delta(
                current_vals, h_c_val, h_c_norm, steering_method, epsilon
            )

            # Get decoder vector
            decoder_vec = decoder_vecs[idx]

            # Apply delta
            if is_batched:
                if isinstance(delta_scalar, (int, float)):
                    # Scalar delta applied to all batch items
                    deltas[:, pos, :] += decoder_vec * delta_scalar
                else:
                    # Batched delta: [batch] * [d_model] -> [batch, d_model]
                    deltas[:, pos, :] += delta_scalar.unsqueeze(-1) * decoder_vec
            else:
                deltas[0, pos, :] += decoder_vec * delta_scalar

        return activations + deltas

    return hook_fn


# =============================================================================
# Forward Pass Functions
# =============================================================================

def run_batched_steered_pass_on_the_fly(
    model,
    batch_token_ids: List[List[int]],
    features: List[Tuple[int, int, int, float]],
    encoder_cache: Dict[int, Dict[str, torch.Tensor]],
    decoder_cache: Dict[int, Tuple[List[int], List[int], torch.Tensor, torch.Tensor]],
    steering_method: str,
    epsilon: float,
    max_seq_len: int = None,
):
    """Batched forward pass for multiple continuations with same steering (On-The-Fly).

    All items in batch share:
    - Same prefix tokens (steering positions are in prefix)
    - Same H_c values and feat_ids
    - Same encoder/decoder weights

    They differ only in continuation tokens.

    Args:
        model: ReplacementModel
        batch_token_ids: List of token ID sequences [batch_size, varying_seq_len]
        features: List of (layer, pos, feat_id, h_c_val) tuples (same for all)
        encoder_cache: Pre-loaded encoder weights
        decoder_cache: Pre-computed decoder vectors
        steering_method: Steering method name
        epsilon: Steering strength
        max_seq_len: Fixed padding length. If None, uses max length in batch.

    Returns:
        logits: [batch, max_seq_len, vocab_size]
        attention_mask: [batch, max_seq_len] (1 for valid, 0 for padding)
    """
    hooks = []
    activation_cache = {}

    # Compute h_c norm
    h_c_values = [h_c_val for _, _, _, h_c_val in features]
    if h_c_values:
        h_c_tensor = torch.tensor(h_c_values, device=model.cfg.device, dtype=model.cfg.dtype)
        h_c_norm = torch.linalg.norm(h_c_tensor).item()
    else:
        h_c_norm = 1.0
    if h_c_norm == 0:
        h_c_norm = 1.0

    # Pad sequences and create attention mask (using shared utility)
    pad_token_id = 0
    if hasattr(model, "tokenizer") and model.tokenizer is not None and model.tokenizer.pad_token_id is not None:
        pad_token_id = model.tokenizer.pad_token_id

    batch_tokens, attention_mask = utils.create_padded_batch(
        batch_token_ids, pad_token_id, model.cfg.device, max_seq_len
    )

    # Group interventions by layer
    interventions_by_layer = {}
    for layer, pos, feat_id, h_c_val in features:
        if layer not in interventions_by_layer:
            interventions_by_layer[layer] = []
        interventions_by_layer[layer].append((pos, feat_id, h_c_val))

    no_hooks = bool(int(os.getenv("STEERING_NO_HOOKS", "0")))

    # Normalize decoder_cache to per-layer mapping if needed (use existing utility function)
    decoder_cache = _normalize_decoder_cache_to_per_layer(
        decoder_cache, model.cfg.device, model.cfg.dtype
    )

    if not no_hooks:
        # Phase 1: Column-wise activation caching hooks (batched)
        for layer in encoder_cache.keys():
            hook_name = f"blocks.{layer}.{model.feature_input_hook}"
            hooks.append((hook_name, get_feature_activation_caching_hook(
                model, layer, encoder_cache, activation_cache
            )))

        # Phase 2: On-the-fly delta application hooks (batched)
        for layer, interventions in interventions_by_layer.items():
            hook_name = f"blocks.{layer}.{model.feature_output_hook}"
            hooks.append((hook_name, make_on_the_fly_steering_hook(
                model, layer, interventions, activation_cache,
                decoder_cache, steering_method, epsilon, h_c_norm
            )))

    with torch.inference_mode():
        with model.hooks(hooks):
            # attention_mask is supported via HookedTransformer.forward inheritance
            # This ensures padding tokens are properly masked in attention computation
            logits = model(batch_tokens, attention_mask=attention_mask)

    return logits, attention_mask


def run_steered_pass_on_the_fly(
    model,
    full_token_ids: List[int],
    features: List[Tuple[int, int, int, float]],
    encoder_cache: Dict[int, Dict[str, torch.Tensor]],
    decoder_cache: Dict[int, Tuple[List[int], List[int], torch.Tensor, torch.Tensor]],
    steering_method: str,
    epsilon: float,
    max_seq_len: int = None,
):
    """Run steering with on-the-fly delta computation using column-wise encoding (Single).

    Args:
        model: ReplacementModel
        full_token_ids: Token IDs WITH BOS at position 0
        features: List of (layer, pos, feat_id, h_c_val) tuples
        encoder_cache: Pre-loaded encoder weights from preload_encoder_weights_for_cluster
        decoder_cache: Pre-computed decoder vectors from precompute_cluster_decoder_vectors
        steering_method: "additive", "multiplicative", "absolute", "sign", or "scaling"
        epsilon: Steering strength
        max_seq_len: Fixed padding length. If None, uses sequence length.

    Returns:
        logits: Model output logits
    """
    # Simple wrapper around batched version
    logits, _ = run_batched_steered_pass_on_the_fly(
        model, [full_token_ids], features, encoder_cache, decoder_cache, 
        steering_method, epsilon, max_seq_len
    )
    return logits


# =============================================================================
# Heterogeneous Steering (Cross-Prefix Batching)
# =============================================================================

def _normalize_decoder_cache_to_per_layer(
    decoder_cache: Dict,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[int, Tuple[List[int], List[int], torch.Tensor, torch.Tensor]]:
    """Normalize decoder cache to per-layer format.

    Handles both formats:
    - Per-layer: {layer: (positions, feat_ids, decoder_vecs, h_c_vals)}
    - Flattened: {'layers': [...], 'positions': [...], 'feat_ids': [...], 'decoder_vecs': [...], 'h_c_values': [...]}

    Returns:
        {layer: (positions_list, feat_ids_list, decoder_vecs, h_c_vals)}
    """
    if not decoder_cache:
        return {}

    # Check if already in per-layer format (first key is int)
    first_key = next(iter(decoder_cache.keys()), None)
    if first_key is not None and isinstance(first_key, int):
        return decoder_cache

    # Flattened format - convert to per-layer
    if "layers" not in decoder_cache:
        return {}

    layers_list = decoder_cache.get("layers", [])
    positions_list = decoder_cache.get("positions", [])
    feat_ids_list = decoder_cache.get("feat_ids", [])
    decoder_vecs_list = decoder_cache.get("decoder_vecs", [])
    h_c_values_list = decoder_cache.get("h_c_values", [])

    # Validate all lists have the same length
    list_lengths = [len(layers_list), len(positions_list), len(feat_ids_list), 
                   len(decoder_vecs_list), len(h_c_values_list)]
    if len(set(list_lengths)) > 1:
        raise ValueError(
            f"Mismatched lengths in decoder cache flattened format: "
            f"layers={list_lengths[0]}, positions={list_lengths[1]}, "
            f"feat_ids={list_lengths[2]}, decoder_vecs={list_lengths[3]}, "
            f"h_c_values={list_lengths[4]}"
        )

    per_layer: Dict[int, Tuple[List[int], List[int], List[torch.Tensor], List[float]]] = {}
    for layer, pos, fid, dvec, hval in zip(layers_list, positions_list, feat_ids_list, decoder_vecs_list, h_c_values_list):
        layer = int(layer)
        if layer not in per_layer:
            per_layer[layer] = ([], [], [], [])
        per_layer[layer][0].append(int(pos))
        per_layer[layer][1].append(int(fid))
        per_layer[layer][2].append(dvec)
        per_layer[layer][3].append(float(hval))

    # Convert lists to tensors
    normalized: Dict[int, Tuple[List[int], List[int], torch.Tensor, torch.Tensor]] = {}
    for layer, (poss, fids, dvecs, hvals) in per_layer.items():
        dvecs_t = torch.stack(dvecs) if dvecs else torch.empty(0, device=device)
        hvals_t = torch.tensor(hvals, device=device, dtype=dtype) if hvals else torch.empty(0, device=device)
        normalized[layer] = (poss, fids, dvecs_t, hvals_t)

    return normalized


def _prepare_heterogeneous_layer_data(
    model,
    batch_items: List[Dict],
) -> Dict[int, Dict[str, Any]]:
    """Prepare union encoder/decoder data per layer for heterogeneous steering.

    Optimized to deduplicate decoder cache normalization and decoder map building
    by cluster_id, since items from the same cluster share the same decoder_cache.
    """
    layer_data: Dict[int, Dict[str, Any]] = {}

    # OPTIMIZATION: Deduplicate decoder cache normalization by cluster_id
    # Items from the same cluster share the same decoder_cache
    cluster_to_normalized_cache: Dict[Any, Dict] = {}
    cluster_to_decoder_maps: Dict[Any, Dict[int, Dict[int, torch.Tensor]]] = {}
    item_cluster_ids: List[Any] = []

    for item in batch_items:
        c_id = item.get("cluster_id")
        item_cluster_ids.append(c_id)

        if c_id is not None and c_id in cluster_to_normalized_cache:
            # Already normalized this cluster's cache
            continue

        # Normalize decoder cache (only once per cluster)
        raw_cache = item.get("decoder_cache", {})
        normalized = _normalize_decoder_cache_to_per_layer(raw_cache, model.cfg.device, model.cfg.dtype)

        if c_id is not None:
            cluster_to_normalized_cache[c_id] = normalized
            # Pre-build decoder maps for all layers (once per cluster)
            cluster_to_decoder_maps[c_id] = {}
            for layer, (_, feat_ids_list, decoder_vecs, _) in normalized.items():
                cluster_to_decoder_maps[c_id][layer] = {
                    int(fid): decoder_vecs[i] for i, fid in enumerate(feat_ids_list)
                }
        else:
            # Fallback for items without cluster_id (shouldn't happen in normal use)
            cluster_to_normalized_cache[id(item)] = normalized
            cluster_to_decoder_maps[id(item)] = {}
            for layer, (_, feat_ids_list, decoder_vecs, _) in normalized.items():
                cluster_to_decoder_maps[id(item)][layer] = {
                    int(fid): decoder_vecs[i] for i, fid in enumerate(feat_ids_list)
                }
            item_cluster_ids[-1] = id(item)  # Use object id as fallback key

    for item_idx, item in enumerate(batch_items):
        for layer, pos, feat_id, h_c_val in item["features"]:
            if layer not in layer_data:
                layer_data[layer] = {
                    "feat_ids": set(),
                    "per_item": defaultdict(list),
                }
            layer_data[layer]["feat_ids"].add(int(feat_id))
            layer_data[layer]["per_item"][item_idx].append((pos, int(feat_id), float(h_c_val)))

    prepared: Dict[int, Dict[str, Any]] = {}
    for layer, data in layer_data.items():
        union_feat_ids = sorted(list(data["feat_ids"]))
        union_feat_ids_t = torch.tensor(
            union_feat_ids, device=model.cfg.device, dtype=torch.long
        )

        # Get encoder weights (using shared utility)
        W_enc_subset, b_enc_subset = utils.get_encoder_weights(model, layer, union_feat_ids_t)

        feat_to_union = {fid: idx for idx, fid in enumerate(union_feat_ids)}

        per_item_entries: List[List[Dict[str, Any]]] = []
        for item_idx in range(len(batch_items)):
            item_entries: List[Dict[str, Any]] = []
            c_id = item_cluster_ids[item_idx]

            # OPTIMIZATION: Use pre-built decoder map from cluster cache
            dec_map = cluster_to_decoder_maps.get(c_id, {}).get(layer, {})
            if not dec_map:
                per_item_entries.append(item_entries)
                continue

            for tup in data["per_item"].get(item_idx, []):
                pos, feat_id, h_c_val = tup
                if feat_id not in feat_to_union or feat_id not in dec_map:
                    continue
                item_entries.append(
                    {
                        "pos": int(pos),
                        "union_idx": feat_to_union[feat_id],
                        "decoder_vec": dec_map[feat_id],
                        "h_c_val": float(h_c_val),
                    }
                )
            per_item_entries.append(item_entries)

        prepared[layer] = {
            "union_feat_ids": union_feat_ids_t,
            "W_enc": W_enc_subset,
            "b_enc": b_enc_subset,
            "per_item_entries": per_item_entries,
        }

    return prepared


def prepare_heterogeneous_layer_metadata(
    model,
    batch_items: List[Dict],
    max_seq_len: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, Dict[str, Any]], List[float]]:
    """Pre-compute layer metadata for heterogeneous steering.

    This can be called once per batch and reused across multiple epsilon values.

    Args:
        model: ReplacementModel
        batch_items: List of dicts with keys: token_ids, features, decoder_cache
        max_seq_len: Fixed padding length. If None, uses max length in batch.

    Returns:
        Tuple of (batch_tokens, attention_mask, layer_metadata, h_c_norms)
    """
    batch_size = len(batch_items)
    if batch_size == 0:
        raise ValueError("Empty batch_items")

    # Compute per-item h_c_norm
    h_c_norms = []
    for item in batch_items:
        h_c_vals = [feat[3] for feat in item["features"]]
        if h_c_vals:
            h_c_tensor = torch.as_tensor(h_c_vals, device=model.cfg.device, dtype=model.cfg.dtype)
            norm_val = torch.linalg.norm(h_c_tensor).item()
            norm_val = norm_val if norm_val != 0 else 1.0
        else:
            norm_val = 1.0
        h_c_norms.append(norm_val)

    # Pad sequences
    pad_token_id = 0
    if hasattr(model, "tokenizer") and model.tokenizer is not None and model.tokenizer.pad_token_id is not None:
        pad_token_id = model.tokenizer.pad_token_id

    batch_token_ids = [it["token_ids"] for it in batch_items]
    batch_tokens, attention_mask = utils.create_padded_batch(
        batch_token_ids, pad_token_id, model.cfg.device, max_seq_len
    )

    # Prepare layer metadata
    layer_metadata = _prepare_heterogeneous_layer_data(model, batch_items)

    return batch_tokens, attention_mask, layer_metadata, h_c_norms


def run_heterogeneous_steered_pass(
    model,
    batch_items: List[Dict],
    steering_method: str,
    epsilon: float,
    max_seq_len: int = None,
    precomputed_metadata: Tuple = None,
):
    """Batched forward pass where each item can have different steering features.

    Args:
        model: ReplacementModel
        batch_items: List of dicts with keys: token_ids, features, decoder_cache
        steering_method: Steering method name
        epsilon: Steering strength
        max_seq_len: Fixed padding length. If None, uses max length in batch.
        precomputed_metadata: Optional tuple from prepare_heterogeneous_layer_metadata.
            If provided, skips the expensive metadata computation.

    Returns:
        logits: [batch, max_seq_len, vocab_size]
        attention_mask: [batch, max_seq_len] (1 for valid, 0 for padding)
    """
    batch_size = len(batch_items)
    if batch_size == 0:
        raise ValueError("Empty batch_items provided to run_heterogeneous_steered_pass")

    # Use precomputed metadata if available, otherwise compute
    if precomputed_metadata is not None:
        batch_tokens, attention_mask, layer_metadata, h_c_norms = precomputed_metadata
    else:
        with profiler.timed("hetero:compute_h_c_norms"):
            # Compute per-item h_c_norm
            h_c_norms = []
            for item in batch_items:
                h_c_vals = [feat[3] for feat in item["features"]]
                if h_c_vals:
                    h_c_tensor = torch.as_tensor(h_c_vals, device=model.cfg.device, dtype=model.cfg.dtype)
                    norm_val = torch.linalg.norm(h_c_tensor).item()
                    norm_val = norm_val if norm_val != 0 else 1.0
                else:
                    norm_val = 1.0
                h_c_norms.append(norm_val)

        with profiler.timed("hetero:create_padded_batch"):
            pad_token_id = 0
            if hasattr(model, "tokenizer") and model.tokenizer is not None and model.tokenizer.pad_token_id is not None:
                pad_token_id = model.tokenizer.pad_token_id

            batch_token_ids = [it["token_ids"] for it in batch_items]
            batch_tokens, attention_mask = utils.create_padded_batch(
                batch_token_ids, pad_token_id, model.cfg.device, max_seq_len
            )

        with profiler.timed("hetero:prepare_layer_metadata"):
            layer_metadata = _prepare_heterogeneous_layer_data(model, batch_items)

    activation_cache: Dict[int, torch.Tensor] = {}
    hooks = []

    # Activation caching hooks
    for layer, meta in layer_metadata.items():
        W_enc = meta["W_enc"]
        b_enc = meta["b_enc"]

        def make_activation_hook(layer_id: int, W: torch.Tensor, b: torch.Tensor):
            def hook_fn(activations, hook=None):
                residual = activations
                pre_acts = torch.einsum("bsd,fd->bsf", residual, W) + b
                acts = model.transcoders.apply_activation_function(layer_id, pre_acts)
                activation_cache[layer_id] = acts.detach()
                return activations

            return hook_fn

        hooks.append((f"blocks.{layer}.{model.feature_input_hook}", make_activation_hook(layer, W_enc, b_enc)))

    # Steering hooks
    for layer, meta in layer_metadata.items():
        per_item_entries = meta["per_item_entries"]

        def make_steering_hook(layer_id: int, entries_per_item: List[List[Dict[str, Any]]]):
            # Pre-compute tensors for vectorized steering
            # Group all valid entries for batch processing
            all_item_idxs = []
            all_positions = []
            all_union_idxs = []
            all_h_c_vals = []
            all_h_c_norms_list = []
            all_decoder_vecs = []

            for item_idx, item_entries in enumerate(entries_per_item):
                if not item_entries:
                    continue
                h_c_norm = h_c_norms[item_idx]
                for entry in item_entries:
                    pos = entry["pos"]
                    if pos <= 0:  # Skip BOS; bounds check done at runtime
                        continue
                    all_item_idxs.append(item_idx)
                    all_positions.append(pos)
                    all_union_idxs.append(entry["union_idx"])
                    all_h_c_vals.append(entry["h_c_val"])
                    all_h_c_norms_list.append(h_c_norm)
                    all_decoder_vecs.append(entry["decoder_vec"])

            if not all_item_idxs:
                # No entries to process
                def noop_hook(activations, hook=None):
                    return activations
                return noop_hook

            # Pre-convert to tensors
            item_idxs_t = torch.tensor(all_item_idxs, device=model.cfg.device, dtype=torch.long)
            positions_t = torch.tensor(all_positions, device=model.cfg.device, dtype=torch.long)
            union_idxs_t = torch.tensor(all_union_idxs, device=model.cfg.device, dtype=torch.long)
            h_c_vals_t = torch.tensor(all_h_c_vals, device=model.cfg.device, dtype=model.cfg.dtype)
            h_c_norms_t = torch.tensor(all_h_c_norms_list, device=model.cfg.device, dtype=model.cfg.dtype)
            decoder_vecs_t = torch.stack(all_decoder_vecs)  # [n_entries, d_model]

            def hook_fn(activations, hook=None):
                if layer_id not in activation_cache:
                    return activations
                acts = activation_cache[layer_id]  # [batch, seq, n_union]
                seq_len = activations.shape[1]

                # Filter entries within sequence bounds
                valid_mask = positions_t < seq_len
                if not valid_mask.any():
                    return activations

                valid_item_idxs = item_idxs_t[valid_mask]
                valid_positions = positions_t[valid_mask]
                valid_union_idxs = union_idxs_t[valid_mask]
                valid_h_c_vals = h_c_vals_t[valid_mask]
                valid_h_c_norms = h_c_norms_t[valid_mask]
                valid_decoder_vecs = decoder_vecs_t[valid_mask]

                # Get current activation values (vectorized)
                current_vals = acts[valid_item_idxs, valid_positions, valid_union_idxs]

                # Compute delta scalars (vectorized for common steering methods)
                delta_scalars = utils.compute_steering_delta_vectorized(
                    current_vals, valid_h_c_vals, valid_h_c_norms, steering_method, epsilon
                )

                # Compute weighted decoder vectors: [n_entries, d_model]
                weighted_deltas = valid_decoder_vecs * delta_scalars.unsqueeze(-1)

                # Accumulate deltas using index_add for efficiency
                deltas = torch.zeros_like(activations)
                # Create flat indices for batch + position -> unique index
                flat_idxs = valid_item_idxs * seq_len + valid_positions

                # Use scatter_add on flattened view
                d_model = activations.shape[2]
                deltas_flat = deltas.view(-1, d_model)  # [batch * seq_len, d_model]
                deltas_flat.index_add_(0, flat_idxs, weighted_deltas)

                return activations + deltas

            return hook_fn

        hooks.append((f"blocks.{layer}.{model.feature_output_hook}", make_steering_hook(layer, per_item_entries)))

    with profiler.timed("hetero:model_forward"):
        with torch.inference_mode():
            with model.hooks(hooks):
                # attention_mask is supported via HookedTransformer.forward inheritance
                # This ensures padding tokens are properly masked in attention computation
                logits = model(batch_tokens, attention_mask=attention_mask)

    return logits, attention_mask


# =============================================================================
# Baseline Computation
# =============================================================================

def compute_per_token_centered_logits_batched(
    logits: torch.Tensor,
    batch_cont_info: List[Tuple[List[int], int]],
    return_per_token: bool = True,
) -> List[Tuple[List[float], float]]:
    """Compute per-token centered logits for a batch (vectorized).

    Centered logit = logit(target_token) - mean(logits across vocabulary)

    Args:
        logits: Batched logits [batch_size, max_seq_len, vocab_size]
        batch_cont_info: List of (continuation_token_ids, continuation_start) tuples

    Returns:
        List of (per_token_centered_logits, mean_centered_logit) tuples, one per batch item
    """
    results = []
    seq_len = logits.shape[1]

    for batch_idx, (cont_ids, cont_start) in enumerate(batch_cont_info):
        if not cont_ids:
            results.append(([], 0.0))
            continue

        # Determine valid positions (within sequence bounds)
        n_tokens = len(cont_ids)
        valid_len = min(n_tokens, seq_len - cont_start)
        if valid_len <= 0:
            results.append(([], 0.0))
            continue

        # Extract logits for all continuation positions at once
        positions = torch.arange(cont_start, cont_start + valid_len, device=logits.device)
        cont_logits = logits[batch_idx, positions, :].float()  # [valid_len, vocab_size]

        # Compute mean across vocabulary for each position (vectorized)
        mean_logits = cont_logits.mean(dim=-1)  # [valid_len]

        # Get target token logits (vectorized)
        token_ids = torch.tensor(cont_ids[:valid_len], device=logits.device, dtype=torch.long)
        target_logits = cont_logits[torch.arange(valid_len, device=logits.device), token_ids]  # [valid_len]

        # Compute centered logits (vectorized)
        centered_logits = target_logits - mean_logits  # [valid_len]

        mean_centered = float(centered_logits.mean().item())
        per_token_centered = centered_logits.tolist() if return_per_token else []
        results.append((per_token_centered, mean_centered))

    return results


def compute_branch_log_probs_batch(
    model,
    branches: List[Dict],
    logger,
    batch_size: int = 32,
    max_seq_len: int = None,
    desc: str | None = None,
    progress=None,
    store_per_token: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """Compute baseline log-probabilities for all branches in a batch.

    This uses run_batched_steered_pass_on_the_fly with empty steering to ensure
    consistency with steered passes (same padding/batching artifacts).

    Args:
        model: ReplacementModel instance
        branches: List of branch dictionaries with full_token_ids and continuation_token_ids
        logger: Logger instance
        batch_size: Batch size for processing
        max_seq_len: Fixed padding length. If None, uses max length in batch.

    Returns:
        {branch_id: {"log_P_original": ..., "per_token_log_probs_original": ...}}
    """
    results = {}
    total_branches = len(branches)
    total_batches = max(1, (total_branches + batch_size - 1) // batch_size) if total_branches else 0
    if logger:
        logger.info(f"  Baseline log_P batching: {total_branches} branches -> {total_batches} batches (batch_size={batch_size})")

    if progress is None:
        progress_desc = desc or "Computing branch log_P"
        iterator = tqdm(
            range(0, total_branches, batch_size),
            total=total_batches,
            desc=progress_desc,
            leave=False,
            dynamic_ncols=True,
        )
    else:
        iterator = range(0, total_branches, batch_size)

    for i in iterator:
        batch = branches[i:i+batch_size]
        batch_tokens = [b.get("full_token_ids", []) for b in batch]
        
        # Validate tokens
        if any(not t for t in batch_tokens):
            raise ValueError("Missing full_token_ids for some branches")

        # Prepare continuation info
        batch_cont_info = []
        for b in batch:
            full_ids = b["full_token_ids"]
            cont_ids = b["continuation_token_ids"]
            cont_start = len(full_ids) - len(cont_ids)
            batch_cont_info.append((cont_ids, cont_start))

        # Run forward pass using SHARED function to ensure consistency
        # Pass empty features/caches -> no hooks attached, but same padding logic
        logits, _ = run_batched_steered_pass_on_the_fly(
            model, batch_tokens, 
            features=[], 
            encoder_cache={}, 
            decoder_cache={}, 
            steering_method="additive", 
            epsilon=0.0,
            max_seq_len=max_seq_len
        )

        # Compute centered logits for the batch (avoid per-token lists unless needed)
        centered_logits_batch = compute_per_token_centered_logits_batched(
            logits, batch_cont_info, return_per_token=store_per_token
        )

        # Compute metrics
        for j, b in enumerate(batch):
            branch_id = b["branch_id"]
            cont_ids, cont_start = batch_cont_info[j]

            # 1. Full continuation log probability (vectorized)
            seq_len = logits.shape[1]
            n_tokens = len(cont_ids)
            valid_len = min(n_tokens, seq_len - cont_start)

            if valid_len > 0:
                positions = torch.arange(cont_start, cont_start + valid_len, device=logits.device)
                cont_logits = logits[j, positions, :].float()  # [valid_len, vocab_size]
                log_probs = F.log_softmax(cont_logits, dim=-1)  # [valid_len, vocab_size]
                token_ids_tensor = torch.tensor(cont_ids[:valid_len], device=logits.device, dtype=torch.long)
                selected_log_probs = log_probs[torch.arange(valid_len, device=logits.device), token_ids_tensor]
                log_P_original = float(selected_log_probs.sum().item())

                # Target logits/probs (teacher-forced), aggregated over continuation tokens
                target_logits = cont_logits[torch.arange(valid_len, device=logits.device), token_ids_tensor]  # [valid_len]
                mean_target_logit = float(target_logits.mean().item())

                target_probs = torch.exp(selected_log_probs)  # [valid_len]
                mean_target_prob = float(target_probs.mean().item())
            else:
                log_P_original = 0.0
                mean_target_logit = 0.0
                mean_target_prob = 0.0

            # 2. Centered logits (already computed)
            per_token_centered_logits, mean_centered_logit = centered_logits_batch[j]

            entry = {
                "log_P_original": log_P_original,
                "mean_centered_logit_original": mean_centered_logit,
                "mean_target_logit_original": mean_target_logit,
                "mean_target_prob_original": mean_target_prob,
            }
            if store_per_token and valid_len > 0:
                entry["per_token_log_probs_original"] = selected_log_probs.tolist()
                entry["per_token_centered_logits_original"] = per_token_centered_logits
                entry["per_token_target_logits_original"] = target_logits.tolist()
                entry["per_token_target_probs_original"] = torch.exp(selected_log_probs).tolist()
            results[branch_id] = entry

        if progress is not None:
            progress.update(1)
        
        # Clear memory after each batch
        del logits
        # Avoid calling gc.collect() in tight loops; throttle any optional cleanup.
        maybe_clear_memory(gc_collect=False, cuda_empty_cache=False, min_interval_s=120.0, logger=logger, tag="baseline_batch")

    return results


def compute_baseline_metadata(
    branches: List[Dict],
    branch_log_probs: Dict[int, Dict[str, Any]],
) -> Dict[int, Dict]:
    """Compute baseline metadata (log_P_original) for each branch.

    This is used with on-the-fly steering which computes activations dynamically.
    Only the log_P_original values are needed, not the activation maps.

    Args:
        branches: List of branch dictionaries
        branch_log_probs: {branch_id: {"log_P_original": ..., ...}} computed on-the-fly

    Returns:
        metadata: {branch_id: {"log_P_original": ...,
                   "mean_centered_logit_original": ..., ...}}
    """
    metadata = {}

    for branch in branches:
        branch_id = branch["branch_id"]
        if branch_id not in branch_log_probs:
            continue
        if "error" in branch_log_probs[branch_id]:
            continue

        lp = branch_log_probs[branch_id]
        entry = {
            "log_P_original": lp["log_P_original"],
            "mean_centered_logit_original": lp.get("mean_centered_logit_original", 0.0),
            "mean_target_logit_original": lp.get("mean_target_logit_original", 0.0),
            "mean_target_prob_original": lp.get("mean_target_prob_original", 0.0),
        }
        # Include per-token fields only if they were computed (reduces memory & GC pressure).
        if "per_token_log_probs_original" in lp:
            entry["per_token_log_probs_original"] = lp["per_token_log_probs_original"]
        if "per_token_centered_logits_original" in lp:
            entry["per_token_centered_logits_original"] = lp["per_token_centered_logits_original"]
        if "per_token_target_logits_original" in lp:
            entry["per_token_target_logits_original"] = lp["per_token_target_logits_original"]
        if "per_token_target_probs_original" in lp:
            entry["per_token_target_probs_original"] = lp["per_token_target_probs_original"]
        metadata[branch_id] = entry

    return metadata


# =============================================================================
# Steered Generation (Sampling)
# =============================================================================

def generate_steered_sequences(
    model,
    prefix_token_ids: List[int],
    features: List[Tuple[int, int, int, float]],
    encoder_cache: Dict[int, Dict[str, torch.Tensor]],
    decoder_cache: Dict[int, Tuple[List[int], List[int], torch.Tensor, torch.Tensor]],
    steering_method: str,
    epsilon: float,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    num_samples: int = 1,
    max_seq_len: int = None,
) -> List[Dict[str, Any]]:
    """Generate sequences with steered sampling using on-the-fly steering hooks.

    This function performs autoregressive generation with steering applied at each step.
    Unlike teacher-forcing evaluation, this actually samples new tokens from the steered
    distribution to see what the model generates.

    Args:
        model: ReplacementModel
        prefix_token_ids: Prefix tokens WITH BOS at position 0
        features: List of (layer, pos, feat_id, h_c_val) tuples for steering
        encoder_cache: Pre-loaded encoder weights
        decoder_cache: Pre-computed decoder vectors
        steering_method: Steering method name
        epsilon: Steering strength
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (1.0 = no temperature)
        top_k: Top-k sampling (0 = disabled)
        top_p: Nucleus sampling (1.0 = disabled)
        num_samples: Number of sequences to sample
        max_seq_len: Fixed max sequence length for batching

    Returns:
        List of dicts with keys:
            - generated_tokens: List of generated token IDs
            - generated_text: Decoded text (if model has tokenizer)
            - full_token_ids: prefix + generated tokens
    """
    hooks = []
    activation_cache = {}

    # Compute h_c norm
    h_c_values = [h_c_val for _, _, _, h_c_val in features]
    if h_c_values:
        h_c_tensor = torch.tensor(h_c_values, device=model.cfg.device, dtype=model.cfg.dtype)
        h_c_norm = torch.linalg.norm(h_c_tensor).item()
    else:
        h_c_norm = 1.0
    if h_c_norm == 0:
        h_c_norm = 1.0

    # Group interventions by layer
    interventions_by_layer = {}
    for layer, pos, feat_id, h_c_val in features:
        if layer not in interventions_by_layer:
            interventions_by_layer[layer] = []
        interventions_by_layer[layer].append((pos, feat_id, h_c_val))

    # Normalize decoder_cache to per-layer mapping if needed
    decoder_cache = _normalize_decoder_cache_to_per_layer(
        decoder_cache, model.cfg.device, model.cfg.dtype
    )

    # Phase 1: Column-wise activation caching hooks
    for layer in encoder_cache.keys():
        hook_name = f"blocks.{layer}.{model.feature_input_hook}"
        hooks.append((hook_name, get_feature_activation_caching_hook(
            model, layer, encoder_cache, activation_cache
        )))

    # Phase 2: On-the-fly delta application hooks
    for layer, interventions in interventions_by_layer.items():
        hook_name = f"blocks.{layer}.{model.feature_output_hook}"
        hooks.append((hook_name, make_on_the_fly_steering_hook(
            model, layer, interventions, activation_cache,
            decoder_cache, steering_method, epsilon, h_c_norm
        )))

    results = []

    with torch.inference_mode():
        with model.hooks(hooks):
            for sample_idx in range(num_samples):
                # Start with prefix
                current_tokens = prefix_token_ids.copy()
                generated = []

                for step in range(max_new_tokens):
                    # Convert to tensor
                    input_tensor = torch.tensor([current_tokens], device=model.cfg.device, dtype=torch.long)

                    # Forward pass with steering
                    logits = model(input_tensor)

                    # Get logits for next token (last position)
                    next_token_logits = logits[0, -1, :] / temperature

                    # Apply top-k filtering
                    if top_k > 0:
                        indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                        next_token_logits[indices_to_remove] = -float('Inf')

                    # Apply top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = -float('Inf')

                    # Sample from the filtered distribution
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()

                    # Add to sequence
                    generated.append(next_token)
                    current_tokens.append(next_token)

                    # Check for EOS token
                    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
                        if model.tokenizer.eos_token_id is not None and next_token == model.tokenizer.eos_token_id:
                            break

                    # Check max length
                    if max_seq_len is not None and len(current_tokens) >= max_seq_len:
                        break

                # Decode generated text if tokenizer available
                generated_text = None
                if hasattr(model, 'tokenizer') and model.tokenizer is not None:
                    try:
                        generated_text = model.tokenizer.decode(generated, skip_special_tokens=True)
                    except:
                        generated_text = str(generated)

                results.append({
                    'sample_idx': sample_idx,
                    'generated_tokens': generated,
                    'generated_text': generated_text,
                    'full_token_ids': current_tokens,
                })

    return results

