#!/usr/bin/env -S uv run python
"""Extract token-wise logit differences for heatmap visualization.

This script runs steering on specific cloze/config combinations and extracts
per-token logit differences (steered - original) for visualization.

Usage:
    python extract_tokenwise_logit_diff.py --cloze cloze_0028 --config beta1.0_gamma0.7 --output heatmap_data.json
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
CIRCUIT_TRACER_PATH = Path(__file__).resolve().parents[1] / "circuit-tracer"
sys.path.insert(0, str(CIRCUIT_TRACER_PATH))

from utils.data_utils import load_json, save_json
from utils.attribution_pooling import load_pooled_attributions
from utils.logging_utils import setup_logger
from utils.model_backend import get_model_device, resolve_backend, resolve_stage_backend
from circuit_tracer import ReplacementModel

# Import from refactored modules
import importlib.util
from pathlib import Path as _Path

def _import_module(name, file_path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

_module_dir = _Path(__file__).parent
graph = _import_module("7c_graph", _module_dir / "7c_graph.py")
steering = _import_module("7c_steering", _module_dir / "7c_steering.py")
metrics = _import_module("7c_metrics", _module_dir / "7c_metrics.py")
utils = _import_module("7c_utils", _module_dir / "7c_utils.py")


def resolve_cli_stage7c_backend(model_name: str, backend_arg: str) -> str:
    """Resolve tokenwise Stage 7c backend through the shared stage helper."""

    config = {
        "attribution": {"backend": backend_arg},
        "stage_7c_steering": {},
    }
    return resolve_backend(
        model_name,
        resolve_stage_backend(config, "stage_7c_steering"),
    )


def load_prefix_data(
    prefix_id: str,
    attribution_graphs_dir: Path,
    samples_dir: Path,
    embeddings_dir: Path,
    logger,
    pooling: str = "mean",
) -> Dict[str, Any]:
    """Load data needed for combined-distance medoid baseline."""
    logger.info(f"Loading data for prefix: {prefix_id}")
    
    branches_file = samples_dir / f"{prefix_id}_branches.json"
    branches_data = load_json(branches_file)
    
    path_probs = []
    for cont in branches_data.get("continuations", []):
        path_probs.append(cont.get("probability", 1.0))
    path_probs = np.array(path_probs)
    
    prefix_context_file = attribution_graphs_dir / f"{prefix_id}_prefix_context.pt"
    pooled_attributions = load_pooled_attributions(
        prefix_context_file,
        pooling=pooling,
        meta_file=attribution_graphs_dir / f"{prefix_id}_attribution.json",
    )
    aggregated_attributions = pooled_attributions.values
    
    embeddings_file = embeddings_dir / f"{prefix_id}_embeddings.npy"
    embeddings = np.load(embeddings_file)
    
    n_samples = len(branches_data.get("continuations", []))
    logger.info(f"  Loaded {n_samples} samples")
    
    return {
        "prefix_id": prefix_id,
        "branches_data": branches_data,
        "aggregated_attributions": aggregated_attributions,
        "embeddings": embeddings,
        "path_probs": path_probs,
        "n_samples": n_samples,
    }


def load_rd_sweep_results(clustering_file: Path) -> Tuple[List[Dict], int]:
    """Load RD sweep results."""
    data = load_json(clustering_file)
    grid = data.get("grid", [])
    sweep_config = data.get("sweep_config", {}) or {}
    K_clamp = int(sweep_config.get("K_clamp", sweep_config.get("K_max", 20)))
    return grid, K_clamp


def compute_per_token_logits_detailed(
    logits: torch.Tensor,
    cont_ids: List[int],
    cont_start: int,
    max_seq_len: int,
) -> Dict[str, Any]:
    """Extract detailed per-token logit information.
    
    Returns:
        Dict with:
            - target_logits: logits for the target tokens
            - mean_logits: mean logit across vocab at each position
            - centered_logits: target - mean
            - max_logits: max logit at each position
            - token_ids: the target token IDs
            - positions: absolute positions in the sequence
    """
    valid_len = min(len(cont_ids), max_seq_len - cont_start) if cont_start < max_seq_len else 0
    
    if valid_len <= 0:
        return {"error": "no_valid_positions"}
    
    positions = torch.arange(cont_start, cont_start + valid_len, device=logits.device)
    cont_logits = logits[positions, :].float()  # [valid_len, vocab_size]
    
    # Get target token logits
    token_ids = torch.tensor(cont_ids[:valid_len], device=logits.device, dtype=torch.long)
    target_logits = cont_logits[torch.arange(valid_len, device=logits.device), token_ids]
    
    # Compute statistics
    mean_logits = cont_logits.mean(dim=-1)
    max_logits = cont_logits.max(dim=-1).values
    centered_logits = target_logits - mean_logits
    
    # Log probabilities
    log_probs = F.log_softmax(cont_logits, dim=-1)
    target_log_probs = log_probs[torch.arange(valid_len, device=logits.device), token_ids]
    
    return {
        "target_logits": target_logits.tolist(),
        "mean_logits": mean_logits.tolist(),
        "centered_logits": centered_logits.tolist(),
        "max_logits": max_logits.tolist(),
        "target_log_probs": target_log_probs.tolist(),
        "token_ids": token_ids.tolist(),
        "positions": positions.tolist(),
        "total_log_prob": float(target_log_probs.sum().item()),
    }


def run_steering_with_tokenwise_output(
    model,
    branches: List[Dict],
    decoder_cache: Dict[int, Dict],
    encoder_cache: Dict[int, Dict],
    epsilons: List[float],
    top_B: int,
    steering_method: str,
    hc_selection: str,
    max_samples_per_cluster: int,
    max_seq_len: int,
    logger,
) -> Dict[str, Any]:
    """Run steering and extract per-token logit differences."""
    
    cluster_ids = sorted(decoder_cache.keys())
    if not cluster_ids:
        return {"error": "no_clusters"}
    
    # Group branches by cluster
    branches_by_cluster = {c: [] for c in cluster_ids}
    for b in branches:
        c = b.get("cluster_id", -1)
        if c in branches_by_cluster:
            branches_by_cluster[c].append(b)
    
    # Limit samples per cluster
    rng = np.random.RandomState(42)
    for c in cluster_ids:
        if max_samples_per_cluster > 0 and len(branches_by_cluster[c]) > max_samples_per_cluster:
            idx = rng.choice(len(branches_by_cluster[c]), max_samples_per_cluster, replace=False)
            branches_by_cluster[c] = [branches_by_cluster[c][i] for i in idx]
    
    # Pre-select features for each cluster
    selected_features_cache = {}
    for c in cluster_ids:
        if c in decoder_cache:
            selected_features_cache[c] = graph.select_features_with_hc_selection(
                decoder_cache[c], top_B, hc_selection
            )
    
    results = {
        "clusters": {},
        "metadata": {
            "epsilons": epsilons,
            "top_B": top_B,
            "steering_method": steering_method,
            "hc_selection": hc_selection,
        }
    }
    
    device = get_model_device(model)
    
    for steer_c in tqdm(cluster_ids, desc="Processing clusters"):
        if steer_c not in selected_features_cache:
            continue
        
        h_c_vals, _, layers, positions, feat_ids = selected_features_cache[steer_c]
        if not h_c_vals:
            continue
        
        features = [(layers[i], positions[i], feat_ids[i], h_c_vals[i]) for i in range(len(h_c_vals))]
        c_decoder_cache = decoder_cache[steer_c]
        c_encoder_cache = encoder_cache[steer_c]  # Per-cluster encoder cache
        
        cluster_branches = branches_by_cluster[steer_c]
        if not cluster_branches:
            continue
        
        cluster_results = {
            "n_branches": len(cluster_branches),
            "branches": []
        }
        
        for branch in cluster_branches:
            full_token_ids = branch["full_token_ids"]
            cont_ids = branch["continuation_token_ids"]
            cont_start = len(full_token_ids) - len(cont_ids)
            
            # Get tokens for this branch
            tokens_tensor = torch.tensor([full_token_ids], device=device, dtype=torch.long)
            
            branch_result = {
                "branch_id": branch["branch_id"],
                "continuation_text": branch.get("continuation_text", ""),
                "tokens": model.tokenizer.convert_ids_to_tokens(full_token_ids) if hasattr(model, 'tokenizer') else None,
                "continuation_tokens": model.tokenizer.convert_ids_to_tokens(cont_ids) if hasattr(model, 'tokenizer') else None,
                "cont_start": cont_start,
                "epsilon_results": {}
            }
            
            # Run original (no steering, epsilon=0)
            with torch.inference_mode():
                logits_original = model(tokens_tensor)[0]  # [seq_len, vocab_size]
            
            original_details = compute_per_token_logits_detailed(
                logits_original, cont_ids, cont_start, logits_original.shape[0]
            )
            branch_result["original"] = original_details
            
            # Run steered for each epsilon
            for eps in epsilons:
                if abs(eps) < 1e-9:
                    # Skip epsilon=0, use original
                    branch_result["epsilon_results"][str(eps)] = {
                        "steered": original_details,
                        "diff": {k: [0.0] * len(v) if isinstance(v, list) else 0.0 
                                for k, v in original_details.items() if k != "error"}
                    }
                    continue
                
                # Run steered pass
                with torch.inference_mode():
                    logits_steered = steering.run_steered_pass_on_the_fly(
                        model=model,
                        full_token_ids=full_token_ids,
                        features=features,
                        encoder_cache=c_encoder_cache,
                        decoder_cache=c_decoder_cache,
                        steering_method=steering_method,
                        epsilon=eps,
                        max_seq_len=max_seq_len,
                    )[0]  # [seq_len, vocab_size]
                
                steered_details = compute_per_token_logits_detailed(
                    logits_steered, cont_ids, cont_start, logits_steered.shape[0]
                )
                
                # Compute differences
                diff = {}
                if "error" not in original_details and "error" not in steered_details:
                    for key in ["target_logits", "mean_logits", "centered_logits", "max_logits", "target_log_probs"]:
                        if key in original_details and key in steered_details:
                            orig_vals = original_details[key]
                            steer_vals = steered_details[key]
                            diff[key] = [s - o for s, o in zip(steer_vals, orig_vals)]
                    
                    diff["total_log_prob"] = steered_details["total_log_prob"] - original_details["total_log_prob"]
                
                branch_result["epsilon_results"][str(eps)] = {
                    "steered": steered_details,
                    "diff": diff,
                }
            
            cluster_results["branches"].append(branch_result)
        
        results["clusters"][str(steer_c)] = cluster_results
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Extract token-wise logit differences for heatmap")
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--attribution-graphs-dir", type=Path, required=True)
    parser.add_argument("--clustering-dir", type=Path, required=True)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cloze", type=str, required=True, help="Cloze ID (e.g., cloze_0028)")
    parser.add_argument("--config", type=str, required=True, help="Config key (e.g., beta1.0_gamma0.7)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--transcoder", type=str, default="mwhanna/qwen3-8b-transcoders")
    parser.add_argument(
        "--backend",
        choices=["auto", "transformerlens", "nnsight"],
        default="auto",
        help="Stage 7c backend. 'auto' picks nnsight for Gemma3 models.",
    )
    parser.add_argument("--epsilons", type=float, nargs="+", default=[-1.0, 1.0])
    parser.add_argument("--top-B", type=int, default=10)
    parser.add_argument("--steering-method", type=str, default="sign")
    parser.add_argument("--hc-selection", type=str, default="full")
    parser.add_argument("--max-cluster-samples", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=96)
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "max", "sum"],
                        help="Pooling method for attributions")
    args = parser.parse_args()
    
    logger = setup_logger("tokenwise_logit_diff", level=logging.INFO)
    
    logger.info("=" * 60)
    logger.info("EXTRACTING TOKEN-WISE LOGIT DIFFERENCES")
    logger.info("=" * 60)
    logger.info(f"Cloze: {args.cloze}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Epsilons: {args.epsilons}")
    
    # Parse config key
    parts = args.config.split("_")
    beta = float(parts[0].replace("beta", ""))
    gamma = float(parts[1].replace("gamma", ""))
    logger.info(f"Beta: {beta}, Gamma: {gamma}")
    
    # Load model
    logger.info("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend = resolve_cli_stage7c_backend(args.model, args.backend)
    logger.info("Using Stage 7c backend=%s for model=%s", backend, args.model)
    model = ReplacementModel.from_pretrained(
        args.model,
        args.transcoder,
        backend=backend,
        device=device,
        dtype=torch.bfloat16,
        lazy_encoder=True,
        lazy_decoder=False,
    )
    
    # Load prefix data
    prefix_data = load_prefix_data(
        args.cloze,
        args.attribution_graphs_dir,
        args.samples_dir,
        args.embeddings_dir,
        logger,
        pooling=args.pooling,
    )
    
    # Load clustering results
    clustering_file = args.clustering_dir / f"{args.cloze}_sweep_results.json"
    grid, K_clamp = load_rd_sweep_results(clustering_file)
    
    # Find matching config
    target_entry = None
    for entry in grid:
        if abs(entry.get("beta", -1) - beta) < 0.001 and abs(entry.get("gamma", -1) - gamma) < 0.001:
            target_entry = entry
            break
    
    if target_entry is None:
        logger.error(f"Config {args.config} not found in clustering results")
        return
    
    assignments = target_entry.get("assignments", [])
    K = target_entry.get("K", len(target_entry.get("components", {})))
    logger.info(f"Found config: K={K}, {len(assignments)} assignments")
    
    # Load attribution context
    active_features, selected_features = graph.load_attribution_context(
        args.attribution_graphs_dir, args.cloze, use_continuation_attribution=True
    )
    n_features = len(selected_features)
    
    # Build semantic graphs using combined-distance medoid
    semantic_graphs, selected_indices = graph.compute_semantic_graphs_combined_medoid(
        assignments,
        prefix_data["aggregated_attributions"],
        prefix_data["embeddings"],
        gamma=gamma,
        weights=None,
        logger=logger
    )
    
    if not semantic_graphs:
        logger.error("No semantic graphs computed")
        return
    
    logger.info(f"Computed {len(semantic_graphs)} semantic graphs")
    
    # Build branches
    branches = utils.build_branches_from_data(prefix_data["branches_data"], assignments)
    
    # Build decoder cache
    max_top_B = args.top_B * 2
    all_needed_indices = set()
    for cluster_id, H_c in semantic_graphs.items():
        H_c_features = H_c[:n_features]
        abs_vals = np.abs(H_c_features)
        top_indices = np.argsort(abs_vals)[-max_top_B:]
        for idx in top_indices:
            if abs(H_c_features[idx]) >= utils.EPSILON_SMALL:
                all_needed_indices.add(int(idx))
    
    global_decoder_cache = {}
    if all_needed_indices:
        layer_to_indices = {}
        for h_c_idx in all_needed_indices:
            feat_idx = selected_features[h_c_idx].item()
            layer, pos, feat_id = active_features[feat_idx].tolist()
            layer = int(layer)
            if layer not in layer_to_indices:
                layer_to_indices[layer] = []
            layer_to_indices[layer].append((h_c_idx, int(feat_id)))
        
        for layer, idx_list in layer_to_indices.items():
            h_c_indices = [x[0] for x in idx_list]
            feat_ids = [x[1] for x in idx_list]
            feat_ids_t = torch.tensor(feat_ids, device=device, dtype=torch.long)
            dec_vecs = model.transcoders._get_decoder_vectors(layer, feat_ids_t)
            for i, h_c_idx in enumerate(h_c_indices):
                global_decoder_cache[h_c_idx] = dec_vecs[i]
    
    decoder_cache = graph.build_cluster_decoder_cache(
        semantic_graphs, global_decoder_cache, active_features, selected_features,
        max_features=max_top_B
    )
    
    features_by_cluster = {}
    for c, cache_data in decoder_cache.items():
        tuples = [(
            cache_data['layers'][i],
            cache_data['positions'][i],
            cache_data['feat_ids'][i],
            cache_data['h_c_values'][i]
        ) for i in range(len(cache_data['h_c_values']))]
        features_by_cluster[c] = tuples
    
    encoder_cache = graph.precompute_cluster_encoder_weights(
        model, features_by_cluster, device
    )
    
    # Run steering with token-wise output
    logger.info("Running steering...")
    results = run_steering_with_tokenwise_output(
        model=model,
        branches=branches,
        decoder_cache=decoder_cache,
        encoder_cache=encoder_cache,
        epsilons=args.epsilons,
        top_B=args.top_B,
        steering_method=args.steering_method,
        hc_selection=args.hc_selection,
        max_samples_per_cluster=args.max_cluster_samples,
        max_seq_len=args.max_seq_len,
        logger=logger,
    )
    
    # Add metadata
    results["cloze_id"] = args.cloze
    results["config"] = args.config
    results["beta"] = beta
    results["gamma"] = gamma
    results["K"] = K
    results["prefix"] = prefix_data["branches_data"].get("prefix", "")
    results["question"] = prefix_data["branches_data"].get("question", "")
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, args.output)
    logger.info(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
