#!/usr/bin/env -S uv run python
"""7c_hypotheses.py - Steering Validation Entry Point.

Entry point for steering validation. Runs H4a/H4b/H4c hypothesis tests.
Routes between cross-prefix batching and normal batching based on config.

Usage:
    python 7c_hypotheses.py --samples-dir ... --clustering-dir ... [options]
"""

import sys
import time
import gc
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from itertools import product

import numpy as np
import torch
from scipy.stats import linregress, spearmanr
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
CIRCUIT_TRACER_PATH = Path(__file__).resolve().parents[1] / "circuit-tracer"
sys.path.insert(0, str(CIRCUIT_TRACER_PATH))

from utils.data_utils import load_json, save_json
from utils.logging_utils import setup_logger, get_log_path
from utils.memory_utils import maybe_clear_memory, reset_cuda_state
from circuit_tracer import ReplacementModel

# Import from refactored modules (use importlib since module names start with digits)
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


# =============================================================================
# Sweep Config Parsing
# =============================================================================

def load_sweep_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate sweep config from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    if "sweeps" not in config:
        config["sweeps"] = []

    config.setdefault("sampling", {})
    config["sampling"].setdefault("max_samples_per_cluster", 30)
    config["sampling"].setdefault("max_prefixes", None)

    config.setdefault("validation", {})
    config["validation"].setdefault("freeze_attention", False)
    config["validation"].setdefault("log_details", False)

    # Validate and normalize each sweep using shared utility
    defaults = {
        "h_c_selections": ["full"],
        "top_B": [10],
        "epsilon_values": [-1.0, 0.0, 1.0]
    }
    config["sweeps"] = [
        utils.validate_and_normalize_sweep_config(sweep, defaults)
        for sweep in config["sweeps"]
    ]

    return config


def parse_sweeps_from_main_config(stage_7c_config: Dict[str, Any]) -> Dict[str, Any]:
    """Parse sweep configuration from main config's stage_7c_steering section."""
    sweeps = stage_7c_config.get("sweeps", [])

    global_feature_selection = stage_7c_config.get("feature_selection", "magnitude")
    global_epsilon_values = stage_7c_config.get("epsilon_values", [-1.0, 0.0, 1.0])

    if sweeps:
        # Validate and normalize each sweep using shared utility
        defaults = {
            "h_c_selections": ["full"],
            "top_B": [10],
            "epsilon_values": global_epsilon_values,
            "feature_selection": global_feature_selection
        }
        normalized_sweeps = [
            utils.validate_and_normalize_sweep_config(sweep, defaults)
            for sweep in sweeps
        ]
        config = {
            "sweeps": normalized_sweeps,
            "sampling": {
                "max_samples_per_cluster": stage_7c_config.get("max_cluster_samples", 30),
                "max_prefixes": stage_7c_config.get("max_samples"),
            },
            "validation": {
                "freeze_attention": stage_7c_config.get("freeze_attention", False),
                "log_details": stage_7c_config.get("log_details", False),
            }
        }
    else:
        # Create single sweep config when no sweeps defined
        steering_method = stage_7c_config.get("steering_method", stage_7c_config.get("method", "multiplicative"))

        single_sweep = {
            "name": "single_config",
            "steering_method": steering_method,
            "h_c_selections": [stage_7c_config.get("h_c_selection", "full")],
            "top_B": [stage_7c_config.get("top_B", 10)],
            "epsilon_values": global_epsilon_values,
            "feature_selection": global_feature_selection,
        }

        # Validate and normalize for consistency
        defaults = {
            "h_c_selections": ["full"],
            "top_B": [10],
            "epsilon_values": global_epsilon_values,
            "feature_selection": global_feature_selection
        }
        normalized_sweep = utils.validate_and_normalize_sweep_config(single_sweep, defaults)

        config = {
            "sweeps": [normalized_sweep],
            "sampling": {
                "max_samples_per_cluster": stage_7c_config.get("max_cluster_samples", 30),
                "max_prefixes": stage_7c_config.get("max_samples"),
            },
            "validation": {
                "freeze_attention": stage_7c_config.get("freeze_attention", False),
                "log_details": stage_7c_config.get("log_details", False),
            }
        }

    return config


def _save_hypothesis_outputs(prefix_results: Dict[str, Any], hypotheses: List[str], output_dir: Path, hypothesis_dirs: Dict[str, str]) -> None:
    """Save per-hypothesis results into separate subfolders."""
    prefix_id = prefix_results.get("prefix_id")
    clustering_runs = prefix_results.get("clustering_runs", {})

    for hypothesis in hypotheses:
        folder_name = hypothesis_dirs.get(hypothesis, hypothesis)
        hypothesis_dir = output_dir / folder_name
        hypothesis_dir.mkdir(parents=True, exist_ok=True)

        if hypothesis == "H4A":
            per_prefix = _compact_h4a_prefix_results(prefix_results)
        else:
            per_prefix = {
                "prefix_id": prefix_id,
                "feature_selection": prefix_results.get("feature_selection"),
                "clustering_runs": {}
            }

        for clustering_key, run in clustering_runs.items():
            if hypothesis == "H4A":
                entry = {
                    "beta": run.get("beta"),
                    "gamma": run.get("gamma"),
                    "K": run.get("K"),
                    "n_clusters": run.get("n_clusters"),
                    "n_branches": run.get("n_branches"),
                    "results": {},
                }
                if run.get("selected_indices"):
                    entry["selected_indices"] = {
                        str(cluster_id): int(sample_idx)
                        for cluster_id, sample_idx in run["selected_indices"].items()
                    }

                for sweep_key, result in run.get("results", {}).items():
                    entry["results"][sweep_key] = _compact_h4a_result(result)

                if entry["results"]:
                    per_prefix["clustering_runs"][clustering_key] = entry
                continue

            entry = {
                "beta": run.get("beta"),
                "gamma": run.get("gamma"),
                "n_clusters": run.get("n_clusters"),
                "n_branches": run.get("n_branches"),
            }
            if "timing" in run:
                entry["timing"] = run["timing"]

            if hypothesis == "H4A":
                entry["results"] = run.get("results", {})
            elif hypothesis == "H4A_GEN":
                if "H4a_generation" not in run:
                    continue
                entry["H4a_generation"] = run["H4a_generation"]
            elif hypothesis == "H4C":
                if "H4c_specificity" not in run:
                    continue
                entry["H4c_specificity"] = run["H4c_specificity"]
            elif hypothesis == "H4C_MASS":
                if "H4c_cluster_mass_pairwise" not in run:
                    continue
                entry["H4c_cluster_mass_pairwise"] = run["H4c_cluster_mass_pairwise"]

            per_prefix["clustering_runs"][clustering_key] = entry

        if per_prefix["clustering_runs"]:
            save_json(per_prefix, hypothesis_dir / f"{prefix_id}_sweep_results.json")


def _paper_method_name(prefix_results: Dict[str, Any]) -> str:
    raw = prefix_results.get("method") or prefix_results.get("baseline_method") or "rd"
    method_map = {
        "combined_medoid": "rd",
        "rd": "rd",
        "kmeans": "km_sem",
        "kmeans_medoid": "km_sem",
        "km_sem": "km_sem",
        "single_continuation": "single",
        "single": "single",
    }
    return method_map.get(str(raw), str(raw))


def _curve_value(curve: Dict[Any, Any], epsilon: Any) -> Optional[float]:
    if not isinstance(curve, dict):
        return None

    candidates = [epsilon]
    try:
        epsilon_float = float(epsilon)
        candidates.extend([epsilon_float, str(epsilon_float), str(epsilon)])
    except (TypeError, ValueError):
        candidates.append(str(epsilon))

    for key in candidates:
        if key in curve:
            value = curve[key]
            return float(value) if value is not None else None
    return None


def _compact_h4a_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if "error" in result:
        return {"error": result["error"]}

    compact = {
        "steering_method": result.get("steering_method"),
        "hc_selection": result.get("hc_selection"),
        "top_B": result.get("top_B"),
        "epsilon_values": result.get("epsilon_values", []),
        "per_cluster": {},
    }

    keep_cluster_keys = [
        "centered_logit_diff",
        "centered_logit_corr",
        "centered_logit_spearman",
        "sum_centered_logit_diff",
        "n_samples",
    ]

    for cluster_id, stats in result.get("per_cluster_logit", {}).items():
        compact["per_cluster"][str(cluster_id)] = {
            key: stats[key]
            for key in keep_cluster_keys
            if key in stats
        }

    metric_map = {
        "centered_logit_corr_mean": result.get("mean_logit_corr"),
        "centered_logit_spearman_mean": result.get("mean_logit_spearman"),
    }
    for key, value in metric_map.items():
        if value is not None:
            compact[key] = float(value)

    for epsilon in compact["epsilon_values"]:
        total = 0.0
        saw_value = False
        for stats in compact["per_cluster"].values():
            value = _curve_value(stats.get("sum_centered_logit_diff", {}), epsilon)
            if value is None:
                continue
            total += value
            saw_value = True
        if saw_value:
            compact[f"sum_diff_eps{epsilon}"] = total

    return compact


def _compact_h4a_prefix_results(prefix_results: Dict[str, Any]) -> Dict[str, Any]:
    compact = {
        "prefix_id": prefix_results.get("prefix_id"),
        "method": _paper_method_name(prefix_results),
        "feature_selection": prefix_results.get("feature_selection"),
        "clustering_runs": {},
    }
    if "random_seed" in prefix_results:
        compact["random_seed"] = prefix_results["random_seed"]
    return compact


def _prefix_outputs_exist(prefix_id: str, hypotheses: List[str], output_dir: Path, hypothesis_dirs: Dict[str, str]) -> bool:
    """Return True if all requested per-prefix output files already exist and are non-empty."""
    for hypothesis in hypotheses:
        folder_name = hypothesis_dirs.get(hypothesis, hypothesis)
        out_path = output_dir / folder_name / f"{prefix_id}_sweep_results.json"
        try:
            if (not out_path.exists()) or out_path.stat().st_size == 0:
                return False
        except OSError:
            return False
    return True


def _round_grid_value(value: Any, ndigits: int = 6) -> float:
    """Normalize beta/gamma values for stable manifest matching."""
    return round(float(value), ndigits)


def _load_clustering_manifest(manifest_path: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    """Load a clustering manifest and index entries by prefix."""
    raw_manifest = load_json(manifest_path)
    if isinstance(raw_manifest, dict):
        entries = raw_manifest.get("clusterings", raw_manifest.get("entries", raw_manifest))
        if isinstance(entries, dict):
            entries = [entries]
    elif isinstance(raw_manifest, list):
        entries = raw_manifest
    else:
        raise ValueError(f"Unsupported manifest format in {manifest_path}")

    if not isinstance(entries, list):
        raise ValueError(f"Manifest must contain a list of entries: {manifest_path}")

    by_prefix: Dict[str, List[Dict[str, Any]]] = {}
    prefix_order: List[str] = []

    for order_idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest entry {order_idx} must be an object")

        prefix_id = entry.get("prefix_id")
        beta = entry.get("beta")
        gamma = entry.get("gamma")
        if prefix_id is None or beta is None or gamma is None:
            raise ValueError(
                f"Manifest entry {order_idx} missing one of prefix_id/beta/gamma: {entry}"
            )

        normalized = {
            "prefix_id": str(prefix_id),
            "beta": float(beta),
            "gamma": float(gamma),
            "K": int(entry["K"]) if entry.get("K") is not None else None,
            "order": order_idx,
        }
        if normalized["prefix_id"] not in by_prefix:
            by_prefix[normalized["prefix_id"]] = []
            prefix_order.append(normalized["prefix_id"])
        by_prefix[normalized["prefix_id"]].append(normalized)

    return by_prefix, prefix_order


def _filter_valid_grid_with_manifest(
    prefix_id: str,
    valid_grid: List[Dict[str, Any]],
    prefix_manifest_entries: List[Dict[str, Any]],
    logger,
) -> List[Dict[str, Any]]:
    """Filter a prefix's grid entries to the exact clusterings listed in the manifest."""
    grid_by_key = {}
    for entry in valid_grid:
        beta = entry.get("beta")
        gamma = entry.get("gamma")
        if beta is None or gamma is None:
            continue
        grid_by_key[(_round_grid_value(beta), _round_grid_value(gamma))] = entry

    filtered_grid: List[Dict[str, Any]] = []
    missing_entries: List[Tuple[float, float, Any]] = []
    mismatched_k: List[Tuple[float, float, Any, Any]] = []

    for manifest_entry in prefix_manifest_entries:
        key = (
            _round_grid_value(manifest_entry["beta"]),
            _round_grid_value(manifest_entry["gamma"]),
        )
        grid_entry = grid_by_key.get(key)
        if grid_entry is None:
            missing_entries.append(
                (manifest_entry["beta"], manifest_entry["gamma"], manifest_entry.get("K"))
            )
            continue

        expected_k = manifest_entry.get("K")
        actual_k = grid_entry.get("K", len(grid_entry.get("components", {})))
        if expected_k is not None and actual_k != expected_k:
            mismatched_k.append((manifest_entry["beta"], manifest_entry["gamma"], expected_k, actual_k))
            continue

        filtered_grid.append(grid_entry)

    logger.info(
        f"  Manifest filtered to {len(filtered_grid)}/{len(prefix_manifest_entries)} "
        f"requested configs for {prefix_id}"
    )
    if missing_entries:
        logger.warning(
            f"  Missing {len(missing_entries)} manifest configs for {prefix_id}: "
            f"{missing_entries[:3]}{'...' if len(missing_entries) > 3 else ''}"
        )
    if mismatched_k:
        logger.warning(
            f"  Skipped {len(mismatched_k)} manifest configs due to K mismatch for {prefix_id}: "
            f"{mismatched_k[:3]}{'...' if len(mismatched_k) > 3 else ''}"
        )

    return filtered_grid


def _resolve_primary_sweep_settings(
    sweeps: List[Dict[str, Any]],
    steering_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Resolve the first sweep's steering settings for H4A_GEN/H4C side experiments."""
    first_sweep = sweeps[0] if sweeps else {}
    method = first_sweep.get(
        "steering_method",
        steering_config.get("steering_method", steering_config.get("method", "multiplicative")),
    )

    hc_values = first_sweep.get(
        "h_c_selections",
        first_sweep.get("hc_selections", [steering_config.get("h_c_selection", "full")]),
    )
    if isinstance(hc_values, list):
        hc_selection = hc_values[0] if hc_values else "full"
    else:
        hc_selection = hc_values or "full"

    top_b_values = first_sweep.get("top_B", [steering_config.get("top_B", 10)])
    if isinstance(top_b_values, list):
        top_B = int(top_b_values[0]) if top_b_values else int(steering_config.get("top_B", 10))
    else:
        top_B = int(top_b_values)

    epsilon_values = first_sweep.get(
        "epsilon_values",
        first_sweep.get("epsilons", steering_config.get("epsilon_values", [-0.5, 0.0, 0.5])),
    )
    epsilons = [float(eps) for eps in epsilon_values]
    positive_epsilons = [eps for eps in epsilons if eps > 0]
    if positive_epsilons:
        default_positive = max(positive_epsilons)
    elif epsilons:
        default_positive = max(epsilons)
    else:
        default_positive = 0.5

    return {
        "method": method,
        "hc_selection": hc_selection,
        "top_B": top_B,
        "epsilons": epsilons,
        "positive_epsilon": default_positive,
    }


def _build_selected_cluster_state(
    model,
    cluster_decoder_cache: Dict[int, Dict[str, Any]],
    device: torch.device,
    top_B: int,
    hc_selection: str,
) -> Tuple[Dict[int, List[Tuple[int, int, int, float]]], Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Build a cluster cache restricted to a specific top_B and H_c sign selection."""
    selected_features_by_cluster: Dict[int, List[Tuple[int, int, int, float]]] = {}
    selected_decoder_cache: Dict[int, Dict[str, Any]] = {}

    for cluster_id, cache_data in cluster_decoder_cache.items():
        h_c_vals, decoder_vecs, layers, positions, feat_ids = graph.select_features_with_hc_selection(
            cache_data, top_B, hc_selection
        )
        if not h_c_vals:
            continue

        selected_features_by_cluster[cluster_id] = [
            (int(layers[i]), int(positions[i]), int(feat_ids[i]), float(h_c_vals[i]))
            for i in range(len(h_c_vals))
        ]
        selected_decoder_cache[cluster_id] = {
            "h_c_values": [float(v) for v in h_c_vals],
            "decoder_vecs": decoder_vecs,
            "layers": [int(layer) for layer in layers],
            "positions": [int(pos) for pos in positions],
            "feat_ids": [int(fid) for fid in feat_ids],
        }

    if not selected_features_by_cluster:
        return {}, {}, {}

    selected_encoder_cache = graph.precompute_cluster_encoder_weights(
        model, selected_features_by_cluster, device
    )
    return selected_features_by_cluster, selected_decoder_cache, selected_encoder_cache


# =============================================================================
# Common Steering Evaluation (Used by all hypotheses)
# =============================================================================

def _run_steering_evaluation(
    model,
    branches: List[Dict],
    features: List[Tuple[int, int, int, float]],
    encoder_cache: Dict,
    decoder_cache: Dict,
    baseline_metadata: Dict[int, Dict],
    epsilon: float,
    steering_method: str,
    cross_prefix_batching: bool = False,
    batch_size: int = 16,
    max_seq_len: int = None,
    logger=None,
) -> Dict[int, Dict[str, float]]:
    """Common steering evaluation used by all hypotheses.

    Routes to appropriate batching strategy and computes effects.

    Returns:
        {branch_id: {"log_P_steered": ..., "log_P_original": ..., "delta": ..., "rel_change": ...}}
    """
    results = {}

    if cross_prefix_batching:
        # Build batch items for heterogeneous steering
        all_items = []
        for branch in branches:
            if branch["branch_id"] not in baseline_metadata:
                continue
            full_token_ids = branch["full_token_ids"]
            cont_ids = branch["continuation_token_ids"]
            cont_start = len(full_token_ids) - len(cont_ids)
            meta = baseline_metadata[branch["branch_id"]]
            baseline = meta.get("log_P_original", 0.0)

            all_items.append({
                "branch_id": branch["branch_id"],
                "token_ids": full_token_ids,
                "cont_ids": cont_ids,
                "cont_start": cont_start,
                "baseline": baseline,
                "features": features,
                "decoder_cache": decoder_cache,
            })

        if not all_items:
            return {}

        # Process in batches
        for i in range(0, len(all_items), batch_size):
            chunk = all_items[i:i+batch_size]
            logits, _ = steering.run_heterogeneous_steered_pass(
                model, chunk, steering_method, epsilon, max_seq_len=max_seq_len
            )
            chunk_cont_info = [(it["cont_ids"], it["cont_start"]) for it in chunk]
            log_probs = metrics.compute_continuation_log_prob_batched(
                logits, chunk_cont_info
            )

            for item, log_P in zip(chunk, log_probs):
                delta = log_P - item["baseline"]
                delta = max(min(delta, utils.CLIP_MAX), utils.CLIP_MIN)
                rel_change = np.exp(delta) - 1.0
                # Absolute change: P_steered - P_original
                P_steered = np.exp(log_P)
                P_original = np.exp(item["baseline"])
                abs_change = P_steered - P_original
                results[item["branch_id"]] = {
                    "log_P_steered": float(log_P),
                    "log_P_original": float(item["baseline"]),
                    "delta": float(delta),
                    "rel_change": float(rel_change),
                    "abs_change": float(abs_change),
                }
            
            # Clear memory after each batch chunk
            del logits, log_probs
            maybe_clear_memory(gc_collect=False, cuda_empty_cache=False, min_interval_s=120.0, logger=logger, tag="h4a_chunk")
    else:
        # Normal batching - all branches share same steering features
        batch_token_ids = []
        batch_branches = []

        for branch in branches:
            if branch["branch_id"] not in baseline_metadata:
                continue
            ft = branch.get("full_token_ids", [])
            if not ft:
                ft = model.ensure_tokenized(branch["sequence"]).squeeze().tolist()
            batch_token_ids.append(ft)
            batch_branches.append(branch)

        if not batch_token_ids:
            return {}

        # Process in batches
        for i in range(0, len(batch_token_ids), batch_size):
            chunk_tokens = batch_token_ids[i:i+batch_size]
            chunk_branches = batch_branches[i:i+batch_size]

            logits, _ = steering.run_batched_steered_pass_on_the_fly(
                model, chunk_tokens, features, encoder_cache, decoder_cache,
                steering_method, epsilon, max_seq_len=max_seq_len
            )

            chunk_cont_info = []
            chunk_baselines = []
            for b in chunk_branches:
                cont_ids = b["continuation_token_ids"]
                full_ids = b["full_token_ids"]
                cont_start = len(full_ids) - len(cont_ids)
                chunk_cont_info.append((cont_ids, cont_start))

                meta = baseline_metadata[b["branch_id"]]
                # Get baseline with proper error handling and logging
                if "log_P_original" not in meta:
                    if logger:
                        logger.warning(f"Missing baseline for branch {b['branch_id']}, using 0.0")
                    baseline = 0.0
                else:
                    baseline = meta["log_P_original"]
                chunk_baselines.append(baseline)

            log_probs = metrics.compute_continuation_log_prob_batched(
                logits, chunk_cont_info
            )

            for j, (branch, log_P, baseline) in enumerate(zip(chunk_branches, log_probs, chunk_baselines)):
                delta = log_P - baseline
                delta = max(min(delta, utils.CLIP_MAX), utils.CLIP_MIN)
                rel_change = np.exp(delta) - 1.0
                # Absolute change: P_steered - P_original
                P_steered = np.exp(log_P)
                P_original = np.exp(baseline)
                abs_change = P_steered - P_original
                results[branch["branch_id"]] = {
                    "log_P_steered": float(log_P),
                    "log_P_original": float(baseline),
                    "delta": float(delta),
                    "rel_change": float(rel_change),
                    "abs_change": float(abs_change),
                }
            
            # Clear memory after each batch chunk
            del logits, log_probs
            maybe_clear_memory(gc_collect=False, cuda_empty_cache=False, min_interval_s=120.0, logger=logger, tag="h4a_chunk")

    return results


# =============================================================================
# Hypothesis Validation Functions
# =============================================================================

def validate_h4a_dose_response(
    model,
    branches: List[Dict],
    features_by_cluster: Dict[int, List[Tuple]],
    cluster_decoder_cache: Dict,
    cluster_encoder_cache: Dict,
    baseline_metadata: Dict[int, Dict],
    epsilon_values: List[float],
    steering_method: str,
    cross_prefix_batching: bool = False,
    max_cluster_samples: int = 20,
    batch_size: int = 16,
    max_seq_len: int = None,
    logger=None
) -> Dict[str, Any]:
    """H4a: Validate dose-response relationship.

    For each cluster, steer with H_c at different epsilon values and measure
    the relationship between epsilon and probability change.
    """
    if logger:
        logger.info("  Running H4a: dose-response validation...")

    cluster_ids = sorted(features_by_cluster.keys())
    per_component = {}

    # Group branches by cluster
    branches_by_cluster = {c: [] for c in cluster_ids}
    for branch in branches:
        c = branch["cluster_id"]
        if c in cluster_ids and branch["branch_id"] in baseline_metadata:
            branches_by_cluster[c].append(branch)

    # Subsample if needed
    rng = np.random.RandomState(42)
    for c in cluster_ids:
        if max_cluster_samples > 0 and len(branches_by_cluster[c]) > max_cluster_samples:
            indices = rng.choice(len(branches_by_cluster[c]), size=max_cluster_samples, replace=False)
            branches_by_cluster[c] = [branches_by_cluster[c][i] for i in indices]

    for comp_id in tqdm(cluster_ids, desc="    H4a Clusters", leave=False):
        features = features_by_cluster[comp_id]
        enc_cache = cluster_encoder_cache.get(comp_id, {})
        dec_cache = cluster_decoder_cache.get(comp_id, {})
        cluster_branches = branches_by_cluster[comp_id]

        if not cluster_branches:
            continue

        delta_probs_by_eps = {eps: [] for eps in epsilon_values}

        for epsilon in epsilon_values:
            eval_results = _run_steering_evaluation(
                model, cluster_branches, features, enc_cache, dec_cache,
                baseline_metadata, epsilon, steering_method,
                cross_prefix_batching=cross_prefix_batching,
                batch_size=batch_size,
                max_seq_len=max_seq_len
            )

            for branch_id, res in eval_results.items():
                delta_probs_by_eps[epsilon].append(res["rel_change"])

        # Aggregate
        comp_stats = {"mean_delta_prob": {}, "std_delta_prob": {}}
        for eps, deltas in delta_probs_by_eps.items():
            if deltas:
                comp_stats["mean_delta_prob"][eps] = float(np.mean(deltas))
                comp_stats["std_delta_prob"][eps] = float(np.std(deltas))
            else:
                comp_stats["mean_delta_prob"][eps] = 0.0
                comp_stats["std_delta_prob"][eps] = 0.0

        if len(epsilon_values) > 2:
            means = [comp_stats["mean_delta_prob"][e] for e in epsilon_values]
            slope, intercept, r_value, _, _ = linregress(epsilon_values, means)
            comp_stats["dose_response_r2"] = float(r_value**2)

        per_component[comp_id] = comp_stats

    return {"per_component": per_component}


# H4b removed - no longer needed


def validate_h4a_generation(
    model,
    prefix_token_ids: List[int],
    prefix_text: str,
    features_by_cluster: Dict[int, List[Tuple]],
    cluster_decoder_cache: Dict,
    cluster_encoder_cache: Dict,
    epsilon_values: List[float],
    steering_method: str,
    num_samples_per_epsilon: int = 5,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    max_seq_len: int = None,
    logger=None
) -> Dict[str, Any]:
    """H4a Generation: Sample sequences with steered generation.

    For each cluster c and each epsilon value, generate N sequences from the prefix
    with steering applied. This tests whether steering actually influences what the
    model generates (not just log probabilities on fixed continuations).

    Args:
        model: ReplacementModel
        prefix_token_ids: Prefix tokens WITH BOS
        prefix_text: Prefix text for display
        features_by_cluster: {cluster_id: [(layer, pos, feat_id, h_c_val), ...]}
        cluster_decoder_cache: Pre-computed decoder cache
        cluster_encoder_cache: Pre-computed encoder cache
        epsilon_values: List of steering strengths to test
        steering_method: Steering method name
        num_samples_per_epsilon: Number of sequences to sample per epsilon
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        max_seq_len: Max sequence length
        logger: Logger instance

    Returns:
        Dict with keys:
            - per_cluster: {cluster_id: {epsilon: [generated_samples]}}
            - prefix_text: The prefix used
    """
    if logger:
        logger.info("  Running H4a Generation: steered sampling experiment...")

    cluster_ids = sorted(features_by_cluster.keys())
    generation_results = {}

    for cluster_id in tqdm(cluster_ids, desc="    H4a Generation Clusters", leave=False):
        features = features_by_cluster[cluster_id]
        enc_cache = cluster_encoder_cache.get(cluster_id, {})
        dec_cache = cluster_decoder_cache.get(cluster_id, {})

        if not features:
            continue

        cluster_generations = {}

        for epsilon in epsilon_values:
            if logger:
                logger.info(f"    Cluster {cluster_id}, epsilon={epsilon}: generating {num_samples_per_epsilon} samples...")

            # Generate samples with steering
            samples = steering.generate_steered_sequences(
                model=model,
                prefix_token_ids=prefix_token_ids,
                features=features,
                encoder_cache=enc_cache,
                decoder_cache=dec_cache,
                steering_method=steering_method,
                epsilon=epsilon,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_samples=num_samples_per_epsilon,
                max_seq_len=max_seq_len,
            )

            cluster_generations[epsilon] = samples

        generation_results[cluster_id] = cluster_generations

    return {
        "per_cluster": generation_results,
        "prefix_text": prefix_text,
        "num_samples_per_epsilon": num_samples_per_epsilon,
        "generation_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }
    }


def validate_h4c_specificity(
    model,
    branches: List[Dict],
    semantic_graphs: Dict[int, np.ndarray],
    active_features: torch.Tensor,
    selected_features: torch.Tensor,
    features_by_cluster: Dict[int, List[Tuple]],
    cluster_decoder_cache: Dict,
    cluster_encoder_cache: Dict,
    baseline_metadata: Dict[int, Dict],
    epsilon: float,
    top_B: int,
    steering_method: str,
    cross_prefix_batching: bool = False,
    max_samples_per_cluster: int = 20,
    batch_size: int = 16,
    max_seq_len: int = None,
    logger=None
) -> Dict[str, Any]:
    """H4c: Validate steering specificity.

    For each cluster c, steer with H_c and measure effect on branches from c (target)
    vs branches from OTHER clusters. Specificity = fraction of clusters where
    |effect on target| > max |effect on others|.
    """
    if logger:
        logger.info("  Running H4c: steering specificity...")

    cluster_ids = sorted(semantic_graphs.keys())
    n_clusters = len(cluster_ids)

    if n_clusters < 2:
        return {"specificity": 0.0, "error": "insufficient_clusters", "n_clusters": n_clusters}

    # Group branches by cluster
    branches_by_cluster = {c: [] for c in cluster_ids}
    for branch in branches:
        c = branch["cluster_id"]
        if c in cluster_ids and branch["branch_id"] in baseline_metadata:
            branches_by_cluster[c].append(branch)

    # Subsample
    rng = np.random.RandomState(42)
    for c in cluster_ids:
        if max_samples_per_cluster > 0 and len(branches_by_cluster[c]) > max_samples_per_cluster:
            indices = rng.choice(len(branches_by_cluster[c]), size=max_samples_per_cluster, replace=False)
            branches_by_cluster[c] = [branches_by_cluster[c][i] for i in indices]

    # Flatten all target branches
    all_target_branches = []
    for c in cluster_ids:
        all_target_branches.extend(branches_by_cluster[c])

    # effect_matrix[steering_c][target_c] = list of effects (both rel and abs)
    effect_matrix_rel = {sc: {tc: [] for tc in cluster_ids} for sc in cluster_ids}
    effect_matrix_abs = {sc: {tc: [] for tc in cluster_ids} for sc in cluster_ids}

    # OUTER LOOP: STEERING CLUSTER
    for steering_c in tqdm(cluster_ids, desc="    H4c Steering", leave=False):
        features = features_by_cluster.get(steering_c, [])
        dec_cache = cluster_decoder_cache.get(steering_c, {})
        enc_cache = cluster_encoder_cache.get(steering_c, {})

        if not features:
            continue

        eval_results = _run_steering_evaluation(
            model, all_target_branches, features, enc_cache, dec_cache,
            baseline_metadata, epsilon, steering_method,
            cross_prefix_batching=cross_prefix_batching,
            batch_size=batch_size,
            max_seq_len=max_seq_len
        )

        for branch in all_target_branches:
            branch_id = branch["branch_id"]
            if branch_id in eval_results:
                target_c = branch["cluster_id"]
                effect_matrix_rel[steering_c][target_c].append(eval_results[branch_id]["rel_change"])
                effect_matrix_abs[steering_c][target_c].append(eval_results[branch_id]["abs_change"])

    # Average effects for both matrices
    for sc in cluster_ids:
        for tc in cluster_ids:
            rel_effects = effect_matrix_rel[sc][tc]
            abs_effects = effect_matrix_abs[sc][tc]
            effect_matrix_rel[sc][tc] = float(np.mean(rel_effects)) if rel_effects else 0.0
            effect_matrix_abs[sc][tc] = float(np.mean(abs_effects)) if abs_effects else 0.0

    # Compute all four specificity variants
    def compute_specificity(effect_matrix, mode="row"):
        """
        Compute specificity from effect matrix.
        mode="row": For each c, check if effect_matrix[c][c] > max(effect_matrix[c][other])
                    "When steering for c, does c benefit most?"
        mode="col": For each c, check if effect_matrix[c][c] > max(effect_matrix[other][c])
                    "For c's continuations, is steering with c the best?"
        """
        specific_count = 0
        per_cluster = {}

        for c in cluster_ids:
            target_effect = effect_matrix[c][c]
            if mode == "row":
                other_effects = [effect_matrix[c][other_c] for other_c in cluster_ids if other_c != c]
            else:  # column
                other_effects = [effect_matrix[other_c][c] for other_c in cluster_ids if other_c != c]

            max_other = max(other_effects) if other_effects else 0.0
            is_specific = target_effect > max_other
            if is_specific:
                specific_count += 1

            per_cluster[c] = {
                "target_effect": target_effect,
                "max_other_effect": max_other,
                "is_specific": is_specific
            }

        specificity = specific_count / n_clusters if n_clusters > 0 else 0.0
        return specificity, specific_count, per_cluster

    # Compute all four variants
    row_rel_spec, row_rel_count, row_rel_clusters = compute_specificity(effect_matrix_rel, mode="row")
    row_abs_spec, row_abs_count, row_abs_clusters = compute_specificity(effect_matrix_abs, mode="row")
    col_rel_spec, col_rel_count, col_rel_clusters = compute_specificity(effect_matrix_rel, mode="col")
    col_abs_spec, col_abs_count, col_abs_clusters = compute_specificity(effect_matrix_abs, mode="col")

    if logger:
        logger.info(f"    H4c specificity (row/rel): {row_rel_spec:.4f} ({row_rel_count}/{n_clusters})")
        logger.info(f"    H4c specificity (row/abs): {row_abs_spec:.4f} ({row_abs_count}/{n_clusters})")
        logger.info(f"    H4c specificity (col/rel): {col_rel_spec:.4f} ({col_rel_count}/{n_clusters})")
        logger.info(f"    H4c specificity (col/abs): {col_abs_spec:.4f} ({col_abs_count}/{n_clusters})")

    return {
        # Original format (row-wise, relative) for backwards compatibility
        "per_cluster": {str(c): v for c, v in row_rel_clusters.items()},
        "specificity": row_rel_spec,
        "n_specific": row_rel_count,
        "n_clusters": n_clusters,
        "epsilon": epsilon,
        # Full effect matrices
        "effect_matrix_rel": {str(sc): {str(tc): effect_matrix_rel[sc][tc] for tc in cluster_ids} for sc in cluster_ids},
        "effect_matrix_abs": {str(sc): {str(tc): effect_matrix_abs[sc][tc] for tc in cluster_ids} for sc in cluster_ids},
        # All four variants
        "variants": {
            "row_rel": {"specificity": row_rel_spec, "n_specific": row_rel_count, "per_cluster": {str(c): v for c, v in row_rel_clusters.items()}},
            "row_abs": {"specificity": row_abs_spec, "n_specific": row_abs_count, "per_cluster": {str(c): v for c, v in row_abs_clusters.items()}},
            "col_rel": {"specificity": col_rel_spec, "n_specific": col_rel_count, "per_cluster": {str(c): v for c, v in col_rel_clusters.items()}},
            "col_abs": {"specificity": col_abs_spec, "n_specific": col_abs_count, "per_cluster": {str(c): v for c, v in col_abs_clusters.items()}},
        }
    }


def validate_h4c_cluster_mass_pairwise(
    model,
    branches: List[Dict],
    semantic_graphs: Dict[int, np.ndarray],
    active_features: torch.Tensor,
    selected_features: torch.Tensor,
    features_by_cluster: Dict[int, List[Tuple]],
    cluster_decoder_cache: Dict,
    cluster_encoder_cache: Dict,
    baseline_metadata: Dict[int, Dict],
    epsilon_values: List[float],
    top_B: int,
    steering_method: str,
    cross_prefix_batching: bool = False,
    max_samples_per_cluster: int = 20,
    batch_size: int = 16,
    max_seq_len: int = None,
    logger=None
) -> Dict[str, Any]:
    """H4c_mass: pairwise H4A-style cluster-mass win metrics over the full (i, j) matrix."""
    if logger:
        logger.info("  Running H4c cluster-mass pairwise...")

    cluster_ids = sorted(semantic_graphs.keys())
    n_clusters = len(cluster_ids)

    if n_clusters < 2:
        return {
            "metric_primary": "cluster_mass_win_corr",
            "metric_secondary": "cluster_mass_win_spearman",
            "error": "insufficient_clusters",
            "n_clusters": n_clusters,
            "epsilon_values": [float(eps) for eps in epsilon_values],
        }

    branches_by_cluster = {c: [] for c in cluster_ids}
    for branch in branches:
        c = branch["cluster_id"]
        if c in cluster_ids and branch["branch_id"] in baseline_metadata:
            branches_by_cluster[c].append(branch)

    rng = np.random.RandomState(42)
    for c in cluster_ids:
        if max_samples_per_cluster > 0 and len(branches_by_cluster[c]) > max_samples_per_cluster:
            indices = rng.choice(len(branches_by_cluster[c]), size=max_samples_per_cluster, replace=False)
            branches_by_cluster[c] = [branches_by_cluster[c][i] for i in indices]

    all_target_branches = []
    for c in cluster_ids:
        all_target_branches.extend(branches_by_cluster[c])

    log_probs_steered = {
        sc: {tc: {float(eps): [] for eps in epsilon_values} for tc in cluster_ids}
        for sc in cluster_ids
    }
    log_probs_original = {
        sc: {tc: {float(eps): [] for eps in epsilon_values} for tc in cluster_ids}
        for sc in cluster_ids
    }

    for steering_c in tqdm(cluster_ids, desc="    H4c Mass", leave=False):
        features = features_by_cluster.get(steering_c, [])
        dec_cache = cluster_decoder_cache.get(steering_c, {})
        enc_cache = cluster_encoder_cache.get(steering_c, {})

        if not features:
            continue

        for epsilon in epsilon_values:
            eval_results = _run_steering_evaluation(
                model, all_target_branches, features, enc_cache, dec_cache,
                baseline_metadata, epsilon, steering_method,
                cross_prefix_batching=cross_prefix_batching,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                logger=logger,
            )

            for branch in all_target_branches:
                branch_id = branch["branch_id"]
                if branch_id not in eval_results:
                    continue
                target_c = branch["cluster_id"]
                log_probs_steered[steering_c][target_c][float(epsilon)].append(
                    eval_results[branch_id]["log_P_steered"]
                )
                log_probs_original[steering_c][target_c][float(epsilon)].append(
                    eval_results[branch_id]["log_P_original"]
                )

    effect_matrix_corr = {sc: {tc: 0.0 for tc in cluster_ids} for sc in cluster_ids}
    effect_matrix_spearman = {sc: {tc: 0.0 for tc in cluster_ids} for sc in cluster_ids}

    for steering_c in cluster_ids:
        for target_c in cluster_ids:
            cell_metrics = metrics.compute_cluster_mass_metrics(
                {target_c: log_probs_steered[steering_c][target_c]},
                {target_c: log_probs_original[steering_c][target_c]},
                [float(eps) for eps in epsilon_values],
            )
            target_metrics = cell_metrics.get("per_cluster_mass", {}).get(target_c, {})
            effect_matrix_corr[steering_c][target_c] = float(
                target_metrics.get("cluster_mass_win_corr", 0.0)
            )
            effect_matrix_spearman[steering_c][target_c] = float(
                target_metrics.get("cluster_mass_win_spearman", 0.0)
            )

    def compute_specificity(effect_matrix, mode="row"):
        specific_count = 0
        per_cluster = {}

        for c in cluster_ids:
            target_effect = effect_matrix[c][c]
            if mode == "row":
                other_effects = [effect_matrix[c][other_c] for other_c in cluster_ids if other_c != c]
            else:
                other_effects = [effect_matrix[other_c][c] for other_c in cluster_ids if other_c != c]

            max_other = max(other_effects) if other_effects else 0.0
            is_specific = target_effect > max_other
            if is_specific:
                specific_count += 1

            per_cluster[c] = {
                "target_effect": target_effect,
                "max_other_effect": max_other,
                "is_specific": is_specific,
            }

        specificity = specific_count / n_clusters if n_clusters > 0 else 0.0
        return specificity, specific_count, per_cluster

    row_corr_spec, row_corr_count, row_corr_clusters = compute_specificity(effect_matrix_corr, mode="row")
    col_corr_spec, col_corr_count, col_corr_clusters = compute_specificity(effect_matrix_corr, mode="col")

    if logger:
        logger.info(f"    H4c mass specificity (row/corr): {row_corr_spec:.4f} ({row_corr_count}/{n_clusters})")
        logger.info(f"    H4c mass specificity (col/corr): {col_corr_spec:.4f} ({col_corr_count}/{n_clusters})")

    return {
        "metric_primary": "cluster_mass_win_corr",
        "metric_secondary": "cluster_mass_win_spearman",
        "epsilon_values": [float(eps) for eps in epsilon_values],
        "n_clusters": n_clusters,
        "n_samples_per_cluster": {str(c): len(branches_by_cluster[c]) for c in cluster_ids},
        "effect_matrix_cluster_mass_win_corr": {
            str(sc): {str(tc): effect_matrix_corr[sc][tc] for tc in cluster_ids}
            for sc in cluster_ids
        },
        "effect_matrix_cluster_mass_win_spearman": {
            str(sc): {str(tc): effect_matrix_spearman[sc][tc] for tc in cluster_ids}
            for sc in cluster_ids
        },
        "specificity": {
            "row_corr": {
                "specificity": row_corr_spec,
                "n_specific": row_corr_count,
                "per_cluster": {str(c): v for c, v in row_corr_clusters.items()},
            },
            "col_corr": {
                "specificity": col_corr_spec,
                "n_specific": col_corr_count,
                "per_cluster": {str(c): v for c, v in col_corr_clusters.items()},
            },
        },
    }


# =============================================================================
# Sweep Mode Functions
# =============================================================================

def run_steering_sweep(
    model,
    branches: List[Dict],
    decoder_cache: Dict[int, Dict],
    encoder_cache: Dict[int, Dict],
    baseline_metadata: Dict[int, Dict],
    epsilons: List[float],
    top_B: int,
    steering_method: str,
    hc_selection: str,
    max_samples_per_cluster: int,
    log_details: bool = False,
    max_batch_size: int = 128,
    cross_prefix_batching: bool = False,
    max_seq_len: int = None,
    logger = None,
) -> Dict[str, Any]:
    """Run steering sweep for a single configuration."""
    cluster_ids = sorted(decoder_cache.keys())
    n_clusters = len(cluster_ids)

    if n_clusters == 0:
        return {"error": "no_clusters"}

    # Group branches by cluster
    branches_by_cluster = {c: [] for c in cluster_ids}
    for b in branches:
        c = b.get("cluster_id", -1)
        if c in branches_by_cluster and b["branch_id"] in baseline_metadata:
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

    # Initialize tracking
    dose_response = {c: {eps: [] for eps in epsilons} for c in cluster_ids}
    log_deltas = {c: {eps: [] for eps in epsilons} for c in cluster_ids}
    log_probs_steered = {c: {eps: [] for eps in epsilons} for c in cluster_ids}
    log_probs_original = {c: {eps: [] for eps in epsilons} for c in cluster_ids}
    centered_logits_steered = {c: {eps: [] for eps in epsilons} for c in cluster_ids}
    centered_logits_original = {c: {eps: [] for eps in epsilons} for c in cluster_ids}
    target_logits_steered = {c: {eps: [] for eps in epsilons} for c in cluster_ids}
    target_logits_original = {c: {eps: [] for eps in epsilons} for c in cluster_ids}
    target_probs_steered = {c: {eps: [] for eps in epsilons} for c in cluster_ids}
    target_probs_original = {c: {eps: [] for eps in epsilons} for c in cluster_ids}
    detailed_logs = [] if log_details else None

    effective_batch_size = max_batch_size if max_batch_size > 0 else 128
    if logger:
        logger.info(f"  Running steering sweep with batch_size={effective_batch_size}, cross_prefix_batching={cross_prefix_batching}")
        logger.info(f"  Epsilons ({len(epsilons)}): {epsilons}")

    if cross_prefix_batching:
        # Build all items for heterogeneous steering
        all_items = []
        for steer_c in cluster_ids:
            if steer_c not in selected_features_cache:
                continue
            h_c_vals, _, layers, positions, feat_ids = selected_features_cache[steer_c]
            if not h_c_vals:
                continue

            features = [(layers[i], positions[i], feat_ids[i], h_c_vals[i]) for i in range(len(h_c_vals))]
            c_decoder_cache = decoder_cache[steer_c]

            for branch in branches_by_cluster[steer_c]:
                if branch["branch_id"] not in baseline_metadata:
                    continue
                full_token_ids = branch["full_token_ids"]
                cont_ids = branch["continuation_token_ids"]
                cont_start = len(full_token_ids) - len(cont_ids)
                meta = baseline_metadata[branch["branch_id"]]
                all_items.append({
                    "cluster_id": steer_c,
                    "branch_id": branch["branch_id"],
                    "token_ids": full_token_ids,
                    "cont_ids": cont_ids,
                    "cont_start": cont_start,
                    "baseline": meta["log_P_original"],
                    "baseline_mean_centered_logit": meta.get("mean_centered_logit_original", 0.0),
                    "baseline_mean_target_logit": meta.get("mean_target_logit_original", 0.0),
                    "baseline_mean_target_prob": meta.get("mean_target_prob_original", 0.0),
                    "features": features,
                    "decoder_cache": c_decoder_cache,
                })

        # Pre-compute layer metadata for each chunk (reused across epsilons)
        chunk_metadata_cache = {}
        for chunk_start in range(0, len(all_items), effective_batch_size):
            chunk_items = all_items[chunk_start:chunk_start + effective_batch_size]
            chunk_metadata_cache[chunk_start] = steering.prepare_heterogeneous_layer_metadata(
                model, chunk_items, max_seq_len=max_seq_len
            )

        for eps_idx, eps in enumerate(epsilons, start=1):
            if logger:
                logger.info(f"    Epsilon {eps_idx}/{len(epsilons)}: {eps}")
            for chunk_start in range(0, len(all_items), effective_batch_size):
                chunk_items = all_items[chunk_start:chunk_start + effective_batch_size]
                precomputed = chunk_metadata_cache[chunk_start]

                logits, _ = steering.run_heterogeneous_steered_pass(
                    model, chunk_items, steering_method, eps, max_seq_len=max_seq_len,
                    precomputed_metadata=precomputed
                )
                chunk_cont_info = [(it["cont_ids"], it["cont_start"]) for it in chunk_items]
                chunk_log_probs = metrics.compute_continuation_log_prob_batched(logits, chunk_cont_info)

                # Compute centered logits for steered pass
                chunk_centered_logits = steering.compute_per_token_centered_logits_batched(
                    logits, chunk_cont_info, return_per_token=log_details
                )
                # Compute mean target logits/probs for steered pass
                chunk_target_logits_probs = metrics.compute_mean_target_logit_and_prob_batched(logits, chunk_cont_info)

                for idx, (item, log_P) in enumerate(zip(chunk_items, chunk_log_probs)):
                    delta = log_P - item["baseline"]
                    delta = max(min(delta, utils.CLIP_MAX), utils.CLIP_MIN)
                    c_id = item["cluster_id"]
                    log_deltas[c_id][eps].append(delta)
                    rel_change = np.exp(delta) - 1.0
                    dose_response[c_id][eps].append(rel_change)

                    # Track log probabilities for cluster mass metrics
                    log_probs_steered[c_id][eps].append(log_P)
                    log_probs_original[c_id][eps].append(item["baseline"])

                    # Track centered logits
                    _, mean_centered_logit_steered = chunk_centered_logits[idx]
                    centered_logits_steered[c_id][eps].append(mean_centered_logit_steered)
                    centered_logits_original[c_id][eps].append(item["baseline_mean_centered_logit"])

                    # Track mean target logits/probs
                    mean_target_logit_steered, mean_target_prob_steered = chunk_target_logits_probs[idx]
                    target_logits_steered[c_id][eps].append(mean_target_logit_steered)
                    target_logits_original[c_id][eps].append(item.get("baseline_mean_target_logit", 0.0))
                    target_probs_steered[c_id][eps].append(mean_target_prob_steered)
                    target_probs_original[c_id][eps].append(item.get("baseline_mean_target_prob", 0.0))

                    if log_details:
                        per_token_centered, _ = chunk_centered_logits[idx]
                        detailed_logs.append({
                            'cluster_id': c_id,
                            'branch_id': item["branch_id"],
                            'epsilon': eps,
                            'log_P_steered': float(log_P),
                            'log_P_original': float(item["baseline"]),
                            'log_delta': float(delta),
                            'rel_change': float(rel_change),
                            'continuation_token_ids': item["cont_ids"],
                            'mean_centered_logit_steered': float(mean_centered_logit_steered),
                            'mean_centered_logit_original': float(item["baseline_mean_centered_logit"]),
                            'per_token_centered_logits_steered': per_token_centered,
                        })

                # Clear memory after each batch chunk
                del logits, chunk_log_probs, chunk_centered_logits, chunk_target_logits_probs
                maybe_clear_memory(gc_collect=False, cuda_empty_cache=False, min_interval_s=120.0, logger=logger, tag="sweep_chunk")
    else:
        # Normal batching path
        for steer_c in cluster_ids:
            if steer_c not in selected_features_cache:
                continue
            h_c_vals, _, layers, positions, feat_ids = selected_features_cache[steer_c]
            if not h_c_vals:
                continue

            cluster_branches = branches_by_cluster[steer_c]
            if not cluster_branches:
                continue

            features = [(layers[i], positions[i], feat_ids[i], h_c_vals[i]) for i in range(len(h_c_vals))]
            c_encoder_cache = encoder_cache.get(steer_c, {})

            batch_items = []
            batch_info = []

            for branch in cluster_branches:
                full_token_ids = branch["full_token_ids"]
                cont_ids = branch["continuation_token_ids"]
                seq_len = len(full_token_ids)
                cont_start = seq_len - len(cont_ids)
                meta = baseline_metadata[branch["branch_id"]]
                log_P_original = meta["log_P_original"]

                batch_items.append(full_token_ids)
                batch_info.append((branch, cont_ids, cont_start, log_P_original, meta))

            if not batch_items:
                continue

            # Clear memory before processing batches for this cluster
            maybe_clear_memory(gc_collect=False, cuda_empty_cache=False, min_interval_s=120.0, logger=logger, tag="sweep_cluster_start")
            
            for chunk_start in range(0, len(batch_items), effective_batch_size):
                chunk_end = min(chunk_start + effective_batch_size, len(batch_items))
                chunk_tokens = batch_items[chunk_start:chunk_end]
                chunk_meta_info = batch_info[chunk_start:chunk_end]
                chunk_cont_info = [(info[1], info[2]) for info in chunk_meta_info]

                # Compute baseline
                logits_base, _ = steering.run_batched_steered_pass_on_the_fly(
                    model, chunk_tokens, features=[],
                    encoder_cache={}, decoder_cache={},
                    steering_method="additive", epsilon=0.0,
                    max_seq_len=max_seq_len
                )
                chunk_log_probs_base = metrics.compute_continuation_log_prob_batched(logits_base, chunk_cont_info)
                # Compute centered logits for baseline
                chunk_centered_logits_base = steering.compute_per_token_centered_logits_batched(
                    logits_base, chunk_cont_info, return_per_token=log_details
                )
                # Compute mean target logits/probs for baseline
                chunk_target_logits_probs_base = metrics.compute_mean_target_logit_and_prob_batched(logits_base, chunk_cont_info)

                for eps_idx, eps in enumerate(epsilons, start=1):
                    if logger:
                        logger.info(f"    Epsilon {eps_idx}/{len(epsilons)}: {eps} (cluster {steer_c})")
                    logits, _ = steering.run_batched_steered_pass_on_the_fly(
                        model, chunk_tokens, features, c_encoder_cache, decoder_cache[steer_c],
                        steering_method, eps, max_seq_len=max_seq_len
                    )
                    chunk_log_probs = metrics.compute_continuation_log_prob_batched(logits, chunk_cont_info)
                    # Compute centered logits for steered pass
                    chunk_centered_logits = steering.compute_per_token_centered_logits_batched(
                        logits, chunk_cont_info, return_per_token=log_details
                    )
                    # Compute mean target logits/probs for steered pass
                    chunk_target_logits_probs = metrics.compute_mean_target_logit_and_prob_batched(logits, chunk_cont_info)

                    for local_idx, (branch, cont_ids, cont_start, _, meta) in enumerate(chunk_meta_info):
                        log_P = chunk_log_probs[local_idx]
                        log_P_original = chunk_log_probs_base[local_idx]

                        delta = log_P - log_P_original
                        delta = max(min(delta, utils.CLIP_MAX), utils.CLIP_MIN)

                        log_deltas[steer_c][eps].append(delta)
                        rel_change = np.exp(delta) - 1.0
                        dose_response[steer_c][eps].append(rel_change)

                        # Track log probabilities for cluster mass metrics
                        log_probs_steered[steer_c][eps].append(log_P)
                        log_probs_original[steer_c][eps].append(log_P_original)

                        # Track centered logits
                        _, mean_centered_logit_steered = chunk_centered_logits[local_idx]
                        _, mean_centered_logit_original = chunk_centered_logits_base[local_idx]
                        centered_logits_steered[steer_c][eps].append(mean_centered_logit_steered)
                        centered_logits_original[steer_c][eps].append(mean_centered_logit_original)

                        # Track mean target logits/probs
                        mean_target_logit_steered, mean_target_prob_steered = chunk_target_logits_probs[local_idx]
                        mean_target_logit_original, mean_target_prob_original = chunk_target_logits_probs_base[local_idx]
                        target_logits_steered[steer_c][eps].append(mean_target_logit_steered)
                        target_logits_original[steer_c][eps].append(mean_target_logit_original)
                        target_probs_steered[steer_c][eps].append(mean_target_prob_steered)
                        target_probs_original[steer_c][eps].append(mean_target_prob_original)

                        if log_details:
                            per_token_centered_steered, _ = chunk_centered_logits[local_idx]
                            per_token_centered_original, _ = chunk_centered_logits_base[local_idx]
                            detailed_logs.append({
                                'cluster_id': steer_c,
                                'branch_id': branch["branch_id"],
                                'epsilon': eps,
                                'log_P_steered': float(log_P),
                                'log_P_original': float(log_P_original),
                                'log_delta': float(delta),
                                'rel_change': float(rel_change),
                                'continuation_token_ids': cont_ids,
                                'mean_centered_logit_steered': float(mean_centered_logit_steered),
                                'mean_centered_logit_original': float(mean_centered_logit_original),
                                'per_token_centered_logits_steered': per_token_centered_steered,
                                'per_token_centered_logits_original': per_token_centered_original,
                            })

                    # Clear memory after each epsilon value
                    del logits, chunk_log_probs, chunk_centered_logits, chunk_target_logits_probs
                    maybe_clear_memory(gc_collect=False, cuda_empty_cache=False, min_interval_s=120.0, logger=logger, tag="sweep_eps")

                # Clear baseline logits after processing all epsilons for this chunk
                del logits_base, chunk_log_probs_base, chunk_centered_logits_base, chunk_target_logits_probs_base
                maybe_clear_memory(gc_collect=False, cuda_empty_cache=False, min_interval_s=120.0, logger=logger, tag="sweep_chunk_end")
            
            # Clear memory after processing all batches for this cluster
            maybe_clear_memory(gc_collect=False, cuda_empty_cache=False, min_interval_s=120.0, logger=logger, tag="sweep_cluster_end")

    # Compute metrics (including cluster mass metrics and centered logit metrics)
    computed_metrics = metrics.compute_steering_metrics(
        dose_response, log_deltas, epsilons,
        log_probs_steered=log_probs_steered,
        log_probs_original=log_probs_original,
        centered_logits_steered=centered_logits_steered,
        centered_logits_original=centered_logits_original,
        target_logits_steered=target_logits_steered,
        target_logits_original=target_logits_original,
        target_probs_steered=target_probs_steered,
        target_probs_original=target_probs_original,
    )

    result = {
        'steering_method': steering_method,
        'hc_selection': hc_selection,
        'top_B': top_B,
        'epsilon_values': epsilons,
        'per_cluster': computed_metrics['per_cluster'],
        'mean_r2': computed_metrics['mean_r2'],
        'mean_corr': computed_metrics['mean_corr'],
        'mean_win_r2': computed_metrics['mean_win_r2'],
        'mean_win_corr': computed_metrics['mean_win_corr'],
        'n_clusters_with_effect': computed_metrics['n_clusters_with_effect']
    }

    # Add cluster mass metrics if available
    if 'per_cluster_mass' in computed_metrics:
        result['per_cluster_mass'] = computed_metrics['per_cluster_mass']

    # Add centered logit metrics if available
    if 'per_cluster_logit' in computed_metrics:
        result['per_cluster_logit'] = computed_metrics['per_cluster_logit']
    if 'mean_logit_r2' in computed_metrics:
        result['mean_logit_r2'] = computed_metrics['mean_logit_r2']
    if 'mean_logit_corr' in computed_metrics:
        result['mean_logit_corr'] = computed_metrics['mean_logit_corr']
    if 'mean_logit_spearman' in computed_metrics:
        result['mean_logit_spearman'] = computed_metrics['mean_logit_spearman']

    if detailed_logs:
        result['detailed_logs'] = detailed_logs

    return result


def run_steering_sweep_prefix_batch(
    model,
    prefix_contexts: List[Dict[str, Any]],
    epsilons: List[float],
    top_B: int,
    steering_method: str,
    hc_selection: str,
    max_samples_per_cluster: int,
    log_details: bool = False,
    max_batch_size: int = 128,
    max_seq_len: int = None,
    logger = None,
) -> Dict[str, Dict[str, Any]]:
    """Run steering sweep for a batch of prefixes using heterogeneous batching."""
    if not prefix_contexts:
        return {}

    rng = np.random.RandomState(42)
    effective_batch_size = max_batch_size if max_batch_size > 0 else 128
    results_by_prefix: Dict[str, Dict[str, Any]] = {}
    detailed_logs_by_prefix: Dict[str, List[Dict[str, Any]]] = {}

    per_prefix: Dict[str, Dict[str, Any]] = {}
    for ctx in prefix_contexts:
        prefix_id = ctx["prefix_id"]
        decoder_cache = ctx["decoder_cache"]
        baseline_metadata = ctx["baseline_metadata"]
        branches = ctx["branches"]

        cluster_ids = sorted(decoder_cache.keys())
        if not cluster_ids:
            results_by_prefix[prefix_id] = {"error": "no_clusters"}
            continue

        branches_by_cluster = {c: [] for c in cluster_ids}
        for b in branches:
            c = b.get("cluster_id", -1)
            if c in branches_by_cluster and b["branch_id"] in baseline_metadata:
                branches_by_cluster[c].append(b)

        # Limit samples per cluster
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

        dose_response = {c: {eps: [] for eps in epsilons} for c in cluster_ids}
        log_deltas = {c: {eps: [] for eps in epsilons} for c in cluster_ids}
        log_probs_steered = {c: {eps: [] for eps in epsilons} for c in cluster_ids}
        log_probs_original = {c: {eps: [] for eps in epsilons} for c in cluster_ids}
        centered_logits_steered = {c: {eps: [] for eps in epsilons} for c in cluster_ids}
        centered_logits_original = {c: {eps: [] for eps in epsilons} for c in cluster_ids}
        target_logits_steered = {c: {eps: [] for eps in epsilons} for c in cluster_ids}
        target_logits_original = {c: {eps: [] for eps in epsilons} for c in cluster_ids}
        target_probs_steered = {c: {eps: [] for eps in epsilons} for c in cluster_ids}
        target_probs_original = {c: {eps: [] for eps in epsilons} for c in cluster_ids}

        per_prefix[prefix_id] = {
            "decoder_cache": decoder_cache,
            "baseline_metadata": baseline_metadata,
            "branches_by_cluster": branches_by_cluster,
            "selected_features_cache": selected_features_cache,
            "dose_response": dose_response,
            "log_deltas": log_deltas,
            "log_probs_steered": log_probs_steered,
            "log_probs_original": log_probs_original,
            "centered_logits_steered": centered_logits_steered,
            "centered_logits_original": centered_logits_original,
            "target_logits_steered": target_logits_steered,
            "target_logits_original": target_logits_original,
            "target_probs_steered": target_probs_steered,
            "target_probs_original": target_probs_original,
        }
        if log_details:
            detailed_logs_by_prefix[prefix_id] = []

    if logger:
        logger.info(
            f"  Running prefix-batched sweep with batch_size={effective_batch_size} "
            f"for {len(per_prefix)} prefixes"
        )

    # Build all items across prefixes/clusters
    all_items = []
    for prefix_id, info in per_prefix.items():
        selected_features_cache = info["selected_features_cache"]
        decoder_cache = info["decoder_cache"]
        branches_by_cluster = info["branches_by_cluster"]
        baseline_metadata = info["baseline_metadata"]

        for steer_c, features_cache in selected_features_cache.items():
            h_c_vals, _, layers, positions, feat_ids = features_cache
            if not h_c_vals:
                continue

            features = [(layers[i], positions[i], feat_ids[i], h_c_vals[i]) for i in range(len(h_c_vals))]
            c_decoder_cache = decoder_cache[steer_c]

            for branch in branches_by_cluster.get(steer_c, []):
                if branch["branch_id"] not in baseline_metadata:
                    continue
                full_token_ids = branch["full_token_ids"]
                cont_ids = branch["continuation_token_ids"]
                cont_start = len(full_token_ids) - len(cont_ids)
                meta = baseline_metadata[branch["branch_id"]]
                all_items.append({
                    "prefix_id": prefix_id,
                    "cluster_id": steer_c,
                    "branch_id": branch["branch_id"],
                    "token_ids": full_token_ids,
                    "cont_ids": cont_ids,
                    "cont_start": cont_start,
                    "baseline": meta["log_P_original"],
                    "baseline_mean_centered_logit": meta.get("mean_centered_logit_original", 0.0),
                    "baseline_mean_target_logit": meta.get("mean_target_logit_original", 0.0),
                    "baseline_mean_target_prob": meta.get("mean_target_prob_original", 0.0),
                    "features": features,
                    "decoder_cache": c_decoder_cache,
                })

    if not all_items:
        return {pid: {"error": "no_items"} for pid in per_prefix.keys()}

    # Pre-compute layer metadata for each chunk (reused across epsilons)
    chunk_metadata_cache = {}
    for chunk_start in range(0, len(all_items), effective_batch_size):
        chunk_items = all_items[chunk_start:chunk_start + effective_batch_size]
        chunk_metadata_cache[chunk_start] = steering.prepare_heterogeneous_layer_metadata(
            model, chunk_items, max_seq_len=max_seq_len
        )

    for eps in epsilons:
        for chunk_start in range(0, len(all_items), effective_batch_size):
            chunk_items = all_items[chunk_start:chunk_start + effective_batch_size]
            precomputed = chunk_metadata_cache[chunk_start]

            logits, _ = steering.run_heterogeneous_steered_pass(
                model, chunk_items, steering_method, eps, max_seq_len=max_seq_len,
                precomputed_metadata=precomputed
            )
            chunk_cont_info = [(it["cont_ids"], it["cont_start"]) for it in chunk_items]
            chunk_log_probs = metrics.compute_continuation_log_prob_batched(logits, chunk_cont_info)
            chunk_centered_logits = steering.compute_per_token_centered_logits_batched(
                logits, chunk_cont_info, return_per_token=log_details
            )
            chunk_target_logits_probs = metrics.compute_mean_target_logit_and_prob_batched(logits, chunk_cont_info)

            for idx, (item, log_P) in enumerate(zip(chunk_items, chunk_log_probs)):
                prefix_id = item["prefix_id"]
                c_id = item["cluster_id"]
                delta = log_P - item["baseline"]
                delta = max(min(delta, utils.CLIP_MAX), utils.CLIP_MIN)

                info = per_prefix.get(prefix_id)
                if info is None:
                    continue

                info["log_deltas"][c_id][eps].append(delta)
                rel_change = np.exp(delta) - 1.0
                info["dose_response"][c_id][eps].append(rel_change)

                info["log_probs_steered"][c_id][eps].append(log_P)
                info["log_probs_original"][c_id][eps].append(item["baseline"])

                _, mean_centered_logit_steered = chunk_centered_logits[idx]
                info["centered_logits_steered"][c_id][eps].append(mean_centered_logit_steered)
                info["centered_logits_original"][c_id][eps].append(item["baseline_mean_centered_logit"])

                mean_target_logit_steered, mean_target_prob_steered = chunk_target_logits_probs[idx]
                info["target_logits_steered"][c_id][eps].append(mean_target_logit_steered)
                info["target_logits_original"][c_id][eps].append(item.get("baseline_mean_target_logit", 0.0))
                info["target_probs_steered"][c_id][eps].append(mean_target_prob_steered)
                info["target_probs_original"][c_id][eps].append(item.get("baseline_mean_target_prob", 0.0))

                if log_details:
                    per_token_centered, _ = chunk_centered_logits[idx]
                    detailed_logs_by_prefix[prefix_id].append({
                        "cluster_id": c_id,
                        "branch_id": item["branch_id"],
                        "epsilon": eps,
                        "log_P_steered": float(log_P),
                        "log_P_original": float(item["baseline"]),
                        "log_delta": float(delta),
                        "rel_change": float(rel_change),
                        "continuation_token_ids": item["cont_ids"],
                        "mean_centered_logit_steered": float(mean_centered_logit_steered),
                        "mean_centered_logit_original": float(item["baseline_mean_centered_logit"]),
                        "per_token_centered_logits_steered": per_token_centered,
                        "mean_target_logit_steered": float(mean_target_logit_steered),
                        "mean_target_logit_original": float(item.get("baseline_mean_target_logit", 0.0)),
                        "mean_target_prob_steered": float(mean_target_prob_steered),
                        "mean_target_prob_original": float(item.get("baseline_mean_target_prob", 0.0)),
                    })

            del logits, chunk_log_probs, chunk_centered_logits, chunk_target_logits_probs
            maybe_clear_memory(gc_collect=False, cuda_empty_cache=False, min_interval_s=120.0, logger=logger, tag="prefix_sweep_chunk")

    for prefix_id, info in per_prefix.items():
        result = metrics.compute_steering_metrics(
            info["dose_response"],
            info["log_deltas"],
            epsilons,
            log_probs_steered=info["log_probs_steered"],
            log_probs_original=info["log_probs_original"],
            centered_logits_steered=info["centered_logits_steered"],
            centered_logits_original=info["centered_logits_original"],
            target_logits_steered=info["target_logits_steered"],
            target_logits_original=info["target_logits_original"],
            target_probs_steered=info["target_probs_steered"],
            target_probs_original=info["target_probs_original"],
        )
        if log_details:
            result["detailed_logs"] = detailed_logs_by_prefix.get(prefix_id, [])
        results_by_prefix[prefix_id] = result

    return results_by_prefix


def run_sweep_mode(
    model,
    args,
    config: Dict,
    steering_config: Dict,
    sweeps: List[Dict],
    feature_selection: str,
    log_details: bool,
    device: torch.device,
    logger,
    cross_prefix_batching: bool,
    K_clamp: int = None,
    center_Hc: bool = True,
):
    """Run sweep mode - iterate over multiple configurations."""
    max_cluster_samples = args.max_cluster_samples if args.max_cluster_samples is not None else steering_config.get("max_cluster_samples", 30)
    max_samples = args.max_samples if args.max_samples is not None else steering_config.get("max_samples")
    
    # Get batch size and max_seq_len from global config
    global_config = config.get("global", {})
    global_batch_size = global_config.get("batch_size", 128)
    max_seq_len = global_config.get("max_seq_len", None)  # None means dynamic padding
    
    max_batch_size = steering_config.get("max_batch_size", -1)
    if max_batch_size <= 0:
        max_batch_size = global_batch_size
    prefix_batch_size = steering_config.get("prefix_batch_size")
    
    # Log center_Hc setting
    logger.info(f"H_c centering: {'enabled (Delta_H_c)' if center_Hc else 'disabled (H_0 + mu_a)'}")

    if not sweeps:
        sweep_config = parse_sweeps_from_main_config(steering_config)
        sweeps = sweep_config.get("sweeps", [])

    hypotheses = steering_config.get("hypotheses", [])
    hypotheses = [h.upper() for h in hypotheses] if hypotheses else ["H4A"]
    run_h4a_sweeps = "H4A" in hypotheses
    run_h4a_gen = "H4A_GEN" in hypotheses
    run_h4c = "H4C" in hypotheses
    run_h4c_mass = "H4C_MASS" in hypotheses
    need_full_cluster_state = run_h4a_sweeps or run_h4a_gen
    hypothesis_dirs = {
        "H4A": "H4a",
        "H4A_GEN": "H4a_gen",
        "H4C": "H4c",
        "H4C_MASS": "H4c_mass",
    }

    manifest_by_prefix = None
    manifest_prefix_order = None
    if getattr(args, "clustering_manifest", None):
        manifest_by_prefix, manifest_prefix_order = _load_clustering_manifest(args.clustering_manifest)
        manifest_entry_count = sum(len(entries) for entries in manifest_by_prefix.values())
        logger.info(
            f"Loaded clustering manifest with {manifest_entry_count} entries across "
            f"{len(manifest_prefix_order)} prefixes from {args.clustering_manifest}"
        )

    # Log sweep info
    logger.info(f"Sweep configurations: {len(sweeps)}")
    for i, sw in enumerate(sweeps):
        logger.info(f"  [{i+1}] {sw.get('name', 'unnamed')}: {sw.get('steering_method')} | "
                    f"hc={sw.get('h_c_selections', sw.get('hc_selections'))} | B={sw.get('top_B')} | "
                    f"eps={sw.get('epsilon_values', sw.get('epsilons'))}")
    logger.info(f"Max samples/cluster: {max_cluster_samples}")
    primary_sweep_settings = _resolve_primary_sweep_settings(sweeps, steering_config)

    # Build aggregation keys (steering sweep keys)
    all_keys = []
    for sw in sweeps:
        method = sw.get("steering_method")
        for hc_sel in sw.get("h_c_selections", sw.get("hc_selections", ["full"])):
            for tb in sw.get("top_B", [10]):
                all_keys.append((method, hc_sel, tb))

    aggregated_by_clustering = {}

    # Find prefixes from attribution files
    attr_files = sorted(args.attribution_graphs_dir.glob("*_prefix_context.pt"))
    prefix_ids = [f.stem.replace("_prefix_context", "") for f in attr_files]
    logger.info(f"Discovered {len(prefix_ids)} prefixes from attribution files")

    if manifest_prefix_order is not None and not args.prefix_id:
        discovered_prefixes = set(prefix_ids)
        missing_prefixes = [pid for pid in manifest_prefix_order if pid not in discovered_prefixes]
        if missing_prefixes:
            logger.warning(
                f"Manifest includes {len(missing_prefixes)} prefixes missing attribution files: "
                f"{missing_prefixes[:3]}{'...' if len(missing_prefixes) > 3 else ''}"
            )
        prefix_ids = [pid for pid in manifest_prefix_order if pid in discovered_prefixes]

    if args.prefix_id:
        prefix_ids = [args.prefix_id]

    if max_samples is not None and max_samples > 0 and len(prefix_ids) > max_samples:
        prefix_ids = prefix_ids[:max_samples]
        logger.info(f"Limited to first {max_samples} prefixes")

    if getattr(args, "skip_existing", False):
        before = len(prefix_ids)
        prefix_ids = [pid for pid in prefix_ids if not _prefix_outputs_exist(pid, hypotheses, args.output_dir, hypothesis_dirs)]
        skipped = before - len(prefix_ids)
        if skipped:
            logger.info(f"Skipping {skipped}/{before} prefixes with existing outputs (sample-wise resume)")

    logger.info(f"Processing {len(prefix_ids)} prefixes")

    all_results = {}
    prefix_batch_size = steering_config.get("prefix_batch_size")
    try:
        prefix_batch_size = int(prefix_batch_size) if prefix_batch_size is not None else 1
    except (ValueError, TypeError):
        prefix_batch_size = 1

    if prefix_batch_size <= 0:
        prefix_batch_size = 1

    prefix_batches = [prefix_ids[i:i + prefix_batch_size] for i in range(0, len(prefix_ids), prefix_batch_size)]
    logger.info(f"Prefix batching: {prefix_batch_size if prefix_batch_size else 'disabled'}")

    total_batches = len(prefix_batches)
    for batch_idx, prefix_batch in enumerate(tqdm(prefix_batches, desc="Prefix batches", total=total_batches, dynamic_ncols=True), start=1):
        logger.info(f"Prefix batch {batch_idx}/{total_batches}: {len(prefix_batch)} prefixes")
        batch_contexts_by_key: Dict[str, List[Dict[str, Any]]] = {}
        batch_prefix_state: Dict[str, Dict[str, Any]] = {}
        baseline_pbar = None

        effective_batch_size = max_batch_size if max_batch_size > 0 else 32
        total_prefixes_in_batch = len(prefix_batch)
        baseline_total_batches = 0
        for prefix_idx, prefix_id in enumerate(prefix_batch, start=1):
            logger.info(f"Processing {prefix_id} ({prefix_idx}/{total_prefixes_in_batch})...")

            # Load files
            branches_file = args.samples_dir / f"{prefix_id}_branches.json"
            clustering_file = args.clustering_dir / f"{prefix_id}_sweep_results.json"
            attr_file = args.attribution_graphs_dir / f"{prefix_id}_prefix_context.pt"

            required_files = [branches_file, clustering_file, attr_file]
            if not all(f.exists() for f in required_files):
                logger.warning(f"Missing files for {prefix_id}")
                continue

            branches_data = load_json(branches_file)
            clustering_sweep = load_json(clustering_file)
            active_features, selected_features = graph.load_attribution_context(
                args.attribution_graphs_dir, prefix_id, use_continuation_attribution=True
            )

            H_0 = None
            H_0_raw = clustering_sweep.get("H_0")
            if H_0_raw is not None:
                H_0 = np.array(H_0_raw)

            grid_results = clustering_sweep.get("grid", [])

            # Get K_clamp for downstream filtering
            # Priority: CLI arg > config > sweep_config K_clamp > sweep_config K_max > default 20
            sweep_config = clustering_sweep.get("sweep_config", {})
            effective_K_clamp = K_clamp if K_clamp is not None else sweep_config.get(
                "K_clamp", sweep_config.get("K_max", 20)
            )

            # Filter: valid entries with K in range (1, K_clamp].
            # We skip degenerate clusterings (K<=1) and those exceeding K_clamp.
            valid_grid = []
            skipped_by_K = []
            for entry in grid_results:
                if not entry.get("components") or not entry.get("assignments") or "error" in entry:
                    continue
                K = entry.get("K", len(entry.get("components", {})))
                if K is None:
                    continue
                if K <= 1 or K > effective_K_clamp:
                    skipped_by_K.append((entry.get("beta"), entry.get("gamma"), K))
                    continue
                valid_grid.append(entry)

            if skipped_by_K:
                logger.info(
                    f"  Skipped {len(skipped_by_K)} configs outside (1, {effective_K_clamp}] for K: "
                    f"{skipped_by_K[:3]}{'...' if len(skipped_by_K) > 3 else ''}"
                )

            # Filter by beta/gamma values if specified
            if args.beta_values is not None or args.gamma_values is not None:
                filtered_grid = []
                for entry in valid_grid:
                    beta = entry.get("beta")
                    gamma = entry.get("gamma")
                    beta_ok = args.beta_values is None or any(abs(beta - b) < 0.001 for b in args.beta_values)
                    gamma_ok = args.gamma_values is None or any(abs(gamma - g) < 0.001 for g in args.gamma_values)
                    if beta_ok and gamma_ok:
                        filtered_grid.append(entry)
                skipped_by_beta_gamma = len(valid_grid) - len(filtered_grid)
                if skipped_by_beta_gamma > 0:
                    logger.info(
                        f"  Filtered to {len(filtered_grid)}/{len(valid_grid)} configs "
                        f"(beta={args.beta_values}, gamma={args.gamma_values})"
                    )
                valid_grid = filtered_grid

            if manifest_by_prefix is not None:
                prefix_manifest_entries = manifest_by_prefix.get(prefix_id, [])
                if not prefix_manifest_entries:
                    logger.warning(f"  Prefix {prefix_id} is not present in the clustering manifest")
                    valid_grid = []
                else:
                    valid_grid = _filter_valid_grid_with_manifest(
                        prefix_id=prefix_id,
                        valid_grid=valid_grid,
                        prefix_manifest_entries=prefix_manifest_entries,
                        logger=logger,
                    )

            if not valid_grid:
                logger.warning(f"  No valid clustering results for {prefix_id}")
                continue

            max_top_B = max(max(sw.get("top_B", [10])) for sw in sweeps)
            n_features = len(selected_features)

            baseline_branches = utils.build_branches_from_data(
                branches_data, valid_grid[0].get("assignments", [])
            )
            baseline_batch_count = 0
            if baseline_branches:
                baseline_batch_count = max(
                    1, (len(baseline_branches) + effective_batch_size - 1) // effective_batch_size
                )
            baseline_total_batches += baseline_batch_count

            prefix_results = {
                "prefix_id": prefix_id,
                "feature_selection": feature_selection,
                "clustering_runs": {}
            }

            baseline_metadata = None
            batch_prefix_state[prefix_id] = {
                "prefix_results": prefix_results,
                "branches_data": branches_data,
                "active_features": active_features,
                "selected_features": selected_features,
                "H_0": H_0,
                "n_features": n_features,
                "baseline_metadata": baseline_metadata,
                "baseline_branches": baseline_branches,
                "baseline_time_s": 0.0,
                "valid_grid": valid_grid,
                "max_top_B": max_top_B,
                "clustering_ctx": {},
            }

        if baseline_total_batches > 0:
            baseline_pbar = tqdm(total=baseline_total_batches, desc="Baseline log_P total", dynamic_ncols=True, leave=False)

        for prefix_id in prefix_batch:
            state = batch_prefix_state.get(prefix_id)
            if not state:
                continue
            if state["baseline_metadata"] is None:
                logger.info(
                    f"  Computing branch log_P values (batch_size={effective_batch_size}) "
                    f"for {prefix_id}..."
                )
                t0_baseline = time.perf_counter()
                baseline_branch_log_probs = steering.compute_branch_log_probs_batch(
                    model,
                    state["baseline_branches"],
                    logger,
                    batch_size=effective_batch_size,
                    max_seq_len=max_seq_len,
                    progress=baseline_pbar,
                    store_per_token=bool(log_details),
                )
                state["baseline_metadata"] = steering.compute_baseline_metadata(
                    state["baseline_branches"], baseline_branch_log_probs
                )
                state["baseline_time_s"] = time.perf_counter() - t0_baseline

        for prefix_id in prefix_batch:
            state = batch_prefix_state.get(prefix_id)
            if not state:
                continue
            branches_data = state["branches_data"]
            active_features = state["active_features"]
            selected_features = state["selected_features"]
            H_0 = state["H_0"]
            n_features = state["n_features"]
            baseline_metadata = state["baseline_metadata"]
            valid_grid = state["valid_grid"]
            max_top_B = state["max_top_B"]
            prefix_results = state["prefix_results"]

            total_configs = len(valid_grid)
            for config_idx, grid_entry in enumerate(valid_grid, start=1):
                beta = grid_entry.get("beta")
                gamma = grid_entry.get("gamma")
                clustering_key = f"beta{beta}_gamma{gamma}"
                logger.info(f"  Clustering config {config_idx}/{total_configs}: {clustering_key}")
                components = grid_entry.get("components", {})
                assignments = grid_entry.get("assignments", [])

                semantic_graphs = graph.compute_semantic_graphs(components, H_0, logger, center_Hc=center_Hc)
                if not semantic_graphs:
                    logger.warning(f"  No valid semantic graphs for {prefix_id} ({clustering_key})")
                    continue

                # Collect all feature indices needed
                all_needed_indices = set()
                for cluster_id, H_c in semantic_graphs.items():
                    rank_scores = graph.compute_feature_ranking_scores(
                        cluster_id=cluster_id,
                        semantic_graphs=semantic_graphs,
                        n_features=n_features,
                        selection_mode=feature_selection,
                    )
                    H_c_features = H_c[:n_features]
                    top_indices = np.argsort(rank_scores)[-(max_top_B * 2):]
                    for idx in top_indices:
                        if abs(H_c_features[idx]) >= utils.EPSILON_SMALL:
                            all_needed_indices.add(int(idx))

                logger.info(f"  Precomputing {len(all_needed_indices)} decoder vectors ({clustering_key})...")
                t0_cache = time.perf_counter()

                # Batch-fetch decoder vectors
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
                t1_cache = time.perf_counter()

                decoder_cache = graph.build_cluster_decoder_cache(
                    semantic_graphs, global_decoder_cache, active_features, selected_features,
                    max_features=max_top_B * 2,
                    selection_mode=feature_selection,
                )

                features_by_cluster = {}
                t0_encoder = time.perf_counter()
                encoder_cache = {}
                if need_full_cluster_state:
                    for c, cache_data in decoder_cache.items():
                        tuples = []
                        for i in range(len(cache_data['h_c_values'])):
                            tuples.append((
                                cache_data['layers'][i],
                                cache_data['positions'][i],
                                cache_data['feat_ids'][i],
                                cache_data['h_c_values'][i]
                            ))
                        features_by_cluster[c] = tuples

                    encoder_cache = graph.precompute_cluster_encoder_weights(
                        model, features_by_cluster, device
                    )
                t1_encoder = time.perf_counter()

                # Build branches list (using shared utility)
                branches = utils.build_branches_from_data(branches_data, assignments)

                if run_h4a_sweeps and clustering_key not in aggregated_by_clustering:
                    aggregated_by_clustering[clustering_key] = {
                        key: {'r2': [], 'corr': [], 'win_r2': [], 'win_corr': [], 'logit_r2': [], 'logit_corr': []}
                        for key in all_keys
                    }

                prefix_results["clustering_runs"][clustering_key] = {
                    "beta": beta,
                    "gamma": gamma,
                    "n_clusters": len(semantic_graphs),
                    "n_branches": len(branches),
                    "results": {},
                    "timing": {
                        "decoder_cache_s": t1_cache - t0_cache,
                        "encoder_cache_s": t1_encoder - t0_encoder,
                        "baseline_s": state.get("baseline_time_s", 0.0),
                    }
                }

                ctx = {
                    "prefix_id": prefix_id,
                    "branches": branches,
                    "decoder_cache": decoder_cache,
                    "encoder_cache": encoder_cache,
                    "baseline_metadata": baseline_metadata,
                    "semantic_graphs": semantic_graphs,
                    "features_by_cluster": features_by_cluster,
                }
                batch_contexts_by_key.setdefault(clustering_key, []).append(ctx)
                batch_prefix_state[prefix_id]["clustering_ctx"][clustering_key] = ctx

        # Run sweeps for this prefix batch
        for clustering_key, ctx_list in batch_contexts_by_key.items():
            t0_sweep = time.perf_counter()
            if run_h4a_sweeps:
                total_sweeps = 0
                for sw in sweeps:
                    hc_sels = sw.get("h_c_selections", sw.get("hc_selections", ["full"]))
                    top_B_list = sw.get("top_B", [10])
                    total_sweeps += len(hc_sels) * len(top_B_list)

                sweep_idx = 0
                for sw in sweeps:
                    method = sw.get("steering_method")
                    hc_sels = sw.get("h_c_selections", sw.get("hc_selections", ["full"]))
                    top_B_list = sw.get("top_B", [10])
                    eps_list = sw.get("epsilon_values", sw.get("epsilons", [-1.0, 0.0, 1.0]))

                    for hc_sel, top_B in product(hc_sels, top_B_list):
                        sweep_idx += 1
                        key = metrics.generate_sweep_key(method, hc_sel, top_B)
                        logger.info(f"  Sweep {sweep_idx}/{total_sweeps}: {key} ({clustering_key})...")

                        if cross_prefix_batching and prefix_batch_size and len(ctx_list) > 1:
                            results_by_prefix = run_steering_sweep_prefix_batch(
                                model=model,
                                prefix_contexts=ctx_list,
                                epsilons=eps_list,
                                top_B=top_B,
                                steering_method=method,
                                hc_selection=hc_sel,
                                max_samples_per_cluster=max_cluster_samples,
                                log_details=log_details,
                                max_batch_size=max_batch_size,
                                max_seq_len=max_seq_len,
                                logger=logger,
                            )
                            for ctx in ctx_list:
                                prefix_id = ctx["prefix_id"]
                                prefix_results = batch_prefix_state[prefix_id]["prefix_results"]
                                result = results_by_prefix.get(prefix_id, {"error": "no_result"})
                                prefix_results["clustering_runs"][clustering_key]["results"][key] = result

                                agg_key = (method, hc_sel, top_B)
                                if agg_key in aggregated_by_clustering[clustering_key] and 'mean_r2' in result:
                                    aggregated_by_clustering[clustering_key][agg_key]['r2'].append(result['mean_r2'])
                                    aggregated_by_clustering[clustering_key][agg_key]['corr'].append(result.get('mean_corr', 0.0))
                                    aggregated_by_clustering[clustering_key][agg_key]['win_r2'].append(result.get('mean_win_r2', 0.0))
                                    aggregated_by_clustering[clustering_key][agg_key]['win_corr'].append(result.get('mean_win_corr', 0.0))
                                    aggregated_by_clustering[clustering_key][agg_key]['logit_r2'].append(result.get('mean_logit_r2', 0.0))
                                    aggregated_by_clustering[clustering_key][agg_key]['logit_corr'].append(result.get('mean_logit_corr', 0.0))
                        else:
                            for ctx in ctx_list:
                                prefix_id = ctx["prefix_id"]
                                prefix_results = batch_prefix_state[prefix_id]["prefix_results"]
                                result = run_steering_sweep(
                                    model=model,
                                    branches=ctx["branches"],
                                    decoder_cache=ctx["decoder_cache"],
                                    encoder_cache=ctx["encoder_cache"],
                                    baseline_metadata=ctx["baseline_metadata"],
                                    epsilons=eps_list,
                                    top_B=top_B,
                                    steering_method=method,
                                    hc_selection=hc_sel,
                                    max_samples_per_cluster=max_cluster_samples,
                                    log_details=log_details,
                                    max_batch_size=max_batch_size,
                                    cross_prefix_batching=cross_prefix_batching,
                                    max_seq_len=max_seq_len,
                                    logger=logger,
                                )

                                prefix_results["clustering_runs"][clustering_key]["results"][key] = result

                                agg_key = (method, hc_sel, top_B)
                                if agg_key in aggregated_by_clustering[clustering_key] and 'mean_r2' in result:
                                    aggregated_by_clustering[clustering_key][agg_key]['r2'].append(result['mean_r2'])
                                    aggregated_by_clustering[clustering_key][agg_key]['corr'].append(result.get('mean_corr', 0.0))
                                    aggregated_by_clustering[clustering_key][agg_key]['win_r2'].append(result.get('mean_win_r2', 0.0))
                                    aggregated_by_clustering[clustering_key][agg_key]['win_corr'].append(result.get('mean_win_corr', 0.0))
                                    aggregated_by_clustering[clustering_key][agg_key]['logit_r2'].append(result.get('mean_logit_r2', 0.0))
                                    aggregated_by_clustering[clustering_key][agg_key]['logit_corr'].append(result.get('mean_logit_corr', 0.0))

                        maybe_clear_memory(gc_collect=False, cuda_empty_cache=False, min_interval_s=120.0, logger=logger, tag="prefix_batch_end")

            t1_sweep = time.perf_counter()
            for ctx in ctx_list:
                prefix_id = ctx["prefix_id"]
                prefix_results = batch_prefix_state[prefix_id]["prefix_results"]
                prefix_results["clustering_runs"][clustering_key]["timing"]["sweep_s"] = (
                    t1_sweep - t0_sweep if run_h4a_sweeps else 0.0
                )

            # Run H4a_gen / H4c per prefix
            for ctx in ctx_list:
                prefix_id = ctx["prefix_id"]
                prefix_state = batch_prefix_state[prefix_id]
                prefix_results = prefix_state["prefix_results"]
                branches_data = prefix_state["branches_data"]
                active_features = prefix_state["active_features"]
                selected_features = prefix_state["selected_features"]
                semantic_graphs = ctx["semantic_graphs"]
                decoder_cache = ctx["decoder_cache"]
                encoder_cache = ctx["encoder_cache"]
                branches = ctx["branches"]
                features_by_cluster = ctx["features_by_cluster"]
                baseline_metadata = ctx["baseline_metadata"]

                # H4a_gen (generation experiment)
                if run_h4a_gen:
                    num_samples_per_epsilon = steering_config.get("gen_num_samples_per_epsilon", 5)
                    gen_max_new_tokens = steering_config.get("gen_max_new_tokens", 50)
                    gen_temperature = steering_config.get("gen_temperature", 1.0)
                    gen_top_k = steering_config.get("gen_top_k", 50)
                    gen_top_p = steering_config.get("gen_top_p", 0.95)

                    gen_method = primary_sweep_settings["method"]
                    gen_epsilons = primary_sweep_settings["epsilons"]

                    logger.info(f"  Running H4a Generation experiment in sweep mode ({clustering_key})...")
                    h4a_gen_result = validate_h4a_generation(
                        model=model,
                        prefix_token_ids=branches_data.get("prefix_tokens_with_bos", []),
                        prefix_text=branches_data.get("prefix", ""),
                        features_by_cluster=features_by_cluster,
                        cluster_decoder_cache=decoder_cache,
                        cluster_encoder_cache=encoder_cache,
                        epsilon_values=gen_epsilons,
                        steering_method=gen_method,
                        num_samples_per_epsilon=num_samples_per_epsilon,
                        max_new_tokens=gen_max_new_tokens,
                        temperature=gen_temperature,
                        top_k=gen_top_k,
                        top_p=gen_top_p,
                        max_seq_len=max_seq_len,
                        logger=logger
                    )
                    prefix_results["clustering_runs"][clustering_key]["H4a_generation"] = h4a_gen_result

                if (run_h4c or run_h4c_mass) and len(semantic_graphs) >= 2:
                    h4c_method = primary_sweep_settings["method"]
                    h4c_hc_selection = primary_sweep_settings["hc_selection"]
                    h4c_top_B = primary_sweep_settings["top_B"]
                    h4c_epsilon = primary_sweep_settings["positive_epsilon"]
                    h4c_epsilons = primary_sweep_settings["epsilons"]
                    h4c_features_by_cluster, h4c_decoder_cache, h4c_encoder_cache = _build_selected_cluster_state(
                        model=model,
                        cluster_decoder_cache=decoder_cache,
                        device=device,
                        top_B=h4c_top_B,
                        hc_selection=h4c_hc_selection,
                    )

                    if not h4c_features_by_cluster:
                        logger.warning(
                            f"  No H4c features available after selection "
                            f"(hc={h4c_hc_selection}, top_B={h4c_top_B}) for {prefix_id} ({clustering_key})"
                        )
                        if run_h4c:
                            prefix_results["clustering_runs"][clustering_key]["H4c_specificity"] = {
                                "specificity": 0.0,
                                "error": "no_features_after_selection",
                                "n_clusters": len(semantic_graphs),
                                "epsilon": h4c_epsilon,
                                "top_B": h4c_top_B,
                                "hc_selection": h4c_hc_selection,
                            }
                        if run_h4c_mass:
                            prefix_results["clustering_runs"][clustering_key]["H4c_cluster_mass_pairwise"] = {
                                "metric_primary": "cluster_mass_win_corr",
                                "metric_secondary": "cluster_mass_win_spearman",
                                "error": "no_features_after_selection",
                                "n_clusters": len(semantic_graphs),
                                "epsilon_values": h4c_epsilons,
                                "top_B": h4c_top_B,
                                "hc_selection": h4c_hc_selection,
                            }
                        continue

                    if run_h4c:
                        logger.info(f"  Running H4c specificity ({clustering_key})...")
                        h4c_result = validate_h4c_specificity(
                            model, branches, semantic_graphs, active_features, selected_features,
                            h4c_features_by_cluster, h4c_decoder_cache, h4c_encoder_cache,
                            baseline_metadata, h4c_epsilon, h4c_top_B, h4c_method,
                            cross_prefix_batching=cross_prefix_batching,
                            max_samples_per_cluster=max_cluster_samples,
                            batch_size=global_batch_size, max_seq_len=max_seq_len, logger=logger
                        )
                        h4c_result["feature_selection"] = feature_selection
                        h4c_result["hc_selection"] = h4c_hc_selection
                        h4c_result["top_B"] = h4c_top_B
                        prefix_results["clustering_runs"][clustering_key]["H4c_specificity"] = h4c_result

                    if run_h4c_mass:
                        logger.info(f"  Running H4c cluster-mass pairwise ({clustering_key})...")
                        h4c_mass_result = validate_h4c_cluster_mass_pairwise(
                            model, branches, semantic_graphs, active_features, selected_features,
                            h4c_features_by_cluster, h4c_decoder_cache, h4c_encoder_cache,
                            baseline_metadata, h4c_epsilons, h4c_top_B, h4c_method,
                            cross_prefix_batching=cross_prefix_batching,
                            max_samples_per_cluster=max_cluster_samples,
                            batch_size=global_batch_size, max_seq_len=max_seq_len, logger=logger
                        )
                        h4c_mass_result["feature_selection"] = feature_selection
                        h4c_mass_result["hc_selection"] = h4c_hc_selection
                        h4c_mass_result["top_B"] = h4c_top_B
                        prefix_results["clustering_runs"][clustering_key]["H4c_cluster_mass_pairwise"] = h4c_mass_result

        for prefix_id, state in batch_prefix_state.items():
            prefix_results = state["prefix_results"]
            all_results[prefix_id] = prefix_results
            _save_hypothesis_outputs(prefix_results, hypotheses, args.output_dir, hypothesis_dirs)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if baseline_pbar is not None:
            baseline_pbar.close()

    if "H4A" in hypotheses and not getattr(args, "skip_aggregated_summaries", False):
        h4a_dir = args.output_dir / hypothesis_dirs["H4A"]
        h4a_dir.mkdir(parents=True, exist_ok=True)

        for clustering_key, aggregated in aggregated_by_clustering.items():
            logger.info("\n" + "=" * 100)
            logger.info(f"SUMMARY ({clustering_key}): Metrics across prefixes")
            logger.info("=" * 100)
            logger.info(metrics.format_summary_table(aggregated))

            summary_path = h4a_dir / f"aggregated_summary_{clustering_key}.json"
            metrics.save_aggregated_summary(aggregated, prefix_ids, sweeps, summary_path)
            logger.info(f"\nSaved summary to {summary_path}")
    elif "H4A" in hypotheses:
        logger.info("Skipping aggregated H4a summary files (--skip-aggregated-summaries)")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Steering Validation (7c) - Hypothesis Testing")
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--clustering-dir", type=Path, required=True)
    parser.add_argument("--attribution-graphs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/7_validation/7c_steering"))
    parser.add_argument("--config", type=Path, default=Path("configs/default_config.json"))
    parser.add_argument("--prefix-id", type=str, default=None)
    parser.add_argument("--top-B", type=int, default=None, help="Override top_B from config")
    parser.add_argument("--max-cluster-samples", type=int, default=None, help="Override max_cluster_samples")
    parser.add_argument("--max-samples", type=int, default=None, help="Cut to first N samples total")
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory for log files")
    parser.add_argument("--hypotheses", type=str, nargs="+", default=None,
                        help="List of hypotheses to run: H4a, H4a_gen, H4c, H4c_mass")
    parser.add_argument("--feature-selection", type=str, choices=["magnitude", "distinct"], default=None)
    parser.add_argument("--log-details", action="store_true", help="Log detailed per-sample info")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    parser.add_argument("--cross-prefix-batching", action="store_true",
                        help="Enable heterogeneous batching across clusters/prefixes")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip prefixes whose per-prefix output files already exist (sample-wise resume)")
    parser.add_argument("--K-clamp", type=int, default=None,
                        help="Maximum K for downstream steering (filters out K > K_clamp)")
    parser.add_argument("--beta-values", type=float, nargs="+", default=None,
                        help="Filter to only these beta values (e.g., --beta-values 0.75 1.0)")
    parser.add_argument("--gamma-values", type=float, nargs="+", default=None,
                        help="Filter to only these gamma values (e.g., --gamma-values 0.5 0.7)")
    parser.add_argument("--clustering-manifest", type=Path, default=None,
                        help="JSON manifest of exact prefix/beta/gamma/(optional K) clusterings to run")
    parser.add_argument("--skip-aggregated-summaries", action="store_true",
                        help="Do not write aggregated_summary_*.json files for H4a")
    parser.add_argument("--no-center-hc", action="store_true",
                        help="Disable H_c centering: use H_0 + mu_a (raw centroid) instead of Delta_H_c")

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_file = get_log_path("7c_hypotheses", args.log_dir)
    log_level = logging.WARNING if args.quiet else logging.INFO
    logger = setup_logger("hypotheses_validation", log_file=log_file, level=log_level)

    logger.info("=" * 60)
    logger.info("STEERING VALIDATION (HYPOTHESIS TESTING)")
    logger.info("  - Column-wise encoder with dynamic delta computation")
    logger.info("  - Using prefix_context from Stage 3")
    logger.info("=" * 60)

    # Config
    config = load_json(args.config) if args.config.exists() else {}
    steering_config = config.get("stage_7c_steering", config.get("hypothesis_validation", {}).get("steering", {}))
    
    # Global config for batch_size and max_seq_len
    global_config = config.get("global", {})
    global_batch_size = global_config.get("batch_size", 128)
    max_seq_len = global_config.get("max_seq_len", None)  # None means dynamic padding
    
    epsilon_values = steering_config.get("epsilon_values", [-0.5, -0.1, 0.1, 0.5])
    steering_method = steering_config.get("steering_method", steering_config.get("method", "multiplicative"))
    top_B = args.top_B if args.top_B is not None else steering_config.get("top_B", 50)
    max_cluster_samples = args.max_cluster_samples if args.max_cluster_samples is not None else steering_config.get("max_cluster_samples", 100)
    max_samples_per_token = steering_config.get("max_samples_per_token", 20)  # For H4B π-effect testing
    max_samples = args.max_samples if args.max_samples is not None else steering_config.get("max_samples")
    cross_prefix_batching = args.cross_prefix_batching or steering_config.get("cross_prefix_batching", False)
    h_c_strategy = "H_c"
    
    # K_clamp for downstream filtering (CLI overrides config)
    K_clamp = args.K_clamp if args.K_clamp is not None else steering_config.get("K_clamp", None)
    
    # center_Hc: If True (default), use Delta_H_c (centered). If False, use H_0 + mu_a (non-centered).
    # CLI flag --no-center-hc overrides config.
    center_Hc = not getattr(args, "no_center_hc", False)
    if center_Hc:
        center_Hc = steering_config.get("center_Hc", True)

    if args.hypotheses is not None:
        hypotheses = [h.upper() for h in args.hypotheses]
    elif "hypotheses" in steering_config:
        hypotheses = [h.upper() for h in steering_config["hypotheses"]]
    else:
        hypotheses = ["H4A"]
    # Ensure downstream (run_sweep_mode) sees any CLI override.
    steering_config["hypotheses"] = hypotheses

    feature_selection = args.feature_selection or steering_config.get("feature_selection", "magnitude")

    logger.info(f"Batch config: global_batch_size={global_batch_size}, max_seq_len={max_seq_len}")
    if args.beta_values:
        logger.info(f"Filtering to beta values: {args.beta_values}")
    if args.gamma_values:
        logger.info(f"Filtering to gamma values: {args.gamma_values}")
    if args.clustering_manifest:
        logger.info(f"Using clustering manifest: {args.clustering_manifest}")
    logger.info(f"Sampling limits: max_cluster_samples={max_cluster_samples}, max_samples_per_token={max_samples_per_token}, max_samples={max_samples}")
    logger.info(f"Experiment: top_B={top_B}, hypotheses={hypotheses}, feature_selection={feature_selection}")
    logger.info(f"Semantic graph: h_c_strategy={h_c_strategy}, cross_prefix_batching={cross_prefix_batching}, center_Hc={center_Hc}")

    # Model
    model_config = config.get("model", {})
    model_name = model_config.get("base_model", "Qwen/Qwen3-8B")
    transcoder_set = model_config.get("transcoder", "mwhanna/qwen3-8b-transcoders")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reset_cuda_state()
    logger.info(f"Loading model {model_name}...")
    model = ReplacementModel.from_pretrained(
        model_name, transcoder_set, device=device, dtype=torch.bfloat16,
        lazy_encoder=True, lazy_decoder=False
    )
    try:
        logger.info(f"CUDA available: {torch.cuda.is_available()} (device_count={torch.cuda.device_count()})")
        logger.info(f"Model device: {getattr(model.cfg, 'device', device)}; dtype: {getattr(model.cfg, 'dtype', torch.bfloat16)}")
        if torch.cuda.is_available():
            logger.info(f"CUDA current_device={torch.cuda.current_device()} name={torch.cuda.get_device_name(torch.cuda.current_device())}")
    except Exception as e:
        logger.warning(f"Could not log CUDA/model device details: {e}")

    sweeps = steering_config.get("sweeps", [])
    log_details = args.log_details or steering_config.get("log_details", False)
    logger.info("=" * 60)
    logger.info("SWEEP MODE ENABLED")
    logger.info("=" * 60)
    run_sweep_mode(
        model=model,
        args=args,
        config=config,
        steering_config=steering_config,
        sweeps=sweeps,
        feature_selection=feature_selection,
        log_details=log_details,
        device=device,
        logger=logger,
        cross_prefix_batching=cross_prefix_batching,
        K_clamp=K_clamp,
        center_Hc=center_Hc,
    )
    logger.info("SWEEP MODE COMPLETE")
    return


if __name__ == "__main__":
    main()
