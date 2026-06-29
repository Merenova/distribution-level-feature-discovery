#!/usr/bin/env -S uv run python
"""7c_baseline_kmeans.py - K-means Clustering Baseline for Steering Validation.

This script implements a baseline for 7c steering that uses K-means clustering
on semantic embeddings only (vs. RD-clustering that uses both embeddings and attributions).

The baseline:
1. Loads embeddings from Stage 4a
2. For each (beta, gamma) config, reads K from RD sweep results
3. Runs K-means on embeddings to get cluster assignments
4. Computes weighted median of attributions for each cluster to get H_c
5. Reuses existing 7c steering infrastructure for evaluation

Usage:
    python 7c_baseline_kmeans.py --samples-dir ... --clustering-dir ... [options]
"""

import sys
import time
import gc
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from itertools import product

import numpy as np
import torch
from sklearn.cluster import KMeans
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
CIRCUIT_TRACER_PATH = Path(__file__).resolve().parents[1] / "circuit-tracer"
sys.path.insert(0, str(CIRCUIT_TRACER_PATH))

from utils.data_utils import load_json, save_json
from utils.attribution_pooling import load_pooled_attributions
from utils.logging_utils import setup_logger, get_log_path
from utils.memory_utils import clear_memory
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
hypotheses = _import_module("7c_hypotheses", _module_dir / "7c_hypotheses.py")
prefix_sharding = _import_module("stage7_prefix_sharding", _module_dir / "stage7_prefix_sharding.py")

# Import the medoid function from 7c_graph
compute_semantic_graphs_medoid = graph.compute_semantic_graphs_medoid
select_prefix_shard = prefix_sharding.select_prefix_shard


# =============================================================================
# K-means Clustering Functions
# =============================================================================

def _resolve_pooling(cli_pooling: Optional[str], config: Dict[str, Any]) -> str:
    """Resolve attribution pooling with CLI taking precedence over config."""
    return cli_pooling or config.get("clustering", {}).get("pooling", "mean") or "mean"


def validate_kmeans_k(
    raw_k: Any,
    n_samples: int,
    k_clamp: Optional[int] = None,
) -> Tuple[Optional[int], Optional[str]]:
    """Validate K for sklearn KMeans.

    Returns (valid_k, None) when K is usable, otherwise (None, stable_reason).
    """
    if raw_k is None:
        return None, "invalid_k_missing"

    if isinstance(raw_k, (bool, np.bool_)):
        return None, "invalid_k_not_integer"

    if isinstance(raw_k, (int, np.integer)):
        K = int(raw_k)
    elif isinstance(raw_k, (float, np.floating)):
        if not np.isfinite(raw_k) or not float(raw_k).is_integer():
            return None, "invalid_k_not_integer"
        K = int(raw_k)
    elif isinstance(raw_k, str):
        try:
            numeric_k = float(raw_k)
        except ValueError:
            return None, "invalid_k_not_integer"
        if not np.isfinite(numeric_k) or not numeric_k.is_integer():
            return None, "invalid_k_not_integer"
        K = int(numeric_k)
    else:
        return None, "invalid_k_not_integer"

    if K < 2:
        return None, "invalid_k_less_than_2"

    if k_clamp is not None and K > int(k_clamp):
        return None, "invalid_k_above_clamp"

    if K > int(n_samples):
        return None, "invalid_k_above_samples"

    return K, None


def record_skipped_kmeans_config(
    prefix_results: Dict[str, Any],
    clustering_key: str,
    raw_k: Any,
    reason: str,
    n_samples: int,
    k_clamp: Optional[int],
) -> None:
    """Record a skipped KMeans clustering config in prefix_results."""
    prefix_results.setdefault("clustering_runs", {})[clustering_key] = {
        "skipped": True,
        "skip_reason": reason,
        "K": raw_k,
        "n_samples": int(n_samples),
        "K_clamp": int(k_clamp) if k_clamp is not None else None,
        "results": {},
        "timing": {},
    }


def run_kmeans_clustering(
    embeddings: np.ndarray,
    K: int,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Run K-means clustering on semantic embeddings.
    
    Args:
        embeddings: Semantic embeddings [n_samples, d_embedding]
        K: Number of clusters
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (assignments, cluster_centers):
        - assignments: [n_samples] cluster assignment for each sample
        - cluster_centers: [K, d_embedding] cluster centers
    """
    valid_K, invalid_reason = validate_kmeans_k(K, len(embeddings))
    if invalid_reason is not None:
        raise ValueError(invalid_reason)

    kmeans = KMeans(
        n_clusters=valid_K,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=random_state
    )
    assignments = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_
    
    return assignments, centers


def compute_weighted_median_attribution(
    attributions: np.ndarray,
    assignments: np.ndarray,
    path_probs: np.ndarray,
    cluster_id: int
) -> np.ndarray:
    """Compute probability-weighted median of attributions for a cluster.
    
    Uses weighted median along each dimension independently.
    
    Args:
        attributions: Attribution vectors [n_samples, d_attribution]
        assignments: Cluster assignments [n_samples]
        path_probs: Path probabilities [n_samples]
        cluster_id: Cluster ID to compute median for
        
    Returns:
        Weighted median attribution vector [d_attribution]
    """
    mask = assignments == cluster_id
    if not np.any(mask):
        return np.zeros(attributions.shape[1])
    
    cluster_attributions = attributions[mask]  # [n_cluster, d_attribution]
    cluster_weights = path_probs[mask]  # [n_cluster]
    
    # Normalize weights
    total_weight = cluster_weights.sum()
    if total_weight <= 0:
        return np.mean(cluster_attributions, axis=0)
    
    normalized_weights = cluster_weights / total_weight
    
    # Weighted median per dimension
    d_attribution = attributions.shape[1]
    result = np.zeros(d_attribution)
    
    for dim in range(d_attribution):
        dim_values = cluster_attributions[:, dim]
        
        # Sort by value
        sort_idx = np.argsort(dim_values)
        sorted_values = dim_values[sort_idx]
        sorted_weights = normalized_weights[sort_idx]
        
        # Find weighted median (cumulative weight >= 0.5)
        cumulative = np.cumsum(sorted_weights)
        median_idx = np.searchsorted(cumulative, 0.5)
        
        # Clamp to valid range
        median_idx = min(median_idx, len(sorted_values) - 1)
        result[dim] = sorted_values[median_idx]
    
    return result


def compute_weighted_mean_attribution(
    attributions: np.ndarray,
    assignments: np.ndarray,
    path_probs: np.ndarray,
    cluster_id: int
) -> np.ndarray:
    """Compute probability-weighted mean of attributions for a cluster.
    
    Fallback alternative to weighted median.
    
    Args:
        attributions: Attribution vectors [n_samples, d_attribution]
        assignments: Cluster assignments [n_samples]
        path_probs: Path probabilities [n_samples]
        cluster_id: Cluster ID to compute mean for
        
    Returns:
        Weighted mean attribution vector [d_attribution]
    """
    mask = assignments == cluster_id
    if not np.any(mask):
        return np.zeros(attributions.shape[1])
    
    cluster_attributions = attributions[mask]
    cluster_weights = path_probs[mask]
    
    total_weight = cluster_weights.sum()
    if total_weight <= 0:
        return np.mean(cluster_attributions, axis=0)
    
    # Weighted mean
    weighted_sum = np.sum(cluster_weights[:, None] * cluster_attributions, axis=0)
    return weighted_sum / total_weight


def compute_weighted_global_center(
    attributions: np.ndarray,
    path_probs: np.ndarray,
    use_median: bool = True,
) -> np.ndarray:
    """Compute global H_0 consistent with distortion choice.

    - use_median=True  -> probability-weighted coordinate-wise median (L1)
    - use_median=False -> probability-weighted mean (L2)
    """
    W_total = float(path_probs.sum())
    if W_total <= 0:
        return np.zeros(attributions.shape[1])

    if use_median:
        # Reuse the same weighted-median implementation by treating the full set as one "cluster".
        all_assignments = np.zeros(attributions.shape[0], dtype=np.int32)
        return compute_weighted_median_attribution(attributions, all_assignments, path_probs, cluster_id=0)

    return np.sum(path_probs[:, None] * attributions, axis=0) / W_total


def load_rd_sweep_K_values(clustering_file: Path) -> Tuple[Dict[str, int], int]:
    """Extract K values (and K_clamp) from RD sweep results for each (beta, gamma) config.
    
    Args:
        clustering_file: Path to {prefix}_sweep_results.json
        
    Returns:
        (K_map, K_clamp)
        - K_map: Dict mapping "beta{b}_gamma{g}" -> K
        - K_clamp: K_clamp from sweep_config (falls back to K_max for backward compat)
    """
    data = load_json(clustering_file)
    grid = data.get("grid", [])
    sweep_config = data.get("sweep_config", {}) or {}
    # Priority: K_clamp > K_max > default 20
    K_clamp = int(sweep_config.get("K_clamp", sweep_config.get("K_max", 20)))
    
    K_map = {}
    for entry in grid:
        beta = entry.get("beta")
        gamma = entry.get("gamma")
        K = entry.get("K", len(entry.get("components", {})))
        
        if beta is not None and gamma is not None:
            key = f"beta{beta}_gamma{gamma}"
            K_map[key] = K
    
    return K_map, K_clamp


def load_prefix_data_for_baseline(
    prefix_id: str,
    embeddings_dir: Path,
    attribution_graphs_dir: Path,
    samples_dir: Path,
    logger,
    pooling: str = "mean",
    use_median_center: bool = True,
) -> Dict[str, Any]:
    """Load all data needed for K-means baseline.
    
    Similar to cluster.load_prefix_data but tailored for baseline use.
    """
    logger.info(f"Loading data for prefix: {prefix_id}")
    
    # Load branch samples data
    branches_file = samples_dir / f"{prefix_id}_branches.json"
    branches_data = load_json(branches_file)
    prefix = branches_data.get("prefix", "")
    
    # Extract path probabilities
    path_probs = []
    for cont in branches_data.get("continuations", []):
        path_probs.append(cont.get("probability", 0.0))
    path_probs = np.array(path_probs)
    
    # Load embeddings
    embeddings_file = embeddings_dir / f"{prefix_id}_embeddings.npy"
    embeddings = np.load(embeddings_file)
    
    prefix_context_file = attribution_graphs_dir / f"{prefix_id}_prefix_context.pt"
    pooled_attributions = load_pooled_attributions(
        prefix_context_file,
        pooling=pooling,
        meta_file=attribution_graphs_dir / f"{prefix_id}_attribution.json",
    )
    attributions = pooled_attributions.values
    
    # Compute H_0 (shared attribution center) consistent with distortion choice.
    # IMPORTANT: If you use L1-style centers (weighted median) for H_c, you should also
    # center attributions by weighted median (not mean).
    H_0 = compute_weighted_global_center(attributions, path_probs, use_median=use_median_center)
    
    # Center attributions (clustering operates on Delta_H)
    attributions_centered = attributions - H_0
    
    # Batch normalize
    norms_sq = np.sum(attributions_centered ** 2, axis=1)
    rms_norm = np.sqrt(np.mean(norms_sq))
    if rms_norm > 1e-10:
        attributions_centered = attributions_centered / rms_norm
    
    n_samples = len(path_probs)
    logger.info(f"Loaded {n_samples} samples, emb shape: {embeddings.shape}, attr shape: {attributions.shape}")
    
    return {
        "prefix_id": prefix_id,
        "prefix": prefix,
        "embeddings": embeddings,
        "attributions": attributions,  # Original (uncentered) for H_c computation
        "attributions_centered": attributions_centered,
        "H_0": H_0,
        "rms_norm": rms_norm,
        "path_probs": path_probs,
        "branches_data": branches_data,
        "n_samples": n_samples,
    }


def build_semantic_graphs_from_kmeans(
    assignments: np.ndarray,
    attributions: np.ndarray,
    path_probs: np.ndarray,
    H_0: np.ndarray,
    use_median: bool = True
) -> Dict[int, np.ndarray]:
    """Build semantic graphs (H_c) from K-means assignments.
    
    For each cluster, computes Delta_H_c = weighted_center(attributions - H_0)
    and uses Delta_H_c directly as H_c.
    
    Args:
        assignments: Cluster assignments [n_samples]
        attributions: Original (uncentered) attributions [n_samples, d]
        path_probs: Path probabilities [n_samples]
        H_0: Global mean attribution [d]
        use_median: If True, use weighted median; else weighted mean
        
    Returns:
        {cluster_id: H_c array}
    """
    # Convert to Python int to ensure JSON serializability
    unique_clusters = sorted([int(c) for c in set(assignments)])
    
    # Center attributions for computing Delta_H_c
    attributions_centered = attributions - H_0
    
    semantic_graphs = {}
    for c in unique_clusters:
        if use_median:
            Delta_H_c = compute_weighted_median_attribution(
                attributions_centered, assignments, path_probs, c
            )
        else:
            Delta_H_c = compute_weighted_mean_attribution(
                attributions_centered, assignments, path_probs, c
            )
        
        # Use Delta_H_c directly (no H_0 added back)
        semantic_graphs[c] = Delta_H_c
    
    return semantic_graphs


def build_semantic_graphs_from_kmeans_medoid(
    assignments: np.ndarray,
    attributions: np.ndarray,
    path_probs: np.ndarray,
    H_0: np.ndarray,
    use_weights: bool = False
) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    """Build semantic graphs using L1-medoid for each K-means cluster.
    
    The medoid is the actual data point that minimizes weighted L1 distance
    to all other points in the cluster. This stays on the data manifold.
    
    Args:
        assignments: Cluster assignments [n_samples]
        attributions: Original (uncentered) attributions [n_samples, d]
        path_probs: Path probabilities [n_samples]
        H_0: Global mean attribution [d]
        use_weights: If True, use path_probs as weights for medoid
        
    Returns:
        Tuple of (semantic_graphs, selected_indices)
    """
    # Center attributions
    attributions_centered = attributions - H_0
    
    # Use the medoid function from 7c_graph
    weights = path_probs if use_weights else None
    return compute_semantic_graphs_medoid(
        assignments.tolist(),
        attributions_centered,
        weights=weights,
        logger=None
    )


# =============================================================================
# Main Sweep Functions (reusing 7c infrastructure)
# =============================================================================

# run_steering_sweep is imported from 7c_hypotheses.py via the `hypotheses` module
# This avoids code duplication - the function handles both cross_prefix_batching
# and normal batching modes with all metric computation


def _save_kmeans_hypothesis_outputs(
    prefix_results: Dict[str, Any],
    hypotheses_to_run: List[str],
    output_dir: Path,
    hypothesis_dirs: Dict[str, str],
) -> None:
    """Save kmeans outputs while preserving skipped K metadata."""
    prefix_id = prefix_results.get("prefix_id")
    clustering_runs = prefix_results.get("clustering_runs", {})

    for hypothesis in hypotheses_to_run:
        folder_name = hypothesis_dirs.get(hypothesis, hypothesis)
        hypothesis_dir = output_dir / folder_name
        hypothesis_dir.mkdir(parents=True, exist_ok=True)

        per_prefix = {
            "prefix_id": prefix_id,
            "feature_selection": prefix_results.get("feature_selection"),
            "clustering_runs": {}
        }

        for clustering_key, run in clustering_runs.items():
            if run.get("skipped"):
                per_prefix["clustering_runs"][clustering_key] = {
                    "skipped": True,
                    "skip_reason": run.get("skip_reason"),
                    "K": run.get("K"),
                    "n_samples": run.get("n_samples"),
                    "K_clamp": run.get("K_clamp"),
                    "results": run.get("results", {}),
                    "timing": run.get("timing", {}),
                }
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


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="K-means baseline for 7c steering validation")
    parser.add_argument("--samples-dir", type=Path, required=True,
                        help="Directory with branch samples (2_branch_sampling)")
    parser.add_argument("--embeddings-dir", type=Path, required=True,
                        help="Directory with embeddings (4_feature_extraction/embeddings)")
    parser.add_argument("--attribution-graphs-dir", type=Path, required=True,
                        help="Directory with attribution context (3_attribution_graphs)")
    parser.add_argument("--clustering-dir", type=Path, required=True,
                        help="Directory with RD clustering results (5_clustering)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for baseline results")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config JSON file")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="Model name or path")
    parser.add_argument("--transcoder", type=str, default="mwhanna/qwen3-8b-transcoders",
                        help="Transcoder name or path")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of prefixes to process")
    parser.add_argument("--max-cluster-samples", type=int, default=None,
                        help="Maximum samples per cluster")
    parser.add_argument("--max-batch-size", type=int, default=None,
                        help="Maximum batch size for steering")
    parser.add_argument("--pooling", type=str, default=None,
                        choices=["mean", "max", "sum"],
                        help="Pooling method for attributions")
    parser.add_argument("--use-mean", action="store_true",
                        help="Use weighted mean instead of median for H_c")
    parser.add_argument("--use-medoid", action="store_true",
                        help="Use L1-medoid instead of weighted median for H_c (stays on manifold)")
    parser.add_argument("--prefix-id", type=str, default=None,
                        help="Process only a specific prefix")
    parser.add_argument("--prefix-shard-index", type=int, default=0,
                        help="0-based deterministic prefix shard index to process")
    parser.add_argument("--prefix-shard-count", type=int, default=1,
                        help="Total number of deterministic prefix shards")
    parser.add_argument("--cross-prefix-batching", action="store_true",
                        help="Enable cross-prefix batching")
    parser.add_argument("--prefix-batch-size", type=int, default=None,
                        help="Number of prefixes to batch together")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip prefixes whose per-prefix output file already exists (sample-wise resume)")
    parser.add_argument("--K-clamp", type=int, default=None,
                        help="Maximum K for downstream steering (filters out K > K_clamp)")
    parser.add_argument("--beta-values", type=float, nargs="+", default=None,
                        help="Filter to only these beta values (e.g., --beta-values 0.75 1.0)")
    parser.add_argument("--gamma-values", type=float, nargs="+", default=None,
                        help="Filter to only these gamma values (e.g., --gamma-values 0.5 0.7)")
    parser.add_argument("--hypotheses", type=str, nargs="+", default=None,
                        help="Hypotheses to run: H4a, H4c, and/or H4c_mass")
    parser.add_argument("--feature-selection", type=str, choices=["magnitude", "distinct"], default=None,
                        help="Feature ranking mode for top-B selection")
    parser.add_argument("--clustering-manifest", type=Path, default=None,
                        help="JSON manifest of exact prefix/beta/gamma/(optional K) clusterings to run")
    args = parser.parse_args()
    
    # Setup logging
    log_file = get_log_path("7c_baseline_kmeans", args.log_dir)
    log_level = logging.WARNING if args.quiet else logging.INFO
    logger = setup_logger("kmeans_baseline", log_file=log_file, level=log_level)
    
    logger.info("=" * 60)
    logger.info("K-MEANS BASELINE FOR 7C STEERING")
    logger.info("=" * 60)
    
    # Load config
    config = {}
    sweeps = []
    steering_config = {}
    if args.config and args.config.exists():
        with open(args.config) as f:
            config = json.load(f)
        
        # Parse steering sweep config
        steering_config = config.get("stage_7c_steering", {})
        sweeps = steering_config.get("sweeps", [])
        
        # Normalize sweeps
        defaults = {
            "h_c_selections": ["full"],
            "top_B": [10],
            "epsilon_values": [-1.0, 0.0, 1.0]
        }
        sweeps = [utils.validate_and_normalize_sweep_config(sw, defaults) for sw in sweeps]
        
        # Update args from config
        if args.max_cluster_samples is None:
            args.max_cluster_samples = steering_config.get("max_cluster_samples", 20)
        if args.max_samples is None:
            args.max_samples = steering_config.get("max_samples", None)
        if args.max_batch_size is None:
            args.max_batch_size = steering_config.get("max_batch_size", 512)
        if not args.cross_prefix_batching:
            args.cross_prefix_batching = steering_config.get("cross_prefix_batching", False)
        if args.prefix_batch_size is None or args.prefix_batch_size <= 0:
            args.prefix_batch_size = steering_config.get("prefix_batch_size", 16)
        # K_clamp from config if not provided via CLI
        if args.K_clamp is None:
            args.K_clamp = steering_config.get("K_clamp", None)

    args.pooling = _resolve_pooling(args.pooling, config)
    
    if not sweeps:
        # Default sweep
        sweeps = [{
            "name": "default",
            "steering_method": "sign",
            "h_c_selections": ["full", "positive", "negative"],
            "top_B": [5, 10],
            "epsilon_values": [-0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5]
        }]

    if args.max_cluster_samples is None:
        args.max_cluster_samples = 20
    if args.max_batch_size is None:
        args.max_batch_size = 512
    if args.prefix_batch_size is None or args.prefix_batch_size <= 0:
        args.prefix_batch_size = 16

    if args.hypotheses is not None:
        hypotheses_to_run = [h.upper() for h in args.hypotheses]
    elif "hypotheses" in steering_config:
        hypotheses_to_run = [h.upper() for h in steering_config["hypotheses"]]
    else:
        hypotheses_to_run = ["H4A"]
    run_h4a_sweeps = "H4A" in hypotheses_to_run
    run_h4c = "H4C" in hypotheses_to_run
    run_h4c_mass = "H4C_MASS" in hypotheses_to_run

    if run_h4c_mass and not args.use_medoid:
        raise ValueError("H4C_MASS on the K-means baseline requires --use-medoid")

    feature_selection = args.feature_selection or steering_config.get("feature_selection", "magnitude")
    primary_sweep_settings = hypotheses._resolve_primary_sweep_settings(sweeps, steering_config)
    hypothesis_dirs = {"H4A": "H4a", "H4C": "H4c", "H4C_MASS": "H4c_mass"}
    manifest_by_prefix = None
    manifest_prefix_order = None
    if args.clustering_manifest:
        manifest_by_prefix, manifest_prefix_order = hypotheses._load_clustering_manifest(args.clustering_manifest)
    
    logger.info(f"Sweep configurations: {len(sweeps)}")
    for i, sw in enumerate(sweeps):
        logger.info(f"  [{i+1}] {sw.get('name', 'unnamed')}: {sw.get('steering_method')} | "
                    f"hc={sw.get('h_c_selections')} | B={sw.get('top_B')} | "
                    f"eps={sw.get('epsilon_values')}")
    
    # Find prefixes
    embedding_files = sorted(args.embeddings_dir.glob("*_embeddings.npy"))
    prefix_ids = [f.stem.replace("_embeddings", "") for f in embedding_files]

    if manifest_prefix_order is not None and not args.prefix_id:
        discovered_prefixes = set(prefix_ids)
        prefix_ids = [pid for pid in manifest_prefix_order if pid in discovered_prefixes]

    if args.prefix_id:
        prefix_ids = [args.prefix_id]

    prefixes_before_shard = len(prefix_ids)
    prefix_ids = select_prefix_shard(
        prefix_ids,
        shard_index=args.prefix_shard_index,
        shard_count=args.prefix_shard_count,
    )
    logger.info(
        f"Prefix shard index {args.prefix_shard_index} of {args.prefix_shard_count}: "
        f"selected {len(prefix_ids)}/{prefixes_before_shard} prefixes"
    )

    if args.max_samples and len(prefix_ids) > args.max_samples:
        prefix_ids = prefix_ids[:args.max_samples]
        logger.info(f"Limited to first {args.max_samples} prefixes after sharding")

    logger.info(f"Processing {len(prefix_ids)} prefixes")
    if args.beta_values:
        logger.info(f"Filtering to beta values: {args.beta_values}")
    if args.gamma_values:
        logger.info(f"Filtering to gamma values: {args.gamma_values}")
    if args.clustering_manifest:
        logger.info(f"Using clustering manifest: {args.clustering_manifest}")
    logger.info(f"Hypotheses: {hypotheses_to_run}, feature_selection={feature_selection}")
    
    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing:
        before = len(prefix_ids)
        prefix_ids = [
            pid for pid in prefix_ids
            if not hypotheses._prefix_outputs_exist(pid, hypotheses_to_run, args.output_dir, hypothesis_dirs)
        ]
        skipped = before - len(prefix_ids)
        if skipped:
            logger.info(f"Skipping {skipped}/{before} prefixes with existing outputs (sample-wise resume)")
    
    # Load model
    logger.info("Loading model...")
    global_config = config.get("global", {})
    max_seq_len = global_config.get("max_seq_len", 64)
    store_sequence_logit_values = bool(steering_config.get("store_sequence_logit_values", False))
    
    model_config = config.get("model", {})
    model_name = model_config.get("base_model", args.model)
    transcoder_name = model_config.get("transcoder", args.transcoder)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend_arg = resolve_stage_backend(config, "stage_7c_steering")
    backend = resolve_backend(model_name, backend_arg)
    logger.info("Using Stage 7c backend=%s for model=%s", backend, model_name)
    model = ReplacementModel.from_pretrained(
        model_name,
        transcoder_name,
        backend=backend,
        device=device,
        dtype=torch.bfloat16,
        lazy_encoder=True,
        lazy_decoder=False,
    )
    device = get_model_device(model, fallback=device)
    
    # Build max_top_B from sweeps
    max_top_B = max(max(sw.get("top_B", [10])) for sw in sweeps)
    
    # Split prefixes into batches
    prefix_batch_size = args.prefix_batch_size if args.cross_prefix_batching else 1
    prefix_batches = [prefix_ids[i:i + prefix_batch_size] for i in range(0, len(prefix_ids), prefix_batch_size)]
    logger.info(f"Prefix batching: {prefix_batch_size} prefixes per batch, {len(prefix_batches)} batches")
    
    # Process prefix batches
    for batch_idx, batch_prefix_ids in enumerate(tqdm(prefix_batches, desc="Processing batches")):
        logger.info(f"\n{'='*40}")
        logger.info(f"Batch {batch_idx + 1}/{len(prefix_batches)}: {len(batch_prefix_ids)} prefixes")
        logger.info(f"{'='*40}")
        
        # Load data for all prefixes in batch
        batch_prefix_state = {}  # {prefix_id: {...data...}}
        batch_contexts_by_key = {}  # {clustering_key: [ctx, ctx, ...]}
        
        for prefix_id in batch_prefix_ids:
            # Check required files
            clustering_file = args.clustering_dir / f"{prefix_id}_sweep_results.json"
            if not clustering_file.exists():
                logger.warning(f"Missing clustering file for {prefix_id}")
                continue
            
            # Load data
            try:
                    prefix_data = load_prefix_data_for_baseline(
                    prefix_id,
                    args.embeddings_dir,
                    args.attribution_graphs_dir,
                    args.samples_dir,
                    logger,
                        pooling=args.pooling,
                        use_median_center=not args.use_mean,
                )
            except Exception as e:
                logger.error(f"Error loading data for {prefix_id}: {e}")
                continue
            
            # Load attribution context
            active_features, selected_features = graph.load_attribution_context(
                args.attribution_graphs_dir, prefix_id, use_continuation_attribution=True
            )
            n_features = len(selected_features)
            
            # Load RD sweep K values (baseline uses RD only to choose K and optionally filter by K_clamp)
            K_map, K_clamp_from_sweep = load_rd_sweep_K_values(clustering_file)
            # Use K_clamp if provided via CLI, otherwise use sweep_config K_clamp
            effective_K_clamp = args.K_clamp if args.K_clamp is not None else K_clamp_from_sweep

            # Initialize prefix results before warm-start so invalid configs can be recorded.
            prefix_results = {
                "prefix_id": prefix_id,
                "baseline_method": "kmeans_medoid" if args.use_medoid else "kmeans",
                "feature_selection": feature_selection,
                "clustering_runs": {}
            }

            if manifest_by_prefix is not None:
                prefix_manifest_entries = manifest_by_prefix.get(prefix_id, [])
                if not prefix_manifest_entries:
                    logger.warning(f"  Prefix {prefix_id} is not present in the clustering manifest")
                    continue
                allowed_keys = []
                missing_keys = []
                mismatched_k = []
                for manifest_entry in prefix_manifest_entries:
                    key = f"beta{manifest_entry['beta']}_gamma{manifest_entry['gamma']}"
                    actual_k = K_map.get(key)
                    if actual_k is None:
                        missing_keys.append(key)
                        continue
                    expected_k = manifest_entry.get("K")
                    if expected_k is not None:
                        expected_valid_K, expected_invalid_reason = validate_kmeans_k(
                            expected_k,
                            prefix_data["n_samples"],
                            effective_K_clamp,
                        )
                        if expected_invalid_reason is not None:
                            record_skipped_kmeans_config(
                                prefix_results,
                                key,
                                expected_k,
                                expected_invalid_reason,
                                prefix_data["n_samples"],
                                effective_K_clamp,
                            )
                            continue

                        actual_valid_K, actual_invalid_reason = validate_kmeans_k(
                            actual_k,
                            prefix_data["n_samples"],
                            effective_K_clamp,
                        )
                        if actual_invalid_reason is not None:
                            record_skipped_kmeans_config(
                                prefix_results,
                                key,
                                actual_k,
                                actual_invalid_reason,
                                prefix_data["n_samples"],
                                effective_K_clamp,
                            )
                            continue

                        if actual_valid_K != expected_valid_K:
                            mismatched_k.append((key, expected_k, actual_k))
                            continue
                    allowed_keys.append(key)
                if missing_keys:
                    logger.warning(
                        f"  Missing {len(missing_keys)} manifest configs for {prefix_id}: "
                        f"{missing_keys[:3]}{'...' if len(missing_keys) > 3 else ''}"
                    )
                if mismatched_k:
                    logger.warning(
                        f"  Skipped {len(mismatched_k)} manifest configs due to K mismatch for {prefix_id}: "
                        f"{mismatched_k[:3]}{'...' if len(mismatched_k) > 3 else ''}"
                    )
                logger.info(
                    f"  Manifest filtered to {len(allowed_keys)}/{len(prefix_manifest_entries)} "
                    f"requested configs for {prefix_id}"
                )
                K_map = {key: K_map[key] for key in allowed_keys}

            # IMPORTANT: Do NOT reuse RD clustering's H_0 here.
            # Baseline centers using its own L1-consistent H_0 (weighted median) from attributions + path_probs.
            H_0 = prefix_data["H_0"]
            
            # Compute baseline log_P
            logger.info(f"  {prefix_id}: Computing baseline log_P...")
            # Pick a valid K for the baseline warm-start; skip prefixes with only invalid Ks
            valid_Ks = []
            for clustering_key, raw_k in K_map.items():
                valid_K, invalid_reason = validate_kmeans_k(
                    raw_k,
                    prefix_data["n_samples"],
                    effective_K_clamp,
                )
                if invalid_reason is not None:
                    record_skipped_kmeans_config(
                        prefix_results,
                        clustering_key,
                        raw_k,
                        invalid_reason,
                        prefix_data["n_samples"],
                        effective_K_clamp,
                    )
                    logger.warning(
                        f"  {prefix_id}: Skipping {clustering_key} with K={raw_k}: {invalid_reason}"
                    )
                    continue
                valid_Ks.append(valid_K)

            if not valid_Ks:
                if not K_map and not prefix_results["clustering_runs"]:
                    # Fallback when sweep data is missing: use min(5, K_clamp) to respect K_clamp
                    fallback_clamp = int(effective_K_clamp) if effective_K_clamp is not None else 5
                    fallback_K = min(5, fallback_clamp) if fallback_clamp > 1 else 2
                    valid_fallback_K, fallback_reason = validate_kmeans_k(
                        fallback_K,
                        prefix_data["n_samples"],
                        effective_K_clamp,
                    )
                    if fallback_reason is not None:
                        logger.warning(
                            f"  {prefix_id}: No sweep data and fallback K={fallback_K} is invalid: "
                            f"{fallback_reason}; skipping baseline"
                        )
                        continue
                    valid_Ks = [valid_fallback_K]
                    logger.warning(f"  {prefix_id}: No sweep data, using fallback K={valid_fallback_K}")
                else:
                    logger.warning(
                        f"  {prefix_id}: No valid K for n_samples={prefix_data['n_samples']} "
                        f"and K_clamp={effective_K_clamp}; saving skipped configs"
                    )
                    _save_kmeans_hypothesis_outputs(
                        prefix_results,
                        hypotheses_to_run,
                        args.output_dir,
                        hypothesis_dirs,
                    )
                    continue
            first_K = valid_Ks[0]
            temp_assignments = run_kmeans_clustering(prefix_data["embeddings"], first_K)[0]
            baseline_branches = utils.build_branches_from_data(
                prefix_data["branches_data"], temp_assignments.tolist()
            )
            
            baseline_branch_log_probs = steering.compute_branch_log_probs_batch(
                model, baseline_branches, logger,
                batch_size=args.max_batch_size,
                max_seq_len=max_seq_len,
                store_per_token=store_sequence_logit_values,
            )
            baseline_metadata = steering.compute_baseline_metadata(
                baseline_branches, baseline_branch_log_probs
            )
            
            batch_prefix_state[prefix_id] = {
                "prefix_data": prefix_data,
                "active_features": active_features,
                "selected_features": selected_features,
                "n_features": n_features,
                "K_map": K_map,
                "K_clamp": effective_K_clamp,
                "H_0": H_0,
                "baseline_metadata": baseline_metadata,
                "prefix_results": prefix_results,
                "clustering_ctx": {}
            }
        
        if not batch_prefix_state:
            continue
        
        # Process each (beta, gamma) config - prepare contexts for all prefixes
        all_clustering_keys = set()
        for state in batch_prefix_state.values():
            all_clustering_keys.update(state["K_map"].keys())
        
        # Filter clustering keys by beta/gamma if specified
        if args.beta_values is not None or args.gamma_values is not None:
            filtered_keys = set()
            for key in all_clustering_keys:
                # Parse beta and gamma from key like "beta0.75_gamma0.5"
                parts = key.split("_")
                beta = float(parts[0].replace("beta", ""))
                gamma = float(parts[1].replace("gamma", ""))
                
                beta_ok = args.beta_values is None or any(abs(beta - b) < 0.001 for b in args.beta_values)
                gamma_ok = args.gamma_values is None or any(abs(gamma - g) < 0.001 for g in args.gamma_values)
                
                if beta_ok and gamma_ok:
                    filtered_keys.add(key)
            
            skipped = len(all_clustering_keys) - len(filtered_keys)
            if skipped > 0:
                logger.info(f"  Filtered to {len(filtered_keys)}/{len(all_clustering_keys)} configs (beta={args.beta_values}, gamma={args.gamma_values})")
            all_clustering_keys = filtered_keys
        
        for clustering_key in sorted(all_clustering_keys):
            logger.info(f"  Processing clustering config: {clustering_key}")
            
            for prefix_id, state in batch_prefix_state.items():
                K_map = state["K_map"]
                if clustering_key not in K_map:
                    continue
                    
                K = K_map[clustering_key]
                K_clamp = state.get("K_clamp", 20)
                prefix_data = state["prefix_data"]
                
                valid_K, invalid_reason = validate_kmeans_k(
                    K,
                    prefix_data["n_samples"],
                    K_clamp,
                )
                if invalid_reason is not None:
                    record_skipped_kmeans_config(
                        state["prefix_results"],
                        clustering_key,
                        K,
                        invalid_reason,
                        prefix_data["n_samples"],
                        K_clamp,
                    )
                    logger.warning(
                        f"  {prefix_id}: Skipping {clustering_key} with K={K}: {invalid_reason}"
                    )
                    continue
                K = valid_K
                
                H_0 = state["H_0"]
                active_features = state["active_features"]
                selected_features = state["selected_features"]
                n_features = state["n_features"]
                
                # Run K-means
                t0_kmeans = time.perf_counter()
                assignments, centers = run_kmeans_clustering(
                    prefix_data["embeddings"], K, random_state=42
                )
                t1_kmeans = time.perf_counter()
                
                # Build semantic graphs
                if args.use_medoid:
                    semantic_graphs, selected_indices = build_semantic_graphs_from_kmeans_medoid(
                        assignments, prefix_data["attributions"],
                        prefix_data["path_probs"], H_0,
                        use_weights=False  # Uniform weights for fair comparison
                    )
                else:
                    semantic_graphs = build_semantic_graphs_from_kmeans(
                        assignments, prefix_data["attributions"],
                        prefix_data["path_probs"], H_0,
                        use_median=not args.use_mean
                    )
                    selected_indices = None
                
                if not semantic_graphs:
                    continue
                
                # Collect feature indices
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
                
                # Build decoder cache
                t0_cache = time.perf_counter()
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
                for c, cache_data in decoder_cache.items():
                    tuples = [(
                        cache_data['layers'][i],
                        cache_data['positions'][i],
                        cache_data['feat_ids'][i],
                        cache_data['h_c_values'][i]
                    ) for i in range(len(cache_data['h_c_values']))]
                    features_by_cluster[c] = tuples
                
                t0_encoder = time.perf_counter()
                encoder_cache = graph.precompute_cluster_encoder_weights(
                    model, features_by_cluster, device
                )
                t1_encoder = time.perf_counter()
                
                # Build branches
                branches = utils.build_branches_from_data(
                    prefix_data["branches_data"], assignments.tolist()
                )
                
                # Initialize results for this clustering config
                run_data = {
                    "beta": float(clustering_key.split("_")[0].replace("beta", "")),
                    "gamma": float(clustering_key.split("_")[1].replace("gamma", "")),
                    "K": K,
                    "n_clusters": len(semantic_graphs),
                    "n_branches": len(branches),
                    "results": {},
                    "timing": {
                        "kmeans_s": t1_kmeans - t0_kmeans,
                        "decoder_cache_s": t1_cache - t0_cache,
                        "encoder_cache_s": t1_encoder - t0_encoder,
                    }
                }
                if args.use_medoid and selected_indices is not None:
                    run_data["selected_indices"] = selected_indices
                state["prefix_results"]["clustering_runs"][clustering_key] = run_data
                
                # Build context for batching
                ctx = {
                    "prefix_id": prefix_id,
                    "branches": branches,
                    "decoder_cache": decoder_cache,
                    "encoder_cache": encoder_cache,
                    "baseline_metadata": state["baseline_metadata"],
                    "semantic_graphs": semantic_graphs,
                    "features_by_cluster": features_by_cluster,
                }
                batch_contexts_by_key.setdefault(clustering_key, []).append(ctx)
                state["clustering_ctx"][clustering_key] = ctx
        
        # Run sweeps for this batch
        for clustering_key, ctx_list in batch_contexts_by_key.items():
            t0_sweep = time.perf_counter()
            if run_h4a_sweeps:
                for sw in sweeps:
                    method = sw.get("steering_method")
                    hc_sels = sw.get("h_c_selections", ["full"])
                    top_B_list = sw.get("top_B", [10])
                    eps_list = sw.get("epsilon_values", [-1.0, 0.0, 1.0])
                    
                    for hc_sel, top_B in product(hc_sels, top_B_list):
                        key = metrics.generate_sweep_key(method, hc_sel, top_B)
                        logger.info(f"    Sweep: {key} ({clustering_key}, {len(ctx_list)} prefixes)")
                        
                        if args.cross_prefix_batching and len(ctx_list) > 1:
                            # Batch across prefixes
                            results_by_prefix = hypotheses.run_steering_sweep_prefix_batch(
                                model=model,
                                prefix_contexts=ctx_list,
                                epsilons=eps_list,
                                top_B=top_B,
                                steering_method=method,
                                hc_selection=hc_sel,
                                max_samples_per_cluster=args.max_cluster_samples,
                                log_details=False,
                                max_batch_size=args.max_batch_size,
                                max_seq_len=max_seq_len,
                                store_sequence_logit_values=store_sequence_logit_values,
                                logger=logger,
                            )
                            for ctx in ctx_list:
                                p_id = ctx["prefix_id"]
                                result = results_by_prefix.get(p_id, {"error": "no_result"})
                                batch_prefix_state[p_id]["prefix_results"]["clustering_runs"][clustering_key]["results"][key] = result
                        else:
                            # Process each prefix individually
                            for ctx in ctx_list:
                                p_id = ctx["prefix_id"]
                                result = hypotheses.run_steering_sweep(
                                    model=model,
                                    branches=ctx["branches"],
                                    decoder_cache=ctx["decoder_cache"],
                                    encoder_cache=ctx["encoder_cache"],
                                    baseline_metadata=ctx["baseline_metadata"],
                                    epsilons=eps_list,
                                    top_B=top_B,
                                    steering_method=method,
                                    hc_selection=hc_sel,
                                    max_samples_per_cluster=args.max_cluster_samples,
                                    log_details=False,
                                    max_batch_size=args.max_batch_size,
                                    cross_prefix_batching=False,
                                    max_seq_len=max_seq_len,
                                    store_sequence_logit_values=store_sequence_logit_values,
                                    logger=logger,
                                )
                                batch_prefix_state[p_id]["prefix_results"]["clustering_runs"][clustering_key]["results"][key] = result
                        
                        clear_memory()

            t1_sweep = time.perf_counter()
            for ctx in ctx_list:
                p_id = ctx["prefix_id"]
                if clustering_key in batch_prefix_state[p_id]["prefix_results"]["clustering_runs"]:
                    batch_prefix_state[p_id]["prefix_results"]["clustering_runs"][clustering_key]["timing"]["sweep_s"] = (
                        t1_sweep - t0_sweep if run_h4a_sweeps else 0.0
                    )

            if run_h4c or run_h4c_mass:
                for ctx in ctx_list:
                    p_id = ctx["prefix_id"]
                    state = batch_prefix_state[p_id]
                    run_record = state["prefix_results"]["clustering_runs"].get(clustering_key)
                    if run_record is None:
                        continue

                    semantic_graphs = ctx["semantic_graphs"]
                    if len(semantic_graphs) < 2:
                        continue

                    h4c_features_by_cluster, h4c_decoder_cache, h4c_encoder_cache = hypotheses._build_selected_cluster_state(
                        model=model,
                        cluster_decoder_cache=ctx["decoder_cache"],
                        device=device,
                        top_B=primary_sweep_settings["top_B"],
                        hc_selection=primary_sweep_settings["hc_selection"],
                    )
                    if not h4c_features_by_cluster:
                        if run_h4c:
                            run_record["H4c_specificity"] = {
                                "specificity": 0.0,
                                "error": "no_features_after_selection",
                                "n_clusters": len(semantic_graphs),
                                "epsilon": primary_sweep_settings["positive_epsilon"],
                                "top_B": primary_sweep_settings["top_B"],
                                "hc_selection": primary_sweep_settings["hc_selection"],
                                "feature_selection": feature_selection,
                            }
                        if run_h4c_mass:
                            run_record["H4c_cluster_mass_pairwise"] = {
                                "metric_primary": "cluster_mass_win_corr",
                                "metric_secondary": "cluster_mass_win_spearman",
                                "error": "no_features_after_selection",
                                "n_clusters": len(semantic_graphs),
                                "epsilon_values": primary_sweep_settings["epsilons"],
                                "top_B": primary_sweep_settings["top_B"],
                                "hc_selection": primary_sweep_settings["hc_selection"],
                                "feature_selection": feature_selection,
                            }
                        continue

                    if run_h4c:
                        logger.info(f"  Running H4c specificity ({clustering_key}, {p_id})...")
                        h4c_result = hypotheses.validate_h4c_specificity(
                            model=model,
                            branches=ctx["branches"],
                            semantic_graphs=semantic_graphs,
                            active_features=state["active_features"],
                            selected_features=state["selected_features"],
                            features_by_cluster=h4c_features_by_cluster,
                            cluster_decoder_cache=h4c_decoder_cache,
                            cluster_encoder_cache=h4c_encoder_cache,
                            baseline_metadata=ctx["baseline_metadata"],
                            epsilon=primary_sweep_settings["positive_epsilon"],
                            top_B=primary_sweep_settings["top_B"],
                            steering_method=primary_sweep_settings["method"],
                            cross_prefix_batching=False,
                            max_samples_per_cluster=args.max_cluster_samples,
                            batch_size=args.max_batch_size,
                            max_seq_len=max_seq_len,
                            logger=logger,
                        )
                        h4c_result["feature_selection"] = feature_selection
                        h4c_result["hc_selection"] = primary_sweep_settings["hc_selection"]
                        h4c_result["top_B"] = primary_sweep_settings["top_B"]
                        run_record["H4c_specificity"] = h4c_result

                    if run_h4c_mass:
                        logger.info(f"  Running H4c cluster-mass pairwise ({clustering_key}, {p_id})...")
                        h4c_mass_result = hypotheses.validate_h4c_cluster_mass_pairwise(
                            model=model,
                            branches=ctx["branches"],
                            semantic_graphs=semantic_graphs,
                            active_features=state["active_features"],
                            selected_features=state["selected_features"],
                            features_by_cluster=h4c_features_by_cluster,
                            cluster_decoder_cache=h4c_decoder_cache,
                            cluster_encoder_cache=h4c_encoder_cache,
                            baseline_metadata=ctx["baseline_metadata"],
                            epsilon_values=primary_sweep_settings["epsilons"],
                            top_B=primary_sweep_settings["top_B"],
                            steering_method=primary_sweep_settings["method"],
                            cross_prefix_batching=False,
                            max_samples_per_cluster=args.max_cluster_samples,
                            batch_size=args.max_batch_size,
                            max_seq_len=max_seq_len,
                            logger=logger,
                        )
                        h4c_mass_result["feature_selection"] = feature_selection
                        h4c_mass_result["hc_selection"] = primary_sweep_settings["hc_selection"]
                        h4c_mass_result["top_B"] = primary_sweep_settings["top_B"]
                        run_record["H4c_cluster_mass_pairwise"] = h4c_mass_result
        
        # Save results for all prefixes in batch
        for prefix_id, state in batch_prefix_state.items():
            _save_kmeans_hypothesis_outputs(
                state["prefix_results"], hypotheses_to_run, args.output_dir, hypothesis_dirs
            )
            logger.info(f"  Saved {prefix_id} results under {args.output_dir}")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info("\n" + "=" * 60)
    logger.info("K-MEANS BASELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
