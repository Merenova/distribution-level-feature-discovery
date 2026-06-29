#!/usr/bin/env -S uv run python
"""7c_baseline_medoid.py - L1-Medoid H_c Baseline for Steering Validation.

This script implements a baseline for 7c steering that uses the L1-medoid
of each cluster as H_c (instead of component-wise median centroid).

The baseline:
1. Loads RD clustering assignments from Stage 5
2. For each cluster, computes the L1-medoid (actual data point minimizing
   weighted L1 distance to all other cluster members)
3. Uses that medoid's raw attribution as H_c (no centering)
4. Reuses existing 7c steering infrastructure for evaluation

The medoid stays on the data manifold, unlike the component-wise median
which creates a "Frankenstein vector" from different points.

Usage:
    python 7c_baseline_medoid.py --samples-dir ... --clustering-dir ... [options]
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
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
CIRCUIT_TRACER_PATH = Path(__file__).resolve().parents[1] / "circuit-tracer"
sys.path.insert(0, str(CIRCUIT_TRACER_PATH))

from utils.data_utils import load_json, save_json
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

# Import the medoid H_c function from 7c_graph
compute_semantic_graphs_medoid = graph.compute_semantic_graphs_medoid


# =============================================================================
# Helper Functions
# =============================================================================

def load_rd_sweep_results(clustering_file: Path) -> Tuple[List[Dict], int]:
    """Load RD sweep results including assignments for each (beta, gamma) config.
    
    Args:
        clustering_file: Path to {prefix}_sweep_results.json
        
    Returns:
        (grid, K_clamp)
        - grid: List of entries with beta, gamma, K, assignments, etc.
        - K_clamp: K_clamp from sweep_config
    """
    data = load_json(clustering_file)
    grid = data.get("grid", [])
    sweep_config = data.get("sweep_config", {}) or {}
    K_clamp = int(sweep_config.get("K_clamp", sweep_config.get("K_max", 20)))
    
    return grid, K_clamp


def load_prefix_data_for_medoid_baseline(
    prefix_id: str,
    attribution_graphs_dir: Path,
    samples_dir: Path,
    logger,
) -> Dict[str, Any]:
    """Load data needed for medoid baseline.
    
    Args:
        prefix_id: Prefix identifier
        attribution_graphs_dir: Directory with attribution context
        samples_dir: Directory with branch samples
        logger: Logger instance
        
    Returns:
        Dict with branches_data, aggregated_attributions, path_probs, etc.
    """
    logger.info(f"Loading data for prefix: {prefix_id}")
    
    # Load branch samples data
    branches_file = samples_dir / f"{prefix_id}_branches.json"
    branches_data = load_json(branches_file)
    
    # Extract path probabilities for weighted medoid computation
    path_probs = []
    for cont in branches_data.get("continuations", []):
        path_probs.append(cont.get("probability", 1.0))
    path_probs = np.array(path_probs)
    
    # Load attribution context
    prefix_context_file = attribution_graphs_dir / f"{prefix_id}_prefix_context.pt"
    context_data = torch.load(prefix_context_file, weights_only=False)
    
    # Get aggregated attributions (raw, uncentered)
    aggregated_attributions = context_data["aggregated_attributions"].float().numpy()
    
    n_samples = len(branches_data.get("continuations", []))
    logger.info(f"  Loaded {n_samples} samples, attr shape: {aggregated_attributions.shape}")
    
    return {
        "prefix_id": prefix_id,
        "branches_data": branches_data,
        "aggregated_attributions": aggregated_attributions,
        "path_probs": path_probs,
        "n_samples": n_samples,
    }


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="L1-Medoid H_c baseline for 7c steering")
    parser.add_argument("--samples-dir", type=Path, required=True,
                        help="Directory with branch samples (2_branch_sampling)")
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
    parser.add_argument("--prefix-id", type=str, default=None,
                        help="Process only a specific prefix")
    parser.add_argument("--cross-prefix-batching", action="store_true",
                        help="Enable cross-prefix batching")
    parser.add_argument("--prefix-batch-size", type=int, default=None,
                        help="Number of prefixes to batch together")
    parser.add_argument("--use-weights", action="store_true",
                        help="Use path probabilities as weights for medoid computation")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip prefixes whose output file already exists")
    parser.add_argument("--K-clamp", type=int, default=None,
                        help="Maximum K for downstream steering (filters out K > K_clamp)")
    parser.add_argument("--beta-values", type=float, nargs="+", default=None,
                        help="Filter to only these beta values (e.g., --beta-values 0.75 1.0)")
    parser.add_argument("--gamma-values", type=float, nargs="+", default=None,
                        help="Filter to only these gamma values (e.g., --gamma-values 0.5 0.7)")
    args = parser.parse_args()
    
    # Setup logging
    log_file = get_log_path("7c_baseline_medoid", args.log_dir)
    log_level = logging.WARNING if args.quiet else logging.INFO
    logger = setup_logger("medoid_baseline", log_file=log_file, level=log_level)
    
    logger.info("=" * 60)
    logger.info("L1-MEDOID H_c BASELINE FOR 7C STEERING")
    logger.info("=" * 60)
    logger.info(f"Using weights: {args.use_weights}")
    
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
        if args.max_batch_size is None:
            args.max_batch_size = steering_config.get("max_batch_size", 512)
        if not args.cross_prefix_batching:
            args.cross_prefix_batching = steering_config.get("cross_prefix_batching", False)
        if args.prefix_batch_size is None or args.prefix_batch_size <= 0:
            args.prefix_batch_size = steering_config.get("prefix_batch_size", 16)
        if args.K_clamp is None:
            args.K_clamp = steering_config.get("K_clamp", None)
    
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
    
    logger.info(f"Sweep configurations: {len(sweeps)}")
    for i, sw in enumerate(sweeps):
        logger.info(f"  [{i+1}] {sw.get('name', 'unnamed')}: {sw.get('steering_method')} | "
                    f"hc={sw.get('h_c_selections')} | B={sw.get('top_B')} | "
                    f"eps={sw.get('epsilon_values')}")
    
    # Find prefixes
    attr_files = sorted(args.attribution_graphs_dir.glob("*_prefix_context.pt"))
    prefix_ids = [f.stem.replace("_prefix_context", "") for f in attr_files]
    
    if args.prefix_id:
        prefix_ids = [args.prefix_id]
    
    if args.max_samples and len(prefix_ids) > args.max_samples:
        prefix_ids = prefix_ids[:args.max_samples]
    
    logger.info(f"Processing {len(prefix_ids)} prefixes")
    if args.beta_values:
        logger.info(f"Filtering to beta values: {args.beta_values}")
    if args.gamma_values:
        logger.info(f"Filtering to gamma values: {args.gamma_values}")
    
    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    h4a_output_dir = args.output_dir / "H4a_medoid"
    h4a_output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.skip_existing:
        before = len(prefix_ids)
        keep: List[str] = []
        for pid in prefix_ids:
            out_path = h4a_output_dir / f"{pid}_sweep_results.json"
            try:
                if out_path.exists() and out_path.stat().st_size > 0:
                    continue
            except OSError:
                pass
            keep.append(pid)
        prefix_ids = keep
        skipped = before - len(prefix_ids)
        if skipped:
            logger.info(f"Skipping {skipped}/{before} prefixes with existing outputs")
    
    if not prefix_ids:
        logger.info("No prefixes to process.")
        return
    
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
        batch_prefix_state = {}
        batch_contexts_by_key = {}
        
        for prefix_id in batch_prefix_ids:
            # Check required files
            clustering_file = args.clustering_dir / f"{prefix_id}_sweep_results.json"
            if not clustering_file.exists():
                logger.warning(f"Missing clustering file for {prefix_id}")
                continue
            
            # Load data
            try:
                prefix_data = load_prefix_data_for_medoid_baseline(
                    prefix_id,
                    args.attribution_graphs_dir,
                    args.samples_dir,
                    logger,
                )
            except Exception as e:
                logger.error(f"Error loading data for {prefix_id}: {e}")
                continue
            
            # Load attribution context for feature mapping
            active_features, selected_features = graph.load_attribution_context(
                args.attribution_graphs_dir, prefix_id, use_continuation_attribution=True
            )
            n_features = len(selected_features)
            
            # Load RD sweep results
            grid, K_clamp_from_sweep = load_rd_sweep_results(clustering_file)
            effective_K_clamp = args.K_clamp if args.K_clamp is not None else K_clamp_from_sweep
            
            # Filter grid entries
            valid_grid = []
            for entry in grid:
                if not entry.get("assignments") or "error" in entry:
                    continue
                K = entry.get("K", len(entry.get("components", {})))
                if K is None or K <= 1 or K > effective_K_clamp:
                    continue
                
                beta = entry.get("beta")
                gamma = entry.get("gamma")
                
                # Filter by beta/gamma if specified
                if args.beta_values is not None:
                    if not any(abs(beta - b) < 0.001 for b in args.beta_values):
                        continue
                if args.gamma_values is not None:
                    if not any(abs(gamma - g) < 0.001 for g in args.gamma_values):
                        continue
                
                valid_grid.append(entry)
            
            if not valid_grid:
                logger.warning(f"  No valid clustering configs for {prefix_id}")
                continue
            
            logger.info(f"  {prefix_id}: {len(valid_grid)} valid configs")
            
            # Compute baseline log_P using first valid config
            first_assignments = valid_grid[0].get("assignments", [])
            baseline_branches = utils.build_branches_from_data(
                prefix_data["branches_data"], first_assignments
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
            
            # Initialize prefix results
            prefix_results = {
                "prefix_id": prefix_id,
                "baseline_method": "medoid",
                "use_weights": args.use_weights,
                "feature_selection": "magnitude",
                "clustering_runs": {}
            }
            
            batch_prefix_state[prefix_id] = {
                "prefix_data": prefix_data,
                "active_features": active_features,
                "selected_features": selected_features,
                "n_features": n_features,
                "valid_grid": valid_grid,
                "K_clamp": effective_K_clamp,
                "baseline_metadata": baseline_metadata,
                "prefix_results": prefix_results,
            }
        
        if not batch_prefix_state:
            continue
        
        # Process each clustering config
        for prefix_id, state in batch_prefix_state.items():
            prefix_data = state["prefix_data"]
            active_features = state["active_features"]
            selected_features = state["selected_features"]
            n_features = state["n_features"]
            baseline_metadata = state["baseline_metadata"]
            prefix_results = state["prefix_results"]
            
            for grid_entry in state["valid_grid"]:
                beta = grid_entry.get("beta")
                gamma = grid_entry.get("gamma")
                K = grid_entry.get("K")
                clustering_key = f"beta{beta}_gamma{gamma}"
                assignments = grid_entry.get("assignments", [])
                
                logger.info(f"  {prefix_id} / {clustering_key}: K={K}")
                
                # Build semantic graphs from L1-medoid
                t0_hc = time.perf_counter()
                weights = prefix_data["path_probs"] if args.use_weights else None
                semantic_graphs, selected_indices = compute_semantic_graphs_medoid(
                    assignments,
                    prefix_data["aggregated_attributions"],
                    weights=weights,
                    logger=logger
                )
                t1_hc = time.perf_counter()
                
                if not semantic_graphs:
                    continue
                
                # Build branches
                branches = utils.build_branches_from_data(
                    prefix_data["branches_data"], assignments
                )
                
                # Collect feature indices needed
                all_needed_indices = set()
                for cluster_id, H_c in semantic_graphs.items():
                    H_c_features = H_c[:n_features]
                    abs_vals = np.abs(H_c_features)
                    top_indices = np.argsort(abs_vals)[-(max_top_B * 2):]
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
                    max_features=max_top_B * 2
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
                
                # Initialize results for this clustering config
                prefix_results["clustering_runs"][clustering_key] = {
                    "beta": float(beta),
                    "gamma": float(gamma),
                    "K": K,
                    "n_clusters": len(semantic_graphs),
                    "n_branches": len(branches),
                    "selected_indices": selected_indices,
                    "results": {},
                    "timing": {
                        "hc_build_s": t1_hc - t0_hc,
                        "decoder_cache_s": t1_cache - t0_cache,
                        "encoder_cache_s": t1_encoder - t0_encoder,
                    }
                }
                
                # Build context
                ctx = {
                    "prefix_id": prefix_id,
                    "clustering_key": clustering_key,
                    "branches": branches,
                    "decoder_cache": decoder_cache,
                    "encoder_cache": encoder_cache,
                    "baseline_metadata": baseline_metadata,
                    "semantic_graphs": semantic_graphs,
                    "features_by_cluster": features_by_cluster,
                }
                batch_contexts_by_key.setdefault(clustering_key, []).append(ctx)
        
        # Run sweeps
        for clustering_key, ctx_list in batch_contexts_by_key.items():
            t0_sweep = time.perf_counter()
            
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
                            ck = ctx["clustering_key"]
                            result = results_by_prefix.get(p_id, {"error": "no_result"})
                            batch_prefix_state[p_id]["prefix_results"]["clustering_runs"][ck]["results"][key] = result
                    else:
                        # Process each prefix individually
                        for ctx in ctx_list:
                            p_id = ctx["prefix_id"]
                            ck = ctx["clustering_key"]
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
                            batch_prefix_state[p_id]["prefix_results"]["clustering_runs"][ck]["results"][key] = result
                    
                    clear_memory()
            
            t1_sweep = time.perf_counter()
            for ctx in ctx_list:
                p_id = ctx["prefix_id"]
                ck = ctx["clustering_key"]
                if ck in batch_prefix_state[p_id]["prefix_results"]["clustering_runs"]:
                    batch_prefix_state[p_id]["prefix_results"]["clustering_runs"][ck]["timing"]["sweep_s"] = t1_sweep - t0_sweep
        
        # Save results
        for prefix_id, state in batch_prefix_state.items():
            output_file = h4a_output_dir / f"{prefix_id}_sweep_results.json"
            save_json(state["prefix_results"], output_file)
            logger.info(f"  Saved {prefix_id} results to {output_file}")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info("\n" + "=" * 60)
    logger.info("L1-MEDOID H_c BASELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
