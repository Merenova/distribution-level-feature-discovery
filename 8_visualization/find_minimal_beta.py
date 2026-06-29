#!/usr/bin/env python3
"""Find minimal beta that produces each target K for RD clustering.

Uses binary search on beta to find the smallest beta value that causes
RD clustering to split into at least K clusters.

Higher beta = more distortion penalty = more splits = higher K

Optimized version:
- Uses GPU acceleration when available
- Thread-based parallelism (threads share GPU context)
- Caches clustering results to avoid redundant computations
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "5_gaussian_clustering"))

from cluster import load_prefix_data
from initialize import initialize_single_component
from em_loop import run_em_iteration, check_convergence, GPU_AVAILABLE
from adaptive_control import apply_adaptive_control
from rd_objective import compute_full_rd_statistics, compute_component_masses, compute_component_variance


def run_clustering_get_K(
    data: Dict,
    beta: float,
    gamma: float,
    max_iterations: int = 100,
    convergence_threshold: float = 1e-4,
    metric_a: str = "l1",
    normalize_dims: bool = False,
    use_gpu: bool = True,
) -> Tuple[int, Dict]:
    """Run RD clustering and return resulting K and statistics.
    
    Args:
        data: Data dict with embeddings_e, attributions_a, path_probs
        beta: Overall distortion weight
        gamma: Semantic vs attribution trade-off (beta_e = gamma*beta, beta_a = (1-gamma)*beta)
        max_iterations: Maximum EM iterations
        convergence_threshold: Convergence threshold
        metric_a: Attribution metric ("l1" or "l2")
        normalize_dims: Whether to normalize beta by dimensions
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Tuple of (K, stats_dict)
    """
    embeddings_e = data["embeddings_e"]
    attributions_a = data["attributions_a"]
    path_probs = data["path_probs"]
    
    # Get dimensions for normalization
    d_e = embeddings_e.shape[1]
    d_a = attributions_a.shape[1]
    
    # Compute beta_e, beta_a with optional dimension normalization
    if normalize_dims:
        beta_e = gamma * beta / np.sqrt(d_e)
        beta_a = (1 - gamma) * beta / d_a
    else:
        beta_e = gamma * beta
        beta_a = (1 - gamma) * beta
    
    # Initialize with single component (K=1)
    components, assignments = initialize_single_component(
        embeddings_e, attributions_a, path_probs, metric_a=metric_a
    )
    
    next_component_id = max(components.keys()) + 1 if components else 2
    L_RD_prev = np.inf
    
    # EM loop with GPU acceleration
    for iteration in range(max_iterations):
        assignments, components, rd_stats = run_em_iteration(
            embeddings_e, attributions_a, path_probs,
            components, beta_e, beta_a,
            metric_a=metric_a,
            use_gpu=use_gpu,
        )
        
        L_RD_curr = rd_stats['L_RD']
        
        # Adaptive control (split/junk)
        P_bar = rd_stats['P_bar']
        Var_e = rd_stats.get('Var_e', {})
        Var_a = rd_stats.get('Var_a', {})
        
        if not Var_e or not Var_a:
            W_c, _ = compute_component_masses(assignments, path_probs, list(components.keys()))
            for c, comp in components.items():
                indices = [i for i, a in enumerate(assignments) if a == c]
                W_c_val = W_c.get(c, 0)
                if not indices:
                    Var_e[c] = 0.0
                    Var_a[c] = 0.0
                    continue
                Var_e[c] = compute_component_variance(
                    embeddings_e[indices], comp['mu_e'], path_probs[indices], W_c_val, "l2"
                )
                Var_a[c] = compute_component_variance(
                    attributions_a[indices], comp['mu_a'], path_probs[indices], W_c_val, metric_a
                )
        
        components, assignments, next_component_id = apply_adaptive_control(
            embeddings_e, attributions_a, path_probs,
            assignments, components, P_bar, Var_e, Var_a,
            beta_e, beta_a,
            next_component_id=next_component_id,
            metric_a=metric_a,
        )
        
        if len(components) > 0:
            rd_stats = compute_full_rd_statistics(
                embeddings_e, attributions_a, assignments, path_probs,
                components, beta_e, beta_a,
                metric_a=metric_a,
            )
            L_RD_curr = rd_stats['L_RD']
        
        if check_convergence(L_RD_prev, L_RD_curr, convergence_threshold):
            break
        
        L_RD_prev = L_RD_curr
    
    K = len(components)
    stats = {
        "K": K,
        "H": float(rd_stats["H"]),
        "D_e": float(rd_stats["D_e"]),
        "D_a": float(rd_stats["D_a"]),
        "L_RD": float(rd_stats["L_RD"]),
        "n_iterations": iteration + 1,
    }
    
    return K, stats


def binary_search_beta(
    data: Dict,
    target_K: int,
    gamma: float,
    beta_lo: float = 0.1,
    beta_hi: float = 3.0,
    tolerance: float = 0.05,
    max_iterations: int = 50,
    metric_a: str = "l1",
    normalize_dims: bool = False,
    use_gpu: bool = True,
    cache: Optional[Dict] = None,
) -> Dict:
    """Binary search for minimal beta that produces at least target_K clusters.
    
    Args:
        data: Data dict with embeddings
        target_K: Target number of clusters
        gamma: Semantic vs attribution trade-off
        beta_lo: Lower bound for beta search
        beta_hi: Upper bound for beta search
        tolerance: Stop when beta_hi - beta_lo < tolerance
        max_iterations: Max clustering iterations
        metric_a: Attribution metric
        normalize_dims: Whether to normalize beta by dimensions
        use_gpu: Whether to use GPU acceleration
        cache: Optional cache dict to store/retrieve results
        
    Returns:
        Dict with minimal_beta, actual_K, stats, and search_history with all tested betas
    """
    # Track all tested betas with full stats
    search_history = []
    
    def get_K_at_beta(beta: float) -> Tuple[int, Dict]:
        """Get K at given beta, using cache if available."""
        cache_key = f"{beta:.4f}"
        if cache is not None and cache_key in cache:
            cached = cache[cache_key]
            # Add to search history
            search_history.append({
                "beta": beta,
                "K": cached["K"],
                "H": cached.get("H", 0),
                "D_e": cached.get("D_e", 0),
                "D_a": cached.get("D_a", 0),
                "L_RD": cached.get("L_RD", 0),
            })
            return cached["K"], cached
        
        K, stats = run_clustering_get_K(
            data, beta, gamma, max_iterations, metric_a=metric_a, 
            normalize_dims=normalize_dims, use_gpu=use_gpu
        )
        
        # Store in cache with full stats
        result = {"K": K, **stats}
        if cache is not None:
            cache[cache_key] = result
        
        # Add to search history
        search_history.append({
            "beta": beta,
            "K": K,
            "H": stats.get("H", 0),
            "D_e": stats.get("D_e", 0),
            "D_a": stats.get("D_a", 0),
            "L_RD": stats.get("L_RD", 0),
        })
        
        return K, stats
    
    # First check if target K is achievable at beta_hi
    K_hi, stats_hi = get_K_at_beta(beta_hi)
    
    if K_hi < target_K:
        # Cannot achieve target K even at max beta
        return {
            "found": False,
            "target_K": target_K,
            "max_K_at_beta_hi": K_hi,
            "beta_hi": beta_hi,
            "search_history": search_history,
        }
    
    # Check if we already have K >= target at beta_lo
    K_lo, stats_lo = get_K_at_beta(beta_lo)
    
    if K_lo >= target_K:
        # Already at target with lowest beta
        return {
            "found": True,
            "minimal_beta": beta_lo,
            "actual_K": K_lo,
            "target_K": target_K,
            "H": stats_lo.get("H", 0),
            "D_e": stats_lo.get("D_e", 0),
            "D_a": stats_lo.get("D_a", 0),
            "L_RD": stats_lo.get("L_RD", 0),
            "search_history": search_history,
        }
    
    # Binary search
    best_beta = beta_hi
    best_K = K_hi
    best_stats = stats_hi
    
    while beta_hi - beta_lo > tolerance:
        beta_mid = (beta_lo + beta_hi) / 2
        K_mid, stats_mid = get_K_at_beta(beta_mid)
        
        if K_mid >= target_K:
            # Found a valid beta, try to find smaller
            best_beta = beta_mid
            best_K = K_mid
            best_stats = stats_mid
            beta_hi = beta_mid
        else:
            # Need higher beta
            beta_lo = beta_mid
    
    return {
        "found": True,
        "minimal_beta": best_beta,
        "actual_K": best_K,
        "target_K": target_K,
        "H": best_stats.get("H", 0),
        "D_e": best_stats.get("D_e", 0),
        "D_a": best_stats.get("D_a", 0),
        "L_RD": best_stats.get("L_RD", 0),
        "search_history": search_history,
    }


def _process_prefix_worker(args_tuple):
    """Worker function for ProcessPoolExecutor."""
    (prefix_id, embeddings_dir, attribution_graphs_dir, samples_dir,
     target_K_values, gamma_values, beta_lo, beta_hi, beta_tolerance,
     max_iterations, metric_a, normalize_dims, use_gpu) = args_tuple
    
    # Convert paths back to Path objects
    embeddings_dir = Path(embeddings_dir)
    attribution_graphs_dir = Path(attribution_graphs_dir)
    samples_dir = Path(samples_dir)
    
    # Create a minimal logger for worker
    logger = logging.getLogger(f"worker_{prefix_id}")
    logger.setLevel(logging.WARNING)
    
    try:
        # Load data once for this prefix
        data = load_prefix_data(
            prefix_id, embeddings_dir, attribution_graphs_dir, samples_dir,
            logger, metric_a=metric_a
        )
        
        results = {"prefix_id": prefix_id}
        
        for gamma in gamma_values:
            gamma_key = f"{gamma:.1f}"
            results[gamma_key] = {}
            
            # Cache for this (prefix, gamma) to avoid redundant clustering
            beta_cache = {}
            
            for target_K in target_K_values:
                result = binary_search_beta(
                    data, target_K, gamma,
                    beta_lo=beta_lo, beta_hi=beta_hi, tolerance=beta_tolerance,
                    max_iterations=max_iterations, metric_a=metric_a,
                    normalize_dims=normalize_dims, use_gpu=use_gpu,
                    cache=beta_cache
                )
                results[gamma_key][str(target_K)] = result
        
        return results
        
    except Exception as e:
        return {"prefix_id": prefix_id, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Find minimal beta for target K values")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Results directory containing pipeline outputs")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for results")
    parser.add_argument("--target-K", type=str, default="2,4,6,8,10",
                        help="Comma-separated target K values")
    parser.add_argument("--gamma-values", type=str, default="0.1,0.3,0.5,0.7,0.9",
                        help="Comma-separated gamma values")
    parser.add_argument("--beta-range", type=str, default="0.1,3.0",
                        help="Beta search range as 'lo,hi'")
    parser.add_argument("--beta-tolerance", type=float, default=0.05,
                        help="Binary search tolerance for beta")
    parser.add_argument("--max-iterations", type=int, default=50,
                        help="Maximum EM iterations per clustering run")
    parser.add_argument("--metric-a", type=str, default="l1",
                        help="Attribution metric (l1 or l2)")
    parser.add_argument("--normalize-dims", action="store_true", default=False,
                        help="Normalize beta by dimensions (default: False)")
    parser.add_argument("--no-gpu", action="store_true",
                        help="Disable GPU acceleration")
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Number of parallel workers (threads)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of prefixes to process")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save intermediate results every N prefixes")
    args = parser.parse_args()
    
    # Parse arguments
    target_K_values = [int(k) for k in args.target_K.split(",")]
    gamma_values = [float(g) for g in args.gamma_values.split(",")]
    beta_lo, beta_hi = [float(b) for b in args.beta_range.split(",")]
    use_gpu = GPU_AVAILABLE and not args.no_gpu
    
    # Setup directories (matching actual pipeline structure)
    embeddings_dir = args.results_dir / "4_feature_extraction" / "embeddings"
    attribution_graphs_dir = args.results_dir / "3_attribution_graphs"
    samples_dir = args.results_dir / "2_branch_sampling"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all prefixes
    branch_files = sorted(samples_dir.glob("*_branches.json"))
    prefix_ids = [f.stem.replace("_branches", "") for f in branch_files]
    
    if args.limit:
        prefix_ids = prefix_ids[:args.limit]
    
    print(f"Found {len(prefix_ids)} prefixes to process")
    print(f"Target K values: {target_K_values}")
    print(f"Gamma values: {gamma_values}")
    print(f"Beta range: [{beta_lo}, {beta_hi}], tolerance: {args.beta_tolerance}")
    print(f"GPU acceleration: {'enabled' if use_gpu else 'disabled'}")
    print(f"Workers: {args.n_workers}")
    
    # Create a minimal logger
    logger = logging.getLogger("find_minimal_beta")
    logger.setLevel(logging.WARNING)
    
    # Prepare work items for multiprocessing
    work_items = [
        (prefix_id, str(embeddings_dir), str(attribution_graphs_dir), str(samples_dir),
         target_K_values, gamma_values, beta_lo, beta_hi, args.beta_tolerance,
         args.max_iterations, args.metric_a, args.normalize_dims, use_gpu)
        for prefix_id in prefix_ids
    ]
    
    # Process prefixes in parallel using ProcessPoolExecutor
    all_results = {}
    completed_count = 0
    
    # Use 'fork' context for faster startup (no need to re-import modules)
    mp_context = multiprocessing.get_context('fork')
    
    with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=mp_context) as executor:
        futures = {executor.submit(_process_prefix_worker, item): item[0] for item in work_items}
        
        with tqdm(total=len(prefix_ids), desc="Processing prefixes") as pbar:
            for future in as_completed(futures):
                prefix_id = futures[future]
                try:
                    result = future.result()
                    all_results[prefix_id] = result
                except Exception as e:
                    all_results[prefix_id] = {"prefix_id": prefix_id, "error": str(e)}
                
                completed_count += 1
                
                # Save intermediate results
                if completed_count % args.save_every == 0:
                    _save_results(all_results, args.output_dir, target_K_values, gamma_values, 
                                 beta_lo, beta_hi, args, prefix_ids, partial=True)
                
                if "error" in all_results[prefix_id]:
                    pbar.write(f"Error processing {prefix_id}: {all_results[prefix_id]['error']}")
                pbar.update(1)
    
    # Final save
    _save_results(all_results, args.output_dir, target_K_values, gamma_values,
                 beta_lo, beta_hi, args, prefix_ids, partial=False)


def _save_results(all_results, output_dir, target_K_values, gamma_values, 
                  beta_lo, beta_hi, args, prefix_ids, partial=False):
    """Save results and print summary."""
    # Compute aggregates
    aggregate = {}
    for gamma in gamma_values:
        gamma_key = f"{gamma:.1f}"
        aggregate[gamma_key] = {}
        
        for target_K in target_K_values:
            K_key = str(target_K)
            betas = []
            actual_Ks = []
            H_values = []
            D_e_values = []
            D_a_values = []
            L_RD_values = []
            
            for prefix_id, result in all_results.items():
                if "error" in result:
                    continue
                if gamma_key not in result:
                    continue
                if K_key not in result[gamma_key]:
                    continue
                
                entry = result[gamma_key][K_key]
                if entry.get("found", False):
                    betas.append(entry["minimal_beta"])
                    actual_Ks.append(entry["actual_K"])
                    H_values.append(entry.get("H", 0))
                    D_e_values.append(entry.get("D_e", 0))
                    D_a_values.append(entry.get("D_a", 0))
                    L_RD_values.append(entry.get("L_RD", 0))
            
            if betas:
                aggregate[gamma_key][K_key] = {
                    "mean_beta": float(np.mean(betas)),
                    "std_beta": float(np.std(betas)),
                    "median_beta": float(np.median(betas)),
                    "min_beta": float(np.min(betas)),
                    "max_beta": float(np.max(betas)),
                    "mean_actual_K": float(np.mean(actual_Ks)),
                    # Rate and distortion statistics
                    "mean_H": float(np.mean(H_values)),
                    "std_H": float(np.std(H_values)),
                    "mean_D_e": float(np.mean(D_e_values)),
                    "std_D_e": float(np.std(D_e_values)),
                    "mean_D_a": float(np.mean(D_a_values)),
                    "std_D_a": float(np.std(D_a_values)),
                    "mean_L_RD": float(np.mean(L_RD_values)),
                    "std_L_RD": float(np.std(L_RD_values)),
                    "n_found": len(betas),
                    "n_total": len(all_results),
                }
            else:
                aggregate[gamma_key][K_key] = {
                    "n_found": 0,
                    "n_total": len(all_results),
                }
    
    # Save results
    output = {
        "config": {
            "target_K_values": target_K_values,
            "gamma_values": gamma_values,
            "beta_range": [beta_lo, beta_hi],
            "beta_tolerance": args.beta_tolerance,
            "metric_a": args.metric_a,
            "normalize_dims": args.normalize_dims,
        },
        "aggregate": aggregate,
        "per_prefix": all_results,
    }
    
    suffix = "_partial" if partial else ""
    output_file = output_dir / f"minimal_beta_results{suffix}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    if not partial:
        print(f"\nSaved results to: {output_file}")
        
        # Print summary table
        print("\n" + "=" * 120)
        print("AGGREGATE RESULTS: Minimal Beta by (K, γ) with Rate and Distortion")
        print("=" * 120)
        print(f"\n{'K':<4} {'γ':<5} {'β':<8} {'H (Rate)':<12} {'D_e (Sem)':<12} {'D_a (Attr)':<12} {'L_RD':<12} {'n':<8}")
        print("-" * 100)
        
        for target_K in target_K_values:
            K_key = str(target_K)
            for gamma in gamma_values:
                gamma_key = f"{gamma:.1f}"
                entry = aggregate.get(gamma_key, {}).get(K_key, {})
                n_found = entry.get("n_found", 0)
                if n_found > 0:
                    print(f"{target_K:<4} {gamma:<5.1f} {entry['mean_beta']:<8.3f} "
                          f"{entry['mean_H']:<12.4f} {entry['mean_D_e']:<12.4f} "
                          f"{entry['mean_D_a']:<12.4f} {entry['mean_L_RD']:<12.4f} "
                          f"{n_found}/{entry['n_total']}")
                else:
                    print(f"{target_K:<4} {gamma:<5.1f} {'N/A':<8} {'N/A':<12} {'N/A':<12} "
                          f"{'N/A':<12} {'N/A':<12} 0/{entry.get('n_total', len(prefix_ids))}")


if __name__ == "__main__":
    main()
