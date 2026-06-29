#!/usr/bin/env -S uv run python
"""Main clustering orchestrator for rate-distortion Gaussian optimization.

Implements Algorithm 1 from rate_distortion.tex:
Rate-Distortion Two-View Clustering

Objective: L_RD = H(C) + β_e D^(e) + β_a D^(a)

Features:
- Single-component initialization (K emerges from optimization)
- Exact R-D criteria for split (no tunable thresholds)
"""

import sys


import argparse
import json
import logging
import multiprocessing
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Use 'spawn' start method for CUDA compatibility with multiprocessing
_mp_context = multiprocessing.get_context('spawn')
from tqdm import tqdm

# Add local clustering modules to path for direct module imports in tests
CLUSTERING_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(CLUSTERING_PATH))

# Add circuit-tracer to path (relative to project root)
CIRCUIT_TRACER_PATH = Path(__file__).resolve().parents[1] / "circuit-tracer"
sys.path.insert(0, str(CIRCUIT_TRACER_PATH))

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# PathConfig removed - results_dir now derived from embeddings_dir
from utils.data_utils import load_json, save_json
from utils.attribution_pooling import load_pooled_attributions
from utils.logging_utils import setup_logger, get_log_path
from utils.manifest import update_manifest_with_results

# Import R-D clustering modules
from initialize import initialize_single_component
from em_loop import run_em_iteration, check_convergence, GPU_AVAILABLE, weighted_median
from adaptive_control import apply_adaptive_control
from rd_objective import (
    compute_component_masses,
    compute_normalized_masses,
    compute_full_rd_statistics,
    compute_component_variance,
)

from sweep_utils import run_sweep_mode


def _resolve_pooling(cli_pooling: Optional[str], config: Dict[str, Any]) -> str:
    """Resolve attribution pooling with CLI taking precedence over config."""
    return cli_pooling or config.get("pooling", "mean") or "mean"


def compute_global_center(
    attributions_a: np.ndarray,
    path_probs: np.ndarray,
    metric_a: str = "l2",
) -> np.ndarray:
    """Compute H_0 (shared attribution center) consistent with attribution distortion.

    - metric_a == "l1": probability-weighted coordinate-wise median (minimizes weighted L1)
    - metric_a == "l2": probability-weighted mean (minimizes weighted squared L2)
    """
    W_total = float(path_probs.sum())
    if W_total <= 0:
        return np.zeros(attributions_a.shape[1])

    if metric_a == "l1":
        return weighted_median(attributions_a, path_probs)

    return np.sum(path_probs[:, None] * attributions_a, axis=0) / W_total


def load_prefix_data(
    prefix_id: str,
    embeddings_dir: Path,
    attribution_graphs_dir: Path,
    samples_dir: Path,
    logger,
    pooling: str = "mean",
    metric_a: str = "l2",
):
    """Load all data for a single prefix."""
    logger.info(f"Loading data for prefix: {prefix_id}")

    # Load branch samples data
    branches_file = samples_dir / f"{prefix_id}_branches.json"
    branches_data = load_json(branches_file)
    prefix = branches_data["prefix"]

    # Extract path probabilities from all continuations
    path_probs_original = []
    for cont in branches_data.get("continuations", []):
        path_probs_original.append(cont.get("probability", 0.0))

    path_probs_original = np.array(path_probs_original)

    path_probs = path_probs_original

    logger.info(f"Original weights: sum={path_probs_original.sum():.6f}, "
                f"min={path_probs_original.min():.2e}, max={path_probs_original.max():.2e}")
    logger.info(f"Using original weights for clustering (sum={path_probs.sum():.6f})")

    # Load embeddings
    embeddings_file = embeddings_dir / f"{prefix_id}_embeddings.npy"
    embeddings_e = np.load(embeddings_file)

    logger.info("Loading continuation attribution from Stage 3...")
    prefix_context_file = attribution_graphs_dir / f"{prefix_id}_prefix_context.pt"
    pooled_attributions = load_pooled_attributions(
        prefix_context_file,
        pooling=pooling,
        meta_file=attribution_graphs_dir / f"{prefix_id}_attribution.json",
    )
    attributions_a = pooled_attributions.values

    if logger.getEffectiveLevel() <= logging.INFO:
        logger.info(f"  Pooling requested: {pooled_attributions.requested_pooling}")
        logger.info(f"  Pooling effective: {pooled_attributions.effective_pooling}")
        logger.info(f"  Attribution source: {pooled_attributions.source}")

    logger.info(f"Loaded continuation attributions: shape {attributions_a.shape}")

    logger.info(f"Attribution embeddings: shape {attributions_a.shape}")

    # Compute H_0 (shared attribution center) BEFORE clustering.
    # IMPORTANT: for L1 attribution distortion, we center by weighted median (not mean).
    H_0 = compute_global_center(attributions_a, path_probs, metric_a=metric_a)
    logger.info(f"Computed H_0: ||H_0|| = {np.linalg.norm(H_0):.4f}")

    # Center attributions for clustering (clustering operates on Delta_H)
    attributions_a_centered = attributions_a - H_0

    # Batch normalize attributions
    norms_sq = np.sum(attributions_a_centered ** 2, axis=1)
    rms_norm = np.sqrt(np.mean(norms_sq))
    if rms_norm > 1e-10:
        attributions_a_centered = attributions_a_centered / rms_norm
        logger.info(f"Batch normalized attributions: RMS norm = {rms_norm:.4f}")
    else:
        logger.warning("RMS norm near zero, skipping batch normalization")

    attributions_a_original = attributions_a
    attributions_a = attributions_a_centered

    n_samples = len(path_probs)
    assert embeddings_e.shape[0] == n_samples
    assert attributions_a.shape[0] == n_samples

    logger.info(f"Loaded {n_samples} samples")

    return {
        "prefix_id": prefix_id,
        "prefix": prefix,
        "embeddings_e": embeddings_e,
        "attributions_a": attributions_a,  # Centered and batch-normalized for clustering
        "attributions_a_original": attributions_a_original,  # Original for reconstruction
        "H_0": H_0,  # Global mean (shared component)
        "attribution_rms_norm": rms_norm,  # RMS norm used for batch normalization
        "path_probs": path_probs,
        "n_samples": n_samples,
        "prefix_context_file": prefix_context_file,  # Path to PrefixAttributionContext (for intervention)
    }


def run_clustering(
    data: dict,
    beta_e: float,
    beta_a: float,
    K_max: int,
    max_iterations: int,
    convergence_threshold: float,
    logger,
    metric_a: str = "l2",
    save_intermediate: bool = False,
    intermediate_dir: Optional[Path] = None,
    split_random_seed: Optional[int] = None,
    split_event_observer=None,
    collect_runtime_profile: bool = False,
):
    """Run the full R-D clustering algorithm for one prefix.

    Implements Algorithm 1 from rate_distortion.tex.
    All computations use probability-weighted distortion and centers.
    """
    prefix_id = data["prefix_id"]
    embeddings_e = data["embeddings_e"].copy()
    attributions_a = data["attributions_a"].copy()
    path_probs = data["path_probs"].copy()
    total_start_time = time.perf_counter() if collect_runtime_profile else None

    logger.info("=" * 60)
    logger.info(f"R-D CLUSTERING PREFIX: {prefix_id}")
    logger.info("=" * 60)
    logger.info(f"Parameters: β_e={beta_e}, β_a={beta_a}, K_max={K_max}")
    logger.info(f"Metric: {metric_a}")
    logger.info(f"N samples: {len(path_probs)}")

    # Initialization
    logger.info("Initializing with single component (K=1)...")
    components, assignments = initialize_single_component(
        embeddings_e, attributions_a, path_probs, metric_a=metric_a
    )

    # Initialize tracking
    next_component_id = max(components.keys()) + 1 if components else 2
    L_RD_prev = np.inf

    history = {
        "iterations": [],
        "n_components": [],
        "L_RD": [],
        "H": [],
        "D_e": [],
        "D_a": [],
    }
    runtime_iterations: List[Dict[str, Any]] = []

    # Compute initial statistics (always probability-weighted)
    rd_stats = compute_full_rd_statistics(
        embeddings_e, attributions_a, assignments, path_probs,
        components, beta_e, beta_a,
        metric_a=metric_a,
    )

    logger.info(f"Initial: {len(components)} component(s), L_RD={rd_stats['L_RD']:.4f}")

    # Step 2: Main loop
    logger.info("\nStarting R-D EM loop...")
    converged = False
    iteration = 0

    show_pbar = False
    if logger and logger.getEffectiveLevel() >= 30: # logging.WARNING
        show_pbar = True
        
    iterator = range(max_iterations)
    if show_pbar:
        iterator = tqdm(iterator, desc="  EM Iterations", leave=False)

    def serialize_components_snapshot(components_dict: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        snapshot = {}
        for c_id, comp in components_dict.items():
            mu_e = comp.get("mu_e")
            mu_a = comp.get("mu_a")
            snapshot[str(c_id)] = {
                "mu_e": mu_e.tolist() if hasattr(mu_e, "tolist") else mu_e,
                "mu_a": mu_a.tolist() if hasattr(mu_a, "tolist") else mu_a,
                "W_c": float(comp.get("W_c", 0.0)),
            }
        return snapshot

    def maybe_save_intermediate(stage: str, iteration_idx: int, assignments_curr, rd_stats_curr):
        if not save_intermediate or intermediate_dir is None:
            return
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "prefix_id": prefix_id,
            "stage": stage,
            "iteration": iteration_idx + 1,
            "n_components": len(components),
            "assignments": [int(a) for a in assignments_curr],
            "components": serialize_components_snapshot(components),
            "rd_stats": {
                "L_RD": float(rd_stats_curr.get("L_RD", 0.0)),
                "H": float(rd_stats_curr.get("H", 0.0)),
                "D_e": float(rd_stats_curr.get("D_e", 0.0)),
                "D_a": float(rd_stats_curr.get("D_a", 0.0)),
                "beta_e": float(rd_stats_curr.get("beta_e", beta_e)),
                "beta_a": float(rd_stats_curr.get("beta_a", beta_a)),
                "P_bar": {str(k): float(v) for k, v in rd_stats_curr.get("P_bar", {}).items()},
                "Var_e": {str(k): float(v) for k, v in rd_stats_curr.get("Var_e", {}).items()},
                "Var_a": {str(k): float(v) for k, v in rd_stats_curr.get("Var_a", {}).items()},
            },
        }
        out_file = intermediate_dir / f"iter_{iteration_idx + 1:03d}_{stage}.json"
        save_json(payload, out_file)

    prev_assignments = np.array(assignments, copy=True)

    for iteration in iterator:
        logger.info(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")
        iteration_start_time = time.perf_counter() if collect_runtime_profile else None
        k_before_em = len(components)

        # E-step + M-step
        em_start_time = time.perf_counter() if collect_runtime_profile else None
        assignments, components, rd_stats = run_em_iteration(
            embeddings_e,
            attributions_a,
            path_probs,
            components,
            beta_e,
            beta_a,
            metric_a=metric_a,
        )
        em_seconds = (
            float(time.perf_counter() - em_start_time)
            if em_start_time is not None
            else 0.0
        )

        L_RD_curr = rd_stats['L_RD']
        k_after_em = len(components)

        if not np.array_equal(prev_assignments, np.array(assignments)):
            maybe_save_intermediate("em", iteration, assignments, rd_stats)
            prev_assignments = np.array(assignments, copy=True)

        logger.info(f"After EM: K={len(components)}, L_RD={L_RD_curr:.4f}, "
                   f"H={rd_stats['H']:.4f}, D_e={rd_stats['D_e']:.4f}, D_a={rd_stats['D_a']:.4f}")

        # Diagnostic logging
        P_bar_diag = rd_stats['P_bar']
        Var_e_diag = rd_stats.get('Var_e', {})
        Var_a_diag = rd_stats.get('Var_a', {})
        for c, comp in components.items():
            indices = [i for i, a in enumerate(assignments) if a == c]
            p_c = P_bar_diag.get(c, 0)
            v_e = Var_e_diag.get(c, 0)
            v_a = Var_a_diag.get(c, 0)
            logger.info(f"  C{c}: n={len(indices)}, P̄={p_c:.4f}, Var_e={v_e:.4f}, Var_a={v_a:.4f}, "
                       f"contrib_e={p_c*v_e:.4f}, contrib_a={p_c*v_a:.4f}")

        # Adaptive control (Split → Junk)
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
                    embeddings_e[indices], comp['mu_e'], path_probs[indices], W_c_val,
                    "l2",
                )
                Var_a[c] = compute_component_variance(
                    attributions_a[indices], comp['mu_a'], path_probs[indices], W_c_val,
                    metric_a,
                )

        adaptive_runtime: Optional[Dict[str, Any]] = {} if collect_runtime_profile else None
        components, assignments, next_component_id = apply_adaptive_control(
            embeddings_e,
            attributions_a,
            path_probs,
            assignments,
            components,
            P_bar,
            Var_e,
            Var_a,
            beta_e,
            beta_a,
            next_component_id=next_component_id,
            metric_a=metric_a,
            split_random_seed=split_random_seed,
            split_observer=(
                None
                if split_event_observer is None
                else lambda event, iteration_idx=iteration: split_event_observer(
                    {
                        **event,
                        "prefix_id": prefix_id,
                        "prefix": data["prefix"],
                        "iteration": iteration_idx + 1,
                    }
                )
            ),
            runtime_profile_sink=adaptive_runtime,
        )

        logger.info(f"After adaptive: K={len(components)}")
        k_after_adaptive = len(components)

        # Recompute R-D stats after adaptive control
        if len(components) > 0:
            rd_stats = compute_full_rd_statistics(
                embeddings_e, attributions_a, assignments, path_probs,
                components, beta_e, beta_a,
                metric_a=metric_a,
            )
            L_RD_curr = rd_stats['L_RD']

        if not np.array_equal(prev_assignments, np.array(assignments)):
            maybe_save_intermediate("adaptive", iteration, assignments, rd_stats)
            prev_assignments = np.array(assignments, copy=True)

        # Track history
        history["iterations"].append(iteration + 1)
        history["n_components"].append(len(components))
        history["L_RD"].append(L_RD_curr)
        history["H"].append(rd_stats['H'])
        history["D_e"].append(rd_stats['D_e'])
        history["D_a"].append(rd_stats['D_a'])

        # Check convergence
        converged = check_convergence(L_RD_prev, L_RD_curr, convergence_threshold)

        if collect_runtime_profile and iteration_start_time is not None:
            iteration_seconds = float(time.perf_counter() - iteration_start_time)
            split_seconds = float((adaptive_runtime or {}).get("split_seconds", 0.0))
            junk_seconds = float((adaptive_runtime or {}).get("junk_seconds", 0.0))
            residual_seconds = max(
                0.0,
                iteration_seconds - em_seconds - split_seconds - junk_seconds,
            )
            runtime_iterations.append(
                {
                    "iteration": int(iteration + 1),
                    "k_before_em": int(k_before_em),
                    "k_after_em": int(k_after_em),
                    "k_after_adaptive": int(k_after_adaptive),
                    "em_seconds": em_seconds,
                    "split_seconds": split_seconds,
                    "junk_seconds": junk_seconds,
                    "residual_seconds": residual_seconds,
                    "splits_done": int((adaptive_runtime or {}).get("splits_done", 0)),
                    "junks_done": int((adaptive_runtime or {}).get("junks_done", 0)),
                }
            )

        if converged:
            logger.info(f"\nConverged after {iteration + 1} iterations")
            break

        L_RD_prev = L_RD_curr

    # Final statistics
    logger.info("\n" + "=" * 60)
    logger.info("R-D CLUSTERING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final components: {len(components)}")
    logger.info(f"Final L_RD: {rd_stats['L_RD']:.4f}")
    logger.info(f"Iterations: {iteration + 1}")

    result = {
        "prefix_id": prefix_id,
        "prefix": data["prefix"],
        "components": components,
        "rd_stats": rd_stats,
        "assignments": assignments,
        "history": history,
        "n_iterations": iteration + 1,
        "converged": converged,
        # Hierarchical decomposition
        "H_0": data.get("H_0"),  # Global mean (shared attribution)
        "metric_a": metric_a,
    }

    if collect_runtime_profile and total_start_time is not None:
        total_seconds = float(time.perf_counter() - total_start_time)
        em_seconds_total = float(sum(item["em_seconds"] for item in runtime_iterations))
        split_seconds_total = float(sum(item["split_seconds"] for item in runtime_iterations))
        junk_seconds_total = float(sum(item["junk_seconds"] for item in runtime_iterations))
        residual_seconds_total = max(
            0.0,
            total_seconds - em_seconds_total - split_seconds_total - junk_seconds_total,
        )
        result["runtime_profile"] = {
            "total_seconds": total_seconds,
            "em_seconds_total": em_seconds_total,
            "split_seconds_total": split_seconds_total,
            "junk_seconds_total": junk_seconds_total,
            "residual_seconds_total": residual_seconds_total,
            "n_iterations": int(iteration + 1),
            "splits_done_total": int(sum(item["splits_done"] for item in runtime_iterations)),
            "junks_done_total": int(sum(item["junks_done"] for item in runtime_iterations)),
            "iterations": runtime_iterations,
        }

    return result


def serialize_clustering_result(result: dict) -> dict:
    """Convert clustering result to JSON-serializable format."""
    # Get H_0 for computing full mu_a
    H_0 = result.get("H_0")

    def get_mu_a_full(comp):
        """Compute full mu_a = H_0 + Delta_H_c."""
        mu_a = comp["mu_a"]
        if hasattr(mu_a, "tolist"):
            mu_a = mu_a
        else:
            mu_a = np.array(mu_a)
        if H_0 is not None:
            return (mu_a + H_0).tolist() if hasattr(mu_a + H_0, "tolist") else (mu_a + H_0).tolist()
        return mu_a.tolist() if hasattr(mu_a, "tolist") else mu_a

    return {
        "prefix_id": result["prefix_id"],
        "prefix": result["prefix"],
        "method": "rate_distortion",
        "n_iterations": int(result["n_iterations"]),
        "converged": bool(result["converged"]),
        "n_components": len(result["components"]),
        "metric_a": result.get("metric_a", "l2"),
        # Hierarchical decomposition (shared components)
        "H_0": H_0.tolist() if H_0 is not None and hasattr(H_0, "tolist") else H_0,
        "rd_objective": {
            "L_RD": float(result["rd_stats"]["L_RD"]),
            "H": float(result["rd_stats"]["H"]),
            "D_e": float(result["rd_stats"]["D_e"]),
            "D_a": float(result["rd_stats"]["D_a"]),
            "beta_e": float(result["rd_stats"]["beta_e"]),
            "beta_a": float(result["rd_stats"]["beta_a"]),
        },
        "components": {
            str(c): {
                "mu_e": comp["mu_e"].tolist() if hasattr(comp["mu_e"], "tolist") else comp["mu_e"],
                "mu_a": comp["mu_a"].tolist() if hasattr(comp["mu_a"], "tolist") else comp["mu_a"],  # Delta_H_c (centered)
                "mu_a_full": get_mu_a_full(comp),  # H_0 + Delta_H_c
                "W_c": float(comp["W_c"]),
                "indices": [int(i) for i in comp.get("indices", [])],
            }
            for c, comp in result["components"].items()
        },
        "assignments": [int(a) for a in result["assignments"]],
        "statistics": {
            str(c): {
                "Var_e_w": float(result["rd_stats"]["Var_e"].get(c, 0.0)),
                "Var_a_w": float(result["rd_stats"]["Var_a"].get(c, 0.0)),
                "Err_c_w": 0.0,
                "n_samples": len(comp.get("indices", [])),
            }
            for c, comp in result["components"].items()
        },
    }


def process_prefix(meta_file, args, sweeps_config, n_sweep_workers):
    """Process a single prefix (load data, run clustering/sweep, save results)."""
    prefix_id = meta_file.stem.replace("_embeddings_meta", "")
    
    # Create worker-specific logger
    log_file = args.output_dir / "logs" / f"{prefix_id}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(f"clustering_{prefix_id}", log_file=log_file, level=logging.INFO)
    
    try:
        data = load_prefix_data(
            prefix_id,
            args.embeddings_dir,
            args.attribution_graphs_dir,
            args.samples_dir,
            logger,
            pooling=args.pooling,
            metric_a=args.attribution_metric,
        )

        intermediate_dir = args.intermediate_dir
        if args.save_intermediate and intermediate_dir is None:
            intermediate_dir = args.output_dir / "intermediate"

        # Sweep mode: run sweep over (beta, gamma) grid
        sweep_results = run_sweep_mode(
            data,
            sweeps_config,
            args.K_max,
            args.max_iterations,
            args.convergence_threshold,
            logger,
            n_workers=n_sweep_workers,
            metric_a=args.attribution_metric,
            prefix_id=prefix_id,
            intermediate_dir=intermediate_dir,
            save_intermediate=args.save_intermediate,
            normalize_dims=args.normalize_dims,
            K_clamp=getattr(args, 'K_clamp', None)
        )

        # Save sweep results
        sweep_file = args.output_dir / f"{prefix_id}_sweep_results.json"
        save_json(sweep_results, sweep_file)
        logger.info(f"Saved sweep results to: {sweep_file}")
        
        # Close handlers to avoid leaking
        for handler in logger.handlers:
            handler.close()
            
        return {
            "status": "success",
            "prefix_id": prefix_id,
            "result": sweep_results,
        }

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Error processing {prefix_id}: {error_msg}")
        logger.error(traceback.format_exc())
        
        # Close handlers
        for handler in logger.handlers:
            handler.close()
            
        return {
            "status": "failed",
            "prefix_id": prefix_id,
            "error": error_msg
        }


def main():
    parser = argparse.ArgumentParser(description="Run Rate-Distortion Gaussian clustering")
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--attribution-graphs-dir", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)

    # R-D specific parameters
    parser.add_argument("--beta", type=float, default=2.0, help="Total β")
    parser.add_argument("--gamma", type=float, default=0.5, help="View ratio: β_e = γβ, β_a = (1-γ)β")
    parser.add_argument("--K-max", type=int, default=20, 
                        help="DEPRECATED: K_max no longer constrains clustering. "
                             "Clustering converges naturally based on R-D criterion. "
                             "This value is stored in sweep_config for downstream K_clamp filtering.")
    parser.add_argument("--K-clamp", type=int, default=None,
                        help="Maximum K for downstream steering (stored in sweep_config). "
                             "If not provided, defaults to K_max value.")
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--convergence-threshold", type=float, default=1e-6)
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory for log files")
    parser.add_argument("--pooling", type=str, default=None,
                        choices=["mean", "max", "sum"],
                        help="Pooling method for aggregating token attributions (default: mean)")
    parser.add_argument("--n-workers", type=int, default=1, help="Number of workers for parallel prefix processing")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (only progress bars)")
    parser.add_argument("--save-intermediate", action="store_true",
                        help="Save intermediate clustering snapshots when assignments change")
    parser.add_argument("--intermediate-dir", type=Path, default=None,
                        help="Base directory for intermediate snapshots (default: output_dir/intermediate)")
    
    # New Design Arguments
    parser.add_argument("--attribution-metric", type=str, default="l1", choices=["l2", "l1"],
                        help="Metric for attribution distance (default: l1)")
    parser.add_argument("--normalize-dims", action="store_true",
                        help="Normalize beta by dimensions: beta_e /= sqrt(d_e), beta_a /= d_a. "
                             "This accounts for L2 scaling as sqrt(d) and L1 scaling as d.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip prefixes that already have *_sweep_results.json in output directory")
    args = parser.parse_args()

    # Load config if provided
    sweeps_config = {}
    n_sweep_workers = 4
    clustering = {}

    if args.config and args.config.exists():
        with open(args.config) as f:
            config = json.load(f)

        # Clustering config
        clustering = config.get("clustering", {})
        args.beta = clustering.get("beta", args.beta)
        args.gamma = clustering.get("gamma", args.gamma)
        args.K_max = clustering.get("K_max", args.K_max)
        # Read K_clamp from config if not set via CLI
        if args.K_clamp is None:
            args.K_clamp = clustering.get("K_clamp", None)
        args.max_iterations = clustering.get("max_iterations", args.max_iterations)
        args.convergence_threshold = clustering.get("convergence_threshold", args.convergence_threshold)

        # Design arguments in config
        args.attribution_metric = clustering.get("attribution_metric", args.attribution_metric)
        args.normalize_dims = clustering.get("normalize_dims", args.normalize_dims)

        # Worker config from config file (if provided)
        # Note: --n-workers cli arg overrides this for prefix parallelism
        
        # Detect sweep mode (like 7C pattern)
        sweeps = clustering.get("sweeps", {})
        if sweeps and sweeps.get("beta_values") and sweeps.get("gamma_values"):
            sweeps_config = sweeps
            n_sweep_workers = sweeps.get("n_workers", 4)

    args.pooling = _resolve_pooling(args.pooling, clustering)

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = Path("test_results")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup main logger
    log_file = get_log_path("5_clustering", args.log_dir)
    import logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logger = setup_logger("rd_clustering", log_file=log_file, level=log_level)

    logger.info("=" * 60)
    logger.info("RATE-DISTORTION GAUSSIAN CLUSTERING")
    logger.info("=" * 60)
    
    # Configure parallelism strategy
    # If parallelizing prefixes (n_workers > 1), force sequential sweeps (n_sweep_workers = 1)
    # If serial prefixes (n_workers = 1), allow parallel sweeps (n_sweep_workers from config)
    if args.n_workers > 1:
        logger.info(f"PARALLEL STRATEGY: Parallel Prefixes ({args.n_workers} workers) -> Serial Sweeps")
        n_sweep_workers = 1
    else:
        logger.info(f"PARALLEL STRATEGY: Serial Prefixes -> Parallel Sweeps ({n_sweep_workers} workers)")
    
    if not sweeps_config:
        sweeps_config = {"beta_values": [args.beta], "gamma_values": [args.gamma]}
    logger.info("MODE: SWEEP (hyperparameter search)")
    logger.info(f"Beta values: {sweeps_config.get('beta_values')}")
    logger.info(f"Gamma values: {sweeps_config.get('gamma_values')}")
    logger.info(f"K_max: {args.K_max}")
    logger.info(f"K_clamp: {args.K_clamp}")
    logger.info(f"Attribution Metric: {args.attribution_metric}")
    logger.info(f"Dimension Normalization: {args.normalize_dims}")
    logger.info("Weighted Distortion: False")
    logger.info(f"GPU acceleration: {'enabled' if GPU_AVAILABLE else 'disabled'}")

    # Find all embedding metadata files
    embedding_meta_files = sorted(args.embeddings_dir.glob("*_embeddings_meta.json"))
    logger.info(f"\nFound {len(embedding_meta_files)} prefixes to process")

    # Filter based on Stage 4a manifest (embeddings extraction)
    # Derive results_dir from embeddings_dir to respect --output-dir
    # embeddings_dir is typically {output_dir}/results/4_feature_extraction/embeddings/
    results_dir = args.embeddings_dir.parent.parent
    # all_prefix_ids = [f.stem.replace("_embeddings_meta", "") for f in embedding_meta_files]
    # available_ids, skipped_ids = filter_samples_by_manifest(
    #     all_prefix_ids, results_dir, "stage4a", logger
    # )
    # # Filter embedding files to only available ones
    # available_id_set = set(available_ids)
    # embedding_meta_files = [f for f in embedding_meta_files if f.stem.replace("_embeddings_meta", "") in available_id_set]
    # logger.info(f"Processing {len(embedding_meta_files)} available prefixes (skipped {len(skipped_ids)})")
    
    # SKIP MANIFEST CHECK FOR DESIGN CHECK
    skipped_ids = []
    logger.info("Skipping manifest check for design check...")

    # Skip existing results if --skip-existing is set
    if args.skip_existing:
        original_count = len(embedding_meta_files)
        filtered_files = []
        for meta_file in embedding_meta_files:
            prefix_id = meta_file.stem.replace("_embeddings_meta", "")
            output_file = args.output_dir / f"{prefix_id}_sweep_results.json"
            if output_file.exists():
                skipped_ids.append(prefix_id)
            else:
                filtered_files.append(meta_file)
        embedding_meta_files = filtered_files
        logger.info(f"--skip-existing: Skipping {len(skipped_ids)} already completed prefixes")
        logger.info(f"Remaining to process: {len(embedding_meta_files)} / {original_count}")

    # Process prefixes
    results = []
    completed_ids = []
    failed_ids = []
    errors = {}
    
    if args.n_workers > 1:
        # Parallel execution with 'spawn' context for CUDA compatibility
        logger.info(f"Starting parallel processing with {args.n_workers} workers...")
        with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=_mp_context) as executor:
            futures = {
                executor.submit(
                    process_prefix,
                    meta_file,
                    args,
                    sweeps_config,
                    n_sweep_workers
                ): meta_file
                for meta_file in embedding_meta_files
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Clustering prefixes"):
                res = future.result()
                if res["status"] == "success":
                    completed_ids.append(res["prefix_id"])
                    results.append(res["result"])
                else:
                    failed_ids.append(res["prefix_id"])
                    errors[res["prefix_id"]] = res.get("error", "Unknown error")
    else:
        # Sequential execution
        logger.info("Starting sequential processing...")
        for meta_file in tqdm(embedding_meta_files, desc="Clustering prefixes"):
            res = process_prefix(meta_file, args, sweeps_config, n_sweep_workers)
            if res["status"] == "success":
                completed_ids.append(res["prefix_id"])
                results.append(res["result"])
            else:
                failed_ids.append(res["prefix_id"])
                errors[res["prefix_id"]] = res.get("error", "Unknown error")

    # Save summary
    summary = {
        "method": "rate_distortion",
        "mode": "sweep",
        "n_prefixes": len(results),
        "parameters": {
            "K_max": args.K_max,
            "attribution_metric": args.attribution_metric,
        },
        "sweep_config": sweeps_config,
        "results": [
            {
                "prefix_id": r.get("prefix_id"),
                "n_configs": len(r.get("grid", [])),
            }
            for r in results
        ]
    }

    summary_file = args.output_dir / "clustering_summary.json"
    save_json(summary, summary_file)

    # Write Stage 5 manifest
    update_manifest_with_results(
        results_dir=results_dir,
        stage_name="stage5",
        processed=completed_ids,
        failed=failed_ids,
        skipped=skipped_ids,
        logger=logger,
        errors=errors,
    )

    logger.info("=" * 60)
    logger.info("ALL PREFIXES COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Processed: {len(completed_ids)} prefixes")
    logger.info(f"Completed: {len(completed_ids)}, Failed: {len(failed_ids)}, Skipped: {len(skipped_ids)}")
    logger.info("Sweep results saved.")


if __name__ == "__main__":
    main()
