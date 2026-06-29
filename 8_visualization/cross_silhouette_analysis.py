#!/usr/bin/env python
"""Cross-Silhouette Analysis: Compare RD Clustering with K-Means Baselines.

Computes cross-modality silhouette scores:
- Silhouette(E): Cluster quality measured in embedding space
- Silhouette(A): Cluster quality measured in attribution space

Methods compared:
1. RD Clustering (from 5_clustering sweep)
2. K-Means on Embeddings only (baseline)
3. K-Means on Attributions only (baseline)

Usage:
    python cross_silhouette_analysis.py \
        --clustering-dir AmbigQA_Qwen3-8B/results/5_clustering \
        --embeddings-dir AmbigQA_Qwen3-8B/results/4_feature_extraction/embeddings \
        --attribution-dir AmbigQA_Qwen3-8B/results/3_attribution_graphs \
        --output-dir AmbigQA_Qwen3-8B/results/cross_silhouette
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data_utils import load_json, save_json
from utils.attribution_pooling import load_pooled_attributions
from utils.logging_utils import setup_logger


def compute_silhouette_both_spaces(
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    assignments: np.ndarray,
) -> Tuple[float, float]:
    """Compute silhouette scores in both embedding and attribution spaces.
    
    Args:
        embeddings_e: Semantic embeddings [n_samples, d_e]
        attributions_a: Attribution vectors [n_samples, d_a]
        assignments: Cluster assignments [n_samples]
        
    Returns:
        (silhouette_e, silhouette_a): Silhouette scores in each space
    """
    unique_labels = np.unique(assignments)
    
    # Need at least 2 clusters
    if len(unique_labels) < 2:
        return -1.0, -1.0
    
    # Need enough samples
    if len(assignments) < 3:
        return -1.0, -1.0
    
    try:
        sil_e = silhouette_score(embeddings_e, assignments)
    except Exception:
        sil_e = -1.0
    
    try:
        sil_a = silhouette_score(attributions_a, assignments)
    except Exception:
        sil_a = -1.0
    
    return float(sil_e), float(sil_a)


def compute_harmonic_silhouette(sil_e: float, sil_a: float) -> float:
    """Compute harmonic mean of silhouette scores.
    
    Handles negative values by returning -1.0 if either is negative.
    """
    if sil_e <= 0 or sil_a <= 0:
        return -1.0
    return 2 * sil_e * sil_a / (sil_e + sil_a)


def run_kmeans_sweep(
    data: np.ndarray,
    K_range: List[int],
    random_state: int = 42,
) -> Dict[int, np.ndarray]:
    """Run K-means clustering for a range of K values.
    
    Args:
        data: Feature matrix [n_samples, d]
        K_range: List of K values to try
        random_state: Random seed
        
    Returns:
        Dict mapping K -> assignments array
    """
    results = {}
    n_samples = data.shape[0]
    
    for K in K_range:
        if K > n_samples:
            continue
        if K < 2:
            continue
            
        kmeans = KMeans(
            n_clusters=K,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=random_state
        )
        assignments = kmeans.fit_predict(data)
        results[K] = assignments
    
    return results


def load_attributions(
    attribution_dir: Path,
    prefix_id: str,
    pooling: str = "mean",
) -> np.ndarray:
    """Load attributions from Stage 3 output.
    
    Args:
        attribution_dir: Path to attribution graphs directory
        prefix_id: Prefix ID
        pooling: Pooling method for token-level attributions
        
    Returns:
        Attribution matrix [n_samples, d_a]
    """
    context_file = attribution_dir / f"{prefix_id}_prefix_context.pt"
    meta_file = attribution_dir / f"{prefix_id}_attribution.json"

    pooled_attributions = load_pooled_attributions(
        context_file,
        pooling=pooling,
        meta_file=meta_file,
    )
    return pooled_attributions.values


def analyze_prefix(
    prefix_id: str,
    clustering_dir: Path,
    embeddings_dir: Path,
    attribution_dir: Path,
    K_range: List[int],
    logger,
    pooling: str = "mean",
) -> Optional[Dict[str, Any]]:
    """Analyze cross-modality silhouettes for a single prefix.
    
    Args:
        prefix_id: Prefix ID
        clustering_dir: Directory with RD sweep results
        embeddings_dir: Directory with embeddings
        attribution_dir: Directory with attribution graphs
        K_range: Range of K values for K-means
        logger: Logger instance
        pooling: Pooling method for attribution loading
        
    Returns:
        Dict with analysis results or None if error
    """
    # Load sweep results
    sweep_file = clustering_dir / f"{prefix_id}_sweep_results.json"
    if not sweep_file.exists():
        logger.warning(f"Missing sweep file: {sweep_file}")
        return None
    
    sweep_data = load_json(sweep_file)
    grid = sweep_data.get("grid", [])
    
    if not grid:
        logger.warning(f"Empty grid for {prefix_id}")
        return None
    
    # Load embeddings
    emb_file = embeddings_dir / f"{prefix_id}_embeddings.npy"
    if not emb_file.exists():
        logger.warning(f"Missing embeddings: {emb_file}")
        return None
    
    embeddings = np.load(emb_file)
    
    # Load attributions
    try:
        attributions = load_attributions(attribution_dir, prefix_id, pooling=pooling)
    except Exception as e:
        logger.warning(f"Error loading attributions for {prefix_id}: {e}")
        return None
    
    n_samples = embeddings.shape[0]
    
    if n_samples != attributions.shape[0]:
        logger.warning(f"Shape mismatch for {prefix_id}: emb={embeddings.shape[0]}, attr={attributions.shape[0]}")
        return None
    
    # =====================================================================
    # 1. Extract RD clustering results from sweep
    # =====================================================================
    rd_results = []
    for entry in grid:
        beta = entry.get("beta")
        gamma = entry.get("gamma")
        K = entry.get("K", 1)
        sil_e = entry.get("sil_e", -1.0)
        sil_a = entry.get("sil_a", -1.0)
        harmonic = entry.get("harmonic", -1.0)
        
        rd_results.append({
            "beta": beta,
            "gamma": gamma,
            "K": K,
            "sil_e": sil_e,
            "sil_a": sil_a,
            "harmonic": harmonic,
        })
    
    # =====================================================================
    # 2. Run K-Means baselines
    # =====================================================================
    
    # K-Means on Embeddings
    logger.debug(f"  Running K-Means on embeddings...")
    kmeans_e_results = run_kmeans_sweep(embeddings, K_range)
    
    kmeans_e_sweep = []
    for K, assignments in kmeans_e_results.items():
        sil_e, sil_a = compute_silhouette_both_spaces(embeddings, attributions, assignments)
        kmeans_e_sweep.append({
            "K": K,
            "sil_e": sil_e,
            "sil_a": sil_a,
            "harmonic": compute_harmonic_silhouette(sil_e, sil_a),
        })
    
    # K-Means on Attributions
    logger.debug(f"  Running K-Means on attributions...")
    kmeans_a_results = run_kmeans_sweep(attributions, K_range)
    
    kmeans_a_sweep = []
    for K, assignments in kmeans_a_results.items():
        sil_e, sil_a = compute_silhouette_both_spaces(embeddings, attributions, assignments)
        kmeans_a_sweep.append({
            "K": K,
            "sil_e": sil_e,
            "sil_a": sil_a,
            "harmonic": compute_harmonic_silhouette(sil_e, sil_a),
        })
    
    # =====================================================================
    # 3. Find best configs for each method
    # =====================================================================
    
    # Best RD by harmonic silhouette
    valid_rd = [r for r in rd_results if r["harmonic"] > 0]
    if valid_rd:
        best_rd = max(valid_rd, key=lambda x: x["harmonic"])
    else:
        best_rd = max(rd_results, key=lambda x: (x["sil_e"] + x["sil_a"]) / 2 if x["sil_e"] > -1 else -2)
    
    # Best K-Means(E) by sil_e (native metric)
    if kmeans_e_sweep:
        best_kmeans_e_native = max(kmeans_e_sweep, key=lambda x: x["sil_e"])
        # Also get best by harmonic
        valid_kme = [r for r in kmeans_e_sweep if r["harmonic"] > 0]
        best_kmeans_e_harmonic = max(valid_kme, key=lambda x: x["harmonic"]) if valid_kme else best_kmeans_e_native
    else:
        best_kmeans_e_native = {"K": 0, "sil_e": -1, "sil_a": -1, "harmonic": -1}
        best_kmeans_e_harmonic = best_kmeans_e_native
    
    # Best K-Means(A) by sil_a (native metric)
    if kmeans_a_sweep:
        best_kmeans_a_native = max(kmeans_a_sweep, key=lambda x: x["sil_a"])
        # Also get best by harmonic
        valid_kma = [r for r in kmeans_a_sweep if r["harmonic"] > 0]
        best_kmeans_a_harmonic = max(valid_kma, key=lambda x: x["harmonic"]) if valid_kma else best_kmeans_a_native
    else:
        best_kmeans_a_native = {"K": 0, "sil_e": -1, "sil_a": -1, "harmonic": -1}
        best_kmeans_a_harmonic = best_kmeans_a_native
    
    return {
        "prefix_id": prefix_id,
        "n_samples": n_samples,
        "rd_sweep": rd_results,
        "kmeans_e_sweep": kmeans_e_sweep,
        "kmeans_a_sweep": kmeans_a_sweep,
        "best": {
            "rd": best_rd,
            "kmeans_e_native": best_kmeans_e_native,
            "kmeans_e_harmonic": best_kmeans_e_harmonic,
            "kmeans_a_native": best_kmeans_a_native,
            "kmeans_a_harmonic": best_kmeans_a_harmonic,
        }
    }


def aggregate_results(all_results: List[Dict]) -> Dict[str, Any]:
    """Aggregate cross-silhouette results across prefixes.
    
    Args:
        all_results: List of per-prefix results
        
    Returns:
        Aggregated statistics
    """
    # Collect best results
    rd_sil_e = []
    rd_sil_a = []
    rd_harmonic = []
    
    kme_native_sil_e = []
    kme_native_sil_a = []
    kme_harmonic_sil_e = []
    kme_harmonic_sil_a = []
    
    kma_native_sil_e = []
    kma_native_sil_a = []
    kma_harmonic_sil_e = []
    kma_harmonic_sil_a = []
    
    for r in all_results:
        best = r["best"]
        
        # RD
        if best["rd"]["sil_e"] > -1:
            rd_sil_e.append(best["rd"]["sil_e"])
            rd_sil_a.append(best["rd"]["sil_a"])
            if best["rd"]["harmonic"] > 0:
                rd_harmonic.append(best["rd"]["harmonic"])
        
        # K-Means(E) native
        if best["kmeans_e_native"]["sil_e"] > -1:
            kme_native_sil_e.append(best["kmeans_e_native"]["sil_e"])
            kme_native_sil_a.append(best["kmeans_e_native"]["sil_a"])
        
        # K-Means(E) harmonic
        if best["kmeans_e_harmonic"]["sil_e"] > -1:
            kme_harmonic_sil_e.append(best["kmeans_e_harmonic"]["sil_e"])
            kme_harmonic_sil_a.append(best["kmeans_e_harmonic"]["sil_a"])
        
        # K-Means(A) native
        if best["kmeans_a_native"]["sil_a"] > -1:
            kma_native_sil_e.append(best["kmeans_a_native"]["sil_e"])
            kma_native_sil_a.append(best["kmeans_a_native"]["sil_a"])
        
        # K-Means(A) harmonic
        if best["kmeans_a_harmonic"]["sil_a"] > -1:
            kma_harmonic_sil_e.append(best["kmeans_a_harmonic"]["sil_e"])
            kma_harmonic_sil_a.append(best["kmeans_a_harmonic"]["sil_a"])
    
    def stats(arr):
        if not arr:
            return {"mean": -1, "std": 0, "n": 0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "n": len(arr),
        }
    
    return {
        "n_prefixes": len(all_results),
        "rd_clustering": {
            "sil_e": stats(rd_sil_e),
            "sil_a": stats(rd_sil_a),
            "harmonic": stats(rd_harmonic),
            "description": "RD clustering (best by harmonic silhouette)"
        },
        "kmeans_e_native": {
            "sil_e": stats(kme_native_sil_e),
            "sil_a": stats(kme_native_sil_a),
            "description": "K-Means on Embeddings (best by sil_e)"
        },
        "kmeans_e_harmonic": {
            "sil_e": stats(kme_harmonic_sil_e),
            "sil_a": stats(kme_harmonic_sil_a),
            "description": "K-Means on Embeddings (best by harmonic)"
        },
        "kmeans_a_native": {
            "sil_e": stats(kma_native_sil_e),
            "sil_a": stats(kma_native_sil_a),
            "description": "K-Means on Attributions (best by sil_a)"
        },
        "kmeans_a_harmonic": {
            "sil_e": stats(kma_harmonic_sil_e),
            "sil_a": stats(kma_harmonic_sil_a),
            "description": "K-Means on Attributions (best by harmonic)"
        },
    }


def print_summary_table(agg: Dict[str, Any], logger):
    """Print a formatted summary table."""
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-MODALITY SILHOUETTE SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Method':<35} {'Sil(E) mean±std':<20} {'Sil(A) mean±std':<20} {'N':<5}")
    logger.info("-" * 80)
    
    def fmt(s):
        if s["n"] == 0:
            return "N/A"
        return f"{s['mean']:.4f}±{s['std']:.4f}"
    
    methods = [
        ("RD Clustering (harmonic best)", "rd_clustering"),
        ("K-Means(E) - native best", "kmeans_e_native"),
        ("K-Means(E) - harmonic best", "kmeans_e_harmonic"),
        ("K-Means(A) - native best", "kmeans_a_native"),
        ("K-Means(A) - harmonic best", "kmeans_a_harmonic"),
    ]
    
    for name, key in methods:
        m = agg.get(key, {})
        sil_e = m.get("sil_e", {})
        sil_a = m.get("sil_a", {})
        n = sil_e.get("n", 0)
        logger.info(f"{name:<35} {fmt(sil_e):<20} {fmt(sil_a):<20} {n:<5}")
    
    logger.info("=" * 80)
    
    # Key findings
    logger.info("\nKEY FINDINGS:")
    rd = agg.get("rd_clustering", {})
    kme = agg.get("kmeans_e_native", {})
    kma = agg.get("kmeans_a_native", {})
    
    rd_sil_e = rd.get("sil_e", {}).get("mean", -1)
    rd_sil_a = rd.get("sil_a", {}).get("mean", -1)
    kme_sil_e = kme.get("sil_e", {}).get("mean", -1)
    kme_sil_a = kme.get("sil_a", {}).get("mean", -1)
    kma_sil_e = kma.get("sil_e", {}).get("mean", -1)
    kma_sil_a = kma.get("sil_a", {}).get("mean", -1)
    
    logger.info(f"  - K-Means(E) optimizes Sil(E)={kme_sil_e:.4f}, but cross-Sil(A)={kme_sil_a:.4f}")
    logger.info(f"  - K-Means(A) optimizes Sil(A)={kma_sil_a:.4f}, but cross-Sil(E)={kma_sil_e:.4f}")
    logger.info(f"  - RD Clustering balances: Sil(E)={rd_sil_e:.4f}, Sil(A)={rd_sil_a:.4f}")
    
    if rd_sil_e > 0 and rd_sil_a > 0:
        rd_harmonic = 2 * rd_sil_e * rd_sil_a / (rd_sil_e + rd_sil_a)
        logger.info(f"  - RD harmonic mean: {rd_harmonic:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Cross-Silhouette Analysis")
    parser.add_argument("--clustering-dir", type=Path, required=True,
                        help="Directory with RD clustering results (5_clustering)")
    parser.add_argument("--embeddings-dir", type=Path, required=True,
                        help="Directory with embeddings (4_feature_extraction/embeddings)")
    parser.add_argument("--attribution-dir", type=Path, required=True,
                        help="Directory with attribution graphs (3_attribution_graphs)")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of prefixes to process")
    parser.add_argument("--k-range", type=int, nargs=2, default=[2, 15],
                        help="K range for K-Means (default: 2 15)")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["mean", "max", "sum"],
                        help="Pooling method for attributions")
    parser.add_argument("--log-file", type=Path, default=None)
    
    args = parser.parse_args()
    
    # Setup output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.log_file or args.output_dir / "cross_silhouette_analysis.log"
    logger = setup_logger("cross_sil", log_file=log_file)
    
    logger.info("=" * 60)
    logger.info("CROSS-SILHOUETTE ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Clustering dir: {args.clustering_dir}")
    logger.info(f"Embeddings dir: {args.embeddings_dir}")
    logger.info(f"Attribution dir: {args.attribution_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    
    # Find prefixes
    sweep_files = sorted(args.clustering_dir.glob("*_sweep_results.json"))
    prefix_ids = [f.stem.replace("_sweep_results", "") for f in sweep_files]
    
    if args.max_samples and len(prefix_ids) > args.max_samples:
        prefix_ids = prefix_ids[:args.max_samples]
    
    logger.info(f"Processing {len(prefix_ids)} prefixes")
    
    K_range = list(range(args.k_range[0], args.k_range[1] + 1))
    logger.info(f"K-Means K range: {K_range}")
    
    # Process prefixes
    all_results = []
    for prefix_id in tqdm(prefix_ids, desc="Analyzing prefixes"):
        result = analyze_prefix(
            prefix_id,
            args.clustering_dir,
            args.embeddings_dir,
            args.attribution_dir,
            K_range,
            logger,
            pooling=args.pooling,
        )
        if result:
            all_results.append(result)
    
    logger.info(f"Successfully processed {len(all_results)}/{len(prefix_ids)} prefixes")
    
    # Aggregate results
    aggregated = aggregate_results(all_results)
    
    # Print summary
    print_summary_table(aggregated, logger)
    
    # Save results
    results_file = args.output_dir / "cross_silhouette_results.json"
    save_json({
        "summary": aggregated,
        "per_prefix": all_results,
    }, results_file)
    logger.info(f"\nSaved results to {results_file}")
    
    # Save summary separately
    summary_file = args.output_dir / "cross_silhouette_summary.json"
    save_json(aggregated, summary_file)
    logger.info(f"Saved summary to {summary_file}")


if __name__ == "__main__":
    main()
