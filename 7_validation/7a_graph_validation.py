#!/usr/bin/env -S uv run python
"""Attribution graph quality validation using circuit-tracer metrics."""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.attribution_pooling import load_pooled_attributions
from utils.data_utils import save_json
from utils.logging_utils import setup_logger, get_log_path

# Add circuit-tracer to path (relative to project root)
CIRCUIT_TRACER_PATH = Path(__file__).resolve().parents[1] / "circuit-tracer"
sys.path.insert(0, str(CIRCUIT_TRACER_PATH))

from circuit_tracer.graph import Graph, compute_graph_scores


def compute_attribution_graph_quality(
    graph_path: Path,
    logger,
    pooling: str = "mean",
) -> Dict[str, float]:
    """Compute quality scores for a single attribution graph.

    Uses compute_graph_scores from circuit-tracer to evaluate:
    - replacement_score: Fraction of influence through features vs. errors
    - completeness_score: Fraction of edges from features/tokens vs. errors

    Args:
        graph_path: Path to attribution graph .pt file
        logger: Logger instance

    Returns:
        Dictionary with replacement_score and completeness_score
    """
    logger.info(f"Loading graph from {graph_path}")
    
    # Check if this is a prefix context file
    is_prefix_context = str(graph_path).endswith("_prefix_context.pt")
    
    if is_prefix_context:
        import torch
        import numpy as np
        
        data = torch.load(graph_path, weights_only=False)
        
        # Extract dimensions
        n_features = data["n_prefix_features"]
        n_errors = data["n_prefix_errors"]
        n_tokens = data["n_prefix_tokens"]
        
        pooled_attributions = load_pooled_attributions(graph_path, pooling=pooling)
        mean_attr = np.abs(pooled_attributions.values).mean(axis=0)
            
        # Calculate influence sums
        feature_inf = mean_attr[:n_features].sum()
        error_inf = mean_attr[n_features : n_features + n_errors].sum()
        token_inf = mean_attr[n_features + n_errors :].sum()
        
        total_inf = feature_inf + error_inf + token_inf
        
        # replacement_score: tokens / (tokens + errors)
        # Measures how much we rely on original tokens (good) vs errors (bad) 
        # when features aren't used?
        # Actually, if replacement is good, errors should be 0.
        if token_inf + error_inf > 0:
            replacement_score = token_inf / (token_inf + error_inf)
        else:
            replacement_score = 1.0 # No errors or tokens (pure feature?)
            
        # completeness_score: Approx as fraction of influence from non-error sources
        if total_inf > 0:
            completeness_score = (feature_inf + token_inf) / total_inf
        else:
            completeness_score = 0.0
            
        logger.info(f"  Replacement score (approx): {replacement_score:.4f}")
        logger.info(f"  Completeness score (approx): {completeness_score:.4f}")
        
        return {
            "replacement_score": float(replacement_score),
            "completeness_score": float(completeness_score),
        }

    graph = Graph.from_pt(graph_path)

    logger.info("Computing graph quality scores...")
    replacement_score, completeness_score = compute_graph_scores(graph)

    logger.info(f"  Replacement score: {replacement_score:.4f}")
    logger.info(f"  Completeness score: {completeness_score:.4f}")

    return {
        "replacement_score": float(replacement_score),
        "completeness_score": float(completeness_score),
    }


def validate_all_graphs(
    graph_dir: Path,
    prefix_ids: list,
    logger,
    pooling: str = "mean",
) -> Dict[str, Any]:
    """Validate attribution graph quality for all prefixes.

    Args:
        graph_dir: Directory containing attribution graph .pt files
        prefix_ids: List of prefix IDs to validate
        logger: Logger instance

    Returns:
        Dictionary with per-prefix scores and aggregate statistics
    """
    results = {}
    replacement_scores = []
    completeness_scores = []

    # Determine if we should show progress bar (quiet mode check via logger level)
    show_pbar = False
    if logger and logger.getEffectiveLevel() >= 30: # logging.WARNING
        show_pbar = True
        
    iterator = prefix_ids
    if show_pbar:
        from tqdm import tqdm
        iterator = tqdm(prefix_ids, desc="Validating graphs")

    for prefix_id in iterator:
        # Try finding graph file or prefix context file
        graph_file = graph_dir / f"{prefix_id}_graph.pt"
        if not graph_file.exists():
            graph_file = graph_dir / f"{prefix_id}_prefix_context.pt"

        if not graph_file.exists():
            logger.warning(f"Graph file not found for {prefix_id}, skipping")
            continue

        try:
            scores = compute_attribution_graph_quality(graph_file, logger, pooling=pooling)
            results[prefix_id] = scores

            replacement_scores.append(scores["replacement_score"])
            completeness_scores.append(scores["completeness_score"])

        except Exception as e:
            logger.error(f"Error processing {prefix_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    # Compute aggregate statistics
    import numpy as np

    summary = {
        "n_graphs": len(results),
        "replacement_score": {
            "mean": float(np.mean(replacement_scores)) if replacement_scores else 0.0,
            "std": float(np.std(replacement_scores)) if replacement_scores else 0.0,
            "min": float(np.min(replacement_scores)) if replacement_scores else 0.0,
            "max": float(np.max(replacement_scores)) if replacement_scores else 0.0,
        },
        "completeness_score": {
            "mean": float(np.mean(completeness_scores)) if completeness_scores else 0.0,
            "std": float(np.std(completeness_scores)) if completeness_scores else 0.0,
            "min": float(np.min(completeness_scores)) if completeness_scores else 0.0,
            "max": float(np.max(completeness_scores)) if completeness_scores else 0.0,
        },
        "per_prefix": results,
    }

    logger.info("\n" + "=" * 60)
    logger.info("ATTRIBUTION GRAPH QUALITY SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Processed {len(results)} graphs")
    logger.info(f"Replacement score: {summary['replacement_score']['mean']:.4f} ± {summary['replacement_score']['std']:.4f}")
    logger.info(f"Completeness score: {summary['completeness_score']['mean']:.4f} ± {summary['completeness_score']['std']:.4f}")

    return summary


def get_prefix_ids_from_dir(directory: Path, suffix: str) -> List[str]:
    """Extract prefix IDs from files in a directory.

    Args:
        directory: Directory to search
        suffix: File suffix to match (e.g., '_graph.pt')

    Returns:
        List of prefix IDs
    """
    prefix_ids = set()
    
    # Search for specified suffix
    for file_path in directory.glob(f"*{suffix}"):
        prefix_id = file_path.stem.replace(suffix.replace(".pt", "").replace(".json", ""), "")
        prefix_ids.add(prefix_id)
        
    # Also search for _prefix_context.pt if suffix is _graph.pt
    if suffix == "_graph.pt":
        for file_path in directory.glob("*_prefix_context.pt"):
             prefix_id = file_path.stem.replace("_prefix_context", "")
             prefix_ids.add(prefix_id)
             
    return sorted(list(prefix_ids))


def main():
    parser = argparse.ArgumentParser(description="Validate attribution graph quality")
    parser.add_argument(
        "--attribution-graphs-dir",
        type=Path,
        required=True,
        help="Directory with attribution graphs (Stage 2)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/7_validation/7a_graph_validation/)"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for log files"
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "sum", "max"],
        default="mean",
        help="Attribution pooling mode for prefix-context graphs",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode (only progress bars)"
    )
    args = parser.parse_args()

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = Path("results/7_validation/7a_graph_validation")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    log_file = get_log_path("7a_graph_validation", args.log_dir)
    import logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logger = setup_logger("graph_validation", log_file=log_file, level=log_level)

    logger.info("=" * 60)
    logger.info("ATTRIBUTION GRAPH QUALITY VALIDATION")
    logger.info("=" * 60)
    logger.info(f"Attribution graphs dir: {args.attribution_graphs_dir}")
    logger.info(f"Output dir: {args.output_dir}")

    # Get list of prefix IDs from attribution graphs directory
    prefix_ids = get_prefix_ids_from_dir(args.attribution_graphs_dir, "_graph.pt")
    logger.info(f"\nFound {len(prefix_ids)} graphs to validate")

    # Run validation
    validation_result = validate_all_graphs(
        args.attribution_graphs_dir,
        prefix_ids,
        logger,
        pooling=args.pooling,
    )

    # Save results
    output_file = args.output_dir / "graph_validation.json"
    save_json(validation_result, output_file)
    logger.info(f"\nSaved graph validation results to: {output_file}")

    logger.info("\n" + "=" * 60)
    logger.info("GRAPH VALIDATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
