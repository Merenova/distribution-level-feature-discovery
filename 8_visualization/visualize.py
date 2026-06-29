#!/usr/bin/env -S uv run python
"""Stage 8: Comprehensive Visualization of Pipeline Results.

Consolidates all visualizations from the pipeline, organized by source stage:
- 5_clustering/: t-SNE plots, Sankey diagrams, clustering history
- 6_semantic_graphs/: Semantic graph heatmaps, token scores
- parameter_sweep/: Sweep analysis plots (if sweep results exist)
- interactive/: HTML cluster explorers
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import PathConfig
from utils.data_utils import load_json, load_torch
from utils.attribution_pooling import load_pooled_attributions
from utils.logging_utils import setup_logger, get_log_path

# Import local visualization modules
from tsne_plots import plot_tsne_clusters, plot_tsne_comparison
from sankey_plots import plot_sankey_cluster_to_token, get_top_continuations_per_cluster
from cluster_plots import plot_clustering_history
from semantic_graph_plots import plot_semantic_graph_heatmap, plot_token_scores
from parameter_sweep_plots import plot_harmonic_scores, plot_rd_curve, plot_rd_curve_2d, plot_heatmaps
from summary_plots import create_summary, plot_best_params, plot_aggregated_metrics, load_sweep_results

# circuit-tracer for loading graphs (relative to project root)
CIRCUIT_TRACER_PATH = Path(__file__).resolve().parents[1] / "circuit-tracer"
sys.path.insert(0, str(CIRCUIT_TRACER_PATH))


def load_prefix_data(prefix_id: str, paths: PathConfig, logger, pooling: str = "mean") -> Optional[Dict[str, Any]]:
    """Load all data needed for visualization.

    Args:
        prefix_id: Prefix identifier (e.g., "cloze_0029")
        paths: PathConfig with directory locations
        logger: Logger

    Returns:
        Dict with embeddings, attributions, assignments, token data, etc.
    """
    # Define result paths (stored under results/)
    results_dir = paths.results
    clustering_dir = paths.results_clustering
    embeddings_dir = results_dir / "4_feature_extraction" / "embeddings"
    branches_dir = results_dir / "2_branch_sampling"
    attribution_graphs_dir = results_dir / "3_attribution_graphs"
    semantic_graphs_dir = paths.results_semantic_graphs

    # Load clustering results
    clustering_file = clustering_dir / f"{prefix_id}_clustering.json"
    if not clustering_file.exists():
        logger.warning(f"  Clustering file not found: {clustering_file}")
        return None

    clustering = load_json(clustering_file)
    assignments = np.array(clustering["assignments"])
    components = clustering["components"]
    history = clustering.get("history", {})

    # Load embeddings
    embeddings_file = embeddings_dir / f"{prefix_id}_embeddings.npy"
    if not embeddings_file.exists():
        logger.warning(f"  Embeddings file not found: {embeddings_file}")
        return None
    embeddings = np.load(embeddings_file)

    # Load branches for continuation text and token info
    branches_file = branches_dir / f"{prefix_id}_branches.json"
    if not branches_file.exists():
        logger.warning(f"  Branches file not found: {branches_file}")
        return None
    branches = load_json(branches_file)

    # Extract continuation probabilities
    continuation_probs = np.array([
        cont.get("probability", 0.0)
        for cont in branches.get("continuations", [])
    ])

    # Load attribution graph and extract attributions (optional - for t-SNE)
    attributions = None
    
    # Try loading from Stage 3 output (newer format used in pipeline)
    prefix_context_file = attribution_graphs_dir / f"{prefix_id}_prefix_context.pt"
    if prefix_context_file.exists():
        try:
            logger.info(f"  Loading attribution context from {prefix_context_file}...")
            pooled_attributions = load_pooled_attributions(
                prefix_context_file,
                pooling=pooling,
                meta_file=attribution_graphs_dir / f"{prefix_id}_attribution.json",
            )
            attributions = pooled_attributions.values
            logger.info(
                "  Loaded attributions: shape %s, source=%s, effective_pooling=%s",
                attributions.shape,
                pooled_attributions.source,
                pooled_attributions.effective_pooling,
            )
        except Exception as e:
            logger.warning(f"  Failed to load attribution context: {e}")

    # Load semantic graphs for token_scores and semantic_graphs
    semantic_file = semantic_graphs_dir / f"{prefix_id}_semantic_graphs.pt"
    semantic_graphs = {}
    token_scores = {}
    if semantic_file.exists():
        semantic_data = load_torch(semantic_file)
        token_scores = semantic_data.get("token_scores", {})
        # Convert keys to int
        token_scores = {int(k): {int(c): v for c, v in scores.items()}
                        for k, scores in token_scores.items()}
        # Get semantic graphs (attribution centers)
        semantic_graphs = semantic_data.get("semantic_graphs", {})
        if semantic_graphs:
            # Convert keys to int and values to numpy
            semantic_graphs = {
                int(k): np.array(v) if not isinstance(v, np.ndarray) else v
                for k, v in semantic_graphs.items()
            }
    else:
        logger.warning(f"  Semantic graphs file not found: {semantic_file}")

    # Extract token info from branches
    token_probs = {}
    token_id_to_text = {}

    # Get P_bar from components
    P_bar = {}
    for c_str, comp in components.items():
        c = int(c_str)
        if c > 0:  # Valid cluster ID
            P_bar[c] = comp.get("W_c", 0.0)

    # Get actual prefix text from branches
    prefix_text = branches.get("prefix", "")

    return {
        "prefix_id": prefix_id,
        "prefix_text": prefix_text,
        "embeddings": embeddings,
        "attributions": attributions,
        "assignments": assignments,
        "components": components,
        "history": history,
        "branches": branches,
        "token_scores": token_scores,
        "semantic_graphs": semantic_graphs,
        "token_probs": token_probs,
        "token_id_to_text": token_id_to_text,
        "P_bar": P_bar,
        "continuation_probs": continuation_probs,
    }


def visualize_clustering(
    data: Dict[str, Any],
    output_dir: Path,
    logger,
):
    """Generate Stage 5 clustering visualizations.

    Args:
        data: Loaded prefix data
        output_dir: Output directory (5_clustering/{prefix_id}/)
        logger: Logger
    """
    prefix_id = data["prefix_id"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clustering history
    if data["history"]:
        logger.info(f"    Clustering history plot...")
        plot_clustering_history(
            data["history"],
            prefix_id,
            output_dir / f"{prefix_id}_clustering_history.png"
        )

    # t-SNE for embeddings (768d)
    logger.info(f"    t-SNE on embedding space (768d)...")
    plot_tsne_clusters(
        data["embeddings"],
        data["assignments"],
        output_dir / f"{prefix_id}_tsne_embedding.png",
        title=f"{prefix_id}: t-SNE (Embedding Space, 768d)",
        probabilities=data["continuation_probs"],
        logger=logger
    )

    # t-SNE for attributions (~8500d)
    if data["attributions"] is not None:
        logger.info(f"    t-SNE on attribution space ({data['attributions'].shape[1]}d)...")
        plot_tsne_clusters(
            data["attributions"],
            data["assignments"],
            output_dir / f"{prefix_id}_tsne_attribution.png",
            title=f"{prefix_id}: t-SNE (Attribution Space)",
            probabilities=data["continuation_probs"],
            logger=logger
        )

        # Side-by-side comparison
        logger.info(f"    Creating t-SNE comparison plot...")
        plot_tsne_comparison(
            data["embeddings"],
            data["attributions"],
            data["assignments"],
            output_dir / f"{prefix_id}_tsne_comparison.png",
            prefix_id,
            probabilities=data["continuation_probs"],
            logger=logger
        )

    # Sankey diagram
    if data["token_scores"] and data["P_bar"]:
        logger.info(f"    Sankey diagram...")
        top_conts = get_top_continuations_per_cluster(
            data["branches"],
            data["assignments"],
            data["token_scores"]
        )
        plot_sankey_cluster_to_token(
            data["token_scores"],
            data["token_probs"],
            data["P_bar"],
            top_conts,
            output_dir / f"{prefix_id}_sankey.png",
            prefix_id,
            prefix_text=data.get("prefix_text", ""),
            token_id_to_text=data["token_id_to_text"],
            logger=logger
        )


def visualize_semantic_graphs(
    data: Dict[str, Any],
    output_dir: Path,
    logger,
):
    """Generate Stage 6 semantic graph visualizations.

    Args:
        data: Loaded prefix data
        output_dir: Output directory (6_semantic_graphs/{prefix_id}/)
        logger: Logger
    """
    prefix_id = data["prefix_id"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Semantic graph heatmap
    if data["semantic_graphs"]:
        logger.info(f"    Semantic graph heatmap...")
        plot_semantic_graph_heatmap(
            data["semantic_graphs"],
            prefix_id,
            output_dir / f"{prefix_id}_semantic_graph_heatmap.png"
        )

    # Token scores
    if data["token_scores"]:
        logger.info(f"    Token scores plot...")
        plot_token_scores(
            data["token_scores"],
            prefix_id,
            output_dir / f"{prefix_id}_token_scores.png",
            token_id_to_text=data["token_id_to_text"]
        )


def visualize_sweep_results(
    sweep_results_dir: Path,
    output_dir: Path,
    logger,
):
    """Generate parameter sweep visualizations.

    Args:
        sweep_results_dir: Directory containing *_sweep_results.json files
        output_dir: Output directory (parameter_sweep/)
        logger: Logger
    """
    if not sweep_results_dir.exists():
        logger.info("  No sweep results directory found, skipping sweep visualizations")
        return

    sweep_files = list(sweep_results_dir.glob("*_sweep_results.json"))
    if not sweep_files:
        logger.info("  No sweep results found, skipping sweep visualizations")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each prefix's sweep results
    for sweep_file in sweep_files:
        prefix_id = sweep_file.stem.replace("_sweep_results", "")
        logger.info(f"  Processing sweep results for {prefix_id}...")

        with open(sweep_file) as f:
            results = json.load(f)

        prefix_output_dir = output_dir / prefix_id
        prefix_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate individual plots
        logger.info(f"    Harmonic scores plot...")
        plot_harmonic_scores(results, prefix_output_dir / f"{prefix_id}_harmonic_scores.png")

        logger.info(f"    R-D curves plot...")
        plot_rd_curve(results, prefix_output_dir / f"{prefix_id}_rd_curves.png")

        logger.info(f"    R-D surface plot...")
        plot_rd_curve_2d(results, prefix_output_dir / f"{prefix_id}_rd_surface.png")

        logger.info(f"    Heatmaps plot...")
        plot_heatmaps(results, prefix_output_dir / f"{prefix_id}_heatmaps.png")

    # Generate cross-prefix summary plots
    if len(sweep_files) > 1:
        logger.info("  Generating cross-prefix summary plots...")
        all_results = load_sweep_results(sweep_results_dir)

        summary = create_summary(all_results)
        plot_best_params(summary, output_dir / "summary_best_params.png")
        plot_aggregated_metrics(all_results, output_dir / "summary_aggregated.png")


def visualize_prefix(
    prefix_id: str,
    paths: PathConfig,
    output_base: Path,
    config: Dict,
    logger,
):
    """Generate all visualizations for a single prefix.

    Args:
        prefix_id: Prefix identifier
        paths: PathConfig with directory locations
        output_base: Base output directory
        config: Configuration dict
        logger: Logger
    """
    logger.info(f"Loading data for {prefix_id}...")

    pooling = config.get("clustering", {}).get("pooling", "mean")

    data = load_prefix_data(prefix_id, paths, logger, pooling=pooling)
    if data is None:
        logger.error(f"  Could not load data for {prefix_id}")
        return

    # === Stage 5 Clustering Visualizations ===
    logger.info(f"  Generating clustering visualizations (5_clustering/)...")
    clustering_output = output_base / "5_clustering" / prefix_id
    visualize_clustering(data, clustering_output, logger)

    # === Stage 6 Semantic Graph Visualizations ===
    logger.info(f"  Generating semantic graph visualizations (6_semantic_graphs/)...")
    semantic_output = output_base / "6_semantic_graphs" / prefix_id
    visualize_semantic_graphs(data, semantic_output, logger)

    logger.info(f"  Visualizations saved for {prefix_id}")


def main():
    parser = argparse.ArgumentParser(description="Stage 8: Comprehensive Visualization")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default_config.json"),
        help="Path to config file"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="Single prefix to visualize (default: all)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--sweep-results-dir",
        type=Path,
        default=None,
        help="Directory containing sweep results (for parameter sweep visualizations)"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for log files"
    )
    args = parser.parse_args()

    # Load config
    if args.config.exists():
        config = load_json(args.config)
    else:
        config = {}

    # Setup paths
    paths = PathConfig()
    
    # Override paths if output_dir is provided (assuming output_dir is inside results/)
    if args.output_dir:
        results_dir = args.output_dir.parent
        # logger not initialized yet
        paths.results = results_dir
        paths.results_clustering = results_dir / "5_clustering"
        paths.results_semantic_graphs = results_dir / "6_semantic_graphs"
        paths.results_branch_sampling = results_dir / "2_branch_sampling"
        paths.results_attribution_graphs = results_dir / "3_attribution_graphs"
        paths.results_feature_extraction = results_dir / "4_feature_extraction"
        paths.results_visualization = args.output_dir
    
    paths.ensure_dirs()

    # Setup logger
    import logging
    log_file = get_log_path("8_visualization", args.log_dir)
    log_level = logging.WARNING if args.quiet else logging.INFO
    logger = setup_logger("visualization", log_file=log_file, level=log_level)

    logger.info("=" * 60)
    logger.info("STAGE 8: COMPREHENSIVE VISUALIZATION")
    logger.info("=" * 60)

    # Determine output directory
    output_base = args.output_dir if args.output_dir else paths.results_visualization
    output_base.mkdir(parents=True, exist_ok=True)

    # Create output subdirectories
    (output_base / "5_clustering").mkdir(parents=True, exist_ok=True)
    (output_base / "6_semantic_graphs").mkdir(parents=True, exist_ok=True)
    (output_base / "parameter_sweep").mkdir(parents=True, exist_ok=True)
    (output_base / "interactive").mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_base}")

    # Find prefixes to process
    clustering_dir = paths.results_clustering
    if args.prefix:
        prefixes = [args.prefix]
    else:
        # Find all clustering results
        clustering_files = sorted(clustering_dir.glob("*_clustering.json"))
        prefixes = [f.stem.replace("_clustering", "") for f in clustering_files]

    if not prefixes:
        logger.warning("No clustering results found to visualize")
        return

    logger.info(f"Found {len(prefixes)} prefixes to visualize")

    # Process each prefix
    from tqdm import tqdm
    
    iterator = prefixes
    if args.quiet:
        iterator = tqdm(prefixes, desc="Generating Visualizations")
        
    for item in iterator:
        # Handle both iterator types (enumerate vs direct)
        if args.quiet:
            prefix_id = item
            i = 0 # Dummy index
        else:
            # If not quiet, we might not have used enumerate in the iterator assignment above
            # but let's just stick to the original logic if not quiet
            pass

    if not args.quiet:
        for i, prefix_id in enumerate(prefixes, 1):
            logger.info(f"\n[{i}/{len(prefixes)}] {prefix_id}")
            try:
                visualize_prefix(prefix_id, paths, output_base, config, logger)
            except Exception as e:
                logger.error(f"  Error visualizing {prefix_id}: {e}")
                import traceback
                traceback.print_exc()
    else:
        for prefix_id in iterator:
            try:
                visualize_prefix(prefix_id, paths, output_base, config, logger)
            except Exception as e:
                # Still log errors even in quiet mode (logger level usually allows ERROR)
                logger.error(f"  Error visualizing {prefix_id}: {e}")

    # Process sweep results (if available)
    sweep_dir = args.sweep_results_dir
    if sweep_dir is None:
        # Try default location
        sweep_dir = paths.results_clustering
    logger.info(f"\nChecking for sweep results in: {sweep_dir}")
    visualize_sweep_results(sweep_dir, output_base / "parameter_sweep", logger)

    logger.info("\n" + "=" * 60)
    logger.info("VISUALIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output structure:")
    logger.info(f"  5_clustering/       - t-SNE, Sankey, clustering history")
    logger.info(f"  6_semantic_graphs/  - Heatmaps, token scores")
    logger.info(f"  parameter_sweep/    - Sweep analysis (if available)")
    logger.info(f"  interactive/        - HTML explorers (if generated)")


if __name__ == "__main__":
    main()
