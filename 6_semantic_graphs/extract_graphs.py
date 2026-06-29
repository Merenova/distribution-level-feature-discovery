#!/usr/bin/env -S uv run python
"""Extract semantic graphs from clustering results.

For each prefix and component c, the semantic graph H_c is defined as the
attribution center μ_c^(a). This script also computes:
- Soft node memberships σ_{j,c} = |H_c[j]| / (Σ_{c'} |H_{c'}[j]| + ε)
- Token-level attribution embeddings G_s (probability-weighted mean)
- Attribution mixture reconstruction G_s ≈ Σ_c π_{s,c} * H_c
- Token scores π_{s,c} from path probabilities
"""

import sys

# Fix Python path issue - remove Python 3.12 global packages before importing
# This prevents version conflicts with packages in the venv
sys.path = [p for p in sys.path if 'python3.12' not in p]

import argparse
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Add circuit-tracer to path (relative to project root)
CIRCUIT_TRACER_PATH = Path(__file__).resolve().parents[1] / "circuit-tracer"
sys.path.insert(0, str(CIRCUIT_TRACER_PATH))

from utils.config import PathConfig
from utils.attribution_pooling import load_pooled_attributions
from utils.data_utils import load_json, save_json, save_torch, reconstruct_active_features
from utils.logging_utils import setup_logger, get_log_path
from utils.manifest import filter_samples_by_manifest, update_manifest_with_results


def extract_semantic_graphs(
    clustering_result: dict,
    branches_data: dict,
    attribution_graph_path: Path,
    logger,
    pooling: str = "mean",
) -> dict:
    """Extract semantic graphs and related quantities from clustering result.

    Args:
        clustering_result: Clustering result dictionary
        branches_data: Branch samples data with continuations
        attribution_graph_path: Path to attribution graph .pt file
        logger: Logger instance
        pooling: Attribution pooling mode used by Stage 5 clustering

    Returns:
        Dictionary with semantic graphs, soft node memberships, token attributions, etc.
    """
    components = clustering_result["components"]
    assignments = clustering_result["assignments"]


    # Load hierarchical decomposition components (if available)
    H_0 = clustering_result.get("H_0")
    if H_0 is not None:
        H_0 = np.array(H_0)
        logger.info(f"Loaded H_0 (shared attribution): ||H_0|| = {np.linalg.norm(H_0):.4f}")

    # Extract path probabilities from all continuations
    path_probs_original = []
    for cont in branches_data.get("continuations", []):
        path_probs_original.append(cont.get("probability", 0.0))

    path_probs_original = np.array(path_probs_original)
    n_samples = len(path_probs_original)

    # REVISION 2026-01-13: Use path_probs_original directly.
    # We no longer normalize/average within token groups.
    path_probs = path_probs_original

    # Log both original and normalized probabilities
    logger.info(f"Original probabilities: sum={path_probs_original.sum():.6f}, "
                f"min={path_probs_original.min():.2e}, max={path_probs_original.max():.2e}")

    # Load attribution graph and extract a_n for each continuation
    # We need the individual attributions (a_n) to compute token-level attribution embeddings (G_s)
    # G_s is the weighted average of attributions for continuations starting with token s.
    # While H_c (semantic graphs) are derived from Stage 5 clusters, G_s is the "ground truth"
    # we want to reconstruct: G_s ≈ Σ_c π_{s,c} * H_c
    logger.info(f"Loading attribution graph from {attribution_graph_path}...")

    assert str(attribution_graph_path).endswith("_prefix_context.pt"), "Only prefix context format is supported"

    # Load dictionary
    data = torch.load(attribution_graph_path, weights_only=False)

    # Extract dimensions
    n_features = data["n_prefix_features"]
    n_error = data["n_prefix_errors"]
    n_token = data["n_prefix_tokens"]
    n_attribution_nodes = data["n_prefix_sources"]

    pooled_attributions = load_pooled_attributions(
        attribution_graph_path,
        pooling=pooling,
    )
    attributions_a = pooled_attributions.values
    logger.info(
        "Loaded attribution embeddings from prefix context: shape %s, source=%s, "
        "requested_pooling=%s, effective_pooling=%s",
        attributions_a.shape,
        pooled_attributions.source,
        pooled_attributions.requested_pooling,
        pooled_attributions.effective_pooling,
    )

    # Verify shape matches n_samples (continuations)
    if attributions_a.shape[0] != n_samples:
            logger.warning(f"Mismatch in samples: branches={n_samples}, attributions={attributions_a.shape[0]}")

    # Extract feature mapping
    # In prefix_context, we found that:
    # - 'decoder_locations' is (2, N) where row 0 = layer, row 1 = position (0..prefix_len)
    # - 'selected_features' is (N,) containing the actual feature IDs
    # So we can construct active_features as (N, 3) = [layer, pos, feat_id]
    
    active_features = None
    
    if "active_features" in data:
            active_features = data["active_features"]
    
    # If not present (new format), reconstruct from decoder_locations, selected_features, and activation_matrix
    if active_features is None and "decoder_locations" in data and "selected_features" in data:
        try:
            # Pass activation_matrix to extract actual feature IDs
            # (selected_features contains indices into the sparse tensor, not actual feature IDs)
            activation_matrix = data.get("activation_matrix", None)
            active_features = reconstruct_active_features(
                data["decoder_locations"],
                data["selected_features"],
                activation_matrix=activation_matrix,
                return_numpy=True
            )
            logger.info(f"Reconstructed active_features from decoder_locations and selected_features: {active_features.shape}")
        except ValueError as e:
            logger.warning(f"Could not reconstruct active_features: {e}")

    if active_features is not None:
            if isinstance(active_features, torch.Tensor):
                active_features = active_features.cpu().numpy()
    else:
            logger.warning("active_features mapping not found and could not be reconstructed")

    logger.info(f"Extracted attribution embeddings: shape {attributions_a.shape}")

    # Convert components to numpy arrays
    # With hierarchical decomposition:
    #   - mu_a in clustering is Delta_H_c (centered)
    # We use Delta_H_c directly as H_c (no H_0 added back).
    semantic_graphs = {}  # Delta_H_c directly
    semantic_graphs_centered = {}  # Same as semantic_graphs
    component_ids = []

    for c_str, comp in components.items():
        c = int(c_str)
        component_ids.append(c)

        # Delta_H_c (centered, from clustering on centered attributions)
        Delta_H_c = np.array(comp["mu_a"])
        semantic_graphs_centered[c] = Delta_H_c

        # Use Delta_H_c directly (no H_0 added back)
        semantic_graphs[c] = Delta_H_c

    # Compute soft node memberships σ_{j,c}
    soft_node_memberships = None
    if len(semantic_graphs) > 0:
        # Stack all graphs
        graphs_stacked = np.stack([semantic_graphs[c] for c in component_ids], axis=0)
        # graphs_stacked shape: (n_components, E_x)

        # Compute soft memberships: σ_{j,c} = |H_c[j]| / (Σ_{c'} |H_{c'}[j]| + ε)
        abs_graphs = np.abs(graphs_stacked)  # (n_components, E_x)
        epsilon = 1e-8
        denominators = np.sum(abs_graphs, axis=0) + epsilon  # (E_x,)
        soft_node_memberships = abs_graphs / denominators[None, :]  # (n_components, E_x)
        logger.info(f"Computed soft node memberships: shape {soft_node_memberships.shape}")

        # Verify normalization (each column should sum to ~1)
        column_sums = np.sum(soft_node_memberships, axis=0)
        logger.info(f"  Membership sums: min={column_sums.min():.6f}, max={column_sums.max():.6f}, mean={column_sums.mean():.6f}")
    else:
        soft_node_memberships = np.array([])

    # Token-level summaries are disabled when token grouping is removed
    token_scores = {}
    token_attributions = {}
    token_attributions_reconstructed = {}
    attribution_reconstruction_errors = {}

    return {
        # Hierarchical decomposition (shared components)
        "H_0": H_0,  # Global mean (shared attribution)
        # Delta_H_c directly
        "semantic_graphs": semantic_graphs,
        # Same as semantic_graphs
        "semantic_graphs_centered": semantic_graphs_centered,
        # Other fields
        "soft_node_memberships": soft_node_memberships,
        "token_attributions": token_attributions,
        "token_attributions_reconstructed": token_attributions_reconstructed,
        "attribution_reconstruction_errors": attribution_reconstruction_errors,
        "token_scores": token_scores,
        "component_ids": component_ids,
        # Feature index mapping: active_features[i] = [layer, pos, feat_id]
        # Allows interpreting H_c[i] as a specific model feature
        "active_features": active_features,
        "n_features": n_features,
        "n_error_nodes": n_error,
        "n_token_nodes": n_token,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract semantic graphs")
    parser.add_argument(
        "--clustering-dir",
        type=Path,
        required=True,
        help="Directory with clustering results"
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        required=True,
        help="Directory with original samples"
    )
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
        help="Output directory (default: semantic_graphs/)"
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
        help="Attribution pooling mode used by Stage 5 clustering",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode (only progress bars)"
    )
    args = parser.parse_args()

    # Setup paths
    paths = PathConfig()
    paths.ensure_dirs()

    if args.output_dir is None:
        args.output_dir = paths.results_semantic_graphs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    log_file = get_log_path("6_semantic_graphs", args.log_dir)
    import logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logger = setup_logger("semantic_graphs", log_file=log_file, level=log_level)

    logger.info("=" * 60)
    logger.info("SEMANTIC GRAPH EXTRACTION")
    logger.info("=" * 60)
    logger.info(f"Clustering dir: {args.clustering_dir}")
    logger.info(f"Samples dir: {args.samples_dir}")
    logger.info(f"Attribution graphs dir: {args.attribution_graphs_dir}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Attribution pooling: {args.pooling}")

    # Find all clustering sweep result files
    clustering_files = sorted(args.clustering_dir.glob("*_sweep_results.json"))
    logger.info(f"\nFound {len(clustering_files)} sweep results")

    # Filter based on Stage 5 manifest
    # Derive results_dir from clustering_dir to respect --output-dir
    # clustering_dir is typically {output_dir}/results/5_clustering/
    results_dir = args.clustering_dir.parent
    all_prefix_ids = [f.stem.replace("_sweep_results", "") for f in clustering_files]
    available_ids, skipped_ids = filter_samples_by_manifest(
        all_prefix_ids, results_dir, "stage5", logger
    )
    # Filter clustering files to only available ones
    available_id_set = set(available_ids)
    clustering_files = [f for f in clustering_files if f.stem.replace("_sweep_results", "") in available_id_set]
    logger.info(f"Processing {len(clustering_files)} available results (skipped {len(skipped_ids)})")

    # Process each file
    completed_ids = []
    failed_ids = []
    errors = {}
    
    # Determine if we should show progress bar (quiet mode check via logger level)
    show_pbar = False
    if logger and logger.getEffectiveLevel() >= 30: # logging.WARNING
        show_pbar = True
        
    iterator = clustering_files
    if show_pbar:
        from tqdm import tqdm
        iterator = tqdm(clustering_files, desc="Extracting graphs")

    for clustering_file in iterator:
        prefix_id = clustering_file.stem.replace("_sweep_results", "")
        logger.info(f"\nProcessing: {prefix_id}")

        try:
            # Load clustering sweep results
            clustering_sweep = load_json(clustering_file)

            # Load branch samples data
            branches_file = args.samples_dir / f"{prefix_id}_branches.json"
            if not branches_file.exists():
                logger.warning(f"Missing samples for {prefix_id}, skipping")
                failed_ids.append(prefix_id)
                errors[prefix_id] = f"FileNotFoundError: {branches_file}"
                continue
            branches_data = load_json(branches_file)

            # Load attribution graph. The prefix-context format is the only
            # supported format for pooled continuation attribution.
            attribution_graph_file = args.attribution_graphs_dir / f"{prefix_id}_prefix_context.pt"

            if not attribution_graph_file.exists():
                logger.warning(f"Missing attribution graph for {prefix_id}, skipping")
                failed_ids.append(prefix_id)
                errors[prefix_id] = f"FileNotFoundError: {attribution_graph_file}"
                continue

            grid_results = clustering_sweep.get("grid", [])

            # Get K_max from sweep config (default 20)
            sweep_config = clustering_sweep.get("sweep_config", {})
            K_max = sweep_config.get("K_max", 20)

            # Filter: valid entries with intermediate K (not 1 and not K_max)
            valid_grid = []
            skipped_boundary = []
            for entry in grid_results:
                if not entry.get("components") or not entry.get("assignments") or "error" in entry:
                    continue
                K = entry.get("K", len(entry.get("components", {})))
                if K == 1 or K == K_max:
                    skipped_boundary.append((entry.get("beta"), entry.get("gamma"), K))
                    continue
                valid_grid.append(entry)

            if skipped_boundary:
                logger.info(f"  Skipped {len(skipped_boundary)} configs with K=1 or K={K_max}: {skipped_boundary[:5]}{'...' if len(skipped_boundary) > 5 else ''}")

            if not valid_grid:
                logger.warning(f"  No valid clustering results for {prefix_id} (all K=1 or K={K_max})")
                failed_ids.append(prefix_id)
                errors[prefix_id] = f"No valid clustering results (all K=1 or K={K_max})"
                continue

            processed_any = False
            for grid_entry in valid_grid:
                beta = grid_entry.get("beta")
                gamma = grid_entry.get("gamma")
                clustering_key = f"beta{beta}_gamma{gamma}"
                clustering_result = {
                    "components": grid_entry.get("components", {}),
                    "assignments": grid_entry.get("assignments", []),
                    "H_0": clustering_sweep.get("H_0"),
                }

                # Extract semantic graphs
                graphs_data = extract_semantic_graphs(
                    clustering_result,
                    branches_data,
                    attribution_graph_file,
                    logger,
                    pooling=args.pooling,
                )

                logger.info(f"  Components ({clustering_key}): {len(graphs_data['component_ids'])}")
                logger.info(f"  Unique tokens ({clustering_key}): {len(graphs_data['token_scores'])}")

                # Save graphs (as PyTorch .pt file, consistent with knowledge_attribution)
                output_file_pt = args.output_dir / f"{prefix_id}_{clustering_key}_semantic_graphs.pt"
                save_torch(graphs_data, output_file_pt)
                logger.info(f"  Saved to: {output_file_pt}")

                # Also save JSON summary
                output_file_json = args.output_dir / f"{prefix_id}_{clustering_key}_semantic_graphs.json"
                # Compute soft membership stats if available
                soft_membership_stats = {}
                if graphs_data["soft_node_memberships"] is not None and len(graphs_data["soft_node_memberships"]) > 0:
                    sigma = graphs_data["soft_node_memberships"]
                    soft_membership_stats = {
                        "shape": list(sigma.shape),
                        "max_membership_per_node": {
                            "mean": float(np.max(sigma, axis=0).mean()),
                            "min": float(np.max(sigma, axis=0).min()),
                            "max": float(np.max(sigma, axis=0).max()),
                        },
                    }

                # Feature mapping info for JSON summary
                feature_mapping_info = {
                    "n_features": graphs_data["n_features"],
                    "n_error_nodes": graphs_data["n_error_nodes"],
                    "n_token_nodes": graphs_data["n_token_nodes"],
                    "has_active_features": graphs_data["active_features"] is not None,
                }
                if graphs_data["active_features"] is not None:
                    feature_mapping_info["active_features_shape"] = list(graphs_data["active_features"].shape)

                # Prepare H_0 for JSON
                H_0_json = None
                if graphs_data["H_0"] is not None:
                    H_0_json = {
                        "norm": float(np.linalg.norm(graphs_data["H_0"])),
                        "shape": list(graphs_data["H_0"].shape),
                    }

                graphs_data_json = {
                    "prefix_id": prefix_id,
                    "prefix": clustering_sweep.get("prefix", ""),
                    "clustering_key": clustering_key,
                    "beta": beta,
                    "gamma": gamma,
                    "n_components": len(graphs_data["component_ids"]),
                    "component_ids": graphs_data["component_ids"],
                    "n_tokens": len(graphs_data["token_scores"]),
                    # Hierarchical decomposition
                    "H_0": H_0_json,
                    "semantic_graphs_shape": {
                        str(c): list(H.shape) for c, H in graphs_data["semantic_graphs"].items()
                    },
                    "feature_mapping": feature_mapping_info,
                    "soft_node_memberships": soft_membership_stats,
                    "token_scores": {
                        str(token): {str(c): float(score) for c, score in scores.items()}
                        for token, scores in graphs_data["token_scores"].items()
                    },
                    "attribution_reconstruction_errors": {
                        str(token): float(error)
                        for token, error in graphs_data["attribution_reconstruction_errors"].items()
                    },
                    "reconstruction_error_stats": {
                        "mean": float(np.mean(list(graphs_data["attribution_reconstruction_errors"].values()))),
                        "max": float(np.max(list(graphs_data["attribution_reconstruction_errors"].values()))),
                        "min": float(np.min(list(graphs_data["attribution_reconstruction_errors"].values()))),
                    } if len(graphs_data["attribution_reconstruction_errors"]) > 0 else {},
                }
                save_json(graphs_data_json, output_file_json)
                processed_any = True

            if processed_any:
                completed_ids.append(prefix_id)

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Failed to process {prefix_id}: {error_msg}")
            import traceback
            logger.error(traceback.format_exc())
            failed_ids.append(prefix_id)
            errors[prefix_id] = error_msg

    # Write Stage 6 manifest
    update_manifest_with_results(
        results_dir=results_dir,
        stage_name="stage6",
        processed=completed_ids,
        failed=failed_ids,
        skipped=skipped_ids,
        logger=logger,
        errors=errors,
    )

    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Processed {len(completed_ids)} results")
    logger.info(f"Completed: {len(completed_ids)}, Failed: {len(failed_ids)}, Skipped: {len(skipped_ids)}")
    logger.info(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
