#!/usr/bin/env -S uv run python
"""t-SNE visualization functions for clustering results."""

from pathlib import Path
from typing import Dict, Optional, Set

import matplotlib.pyplot as plt
import numpy as np


def plot_tsne_clusters(
    data: np.ndarray,
    assignments: np.ndarray,
    output_path: Path,
    title: str = "t-SNE Cluster Visualization",
    perplexity: int = 30,
    n_iter: int = 1000,
    figsize: tuple = (12, 10),
    show_centroids: bool = True,
    probabilities: np.ndarray = None,
    size_range: tuple = (10, 200),
    logger=None
) -> Path:
    """
    Create t-SNE visualization colored by cluster assignment.

    Args:
        data: High-dimensional vectors (embeddings or attributions), shape (n_samples, d)
        assignments: Cluster ID per sample, shape (n_samples,). Use -1 for invalid samples.
        output_path: Where to save the plot
        title: Plot title
        perplexity: t-SNE perplexity (default 30)
        n_iter: Number of iterations (default 1000)
        figsize: Figure size
        show_centroids: Whether to mark cluster centroids with stars
        probabilities: Optional probability per sample, shape (n_samples,). Point sizes scale with probability.
        size_range: Min and max point sizes when using probabilities (default 10, 200)
        logger: Optional logger

    Returns:
        Path to saved plot
    """
    from sklearn.manifold import TSNE

    if logger:
        logger.info(f"    Running t-SNE (perplexity={perplexity}, n_iter={n_iter})...")

    # Filter out invalid samples (assignments == -1)
    valid_mask = assignments >= 0
    X_valid = data[valid_mask]
    y_valid = assignments[valid_mask]

    # Filter probabilities if provided
    probs_valid = None
    if probabilities is not None:
        probs_valid = probabilities[valid_mask]

    if len(X_valid) == 0:
        if logger:
            logger.warning("    No valid samples for t-SNE")
        return None

    # Adjust perplexity if needed (must be < n_samples)
    actual_perplexity = min(perplexity, len(X_valid) - 1)
    if actual_perplexity < 5:
        if logger:
            logger.warning(f"    Too few samples ({len(X_valid)}) for meaningful t-SNE")
        return None

    # Handle NaNs and Infs
    X_valid = np.nan_to_num(X_valid, nan=0.0, posinf=0.0, neginf=0.0)

    # Check for zero variance or empty data
    if X_valid.size == 0:
        if logger:
            logger.warning("    Empty data after filtering. Skipping t-SNE.")
        return None

    if np.allclose(X_valid, X_valid[0]):
        if logger:
            logger.warning("    Data has zero variance (all samples identical). Skipping t-SNE.")
        return None

    # Run t-SNE
    # Use init='random' to avoid potential PCA segfaults on sparse/degenerate data
    tsne = TSNE(
        n_components=2,
        perplexity=actual_perplexity,
        max_iter=n_iter,
        random_state=42,
        init='random',
        learning_rate='auto'
    )
    X_2d = tsne.fit_transform(X_valid)

    if logger:
        logger.info(f"    t-SNE complete. Plotting...")

    # Compute point sizes based on probabilities
    if probs_valid is not None:
        # Normalize probabilities to [0, 1] range for size scaling
        # Use sqrt scaling for better visual perception (area ~ probability)
        probs_normalized = probs_valid / (probs_valid.max() + 1e-10)
        sizes = size_range[0] + (size_range[1] - size_range[0]) * np.sqrt(probs_normalized)
    else:
        sizes = np.full(len(X_valid), 20)  # Default uniform size

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    unique_clusters = sorted(set(y_valid))
    n_clusters = len(unique_clusters)

    # Choose colormap
    if n_clusters <= 10:
        cmap = plt.cm.tab10
    elif n_clusters <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.turbo

    # Plot each cluster
    for i, c in enumerate(unique_clusters):
        mask = y_valid == c
        color = cmap(i / max(n_clusters - 1, 1))
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=[color],
            label=f"C{c} (n={mask.sum()})",
            alpha=0.6,
            s=sizes[mask],
            edgecolors='none'
        )

    # Mark centroids (mean of cluster points in 2D space)
    if show_centroids:
        for i, c in enumerate(unique_clusters):
            mask = y_valid == c
            if mask.sum() > 0:
                centroid_2d = X_2d[mask].mean(axis=0)
                ax.scatter(
                    centroid_2d[0], centroid_2d[1],
                    marker='*',
                    s=200,
                    c='black',
                    edgecolors='white',
                    linewidth=1,
                    zorder=10
                )
                ax.annotate(
                    f"C{c}",
                    centroid_2d,
                    fontsize=9,
                    ha='center',
                    va='bottom',
                    fontweight='bold'
                )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    # Legend
    if n_clusters <= 15:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    else:
        # Too many clusters - skip legend
        pass

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"    Saved: {output_path}")

    return output_path


def plot_tsne_comparison(
    embeddings: np.ndarray,
    attributions: np.ndarray,
    assignments: np.ndarray,
    output_path: Path,
    prefix_id: str,
    figsize: tuple = (20, 8),
    perplexity: int = 30,
    n_iter: int = 1000,
    probabilities: np.ndarray = None,
    size_range: tuple = (10, 200),
    logger=None
) -> Path:
    """
    Create side-by-side t-SNE comparison of embedding and attribution spaces.

    Args:
        embeddings: Embedding vectors, shape (n_samples, 768)
        attributions: Attribution vectors, shape (n_samples, ~8500)
        assignments: Cluster assignments
        output_path: Where to save
        prefix_id: For title
        figsize: Figure size
        perplexity: t-SNE perplexity
        n_iter: Number of iterations
        probabilities: Optional probability per sample for point sizing
        size_range: Min and max point sizes when using probabilities
        logger: Optional logger

    Returns:
        Path to saved plot
    """
    from sklearn.manifold import TSNE

    if logger:
        logger.info(f"    Running t-SNE comparison...")

    # Filter out invalid samples
    valid_mask = assignments >= 0
    X_embed = embeddings[valid_mask]
    X_attr = attributions[valid_mask]
    y_valid = assignments[valid_mask]

    # Filter probabilities if provided
    probs_valid = None
    if probabilities is not None:
        probs_valid = probabilities[valid_mask]

    if len(X_embed) == 0:
        if logger:
            logger.warning("    No valid samples for t-SNE comparison")
        return None

    actual_perplexity = min(perplexity, len(X_embed) - 1)
    if actual_perplexity < 5:
        if logger:
            logger.warning(f"    Too few samples for t-SNE")
        return None

    # Compute point sizes based on probabilities
    if probs_valid is not None:
        probs_normalized = probs_valid / (probs_valid.max() + 1e-10)
        sizes = size_range[0] + (size_range[1] - size_range[0]) * np.sqrt(probs_normalized)
    else:
        sizes = np.full(len(y_valid), 20)

    # Handle NaNs and Infs
    X_embed = np.nan_to_num(X_embed, nan=0.0, posinf=0.0, neginf=0.0)
    X_attr = np.nan_to_num(X_attr, nan=0.0, posinf=0.0, neginf=0.0)

    # Check for zero variance
    if np.allclose(X_embed, X_embed[0]):
        if logger:
            logger.warning("    Embedding data has zero variance. Skipping comparison.")
        return None
    if np.allclose(X_attr, X_attr[0]):
        if logger:
            logger.warning("    Attribution data has zero variance. Skipping comparison.")
        return None

    # Run t-SNE on both (init='random' for stability)
    tsne_embed = TSNE(n_components=2, perplexity=actual_perplexity, max_iter=n_iter,
                      random_state=42, init='random', learning_rate='auto')
    X_embed_2d = tsne_embed.fit_transform(X_embed)

    tsne_attr = TSNE(n_components=2, perplexity=actual_perplexity, max_iter=n_iter,
                     random_state=42, init='random', learning_rate='auto')
    X_attr_2d = tsne_attr.fit_transform(X_attr)

    # Plot side by side
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    unique_clusters = sorted(set(y_valid))
    n_clusters = len(unique_clusters)
    cmap = plt.cm.tab20 if n_clusters <= 20 else plt.cm.turbo

    for ax, X_2d, space_name in [(axes[0], X_embed_2d, "Embedding"),
                                   (axes[1], X_attr_2d, "Attribution")]:
        for i, c in enumerate(unique_clusters):
            mask = y_valid == c
            color = cmap(i / max(n_clusters - 1, 1))
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[color], label=f"C{c}", alpha=0.6, s=sizes[mask])

            # Centroid
            if mask.sum() > 0:
                centroid = X_2d[mask].mean(axis=0)
                ax.scatter(centroid[0], centroid[1], marker='*', s=150,
                           c='black', edgecolors='white', linewidth=1, zorder=10)

        ax.set_title(f"{space_name} Space", fontsize=12)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.3)

    # Shared legend
    if n_clusters <= 10:
        axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)

    plt.suptitle(f"{prefix_id}: t-SNE Comparison", fontsize=14)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"    Saved: {output_path}")

    return output_path
