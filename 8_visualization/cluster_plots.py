"""Clustering history visualization.

Extracted from 6_semantic_graphs/visualize.py for centralized visualization.
"""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def plot_clustering_history(
    history: Dict,
    prefix_id: str,
    output_file: Path,
    figsize: tuple = (12, 4),
    dpi: int = 150,
):
    """Plot clustering history (components and objective over iterations).

    Args:
        history: Clustering history dict with 'iterations', 'n_components', 'L_RD'
        prefix_id: Prefix identifier for title
        output_file: Path to save the plot
        figsize: Figure size
        dpi: Output DPI
    """
    iterations = history.get("iterations", [])
    n_components = history.get("n_components", [])

    if not iterations:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot component count
    ax1.plot(iterations, n_components, marker='o', linewidth=2)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Number of Components")
    ax1.set_title(f"{prefix_id}: Component Count")
    ax1.grid(True, alpha=0.3)

    # Plot R-D objective
    L_RD = history.get("L_RD", [])
    if L_RD:
        ax2.plot(iterations, L_RD, marker='s', linewidth=2, color='green')
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("L_RD (Rate-Distortion Objective)")
        ax2.set_title(f"{prefix_id}: R-D Objective")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
