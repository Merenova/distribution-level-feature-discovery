"""Semantic graph visualization: heatmaps and token scores.

Extracted from 6_semantic_graphs/visualize.py for centralized visualization.
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_semantic_graph_heatmap(
    semantic_graphs: Dict[int, np.ndarray],
    prefix_id: str,
    output_file: Path,
    max_nodes: int = 100,
    figsize: tuple = None,
    dpi: int = 150,
):
    """Plot heatmap of semantic graphs (attribution centers by component).

    Args:
        semantic_graphs: Dict mapping component_id -> attribution center array
        prefix_id: Prefix identifier for title
        output_file: Path to save the plot
        max_nodes: Maximum number of nodes to display
        figsize: Figure size (auto-calculated if None)
        dpi: Output DPI
    """
    component_ids = sorted(semantic_graphs.keys())

    if len(component_ids) == 0:
        return

    # Stack graphs into matrix
    graphs_stacked = np.stack([semantic_graphs[c] for c in component_ids], axis=0)

    # Limit to first max_nodes for visualization
    if graphs_stacked.shape[1] > max_nodes:
        graphs_stacked = graphs_stacked[:, :max_nodes]

    # Auto-calculate figure size based on data
    if figsize is None:
        figsize = (12, min(len(component_ids) * 0.8, 10))

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        graphs_stacked,
        cmap='RdBu_r',
        center=0,
        cbar_kws={'label': 'Attribution Value'},
        yticklabels=[f"C{c}" for c in component_ids],
        ax=ax
    )

    ax.set_xlabel("Node Index")
    ax.set_ylabel("Component")
    ax.set_title(f"{prefix_id}: Semantic Graphs (Attribution Centers)")

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_token_scores(
    token_scores: Dict,
    prefix_id: str,
    output_file: Path,
    token_id_to_text: Optional[Dict] = None,
    top_k: int = 20,
    figsize: tuple = (14, 6),
    dpi: int = 150,
):
    """Plot token score distributions (stacked bar chart of pi_{s,c}).

    Args:
        token_scores: Dict mapping token_id -> {component_id -> score}
        prefix_id: Prefix identifier for title
        output_file: Path to save the plot
        token_id_to_text: Optional dict mapping token_id -> token_text
        top_k: Number of top tokens to show
        figsize: Figure size
        dpi: Output DPI
    """
    if len(token_scores) == 0:
        return

    # Limit to top_k tokens
    token_list = list(token_scores.keys())[:top_k]

    # Collect all valid components
    components = set()
    for scores in token_scores.values():
        components.update(scores.keys())
    # Filter valid components (exclude junk cluster 0 or negative)
    components = sorted([c for c in components if c > 0])

    if len(components) == 0:
        return

    # Build data matrix
    data = []
    for token in token_list:
        scores = token_scores[token]
        data.append([scores.get(c, 0) for c in components])

    data = np.array(data).T  # Shape: (n_components, n_tokens)

    fig, ax = plt.subplots(figsize=figsize)

    bottom = np.zeros(len(token_list))

    # Color palette based on number of components
    n_components = len(components)
    if n_components <= 10:
        cmap = plt.cm.tab10
        colors = [cmap(i) for i in range(n_components)]
    elif n_components <= 20:
        cmap = plt.cm.tab20
        colors = [cmap(i) for i in range(n_components)]
    else:
        # For more than 20, use a continuous colormap
        cmap = plt.cm.turbo
        colors = [cmap(i / (n_components - 1)) for i in range(n_components)]

    # Plot stacked bars in consistent order (sorted component IDs)
    for idx, c in enumerate(components):
        ax.bar(
            range(len(token_list)),
            data[idx],
            bottom=bottom,
            label=f"C{c}",
            color=colors[idx],
            edgecolor='white',
            linewidth=0.5
        )
        bottom += data[idx]

    # Create x-axis labels (use token text if available)
    if token_id_to_text:
        x_labels = []
        for t in token_list:
            text = token_id_to_text.get(t, None)
            if text is not None:
                # Clean up the text for display
                text = text.strip()
                if not text:
                    text = repr(token_id_to_text.get(t, ""))
                # Truncate long tokens
                if len(text) > 15:
                    text = text[:12] + "..."
            else:
                text = str(t)
            x_labels.append(text)
    else:
        x_labels = [str(t) for t in token_list]

    ax.set_xticks(range(len(token_list)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel("Token")
    ax.set_ylabel("Probability Mass Fraction")
    ax.set_title(f"{prefix_id}: Token Semantic Scores (pi_{{s,c}})")
    ax.set_ylim(0, 1.05)

    # Legend with columns for many components
    ncol = max(1, (n_components + 9) // 10)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=ncol, fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close()
