#!/usr/bin/env python3
"""Visualize token-wise logit differences as heatmaps.

Usage:
    python visualize_logit_heatmaps.py --input heatmap_data.json --output heatmap.png
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def truncate_token(token: str, max_len: int = 12) -> str:
    """Truncate token for display."""
    # Clean up special characters
    token = token.replace('Ġ', ' ').replace('Ċ', '\\n').replace('\u010a', '\\n')
    if len(token) > max_len:
        return token[:max_len-2] + '..'
    return token


def plot_single_branch_heatmap(
    branch_data: Dict,
    epsilon: str,
    ax: plt.Axes,
    metric: str = "centered_logits",
    title: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Plot heatmap for a single branch showing logit differences."""
    tokens = branch_data.get("continuation_tokens", [])
    diff_data = branch_data["epsilon_results"][epsilon].get("diff", {})
    
    if metric not in diff_data:
        ax.text(0.5, 0.5, f"No {metric} data", ha='center', va='center')
        return None
    
    values = diff_data[metric]
    if not values:
        ax.text(0.5, 0.5, "No values", ha='center', va='center')
        return None
    
    # Create 1D heatmap as 2D array (1 row)
    data = np.array(values).reshape(1, -1)
    
    # Clean up tokens for display
    token_labels = [truncate_token(t) for t in tokens[:len(values)]]
    
    # Determine color scale
    if vmin is None:
        abs_max = max(abs(data.min()), abs(data.max()), 0.1)
        vmin, vmax = -abs_max, abs_max
    
    # Create heatmap
    im = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)
    
    # Set tick labels
    ax.set_xticks(np.arange(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks([])
    
    # Add value annotations
    for i, val in enumerate(values):
        color = 'white' if abs(val) > abs_max * 0.6 else 'black'
        ax.text(i, 0, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color)
    
    if title:
        ax.set_title(title, fontsize=10)
    
    return im


def plot_multi_branch_heatmap(
    branches: List[Dict],
    epsilon: str,
    metric: str = "centered_logits",
    title: str = "",
    max_branches: int = 10,
    figsize: tuple = (16, 8),
) -> plt.Figure:
    """Plot heatmap comparing multiple branches."""
    
    # Collect all data
    all_data = []
    branch_labels = []
    max_len = 0
    
    for i, branch in enumerate(branches[:max_branches]):
        diff_data = branch["epsilon_results"].get(epsilon, {}).get("diff", {})
        if metric not in diff_data:
            continue
        values = diff_data[metric]
        if not values:
            continue
        all_data.append(values)
        max_len = max(max_len, len(values))
        
        # Create label from continuation text
        cont_tokens = branch.get("continuation_tokens", [])
        label = ' '.join(cont_tokens[:5]) if cont_tokens else f"Branch {branch['branch_id']}"
        label = truncate_token(label, 30)
        branch_labels.append(label)
    
    if not all_data:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return fig
    
    # Pad to same length
    padded_data = []
    for values in all_data:
        padded = values + [np.nan] * (max_len - len(values))
        padded_data.append(padded)
    
    data_matrix = np.array(padded_data)
    
    # Get token labels from first branch
    first_tokens = branches[0].get("continuation_tokens", [])
    token_labels = [truncate_token(t) for t in first_tokens[:max_len]]
    while len(token_labels) < max_len:
        token_labels.append("")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Determine color scale
    valid_data = data_matrix[~np.isnan(data_matrix)]
    if len(valid_data) > 0:
        abs_max = max(abs(valid_data.min()), abs(valid_data.max()), 0.1)
    else:
        abs_max = 1.0
    
    # Plot heatmap
    im = ax.imshow(data_matrix, cmap='RdBu_r', aspect='auto', vmin=-abs_max, vmax=abs_max)
    
    # Set tick labels
    ax.set_xticks(np.arange(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(len(branch_labels)))
    ax.set_yticklabels(branch_labels, fontsize=8)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f'Δ {metric}', fontsize=10)
    
    ax.set_xlabel('Tokens', fontsize=11)
    ax.set_ylabel('Branches', fontsize=11)
    
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_cluster_comparison(
    data: Dict,
    epsilon: str,
    metric: str = "centered_logits",
    figsize: tuple = (18, 12),
) -> plt.Figure:
    """Plot heatmap comparison across all clusters."""
    
    clusters = data.get("clusters", {})
    n_clusters = len(clusters)
    
    if n_clusters == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No clusters", ha='center', va='center')
        return fig
    
    # Create subplots
    fig, axes = plt.subplots(n_clusters, 1, figsize=figsize)
    if n_clusters == 1:
        axes = [axes]
    
    question = data.get("question", data.get("prefix", "")[:50])
    fig.suptitle(f'Token-wise Δ{metric} (ε={epsilon})\n{question}', fontsize=14, fontweight='bold')
    
    for idx, (cluster_id, cluster_data) in enumerate(sorted(clusters.items())):
        ax = axes[idx]
        branches = cluster_data.get("branches", [])
        
        if not branches:
            ax.text(0.5, 0.5, f"Cluster {cluster_id}: No data", ha='center', va='center')
            continue
        
        # Collect data for this cluster
        all_data = []
        branch_labels = []
        max_len = 0
        
        for branch in branches[:8]:  # Limit to 8 branches per cluster
            diff_data = branch["epsilon_results"].get(epsilon, {}).get("diff", {})
            if metric not in diff_data:
                continue
            values = diff_data[metric]
            if not values:
                continue
            all_data.append(values)
            max_len = max(max_len, len(values))
            branch_labels.append(f"B{branch['branch_id']}")
        
        if not all_data:
            ax.text(0.5, 0.5, f"Cluster {cluster_id}: No {metric}", ha='center', va='center')
            continue
        
        # Pad data
        padded_data = [v + [np.nan] * (max_len - len(v)) for v in all_data]
        data_matrix = np.array(padded_data)
        
        # Get token labels
        first_tokens = branches[0].get("continuation_tokens", [])
        token_labels = [truncate_token(t, 8) for t in first_tokens[:max_len]]
        
        # Determine color scale
        valid_data = data_matrix[~np.isnan(data_matrix)]
        abs_max = max(abs(valid_data.min()), abs(valid_data.max()), 0.1) if len(valid_data) > 0 else 1.0
        
        # Plot
        im = ax.imshow(data_matrix, cmap='RdBu_r', aspect='auto', vmin=-abs_max, vmax=abs_max)
        
        ax.set_xticks(np.arange(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(np.arange(len(branch_labels)))
        ax.set_yticklabels(branch_labels, fontsize=8)
        ax.set_ylabel(f'Cluster {cluster_id}', fontsize=10, fontweight='bold')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=7)
    
    plt.tight_layout()
    return fig


def plot_epsilon_comparison(
    data: Dict,
    cluster_id: str,
    branch_idx: int = 0,
    metric: str = "centered_logits",
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Compare epsilon=-1 vs epsilon=+1 for a single branch."""
    
    clusters = data.get("clusters", {})
    if cluster_id not in clusters:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Cluster {cluster_id} not found", ha='center', va='center')
        return fig
    
    branches = clusters[cluster_id].get("branches", [])
    if branch_idx >= len(branches):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"Branch {branch_idx} not found", ha='center', va='center')
        return fig
    
    branch = branches[branch_idx]
    tokens = branch.get("continuation_tokens", [])
    
    # Get data for both epsilons
    eps_neg = branch["epsilon_results"].get("-1.0", {}).get("diff", {}).get(metric, [])
    eps_pos = branch["epsilon_results"].get("1.0", {}).get("diff", {}).get(metric, [])
    
    if not eps_neg or not eps_pos:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Missing epsilon data", ha='center', va='center')
        return fig
    
    max_len = max(len(eps_neg), len(eps_pos))
    token_labels = [truncate_token(t, 10) for t in tokens[:max_len]]
    
    # Create comparison matrix
    data_matrix = np.array([
        eps_neg + [np.nan] * (max_len - len(eps_neg)),
        eps_pos + [np.nan] * (max_len - len(eps_pos)),
    ])
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    valid_data = data_matrix[~np.isnan(data_matrix)]
    abs_max = max(abs(valid_data.min()), abs(valid_data.max()), 0.1) if len(valid_data) > 0 else 1.0
    
    im = ax.imshow(data_matrix, cmap='RdBu_r', aspect='auto', vmin=-abs_max, vmax=abs_max)
    
    ax.set_xticks(np.arange(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['ε = -1', 'ε = +1'], fontsize=11)
    
    # Add value annotations
    for i in range(2):
        for j in range(max_len):
            val = data_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > abs_max * 0.5 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=7, color=color)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f'Δ {metric}', fontsize=11)
    
    question = data.get("question", "")
    cont_text = ' '.join(tokens[:8]) if tokens else ""
    ax.set_title(f'Cluster {cluster_id}, Branch {branch["branch_id"]}\n{cont_text}...', fontsize=11)
    ax.set_xlabel('Tokens', fontsize=11)
    
    plt.tight_layout()
    return fig


def plot_average_effect(
    data: Dict,
    epsilon: str,
    metric: str = "centered_logits",
    figsize: tuple = (16, 5),
) -> plt.Figure:
    """Plot average effect across all branches in each cluster."""
    
    clusters = data.get("clusters", {})
    
    fig, ax = plt.subplots(figsize=figsize)
    
    cluster_means = {}
    max_len = 0
    
    for cluster_id, cluster_data in sorted(clusters.items()):
        branches = cluster_data.get("branches", [])
        all_values = []
        
        for branch in branches:
            diff_data = branch["epsilon_results"].get(epsilon, {}).get("diff", {})
            if metric in diff_data:
                all_values.append(diff_data[metric])
        
        if all_values:
            # Pad to same length
            max_branch_len = max(len(v) for v in all_values)
            max_len = max(max_len, max_branch_len)
            padded = [v + [np.nan] * (max_branch_len - len(v)) for v in all_values]
            mean_vals = np.nanmean(np.array(padded), axis=0)
            std_vals = np.nanstd(np.array(padded), axis=0)
            cluster_means[cluster_id] = (mean_vals, std_vals)
    
    if not cluster_means:
        ax.text(0.5, 0.5, "No data", ha='center', va='center')
        return fig
    
    # Get token labels from first cluster's first branch
    first_cluster = list(clusters.values())[0]
    first_branch = first_cluster["branches"][0] if first_cluster["branches"] else {}
    first_tokens = first_branch.get("continuation_tokens", [])
    token_labels = [truncate_token(t, 10) for t in first_tokens[:max_len]]
    
    # Plot each cluster as a line
    x = np.arange(max_len)
    colors = plt.cm.Set2(np.linspace(0, 1, len(cluster_means)))
    
    for (cluster_id, (mean_vals, std_vals)), color in zip(sorted(cluster_means.items()), colors):
        x_vals = np.arange(len(mean_vals))
        ax.plot(x_vals, mean_vals, 'o-', label=f'Cluster {cluster_id}', color=color, linewidth=2, markersize=5)
        ax.fill_between(x_vals, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2, color=color)
    
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x[:len(token_labels)])
    ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Tokens', fontsize=11)
    ax.set_ylabel(f'Mean Δ {metric}', fontsize=11)
    ax.legend(loc='best', fontsize=9)
    
    question = data.get("question", "")
    ax.set_title(f'Average Steering Effect (ε={epsilon})\n{question}', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize token-wise logit heatmaps")
    parser.add_argument("--input", type=Path, required=True, help="Input JSON file")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for plots")
    parser.add_argument("--metric", type=str, default="centered_logits", 
                        choices=["target_logits", "centered_logits", "target_log_probs"],
                        help="Metric to visualize")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()
    
    # Load data
    with open(args.input) as f:
        data = json.load(f)
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = args.input.parent / "heatmap_plots"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    cloze_id = data.get("cloze_id", args.input.stem)
    config = data.get("config", "")
    
    print(f"Generating heatmaps for {cloze_id} ({config})")
    print(f"Question: {data.get('question', 'N/A')}")
    print(f"Clusters: {list(data.get('clusters', {}).keys())}")
    
    # Generate plots for both epsilons
    for epsilon in ["-1.0", "1.0"]:
        eps_label = "neg1" if epsilon == "-1.0" else "pos1"
        
        # 1. Cluster comparison heatmap
        fig = plot_cluster_comparison(data, epsilon, args.metric)
        fig.savefig(args.output_dir / f"{cloze_id}_{config}_clusters_eps{eps_label}.{args.format}", 
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: clusters_eps{eps_label}.{args.format}")
        
        # 2. Average effect plot
        fig = plot_average_effect(data, epsilon, args.metric)
        fig.savefig(args.output_dir / f"{cloze_id}_{config}_average_eps{eps_label}.{args.format}",
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: average_eps{eps_label}.{args.format}")
    
    # 3. Epsilon comparison for first branch of each cluster
    for cluster_id in data.get("clusters", {}).keys():
        fig = plot_epsilon_comparison(data, cluster_id, branch_idx=0, metric=args.metric)
        fig.savefig(args.output_dir / f"{cloze_id}_{config}_cluster{cluster_id}_eps_compare.{args.format}",
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: cluster{cluster_id}_eps_compare.{args.format}")
    
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

