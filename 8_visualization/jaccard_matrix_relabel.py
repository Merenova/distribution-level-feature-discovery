#!/usr/bin/env python3
"""
Create Jaccard similarity heatmap with hierarchical cluster labels C^K_i.

Labels clusters as C^K_i where:
- K = total number of clusters at that beta level
- i = index within that beta level (1-indexed)

Usage:
    python jaccard_matrix_relabel.py --prefix cloze_0275 --betas 0.5 0.75 1.0 --gamma 0.5
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "AmbigQA_Qwen3-8B" / "results"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_DIR / "7_validation" / "cluster_analysis"
DEFAULT_FIGURES_DIR = Path(os.environ.get("PAPER_FIGURES_DIR", REPO_ROOT / "figures"))

# Colormap: white to deep teal
WHITE_TEAL = LinearSegmentedColormap.from_list('white_teal', ['#ffffff', '#99cccc', '#339999', '#004d4d'])


def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets."""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def load_clustering_data(prefix_id, betas, target_gamma, results_dir):
    """Load clustering data for specified beta values."""
    clustering_file = results_dir / "5_clustering" / f"{prefix_id}_sweep_results.json"
    
    with open(clustering_file) as f:
        clustering = json.load(f)
    
    grid = clustering.get("grid", [])
    
    all_clusters = []
    beta_boundaries = []
    cluster_labels = []
    
    for beta in betas:
        entry = None
        for e in grid:
            if e.get("beta") == beta and abs(e.get("gamma", 0) - target_gamma) < 0.05:
                entry = e
                break
        
        if not entry:
            print(f"Warning: No entry found for beta={beta}, gamma={target_gamma}")
            continue
        
        components = entry.get("components", {})
        K = entry.get("K", 0)
        
        start_idx = len(all_clusters)
        cluster_idx = 1  # 1-indexed within this beta
        
        # Sort by cluster weight (descending)
        for cid, comp in sorted(components.items(), key=lambda x: -x[1].get("W_c", 0)):
            mu_e = comp.get("mu_e", [])
            mu_a = comp.get("mu_a", [])
            W_c = comp.get("W_c", 0)
            
            if W_c > 0.01:
                all_clusters.append({
                    "beta": beta,
                    "mu_e": np.array(mu_e) if mu_e else None,
                    "mu_a": np.array(mu_a) if mu_a else None,
                    "W_c": W_c,
                    "cluster_id": cid,
                    "K": K,
                    "idx_in_beta": cluster_idx
                })
                # New label format: C^K_i
                cluster_labels.append(f"$C^{{{K}}}_{{{cluster_idx}}}$")
                cluster_idx += 1
        
        end_idx = len(all_clusters)
        if end_idx > start_idx:
            beta_boundaries.append((start_idx, end_idx, beta, K))
    
    return all_clusters, beta_boundaries, cluster_labels


def create_triangular_heatmap(similarity, cluster_labels, beta_boundaries, betas,
                               colorbar_label, output_file, vmin=0, vmax=1,
                               show_values=True, fontsize_values=8):
    """Create and save a triangular heatmap with numeric values."""
    n_clusters = len(cluster_labels)
    
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Mask upper triangle
    mask = np.triu(np.ones_like(similarity, dtype=bool), k=1)
    similarity_masked = np.ma.masked_array(similarity, mask)
    
    # Plot with white-to-teal colormap
    im = ax.imshow(similarity_masked, cmap=WHITE_TEAL, aspect='equal', vmin=vmin, vmax=vmax)
    
    # Add numeric values in cells
    if show_values:
        for i in range(n_clusters):
            for j in range(i + 1):  # Lower triangle including diagonal
                val = similarity[i, j]
                # Choose text color based on background
                text_color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                       fontsize=fontsize_values, color=text_color, fontweight='bold')
    
    # Add colorbar with larger font
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(colorbar_label, fontsize=20)
    cbar.ax.tick_params(labelsize=14)
    
    # Set labels
    ax.set_xticks(range(n_clusters))
    ax.set_yticks(range(n_clusters))
    
    # Color labels by beta - use Dark2 colorblind-friendly palette
    dark2_colors = [
        '#1b9e77',  # teal
        '#d95f02',  # orange
        '#7570b3',  # purple
        '#e7298a',  # pink
        '#66a61e',  # green
        '#e6ab02',  # yellow/gold
        '#a6761d',  # brown
        '#666666',  # gray
    ]
    beta_to_color = {b: dark2_colors[i % len(dark2_colors)] for i, b in enumerate(betas)}
    
    ax.set_xticklabels(cluster_labels, rotation=90, fontsize=14)
    ax.set_yticklabels(cluster_labels, fontsize=14)
    
    # Build cluster-to-beta mapping
    cluster_betas = []
    for start, end, beta, K in beta_boundaries:
        cluster_betas.extend([beta] * (end - start))
    
    for i, beta in enumerate(cluster_betas):
        color = beta_to_color.get(beta, 'black')
        ax.get_yticklabels()[i].set_color(color)
        ax.get_xticklabels()[i].set_color(color)
    
    # Add beta boundary lines
    for start, end, beta, K in beta_boundaries:
        if start > 0:
            ax.axhline(y=start - 0.5, color='black', linewidth=2)
            ax.axvline(x=start - 0.5, color='black', linewidth=2)
    
    # Add beta annotations on the LEFT side
    for i, (start, end, beta, K) in enumerate(beta_boundaries):
        mid_y = (start + end - 1) / 2
        ax.text(-2.5, mid_y, f'β={beta}\n(K={K})', 
                fontsize=18, fontweight='bold',
                color=beta_to_color.get(beta, 'black'),
                ha='right', va='center',
                clip_on=False)
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='s', color='w', 
                                   markerfacecolor=beta_to_color[b], markersize=21,
                                   label=f'β={b}') for b in betas if b in beta_to_color]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=18,
              title='Beta Values', title_fontsize=20)
    
    # Adjust layout with left margin for beta labels
    plt.subplots_adjust(left=0.18)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Created {output_file}")
    return output_file


def create_jaccard_heatmap(prefix_id, betas, target_gamma, top_n=100,
                           results_dir=DEFAULT_RESULTS_DIR, 
                           output_dir=DEFAULT_OUTPUT_DIR,
                           copy_to_paper=True,
                           show_values=True):
    """Create Jaccard similarity heatmap for mechanistic features."""
    
    all_clusters, beta_boundaries, cluster_labels = load_clustering_data(
        prefix_id, betas, target_gamma, results_dir)
    
    if len(all_clusters) < 2:
        print(f"Not enough clusters for {prefix_id}")
        return None
    
    n_clusters = len(all_clusters)
    print(f"Jaccard: Found {n_clusters} clusters across {len(betas)} beta values")
    print(f"Cluster labels: {cluster_labels}")
    
    # Extract top-N features for each cluster
    for c in all_clusters:
        if c["mu_a"] is not None:
            top_indices = set(np.argsort(np.abs(c["mu_a"]))[-top_n:])
            c["top_features"] = top_indices
        else:
            c["top_features"] = set()
    
    # Compute Jaccard similarity matrix
    similarity = np.zeros((n_clusters, n_clusters))
    
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i == j:
                similarity[i, j] = 1.0
            else:
                similarity[i, j] = jaccard_similarity(
                    all_clusters[i]["top_features"],
                    all_clusters[j]["top_features"]
                )
    
    # Save CSV with new labels
    csv_output = output_dir / f"{prefix_id}_jaccard_matrix.csv"
    df = pd.DataFrame(similarity, index=cluster_labels, columns=cluster_labels)
    # Clean up LaTeX for CSV
    clean_labels = [f"C{c['K']}_{c['idx_in_beta']}" for c in all_clusters]
    df_clean = pd.DataFrame(similarity, index=clean_labels, columns=clean_labels)
    df_clean.to_csv(csv_output)
    print(f"✅ Saved CSV to {csv_output}")
    
    # Create heatmap
    output_file = output_dir / f"{prefix_id}_jaccard_matrix_numeric.png"
    
    create_triangular_heatmap(
        similarity, cluster_labels, beta_boundaries, betas,
        colorbar_label=f'Jaccard Similarity (Top-{top_n} Features)',
        output_file=output_file,
        show_values=show_values
    )
    
    # Copy to paper figures
    if copy_to_paper and DEFAULT_FIGURES_DIR.exists():
        paper_output = DEFAULT_FIGURES_DIR / output_file.name
        shutil.copy(output_file, paper_output)
        print(f"✅ Copied to {paper_output}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Create Jaccard heatmap with C^K_i labels")
    parser.add_argument("--prefix", type=str, default="cloze_0275", help="Cloze prefix ID")
    parser.add_argument("--betas", type=float, nargs="+", default=[0.5, 0.75, 1.0],
                        help="Beta values to include (default: 0.5 0.75 1.0)")
    parser.add_argument("--gamma", type=float, default=0.5, help="Target gamma value")
    parser.add_argument("--top-n", type=int, default=100,
                        help="Number of top features for Jaccard (default: 100)")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--no-paper-copy", action="store_true")
    parser.add_argument("--no-values", action="store_true", help="Don't show numeric values")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating Jaccard heatmap for {args.prefix}")
    print(f"  Betas: {args.betas}")
    print(f"  Gamma: {args.gamma}")
    print()
    
    create_jaccard_heatmap(
        args.prefix, args.betas, args.gamma, args.top_n,
        results_dir, output_dir, 
        copy_to_paper=not args.no_paper_copy,
        show_values=not args.no_values
    )
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
