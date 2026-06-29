#!/usr/bin/env python3
"""
Create triangular similarity heatmaps for cluster analysis across beta values.

Generates two types of heatmaps:
1. Jaccard similarity of top-N mechanistic features (mu_a)
2. Cosine similarity of semantic centroids (mu_e)

Usage:
    python beta_similarity_heatmaps.py --prefix cloze_0275 --betas 0.5 0.75 1.0 --gamma 0.5
    python beta_similarity_heatmaps.py --prefix cloze_0275 --type semantic
    python beta_similarity_heatmaps.py --prefix cloze_0275 --type both
"""

import argparse
import json
import os
import numpy as np
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


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)


def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets."""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def load_clustering_data(prefix_id, betas, target_gamma, results_dir):
    """Load clustering data for specified beta values.
    
    Labels clusters as C^K_i where:
    - K = total number of clusters at that beta level
    - i = index within that beta level (1-indexed)
    """
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
                # New label format: C^K_i (LaTeX)
                cluster_labels.append(f"$C^{{{K}}}_{{{cluster_idx}}}$")
                cluster_idx += 1
        
        end_idx = len(all_clusters)
        if end_idx > start_idx:
            beta_boundaries.append((start_idx, end_idx, beta, K))
    
    return all_clusters, beta_boundaries, cluster_labels


def create_triangular_heatmap(similarity, cluster_labels, beta_boundaries, betas,
                               title, colorbar_label, output_file, vmin=0, vmax=1):
    """Create and save a triangular heatmap."""
    n_clusters = len(cluster_labels)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Mask upper triangle
    mask = np.triu(np.ones_like(similarity, dtype=bool), k=1)
    similarity_masked = np.ma.masked_array(similarity, mask)
    
    # Plot with white-to-teal colormap
    im = ax.imshow(similarity_masked, cmap=WHITE_TEAL, aspect='equal', vmin=vmin, vmax=vmax)
    
    # Add colorbar with larger font
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(colorbar_label, fontsize=20)
    cbar.ax.tick_params(labelsize=14)
    
    # Set labels
    ax.set_xticks(range(n_clusters))
    ax.set_yticks(range(n_clusters))
    
    # Color labels by beta
    beta_colors = plt.cm.Dark2(np.linspace(0, 1, len(betas)))
    beta_to_color = {b: beta_colors[i] for i, b in enumerate(betas)}
    
    ax.set_xticklabels(cluster_labels, rotation=90, fontsize=16)
    ax.set_yticklabels(cluster_labels, fontsize=16)
    
    # Color the tick labels by beta
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
                           copy_to_paper=True):
    """Create Jaccard similarity heatmap for mechanistic features."""
    
    all_clusters, beta_boundaries, cluster_labels = load_clustering_data(
        prefix_id, betas, target_gamma, results_dir)
    
    if len(all_clusters) < 2:
        print(f"Not enough clusters for {prefix_id}")
        return None
    
    n_clusters = len(all_clusters)
    print(f"Jaccard: Found {n_clusters} clusters across {len(betas)} beta values")
    
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
    
    # Create heatmap
    beta_str = "_".join([str(b) for b in betas[:3]])
    output_file = output_dir / f"{prefix_id}_cluster_jaccard_beta{len(betas)}.png"
    
    create_triangular_heatmap(
        similarity, cluster_labels, beta_boundaries, betas,
        title=f"Jaccard Similarity (Top-{top_n} Mechanistic Features)",
        colorbar_label=f'Jaccard Similarity (Top-{top_n} Features)',
        output_file=output_file
    )
    
    # Copy to paper figures
    if copy_to_paper and DEFAULT_FIGURES_DIR.exists():
        paper_output = DEFAULT_FIGURES_DIR / output_file.name
        shutil.copy(output_file, paper_output)
        print(f"✅ Copied to {paper_output}")
    
    return output_file


def create_semantic_cosine_heatmap(prefix_id, betas, target_gamma,
                                    results_dir=DEFAULT_RESULTS_DIR,
                                    output_dir=DEFAULT_OUTPUT_DIR,
                                    copy_to_paper=True,
                                    auto_vmin=True):
    """Create cosine similarity heatmap for semantic centroids.
    
    Args:
        auto_vmin: If True, set vmin to the minimum similarity value (excluding diagonal).
                   If False, use vmin=0.
    """
    
    all_clusters, beta_boundaries, cluster_labels = load_clustering_data(
        prefix_id, betas, target_gamma, results_dir)
    
    if len(all_clusters) < 2:
        print(f"Not enough clusters for {prefix_id}")
        return None
    
    n_clusters = len(all_clusters)
    print(f"Semantic: Found {n_clusters} clusters across {len(betas)} beta values")
    
    # Compute cosine similarity matrix
    similarity = np.zeros((n_clusters, n_clusters))
    
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i == j:
                similarity[i, j] = 1.0
            elif all_clusters[i]["mu_e"] is not None and all_clusters[j]["mu_e"] is not None:
                similarity[i, j] = cosine_similarity(
                    all_clusters[i]["mu_e"],
                    all_clusters[j]["mu_e"]
                )
            else:
                similarity[i, j] = 0.0
    
    # Compute vmin based on actual data
    if auto_vmin:
        mask_diag = ~np.eye(n_clusters, dtype=bool)
        vmin = similarity[mask_diag].min()
        print(f"  Cosine similarity range: [{vmin:.3f}, 1.000]")
    else:
        vmin = 0.0
    
    # Create heatmap
    output_file = output_dir / f"{prefix_id}_cluster_semantic_cosine_beta{len(betas)}.png"
    
    create_triangular_heatmap(
        similarity, cluster_labels, beta_boundaries, betas,
        title="Cosine Similarity (Semantic Centroids)",
        colorbar_label='Cosine Similarity (Semantic Centroids)',
        output_file=output_file,
        vmin=vmin
    )
    
    # Copy to paper figures
    if copy_to_paper and DEFAULT_FIGURES_DIR.exists():
        paper_output = DEFAULT_FIGURES_DIR / output_file.name
        shutil.copy(output_file, paper_output)
        print(f"✅ Copied to {paper_output}")
    
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Create similarity heatmaps across beta values")
    parser.add_argument("--prefix", type=str, required=True, help="Cloze prefix ID (e.g., cloze_0275)")
    parser.add_argument("--betas", type=float, nargs="+", default=[0.5, 0.75, 1.0],
                        help="Beta values to include (default: 0.5 0.75 1.0)")
    parser.add_argument("--gamma", type=float, default=0.5, help="Target gamma value (default: 0.5)")
    parser.add_argument("--type", type=str, choices=["jaccard", "semantic", "both"], default="both",
                        help="Type of heatmap to create (default: both)")
    parser.add_argument("--top-n", type=int, default=100,
                        help="Number of top features for Jaccard (default: 100)")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR),
                        help="Results directory path")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory path")
    parser.add_argument("--no-paper-copy", action="store_true",
                        help="Don't copy to the paper figures directory")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    copy_to_paper = not args.no_paper_copy
    
    print(f"Creating heatmaps for {args.prefix}")
    print(f"  Betas: {args.betas}")
    print(f"  Gamma: {args.gamma}")
    print(f"  Type: {args.type}")
    print()
    
    if args.type in ["jaccard", "both"]:
        create_jaccard_heatmap(
            args.prefix, args.betas, args.gamma, args.top_n,
            results_dir, output_dir, copy_to_paper
        )
    
    if args.type in ["semantic", "both"]:
        create_semantic_cosine_heatmap(
            args.prefix, args.betas, args.gamma,
            results_dir, output_dir, copy_to_paper
        )
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
