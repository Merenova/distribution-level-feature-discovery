#!/usr/bin/env python
"""Cross-Silhouette Histogram: Compare RD Clustering with K-Means Baselines.

Creates histogram plots comparing silhouette scores in both spaces for:
- RD Clustering (at fixed beta, gamma)
- K-Means on Embeddings (KM-E)
- K-Means on Attributions (KM-A)

Usage:
    python cross_silhouette_histogram.py \
        --results-file AmbigQA_Qwen3-8B/results/cross_silhouette/cross_silhouette_results.json \
        --beta 1.0 --gamma 0.7 \
        --output AmbigQA_Qwen3-8B/results/plots/cross_sil_histogram_b1_g0p7.png
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data_utils import load_json


def extract_rd_at_config(
    per_prefix: List[Dict],
    target_beta: float,
    target_gamma: float,
) -> Tuple[List[float], List[float], List[int]]:
    """Extract RD clustering results at a specific (beta, gamma) config.
    
    Returns:
        Tuple of (sil_e_list, sil_a_list, K_list)
    """
    sil_e_list = []
    sil_a_list = []
    K_list = []
    
    for prefix_result in per_prefix:
        for entry in prefix_result['rd_sweep']:
            if (abs(entry['beta'] - target_beta) < 0.01 and 
                abs(entry['gamma'] - target_gamma) < 0.01):
                K = entry['K']
                if K >= 2:  # Valid clustering
                    sil_e_list.append(entry['sil_e'])
                    sil_a_list.append(entry['sil_a'])
                    K_list.append(K)
                break
    
    return sil_e_list, sil_a_list, K_list


def extract_kmeans_matching_K(
    per_prefix: List[Dict],
    rd_K_list: List[int],
    method: str = 'kmeans_e',
) -> Tuple[List[float], List[float], List[int]]:
    """Extract K-Means results matching the K values from RD.
    
    Args:
        per_prefix: Per-prefix results
        rd_K_list: List of K values from RD (one per prefix)
        method: 'kmeans_e' or 'kmeans_a'
        
    Returns:
        Tuple of (sil_e_list, sil_a_list, K_list)
    """
    sweep_key = f'{method}_sweep'
    sil_e_list = []
    sil_a_list = []
    K_list = []
    
    for i, prefix_result in enumerate(per_prefix):
        if i >= len(rd_K_list):
            break
            
        target_K = rd_K_list[i]
        sweep = prefix_result.get(sweep_key, [])
        
        # Find matching K or closest K
        found = False
        for entry in sweep:
            if entry['K'] == target_K:
                sil_e_list.append(entry['sil_e'])
                sil_a_list.append(entry['sil_a'])
                K_list.append(entry['K'])
                found = True
                break
        
        if not found and sweep:
            # Use closest K
            closest = min(sweep, key=lambda x: abs(x['K'] - target_K))
            sil_e_list.append(closest['sil_e'])
            sil_a_list.append(closest['sil_a'])
            K_list.append(closest['K'])
    
    return sil_e_list, sil_a_list, K_list


def extract_kmeans_best(
    per_prefix: List[Dict],
    method: str = 'kmeans_e',
    criterion: str = 'native',
    k_range: Optional[Tuple[int, int]] = None,
) -> Tuple[List[float], List[float], List[int]]:
    """Extract K-Means best results by criterion.
    
    Args:
        per_prefix: Per-prefix results
        method: 'kmeans_e' or 'kmeans_a'
        criterion: 'native' (best by own metric) or 'harmonic'
        k_range: Optional (min_k, max_k) to filter K values
        
    Returns:
        Tuple of (sil_e_list, sil_a_list, K_list)
    """
    sweep_key = f'{method}_sweep'
    sil_e_list = []
    sil_a_list = []
    K_list = []
    
    for prefix_result in per_prefix:
        sweep = prefix_result.get(sweep_key, [])
        if not sweep:
            continue
        
        # Filter valid entries
        valid = [e for e in sweep if e['K'] >= 2]
        
        # Apply K range filter if specified
        if k_range is not None:
            min_k, max_k = k_range
            valid = [e for e in valid if min_k <= e['K'] <= max_k]
        
        if not valid:
            continue
        
        if criterion == 'native':
            if method == 'kmeans_e':
                best = max(valid, key=lambda x: x['sil_e'])
            else:
                best = max(valid, key=lambda x: x['sil_a'])
        else:
            # Harmonic
            valid_h = [e for e in valid if e['harmonic'] > 0]
            if valid_h:
                best = max(valid_h, key=lambda x: x['harmonic'])
            else:
                best = max(valid, key=lambda x: (x['sil_e'] + x['sil_a']) / 2)
        
        sil_e_list.append(best['sil_e'])
        sil_a_list.append(best['sil_a'])
        K_list.append(best['K'])
    
    return sil_e_list, sil_a_list, K_list


def extract_kmeans_at_K(
    per_prefix: List[Dict],
    method: str = 'kmeans_e',
    target_K: int = 5,
) -> Tuple[List[float], List[float], List[int]]:
    """Extract K-Means results at a specific K value.
    
    Args:
        per_prefix: Per-prefix results
        method: 'kmeans_e' or 'kmeans_a'
        target_K: Specific K value to extract
        
    Returns:
        Tuple of (sil_e_list, sil_a_list, K_list)
    """
    sweep_key = f'{method}_sweep'
    sil_e_list = []
    sil_a_list = []
    K_list = []
    
    for prefix_result in per_prefix:
        sweep = prefix_result.get(sweep_key, [])
        if not sweep:
            continue
        
        # Find entry with target K
        for entry in sweep:
            if entry['K'] == target_K:
                sil_e_list.append(entry['sil_e'])
                sil_a_list.append(entry['sil_a'])
                K_list.append(entry['K'])
                break
    
    return sil_e_list, sil_a_list, K_list


def plot_cross_silhouette_histograms(
    rd_data: Tuple[List[float], List[float], List[int]],
    kme_data: Tuple[List[float], List[float], List[int]],
    kma_data: Tuple[List[float], List[float], List[int]],
    beta: float,
    gamma: float,
    output_path: Path,
    figsize: Tuple[int, int] = (15, 5),
    show_rd: bool = True,
    title_suffix: str = "",
):
    """Create histogram plots comparing cross-silhouette scores."""
    
    rd_sil_e, rd_sil_a, rd_K = rd_data
    kme_sil_e, kme_sil_a, kme_K = kme_data
    kma_sil_e, kma_sil_a, kma_K = kma_data
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Colors
    colors = {
        'RD': '#6495ED',    # Cornflower blue
        'KM-E': '#90EE90',  # Light green
        'KM-A': '#FFB6C1',  # Light pink
    }
    
    # Legend labels
    labels = {
        'RD': 'RD (Ours)',
        'KM-E': 'K-Means semantic-only',
        'KM-A': 'K-Means attribution-only',
    }
    
    # Panel 1: Semantic Space
    ax1 = axes[0]
    bins_e = np.linspace(-0.2, 0.5, 25)
    
    if show_rd and rd_sil_e:
        ax1.hist(rd_sil_e, bins=bins_e, alpha=0.5, label=labels['RD'], color=colors['RD'], edgecolor='white')
    ax1.hist(kme_sil_e, bins=bins_e, alpha=0.5, label=labels['KM-E'], color=colors['KM-E'], edgecolor='white')
    ax1.hist(kma_sil_e, bins=bins_e, alpha=0.5, label=labels['KM-A'], color=colors['KM-A'], edgecolor='white')
    
    # KDE curves
    x_e = np.linspace(-0.3, 0.6, 200)
    data_color_pairs = [(kme_sil_e, 'green'), (kma_sil_e, 'red')]
    if show_rd and rd_sil_e:
        data_color_pairs.insert(0, (rd_sil_e, 'blue'))
    
    for data, color in data_color_pairs:
        if len(data) > 3:
            try:
                kde = stats.gaussian_kde(data)
                ax1.plot(x_e, kde(x_e) * len(data) * (bins_e[1] - bins_e[0]), 
                        color=color, linewidth=2)
            except:
                pass
    
    ax1.set_xlabel('Silhouette Score', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Semantic Space', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Panel 2: Attribution Space
    ax2 = axes[1]
    bins_a = np.linspace(-0.3, 1.0, 25)
    
    if show_rd and rd_sil_a:
        ax2.hist(rd_sil_a, bins=bins_a, alpha=0.5, label=labels['RD'], color=colors['RD'], edgecolor='white')
    ax2.hist(kme_sil_a, bins=bins_a, alpha=0.5, label=labels['KM-E'], color=colors['KM-E'], edgecolor='white')
    ax2.hist(kma_sil_a, bins=bins_a, alpha=0.5, label=labels['KM-A'], color=colors['KM-A'], edgecolor='white')
    
    # KDE curves
    x_a = np.linspace(-0.4, 1.0, 200)
    data_color_pairs = [(kme_sil_a, 'green'), (kma_sil_a, 'red')]
    if show_rd and rd_sil_a:
        data_color_pairs.insert(0, (rd_sil_a, 'blue'))
    
    for data, color in data_color_pairs:
        if len(data) > 3:
            try:
                kde = stats.gaussian_kde(data)
                ax2.plot(x_a, kde(x_a) * len(data) * (bins_a[1] - bins_a[0]), 
                        color=color, linewidth=2)
            except:
                pass
    
    ax2.set_xlabel('Silhouette Score', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Attribution Space', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    # Overall title (disabled)
    # if show_rd:
    #     title = f'Cross-Silhouette Analysis (β={beta}, γ={gamma})'
    # else:
    #     title = f'Cross-Silhouette Analysis (K-Means only)'
    # if title_suffix:
    #     title += f' {title_suffix}'
    # fig.suptitle(title, fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved plot to {output_path}")
    
    # Print statistics
    if show_rd:
        print(f"\nStatistics (β={beta}, γ={gamma}):")
    else:
        print(f"\nStatistics (K-Means only):")
    print(f"{'Method':<10} {'N':<6} {'Sil(E) mean±std':<20} {'Sil(A) mean±std':<20}")
    print("-" * 60)
    
    results_to_show = [('KM-E', kme_data), ('KM-A', kma_data)]
    if show_rd:
        results_to_show.insert(0, ('RD', rd_data))
    
    for name, (sil_e, sil_a, K) in results_to_show:
        n = len(sil_e)
        if n > 0:
            e_mean, e_std = np.mean(sil_e), np.std(sil_e)
            a_mean, a_std = np.mean(sil_a), np.std(sil_a)
            print(f"{name:<10} {n:<6} {e_mean:.4f}±{e_std:.4f}        {a_mean:.4f}±{a_std:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Cross-Silhouette Histogram")
    parser.add_argument("--results-file", type=Path, required=True,
                        help="Path to cross_silhouette_results.json")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Beta value for RD clustering")
    parser.add_argument("--gamma", type=float, default=0.7,
                        help="Gamma value for RD clustering")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for plot")
    parser.add_argument("--match-K", action="store_true",
                        help="Match K-Means K to RD K (default: use best K)")
    parser.add_argument("--kmeans-only", action="store_true",
                        help="Show only K-Means baselines (no RD)")
    parser.add_argument("--k-range", type=int, nargs=2, default=None, metavar=('MIN', 'MAX'),
                        help="K range for K-Means (e.g., --k-range 2 5)")
    parser.add_argument("--fixed-K", type=int, default=None,
                        help="Use a specific K value for K-Means")
    
    args = parser.parse_args()
    
    # Load results
    data = load_json(args.results_file)
    per_prefix = data['per_prefix']
    
    print(f"Loaded {len(per_prefix)} prefixes")
    
    # Parse K range
    k_range = tuple(args.k_range) if args.k_range else None
    title_suffix = ""
    
    # Extract RD data at target config (if not kmeans-only)
    if args.kmeans_only:
        rd_data = ([], [], [])
        print("K-Means only mode (no RD)")
    else:
        rd_data = extract_rd_at_config(per_prefix, args.beta, args.gamma)
        print(f"RD (β={args.beta}, γ={args.gamma}): {len(rd_data[0])} valid prefixes")
    
    # Extract K-Means data
    if args.fixed_K:
        # Use specific K
        kme_data = extract_kmeans_at_K(per_prefix, 'kmeans_e', args.fixed_K)
        kma_data = extract_kmeans_at_K(per_prefix, 'kmeans_a', args.fixed_K)
        print(f"KM-E (K={args.fixed_K}): {len(kme_data[0])} prefixes")
        print(f"KM-A (K={args.fixed_K}): {len(kma_data[0])} prefixes")
        title_suffix = f"[K={args.fixed_K}]"
    elif args.match_K and not args.kmeans_only:
        # K-Means with matching K
        kme_data = extract_kmeans_matching_K(per_prefix, rd_data[2], 'kmeans_e')
        kma_data = extract_kmeans_matching_K(per_prefix, rd_data[2], 'kmeans_a')
        print(f"KM-E (matching K): {len(kme_data[0])} prefixes")
        print(f"KM-A (matching K): {len(kma_data[0])} prefixes")
    else:
        # K-Means best by native criterion
        kme_data = extract_kmeans_best(per_prefix, 'kmeans_e', 'native', k_range)
        kma_data = extract_kmeans_best(per_prefix, 'kmeans_a', 'native', k_range)
        if k_range:
            print(f"KM-E (best sil_e, K∈[{k_range[0]},{k_range[1]}]): {len(kme_data[0])} prefixes")
            print(f"KM-A (best sil_a, K∈[{k_range[0]},{k_range[1]}]): {len(kma_data[0])} prefixes")
            title_suffix = f"[K∈{k_range[0]}-{k_range[1]}]"
        else:
            print(f"KM-E (best sil_e): {len(kme_data[0])} prefixes")
            print(f"KM-A (best sil_a): {len(kma_data[0])} prefixes")
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # Plot
    plot_cross_silhouette_histograms(
        rd_data, kme_data, kma_data,
        args.beta, args.gamma,
        args.output,
        show_rd=not args.kmeans_only,
        title_suffix=title_suffix,
    )


if __name__ == "__main__":
    main()
