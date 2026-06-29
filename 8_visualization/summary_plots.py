"""Summary visualization: best parameters and aggregated metrics across prefixes.

Extracted from gaussian_optimization/tests/clustering_parameter_tuning/summarize_results.py
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def find_best_params(data: Dict) -> Dict:
    """Find best (beta, gamma) by harmonic score.

    Args:
        data: Sweep results dict with 'grid' key

    Returns:
        Best grid point dict
    """
    return max(data['grid'], key=lambda x: x.get('harmonic', -1))


def create_summary(results: List[Dict]) -> List[Dict]:
    """Create summary of best parameters for each prefix.

    Args:
        results: List of {'prefix': str, 'data': sweep_results_dict}

    Returns:
        List of summary dicts per prefix
    """
    summary = []
    for r in results:
        best = find_best_params(r['data'])
        num_continuations = r['data'].get('num_continuations', None)
        summary.append({
            'prefix': r['prefix'],
            'num_continuations': num_continuations,
            'best_beta': best.get('beta'),
            'best_gamma': best.get('gamma'),
            'best_harmonic': best.get('harmonic', -1),
            'best_K': best.get('K', 0),
            'best_H': best.get('H', 0),
            'best_D_e': best.get('D_e', 0),
            'best_D_a': best.get('D_a', 0),
            'best_sil_e': best.get('sil_e', 0),
            'best_sil_a': best.get('sil_a', 0),
        })
    return summary


def plot_best_params(
    summary: List[Dict],
    output_path: Path,
    figsize: tuple = (15, 10),
    dpi: int = 150,
):
    """Plot distribution of best parameters.

    Args:
        summary: List of summary dicts per prefix
        output_path: Output file path
        figsize: Figure size
        dpi: Output DPI
    """
    if len(summary) < 1:
        return

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    prefixes = [s['prefix'] for s in summary]
    x = np.arange(len(prefixes))

    # Plot 1: Best beta
    ax1 = axes[0, 0]
    betas = [s['best_beta'] for s in summary]
    ax1.bar(x, betas, color='steelblue')
    ax1.set_ylabel('Best beta')
    ax1.set_title('Best beta by Prefix')
    ax1.set_xticks(x)
    ax1.set_xticklabels(prefixes, rotation=45, ha='right', fontsize=8)

    # Plot 2: Best gamma
    ax2 = axes[0, 1]
    gammas = [s['best_gamma'] for s in summary]
    ax2.bar(x, gammas, color='coral')
    ax2.set_ylabel('Best gamma')
    ax2.set_title('Best gamma by Prefix')
    ax2.set_xticks(x)
    ax2.set_xticklabels(prefixes, rotation=45, ha='right', fontsize=8)
    ax2.set_ylim(0, 1)

    # Plot 3: Best harmonic score
    ax3 = axes[0, 2]
    harmonics = [s['best_harmonic'] for s in summary]
    ax3.bar(x, harmonics, color='seagreen')
    ax3.set_ylabel('Best Harmonic Score')
    ax3.set_title('Best Harmonic Score by Prefix')
    ax3.set_xticks(x)
    ax3.set_xticklabels(prefixes, rotation=45, ha='right', fontsize=8)
    ax3.set_ylim(-1, 1)

    # Plot 4: Best K
    ax4 = axes[1, 0]
    Ks = [s['best_K'] for s in summary]
    ax4.bar(x, Ks, color='mediumpurple')
    ax4.set_ylabel('Best K')
    ax4.set_title('Number of Clusters at Best Params')
    ax4.set_xticks(x)
    ax4.set_xticklabels(prefixes, rotation=45, ha='right', fontsize=8)

    # Plot 5: Silhouette scores at best params
    ax5 = axes[1, 1]
    sil_e = [s['best_sil_e'] for s in summary]
    sil_a = [s['best_sil_a'] for s in summary]
    width = 0.35
    ax5.bar(x - width / 2, sil_e, width, label='Embedding', color='steelblue')
    ax5.bar(x + width / 2, sil_a, width, label='Attribution', color='coral')
    ax5.set_ylabel('Silhouette Score')
    ax5.set_title('Silhouette Scores at Best Params')
    ax5.set_xticks(x)
    ax5.set_xticklabels(prefixes, rotation=45, ha='right', fontsize=8)
    ax5.legend()
    ax5.set_ylim(-1, 1)

    # Plot 6: Beta-Gamma scatter
    ax6 = axes[1, 2]
    scatter = ax6.scatter(gammas, betas, c=harmonics, cmap='RdYlGn',
                          s=100, vmin=-1, vmax=1, edgecolors='black')
    plt.colorbar(scatter, ax=ax6, label='Harmonic Score')
    ax6.set_xlabel('Best gamma')
    ax6.set_ylabel('Best beta')
    ax6.set_title('Best (beta, gamma) Distribution')
    ax6.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_aggregated_metrics(
    results: List[Dict],
    output_path: Path,
    figsize: tuple = (14, 5),
    dpi: int = 150,
):
    """Plot aggregated metrics across all prefixes.

    Args:
        results: List of {'prefix': str, 'data': sweep_results_dict}
        output_path: Output file path
        figsize: Figure size
        dpi: Output DPI
    """
    if len(results) < 1:
        return

    # Collect all grid points
    all_betas = set()
    all_gammas = set()
    for r in results:
        all_betas.update(r['data'].get('beta_values', []))
        all_gammas.update(r['data'].get('gamma_values', []))

    all_betas = sorted(all_betas)
    all_gammas = sorted(all_gammas)

    if not all_betas or not all_gammas:
        return

    # Aggregate metrics
    n_beta = len(all_betas)
    n_gamma = len(all_gammas)
    n_prefixes = len(results)

    harmonic_matrix = np.zeros((n_beta, n_gamma))
    count_matrix = np.zeros((n_beta, n_gamma))

    beta_to_idx = {b: i for i, b in enumerate(all_betas)}
    gamma_to_idx = {g: j for j, g in enumerate(all_gammas)}

    for r in results:
        for point in r['data'].get('grid', []):
            if point['beta'] in beta_to_idx and point['gamma'] in gamma_to_idx:
                i = beta_to_idx[point['beta']]
                j = gamma_to_idx[point['gamma']]
                harmonic_matrix[i, j] += point.get('harmonic', 0)
                count_matrix[i, j] += 1

    # Average
    with np.errstate(divide='ignore', invalid='ignore'):
        harmonic_matrix = np.where(count_matrix > 0,
                                   harmonic_matrix / count_matrix,
                                   np.nan)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Average harmonic heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(harmonic_matrix, aspect='auto', cmap='RdYlGn',
                     origin='lower', vmin=-1, vmax=1)
    ax1.set_xticks(range(n_gamma))
    ax1.set_xticklabels([f'{g:.1f}' for g in all_gammas])
    ax1.set_yticks(range(n_beta))
    ax1.set_yticklabels([f'{b:.0f}' for b in all_betas])
    ax1.set_xlabel('gamma')
    ax1.set_ylabel('beta')
    ax1.set_title(f'Average Harmonic Score (n={n_prefixes} prefixes)')
    plt.colorbar(im1, ax=ax1)

    # Add annotations
    for i in range(n_beta):
        for j in range(n_gamma):
            if not np.isnan(harmonic_matrix[i, j]):
                ax1.text(j, i, f'{harmonic_matrix[i, j]:.2f}',
                         ha='center', va='center', fontsize=8)

    # Plot 2: Best params histogram
    ax2 = axes[1]
    best_gammas = [find_best_params(r['data'])['gamma'] for r in results]
    ax2.hist(best_gammas, bins=np.arange(0, 1.1, 0.1), edgecolor='black',
             color='steelblue', alpha=0.7)
    ax2.set_xlabel('Best gamma')
    ax2.set_ylabel('Count')
    ax2.set_title('Distribution of Best gamma Across Prefixes')
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def load_sweep_results(results_dir: Path) -> List[Dict]:
    """Load all sweep results from directory.

    Args:
        results_dir: Directory containing *_sweep_results.json files

    Returns:
        List of {'prefix': str, 'data': sweep_results_dict}
    """
    results = []
    for f in sorted(results_dir.glob("*_sweep_results.json")):
        with open(f) as fp:
            data = json.load(fp)
        prefix = f.stem.replace("_sweep_results", "")
        results.append({
            'prefix': prefix,
            'data': data
        })
    return results
