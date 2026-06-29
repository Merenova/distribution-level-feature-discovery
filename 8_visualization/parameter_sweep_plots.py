"""Parameter sweep visualization: harmonic scores, R-D curves, heatmaps.

Extracted from gaussian_optimization/tests/clustering_parameter_tuning/sweep_beta_gamma.py
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def plot_harmonic_scores(
    results: Dict,
    output_path: Path,
    figsize: tuple = (15, 5),
    dpi: int = 150,
):
    """Plot harmonic scores vs gamma for each beta.

    Args:
        results: Sweep results dict with 'beta_values', 'gamma_values', 'grid'
        output_path: Output file path
        figsize: Figure size
        dpi: Output DPI
    """
    beta_values = results['beta_values']
    gamma_values = results['gamma_values']
    grid = results['grid']

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # High-contrast colors for betas
    colors = plt.colormaps.get_cmap('viridis').resampled(len(beta_values))

    # Plot 1: Harmonic mean vs gamma
    ax1 = axes[0]
    for i, beta in enumerate(beta_values):
        gamma_scores = [(r['gamma'], r['harmonic'])
                        for r in grid if r['beta'] == beta]
        gamma_scores.sort(key=lambda x: x[0])
        gammas = [x[0] for x in gamma_scores]
        harmonics = [x[1] for x in gamma_scores]
        ax1.plot(gammas, harmonics, 'o-', color=colors(i),
                 label=f'beta={beta:.0f}', linewidth=2, markersize=6)

    ax1.set_xlabel('gamma (embedding weight)', fontsize=12)
    ax1.set_ylabel('Harmonic Mean of Silhouette Scores', fontsize=12)
    ax1.set_title('Harmonic Score vs gamma', fontsize=14)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)

    # Plot 2: Embedding silhouette vs gamma
    ax2 = axes[1]
    for i, beta in enumerate(beta_values):
        gamma_scores = [(r['gamma'], r['sil_e'])
                        for r in grid if r['beta'] == beta]
        gamma_scores.sort(key=lambda x: x[0])
        gammas = [x[0] for x in gamma_scores]
        sil_e = [x[1] for x in gamma_scores]
        ax2.plot(gammas, sil_e, 'o-', color=colors(i),
                 label=f'beta={beta:.0f}', linewidth=2, markersize=6)

    ax2.set_xlabel('gamma (embedding weight)', fontsize=12)
    ax2.set_ylabel('Silhouette Score (Embeddings)', fontsize=12)
    ax2.set_title('Embedding Silhouette vs gamma', fontsize=14)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)

    # Plot 3: Attribution silhouette vs gamma
    ax3 = axes[2]
    for i, beta in enumerate(beta_values):
        gamma_scores = [(r['gamma'], r['sil_a'])
                        for r in grid if r['beta'] == beta]
        gamma_scores.sort(key=lambda x: x[0])
        gammas = [x[0] for x in gamma_scores]
        sil_a = [x[1] for x in gamma_scores]
        ax3.plot(gammas, sil_a, 'o-', color=colors(i),
                 label=f'beta={beta:.0f}', linewidth=2, markersize=6)

    ax3.set_xlabel('gamma (embedding weight)', fontsize=12)
    ax3.set_ylabel('Silhouette Score (Attributions)', fontsize=12)
    ax3.set_title('Attribution Silhouette vs gamma', fontsize=14)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_rd_curve(
    results: Dict,
    output_path: Path,
    figsize: tuple = (16, 5),
    dpi: int = 150,
):
    """Plot Rate-Distortion curve colored by gamma.

    Args:
        results: Sweep results dict with 'gamma_values', 'grid'
        output_path: Output file path
        figsize: Figure size
        dpi: Output DPI
    """
    grid = results['grid']
    gamma_values = results['gamma_values']

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Colormap for gamma
    colors = plt.colormaps.get_cmap('coolwarm').resampled(len(gamma_values))
    gamma_to_color = {g: colors(i) for i, g in enumerate(gamma_values)}

    # Plot 1: Rate (H) vs Total Distortion (D_e + D_a)
    ax1 = axes[0]
    for gamma in gamma_values:
        points = [(r['H'], r['D_e'] + r['D_a'], r['beta'])
                  for r in grid if r['gamma'] == gamma]
        points.sort(key=lambda x: x[2])  # Sort by beta
        H_vals = [p[0] for p in points]
        D_vals = [p[1] for p in points]
        ax1.plot(D_vals, H_vals, 'o-', color=gamma_to_color[gamma],
                 label=f'gamma={gamma:.1f}', linewidth=2, markersize=6)

    ax1.set_xlabel('Total Distortion (D_e + D_a)', fontsize=12)
    ax1.set_ylabel('Rate H(C)', fontsize=12)
    ax1.set_title('R-D Curve: Rate vs Total Distortion', fontsize=14)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Rate vs Embedding Distortion
    ax2 = axes[1]
    for gamma in gamma_values:
        points = [(r['H'], r['D_e'], r['beta'])
                  for r in grid if r['gamma'] == gamma]
        points.sort(key=lambda x: x[2])
        H_vals = [p[0] for p in points]
        D_e_vals = [p[1] for p in points]
        ax2.plot(D_e_vals, H_vals, 'o-', color=gamma_to_color[gamma],
                 label=f'gamma={gamma:.1f}', linewidth=2, markersize=6)

    ax2.set_xlabel('Embedding Distortion (D_e)', fontsize=12)
    ax2.set_ylabel('Rate H(C)', fontsize=12)
    ax2.set_title('R-D Curve: Rate vs D_e', fontsize=14)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Rate vs Attribution Distortion
    ax3 = axes[2]
    for gamma in gamma_values:
        points = [(r['H'], r['D_a'], r['beta'])
                  for r in grid if r['gamma'] == gamma]
        points.sort(key=lambda x: x[2])
        H_vals = [p[0] for p in points]
        D_a_vals = [p[1] for p in points]
        ax3.plot(D_a_vals, H_vals, 'o-', color=gamma_to_color[gamma],
                 label=f'gamma={gamma:.1f}', linewidth=2, markersize=6)

    ax3.set_xlabel('Attribution Distortion (D_a)', fontsize=12)
    ax3.set_ylabel('Rate H(C)', fontsize=12)
    ax3.set_title('R-D Curve: Rate vs D_a', fontsize=14)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_rd_curve_2d(
    results: Dict,
    output_path: Path,
    figsize: tuple = (10, 8),
    dpi: int = 150,
):
    """Plot 2D R-D surface with D_e and D_a axes.

    Args:
        results: Sweep results dict with 'grid', 'beta_values'
        output_path: Output file path
        figsize: Figure size
        dpi: Output DPI
    """
    grid = results['grid']
    beta_values = sorted(results.get('beta_values', []))
    
    # If beta_values not provided, extract from grid
    if not beta_values:
        beta_values = sorted(list(set(r['beta'] for r in grid)))

    fig, ax = plt.subplots(figsize=figsize)

    # Use a colormap for beta values to distinguish lines
    colors = plt.colormaps.get_cmap('viridis').resampled(len(beta_values))
    
    # Plot lines for each beta
    for i, beta in enumerate(beta_values):
        # Get points for this beta, sorted by gamma
        points = [r for r in grid if r['beta'] == beta]
        points.sort(key=lambda x: x['gamma'])
        
        if not points:
            continue
            
        D_e_vals = [p['D_e'] for p in points]
        D_a_vals = [p['D_a'] for p in points]
        
        # Plot line connecting these points
        ax.plot(D_e_vals, D_a_vals, '-', color=colors(i), alpha=0.6, label=f'beta={beta:.0f}')
        
        # Plot points and error bars if available
        for p in points:
            x = p['D_e']
            y = p['D_a']
            
            # Check for standard deviation
            xerr = p.get('D_e_std', None)
            yerr = p.get('D_a_std', None)
            
            if xerr is not None and yerr is not None:
                ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', 
                           color=colors(i), ecolor='gray', capsize=3, alpha=0.8)
            else:
                ax.plot(x, y, 'o', color=colors(i), alpha=0.8)

    # Annotate some key points with gamma
    # We annotate endpoints (min/max gamma) for each beta curve to reduce clutter
    for beta in beta_values:
        points = [r for r in grid if r['beta'] == beta]
        points.sort(key=lambda x: x['gamma'])
        
        if not points: 
            continue
            
        # Annotate first and last
        for p in [points[0], points[-1]]:
             ax.annotate(f"$\\gamma={p['gamma']:.1f}$",
                        (p['D_e'], p['D_a']),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=8, alpha=0.7)

    ax.set_xlabel('Embedding Distortion (D_e)', fontsize=12)
    ax.set_ylabel('Attribution Distortion (D_a)', fontsize=12)
    ax.set_title('R-D Surface: (D_e, D_a) Sweep trajectories (constant beta)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Beta", loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_heatmaps(
    results: Dict,
    output_path: Path,
    figsize: tuple = (16, 5),
    dpi: int = 150,
):
    """Plot heatmaps of metrics over (beta, gamma) grid.

    Args:
        results: Sweep results dict with 'beta_values', 'gamma_values', 'grid'
        output_path: Output file path
        figsize: Figure size
        dpi: Output DPI
    """
    beta_values = sorted(results['beta_values'])
    gamma_values = sorted(results['gamma_values'])
    grid = results['grid']

    # Create matrices
    n_beta = len(beta_values)
    n_gamma = len(gamma_values)

    harmonic_matrix = np.zeros((n_beta, n_gamma))
    K_matrix = np.zeros((n_beta, n_gamma))
    H_matrix = np.zeros((n_beta, n_gamma))

    beta_to_idx = {b: i for i, b in enumerate(beta_values)}
    gamma_to_idx = {g: j for j, g in enumerate(gamma_values)}

    for r in grid:
        i = beta_to_idx[r['beta']]
        j = gamma_to_idx[r['gamma']]
        harmonic_matrix[i, j] = r['harmonic']
        K_matrix[i, j] = r['K']
        H_matrix[i, j] = r['H']

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Heatmap 1: Harmonic score
    ax1 = axes[0]
    im1 = ax1.imshow(harmonic_matrix, aspect='auto', cmap='RdYlGn',
                     origin='lower', vmin=-1, vmax=1)
    ax1.set_xticks(range(n_gamma))
    ax1.set_xticklabels([f'{g:.1f}' for g in gamma_values])
    ax1.set_yticks(range(n_beta))
    ax1.set_yticklabels([f'{b:.0f}' for b in beta_values])
    ax1.set_xlabel('gamma')
    ax1.set_ylabel('beta')
    ax1.set_title('Harmonic Mean of Silhouette Scores')
    plt.colorbar(im1, ax=ax1)

    # Add value annotations
    for i in range(n_beta):
        for j in range(n_gamma):
            ax1.text(j, i, f'{harmonic_matrix[i, j]:.2f}',
                     ha='center', va='center', fontsize=8)

    # Heatmap 2: Number of clusters K
    ax2 = axes[1]
    im2 = ax2.imshow(K_matrix, aspect='auto', cmap='Blues', origin='lower')
    ax2.set_xticks(range(n_gamma))
    ax2.set_xticklabels([f'{g:.1f}' for g in gamma_values])
    ax2.set_yticks(range(n_beta))
    ax2.set_yticklabels([f'{b:.0f}' for b in beta_values])
    ax2.set_xlabel('gamma')
    ax2.set_ylabel('beta')
    ax2.set_title('Number of Clusters (K)')
    plt.colorbar(im2, ax=ax2)

    for i in range(n_beta):
        for j in range(n_gamma):
            ax2.text(j, i, f'{int(K_matrix[i, j])}',
                     ha='center', va='center', fontsize=8)

    # Heatmap 3: Rate H(C)
    ax3 = axes[2]
    im3 = ax3.imshow(H_matrix, aspect='auto', cmap='Purples', origin='lower')
    ax3.set_xticks(range(n_gamma))
    ax3.set_xticklabels([f'{g:.1f}' for g in gamma_values])
    ax3.set_yticks(range(n_beta))
    ax3.set_yticklabels([f'{b:.0f}' for b in beta_values])
    ax3.set_xlabel('gamma')
    ax3.set_ylabel('beta')
    ax3.set_title('Rate H(C)')
    plt.colorbar(im3, ax=ax3)

    for i in range(n_beta):
        for j in range(n_gamma):
            ax3.text(j, i, f'{H_matrix[i, j]:.2f}',
                     ha='center', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
