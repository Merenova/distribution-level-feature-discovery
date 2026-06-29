#!/usr/bin/env python3
"""
Contrastive H_c computation for improved steering with RD clustering.

The problem: RD clustering groups similar attributions together, making the 
within-cluster H_c (weighted median) weak or inverted.

The solution: Compute H_c as the contrastive direction that distinguishes
this cluster from all other clusters.
"""

import numpy as np
from typing import Dict, Tuple


def compute_contrastive_H_c(
    attributions: np.ndarray,
    assignments: np.ndarray,
    path_probs: np.ndarray,
    cluster_id: int,
    H_0: np.ndarray = None  # No longer used, kept for API compatibility
) -> np.ndarray:
    """Compute H_c as the direction that distinguishes this cluster from others.
    
    H_c = mean_in_cluster - mean_out_of_cluster
    
    This gives a steering direction that moves TOWARD this cluster's semantics,
    even when RD has grouped similar attributions together.
    
    Note: H_0 parameter is kept for API compatibility but no longer used.
    
    Args:
        attributions: Raw attribution vectors [n_samples, d]
        assignments: Cluster assignments [n_samples]
        path_probs: Path probabilities [n_samples]
        cluster_id: Cluster ID to compute H_c for
        H_0: Unused, kept for API compatibility
        
    Returns:
        H_c: Contrastive attribution vector for steering [d]
    """
    in_mask = assignments == cluster_id
    out_mask = ~in_mask
    
    d = attributions.shape[1]
    
    if not np.any(in_mask):
        return np.zeros(d)
    
    if not np.any(out_mask):
        # Only one cluster - fall back to within-cluster mean (centered)
        in_weights = path_probs[in_mask]
        in_total = in_weights.sum()
        if in_total <= 0:
            return np.zeros(d)
        # Return centered mean (subtract global mean)
        global_mean = np.sum(path_probs[:, None] * attributions, axis=0) / path_probs.sum()
        in_mean = np.sum(in_weights[:, None] * attributions[in_mask], axis=0) / in_total
        return in_mean - global_mean
    
    # Weighted means
    in_weights = path_probs[in_mask]
    out_weights = path_probs[out_mask]
    
    in_total = in_weights.sum()
    out_total = out_weights.sum()
    
    if in_total <= 0:
        return np.zeros(d)
    
    in_mean = np.sum(in_weights[:, None] * attributions[in_mask], axis=0) / in_total
    
    if out_total <= 0:
        # Return centered mean
        global_mean = np.sum(path_probs[:, None] * attributions, axis=0) / path_probs.sum()
        return in_mean - global_mean
    
    out_mean = np.sum(out_weights[:, None] * attributions[out_mask], axis=0) / out_total
    
    # Contrastive direction: what makes this cluster different?
    # This is already centered (difference of means)
    H_c = in_mean - out_mean
    
    return H_c


def compute_normalized_contrastive_H_c(
    attributions: np.ndarray,
    assignments: np.ndarray,
    path_probs: np.ndarray,
    cluster_id: int,
    H_0: np.ndarray = None,  # No longer used, kept for API compatibility
    target_norm: float = None
) -> np.ndarray:
    """Compute contrastive H_c with optional normalization.
    
    Note: H_0 parameter is kept for API compatibility but no longer used.
    
    Args:
        attributions: Raw attribution vectors [n_samples, d]
        assignments: Cluster assignments [n_samples]
        path_probs: Path probabilities [n_samples]
        cluster_id: Cluster ID to compute H_c for
        H_0: Unused, kept for API compatibility
        target_norm: If provided, scale H_c to have this norm
        
    Returns:
        H_c: Normalized contrastive attribution vector [d]
    """
    H_c = compute_contrastive_H_c(attributions, assignments, path_probs, cluster_id)
    
    if target_norm is not None:
        current_norm = np.linalg.norm(H_c)
        if current_norm > 1e-10:
            H_c = H_c * (target_norm / current_norm)
    
    return H_c


def build_contrastive_semantic_graphs(
    assignments: np.ndarray,
    attributions: np.ndarray,
    path_probs: np.ndarray,
    H_0: np.ndarray = None,  # No longer used, kept for API compatibility
    normalize: bool = False
) -> Dict[int, np.ndarray]:
    """Build semantic graphs using contrastive H_c.
    
    Drop-in replacement for build_semantic_graphs_from_kmeans() that uses
    contrastive H_c instead of weighted median.
    
    Note: H_0 parameter is kept for API compatibility but no longer used.
    
    Args:
        assignments: Cluster assignments [n_samples]
        attributions: Raw attributions [n_samples, d]
        path_probs: Path probabilities [n_samples]
        H_0: Unused, kept for API compatibility
        normalize: If True, normalize all H_c to have same norm
        
    Returns:
        {cluster_id: H_c array}
    """
    unique_clusters = sorted([int(c) for c in set(assignments)])
    
    semantic_graphs = {}
    
    # First pass: compute all contrastive H_c
    for c in unique_clusters:
        H_c = compute_contrastive_H_c(attributions, assignments, path_probs, c)
        semantic_graphs[c] = H_c
    
    # Optional: normalize all H_c to have consistent scale
    if normalize and len(semantic_graphs) > 1:
        norms = [np.linalg.norm(H_c) for H_c in semantic_graphs.values()]
        median_norm = np.median([n for n in norms if n > 1e-10])
        
        if median_norm > 1e-10:
            for c in semantic_graphs:
                current_norm = np.linalg.norm(semantic_graphs[c])
                if current_norm > 1e-10:
                    semantic_graphs[c] = semantic_graphs[c] * (median_norm / current_norm)
    
    return semantic_graphs


def compute_steering_discriminability(
    attributions: np.ndarray,
    assignments: np.ndarray,
    path_probs: np.ndarray,
    cluster_id: int
) -> float:
    """Measure how well the cluster's H_c distinguishes it from other clusters.
    
    Returns a score similar to Cohen's d: (in_mean - out_mean) / pooled_std
    projected onto the contrastive direction.
    
    High discriminability = the H_c direction cleanly separates this cluster.
    """
    in_mask = assignments == cluster_id
    out_mask = ~in_mask
    
    if not np.any(in_mask) or not np.any(out_mask):
        return 0.0
    
    # Compute contrastive direction
    in_weights = path_probs[in_mask]
    out_weights = path_probs[out_mask]
    
    in_total = in_weights.sum()
    out_total = out_weights.sum()
    
    if in_total <= 0 or out_total <= 0:
        return 0.0
    
    in_mean = np.sum(in_weights[:, None] * attributions[in_mask], axis=0) / in_total
    out_mean = np.sum(out_weights[:, None] * attributions[out_mask], axis=0) / out_total
    
    direction = in_mean - out_mean
    direction_norm = np.linalg.norm(direction)
    
    if direction_norm < 1e-10:
        return 0.0
    
    direction = direction / direction_norm
    
    # Project all points onto this direction
    in_proj = attributions[in_mask] @ direction
    out_proj = attributions[out_mask] @ direction
    
    # Weighted means of projections
    in_mean_proj = np.sum(in_weights * in_proj) / in_total
    out_mean_proj = np.sum(out_weights * out_proj) / out_total
    
    # Weighted variance
    in_var = np.sum(in_weights * (in_proj - in_mean_proj)**2) / in_total
    out_var = np.sum(out_weights * (out_proj - out_mean_proj)**2) / out_total
    
    # Pooled standard deviation
    pooled_var = (in_var * in_total + out_var * out_total) / (in_total + out_total)
    pooled_std = np.sqrt(pooled_var + 1e-10)
    
    # Cohen's d equivalent
    discriminability = (in_mean_proj - out_mean_proj) / pooled_std
    
    return float(discriminability)


# =============================================================================
# Comparison utilities
# =============================================================================

def compare_hc_methods(
    attributions: np.ndarray,
    assignments: np.ndarray,
    path_probs: np.ndarray,
    H_0: np.ndarray
) -> Dict:
    """Compare weighted median vs contrastive H_c for all clusters.
    
    Returns diagnostic info to understand which method produces better H_c.
    """
    from scipy.stats import spearmanr
    
    unique_clusters = sorted([int(c) for c in set(assignments)])
    
    results = {
        'clusters': {},
        'summary': {}
    }
    
    for c in unique_clusters:
        mask = assignments == c
        n_samples = np.sum(mask)
        
        # Contrastive H_c
        H_c_contrastive = compute_contrastive_H_c(
            attributions, assignments, path_probs, c, H_0
        )
        
        # Weighted median H_c (original method)
        H_c_median = _compute_weighted_median_H_c(
            attributions, assignments, path_probs, c, H_0
        )
        
        # Discriminability
        discrim = compute_steering_discriminability(
            attributions, assignments, path_probs, c
        )
        
        # Cosine similarity between methods
        delta_contrastive = H_c_contrastive - H_0
        delta_median = H_c_median - H_0
        
        norm_c = np.linalg.norm(delta_contrastive)
        norm_m = np.linalg.norm(delta_median)
        
        if norm_c > 1e-10 and norm_m > 1e-10:
            cosine_sim = np.dot(delta_contrastive, delta_median) / (norm_c * norm_m)
        else:
            cosine_sim = 0.0
        
        results['clusters'][c] = {
            'n_samples': int(n_samples),
            'discriminability': discrim,
            'delta_norm_contrastive': float(norm_c),
            'delta_norm_median': float(norm_m),
            'cosine_similarity': float(cosine_sim),
            'norm_ratio': float(norm_c / norm_m) if norm_m > 1e-10 else float('inf'),
        }
    
    # Summary stats
    discrims = [r['discriminability'] for r in results['clusters'].values()]
    cosines = [r['cosine_similarity'] for r in results['clusters'].values()]
    ratios = [r['norm_ratio'] for r in results['clusters'].values() if r['norm_ratio'] != float('inf')]
    
    results['summary'] = {
        'mean_discriminability': float(np.mean(discrims)),
        'mean_cosine_similarity': float(np.mean(cosines)),
        'mean_norm_ratio': float(np.mean(ratios)) if ratios else None,
        'n_inverted': sum(1 for c in cosines if c < 0),  # H_c methods point opposite directions
        'n_clusters': len(unique_clusters),
    }
    
    return results


def _compute_weighted_median_H_c(
    attributions: np.ndarray,
    assignments: np.ndarray,
    path_probs: np.ndarray,
    cluster_id: int,
    H_0: np.ndarray
) -> np.ndarray:
    """Compute H_c using weighted median (original method).
    
    Returns centered H_c (Delta_H_c) directly, without adding H_0.
    """
    d = attributions.shape[1]
    mask = assignments == cluster_id
    if not np.any(mask):
        return np.zeros(d)
    
    cluster_attr = attributions[mask] - H_0
    cluster_weights = path_probs[mask]
    
    total_weight = cluster_weights.sum()
    if total_weight <= 0:
        return np.zeros(d)
    
    normalized_weights = cluster_weights / total_weight
    
    # Weighted median per dimension
    H_c = np.zeros(d)
    
    for dim in range(d):
        dim_values = cluster_attr[:, dim]
        sort_idx = np.argsort(dim_values)
        sorted_values = dim_values[sort_idx]
        sorted_weights = normalized_weights[sort_idx]
        cumulative = np.cumsum(sorted_weights)
        median_idx = np.searchsorted(cumulative, 0.5)
        median_idx = min(median_idx, len(sorted_values) - 1)
        H_c[dim] = sorted_values[median_idx]
    
    return H_c

