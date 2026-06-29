"""EM loop implementation with rate-distortion assignment rule.

Implements the rate-distortion EM algorithm from rate_distortion.tex:
- E-step: c(n) = argmin_c [-log P̄_c + β_e ||e_n - μ_c^(e)||² + β_a ||a_n - μ_c^(a)||²]
- M-step: Probability-weighted center updates (same as legacy)

Supports optional GPU acceleration via PyTorch for large datasets.
"""

import numpy as np
from typing import Dict, List, Tuple

from rd_objective import (
    compute_component_masses,
    compute_normalized_masses,
    compute_full_rd_statistics,
    compute_mse_distances,
    compute_l1_distances,
)


def weighted_median_1d(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted median for 1D array.
    
    Args:
        values: 1D array of values
        weights: 1D array of weights (probabilities)
    
    Returns:
        Weighted median value
    """
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    cumsum = np.cumsum(sorted_weights)
    total = weights.sum()
    median_idx = np.searchsorted(cumsum, total / 2)
    median_idx = min(median_idx, len(values) - 1)
    return float(sorted_values[median_idx])


def weighted_median(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute coordinate-wise weighted median.
    
    Args:
        data: 2D array (n_samples, n_features)
        weights: 1D array of weights (n_samples,)
    
    Returns:
        1D array of weighted medians per coordinate
    """
    n_features = data.shape[1]
    result = np.zeros(n_features)
    for j in range(n_features):
        result[j] = weighted_median_1d(data[:, j], weights)
    return result

# Try to import GPU utils
try:
    from gpu_utils import get_compute_backend, TORCH_AVAILABLE
    GPU_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    GPU_AVAILABLE = False
    def get_compute_backend(*args, **kwargs):
        return None


def rd_e_step(
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    components: Dict[int, Dict],
    P_bar: Dict[int, float],
    beta_e: float,
    beta_a: float,
    use_gpu: bool = True,
    metric_a: str = "l2",
) -> List[int]:
    """E-step with rate-distortion assignment rule (vectorized, GPU-accelerated).

    Assignment rule:
    c(n) = argmin_c [-log P̄_c + β_e Dist(e_n, μ_c^(e)) + β_a Dist(a_n, μ_c^(a))]

    Args:
        embeddings_e: Semantic embeddings (n_samples, d_e)
        attributions_a: Attribution embeddings (n_samples, d_a)
        components: Current components with mu_e, mu_a centers
        P_bar: Normalized component masses
        beta_e: Semantic distortion weight
        beta_a: Attribution distortion weight
        use_gpu: Whether to use GPU acceleration if available
        metric_a: Metric for attribution distance ("l2" or "l1")

    Returns:
        assignments: List of component assignments c(n) for each sample
    """
    n_samples = embeddings_e.shape[0]

    # Get component IDs
    component_ids = list(components.keys())
    if len(component_ids) == 0:
        return [0] * n_samples

    K = len(component_ids)
    eps = 1e-10

    # Stack centers into arrays: (K, d_e) and (K, d_a)
    centers_e = np.array([components[c]['mu_e'] for c in component_ids], dtype=np.float32)
    centers_a = np.array([components[c]['mu_a'] for c in component_ids], dtype=np.float32)

    # Rate costs: (K,)
    rate_costs = np.array([-np.log(P_bar.get(c, eps) + eps) for c in component_ids], dtype=np.float32)

    # Try GPU acceleration for datasets
    # GPU path supports both L2 and L1 metrics
    # Threshold lowered to 50 - with large GPUs, transfer overhead is negligible
    backend = None
    if use_gpu and GPU_AVAILABLE and n_samples >= 50:
        backend = get_compute_backend(use_gpu=True, n_samples=n_samples)

    if backend is not None and hasattr(backend, 'name') and backend.name == "torch":
        # GPU-accelerated computation
        best_indices = backend.rd_e_step(
            embeddings_e.astype(np.float32),
            attributions_a.astype(np.float32),
            centers_e, centers_a, rate_costs,
            beta_e, beta_a,
            metric_a=metric_a,
        )
    else:
        # CPU vectorized computation using helper functions
        dist_e_sq = compute_mse_distances(embeddings_e, centers_e)  # (N, K)
        
        if metric_a == "l1":
            dist_a = compute_l1_distances(attributions_a, centers_a) # (N, K)
        else:
            dist_a = compute_mse_distances(attributions_a, centers_a)  # (N, K)

        # Total cost: (N, K)
        total_cost = rate_costs[np.newaxis, :] + beta_e * dist_e_sq + beta_a * dist_a

        # Find best assignment for each sample: (N,)
        best_indices = np.argmin(total_cost, axis=1)

    # Map back to component IDs
    assignments = [int(component_ids[i]) for i in best_indices]

    return assignments


def m_step(
    assignments: List[int],
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    path_probs: np.ndarray,
    component_ids: List[int],
    metric_a: str = "l2",
) -> Tuple[Dict[int, Dict], Dict[int, float]]:
    """M-step: Update components with probability-weighted statistics (vectorized).

    Performs probability-weighted center updates.
    Embedding centers mu_e are L2-normalized for spherical clustering.

    Args:
        assignments: Component assignments for each sample
        embeddings_e: Semantic embeddings (assumed L2-normalized)
        attributions_a: Attribution embeddings
        path_probs: Path probabilities
        component_ids: List of component IDs to update
        metric_a: Metric for attribution center calculation ("l2" -> mean, "l1" -> weighted median)

    Returns:
        Tuple of (updated_components, W_c masses)
    """
    # Convert assignments to numpy array once for vectorized masking
    assignments_arr = np.asarray(assignments)

    components = {}
    W_c = {}

    for c in component_ids:
        # Vectorized mask instead of list comprehension
        mask = assignments_arr == c

        if not np.any(mask):
            continue

        # Extract data for this component using boolean mask
        e_c = embeddings_e[mask]
        a_c = attributions_a[mask]
        P_c = path_probs[mask]

        # Compute total probability mass (always needed for Entropy / W_c)
        W = float(np.sum(P_c))

        if W == 0:
            continue

        W_c[c] = W

        # Update semantic center (probability-weighted mean)
        # Semantic is always L2 spherical -> Mean direction
        mu_e_c = np.sum(P_c[:, None] * e_c, axis=0) / W
            
        # L2-normalize for spherical clustering
        mu_e_norm = np.linalg.norm(mu_e_c)
        if mu_e_norm > 1e-10:
            mu_e_c = mu_e_c / mu_e_norm

        # Update attribution center
        if metric_a == "l1":
            # L1 -> Probability-weighted median (coordinate-wise)
            mu_a_c = weighted_median(a_c, P_c)
        else:
            # L2 -> Probability-weighted mean
            mu_a_c = np.sum(P_c[:, None] * a_c, axis=0) / W

        # Store indices as list for compatibility
        indices = np.where(mask)[0].tolist()

        components[c] = {
            'mu_e': mu_e_c,
            'mu_a': mu_a_c,
            'W_c': W,
            'indices': indices,
        }

    return components, W_c


def run_em_iteration(
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    path_probs: np.ndarray,
    components: Dict[int, Dict],
    beta_e: float,
    beta_a: float,
    use_gpu: bool = True,
    metric_a: str = "l2",
) -> Tuple[List[int], Dict[int, Dict], Dict]:
    """Run one EM iteration with rate-distortion assignment.

    Args:
        embeddings_e: Semantic embeddings
        attributions_a: Attribution embeddings
        path_probs: Path probabilities
        components: Current components
        beta_e: Semantic distortion weight
        beta_a: Attribution distortion weight
        use_gpu: Whether to use GPU acceleration if available
        metric_a: Metric for attribution distortion

    Returns:
        Tuple of (assignments, updated_components, rd_statistics)
    """
    component_ids = list(components.keys())

    # Compute current masses for E-step
    W_c, W_total = compute_component_masses(
        list(range(len(path_probs))),  # dummy assignments for initial calc
        path_probs,
        component_ids
    )

    # Use component indices to compute actual masses
    temp_assignments = []
    for n in range(len(path_probs)):
        # Find which component this sample currently belongs to
        assigned = False
        for c, comp in components.items():
            if 'indices' in comp and n in comp['indices']:
                temp_assignments.append(c)
                assigned = True
                break
        if not assigned:
            temp_assignments.append(component_ids[0] if component_ids else 0)

    W_c, W_total = compute_component_masses(temp_assignments, path_probs, component_ids)
    P_bar = compute_normalized_masses(W_c, W_total)

    # E-step: Rate-distortion assignment (GPU-accelerated if available)
    assignments = rd_e_step(
        embeddings_e,
        attributions_a,
        components,
        P_bar,
        beta_e,
        beta_a,
        use_gpu=use_gpu,
        metric_a=metric_a
    )

    # M-step: Update components (always probability-weighted)
    active_ids = list(set(assignments) - {-1})
    updated_components, W_c = m_step(
        assignments,
        embeddings_e,
        attributions_a,
        path_probs,
        active_ids,
        metric_a=metric_a,
    )

    # Compute R-D statistics (always probability-weighted)
    rd_stats = compute_full_rd_statistics(
        embeddings_e,
        attributions_a,
        assignments,
        path_probs,
        updated_components,
        beta_e,
        beta_a,
        metric_a=metric_a,
    )

    return assignments, updated_components, rd_stats


def check_convergence(
    L_RD_prev: float,
    L_RD_curr: float,
    threshold: float = 1e-6
) -> bool:
    """Check if EM has converged based on R-D objective change.

    Args:
        L_RD_prev: Previous R-D objective
        L_RD_curr: Current R-D objective
        threshold: Relative change threshold

    Returns:
        True if converged, False otherwise
    """
    # Handle special cases: inf, nan, or very small values
    if not np.isfinite(L_RD_prev) or not np.isfinite(L_RD_curr):
        return False  # Can't converge if values are inf/nan

    if abs(L_RD_prev) < 1e-10:
        return abs(L_RD_curr - L_RD_prev) < threshold

    relative_change = abs(L_RD_curr - L_RD_prev) / abs(L_RD_prev)
    return relative_change < threshold


if __name__ == "__main__":
    # Test EM iteration with R-D assignment
    np.random.seed(42)

    n_samples = 100
    d_e = 64
    d_a = 128
    K = 3

    embeddings_e = np.random.randn(n_samples, d_e)
    attributions_a = np.random.randn(n_samples, d_a)
    path_probs = np.random.rand(n_samples)
    path_probs = path_probs / np.sum(path_probs)

    # Initialize with random assignments (probability-weighted)
    init_assignments = list(np.random.choice([1, 2, 3], size=n_samples))
    components = {}
    for c in [1, 2, 3]:
        indices = [i for i, a in enumerate(init_assignments) if a == c]
        P_c = path_probs[indices]
        W_c = np.sum(P_c)
        if W_c > 0:
            components[c] = {
                'mu_e': np.sum(P_c[:, None] * embeddings_e[indices], axis=0) / W_c,
                'mu_a': np.sum(P_c[:, None] * attributions_a[indices], axis=0) / W_c,
                'W_c': W_c,
                'indices': indices,
            }

    print("Testing R-D EM iteration...")
    print(f"Initial components: {len(components)}")

    # Run EM iteration
    assignments, updated_components, rd_stats = run_em_iteration(
        embeddings_e,
        attributions_a,
        path_probs,
        components,
        beta_e=1.0,
        beta_a=1.0,
        metric_a="l2",
    )

    print(f"\nAfter EM iteration (L2):")
    print(f"  Components: {len(updated_components)}")
    print(f"  L_RD: {rd_stats['L_RD']:.4f}")
    print(f"  H(C): {rd_stats['H']:.4f}")
    
    # Run EM iteration L1
    assignments, updated_components, rd_stats = run_em_iteration(
        embeddings_e,
        attributions_a,
        path_probs,
        components,
        beta_e=1.0,
        beta_a=1.0,
        metric_a="l1",
    )
    
    print(f"\nAfter EM iteration (L1):")
    print(f"  Components: {len(updated_components)}")
    print(f"  L_RD: {rd_stats['L_RD']:.4f}")

    print("\nR-D EM iteration test passed!")
