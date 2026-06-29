"""Initialization module for Rate-Distortion Gaussian clustering.

Implements single-component initialization per rate_distortion.tex Algorithm 1:
- Start with K=1 (all points in one component)
- Split operation creates new components as needed

Also supports warm-start from previous solution for β-annealing.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from em_loop import weighted_median


def initialize_single_component(
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    path_probs: np.ndarray,
    metric_a: str = "l2",
) -> Tuple[Dict[int, Dict], List[int]]:
    """Initialize with a single component containing all points.

    Args:
        embeddings_e: Semantic embeddings (N, d_e)
        attributions_a: Attribution embeddings (N, d_a)
        path_probs: Probability weights P_n (N,)

    Returns:
        Tuple of (components dict, assignments list)
    """
    n_samples = len(path_probs)
    W_total = float(np.sum(path_probs))

    if W_total <= 0:
        # Fall back to uniform weights to avoid degenerate all-zero weighting.
        path_probs = np.ones(n_samples, dtype=np.float64)
        W_total = float(np.sum(path_probs))

    # Compute probability-weighted centers
    mu_e = np.sum(path_probs[:, None] * embeddings_e, axis=0) / W_total
    # L2-normalize embedding center for spherical clustering
    mu_e_norm = np.linalg.norm(mu_e)
    if mu_e_norm > 1e-10:
        mu_e = mu_e / mu_e_norm

    if metric_a == "l1":
        mu_a = weighted_median(attributions_a, path_probs)
    else:
        mu_a = np.sum(path_probs[:, None] * attributions_a, axis=0) / W_total

    # All points assigned to component 1
    assignments = [1] * n_samples

    # Single component
    components = {
        1: {
            'mu_e': mu_e,
            'mu_a': mu_a,
            'W_c': W_total,
            'indices': list(range(n_samples)),
        }
    }

    return components, assignments


def initialize_from_previous(
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    path_probs: np.ndarray,
    prev_components: Dict[int, Dict],
    prev_assignments: List[int],
    metric_a: str = "l2",
) -> Tuple[Dict[int, Dict], List[int]]:
    """Warm-start initialization from previous clustering solution.

    Used in β-annealing to provide warm-start from previous β value.
    Copies component structure and recomputes centers from data.

    Args:
        embeddings_e: Semantic embeddings (N, d_e)
        attributions_a: Attribution embeddings (N, d_a)
        path_probs: Probability weights P_n (N,)
        prev_components: Components from previous clustering
        prev_assignments: Assignments from previous clustering

    Returns:
        Tuple of (components dict, assignments list)
    """
    n_samples = len(path_probs)

    # Validate previous assignments length
    if len(prev_assignments) != n_samples:
        # Fall back to single component if mismatch
        return initialize_single_component(embeddings_e, attributions_a, path_probs, metric_a=metric_a)

    # Copy assignments
    assignments = prev_assignments.copy()

    # Recompute component statistics from current data
    components = {}
    component_ids = set(assignments)

    for c in component_ids:
        if c <= 0:  # Skip invalid
            continue

        indices = [i for i, a in enumerate(assignments) if a == c]
        if not indices:
            continue

        P_c = path_probs[indices]
        W_c = np.sum(P_c)

        if W_c == 0:
            continue

        # Compute probability-weighted centers
        mu_e = np.sum(P_c[:, None] * embeddings_e[indices], axis=0) / W_c
        # L2-normalize embedding center for spherical clustering
        mu_e_norm = np.linalg.norm(mu_e)
        if mu_e_norm > 1e-10:
            mu_e = mu_e / mu_e_norm
        if metric_a == "l1":
            mu_a = weighted_median(attributions_a[indices], P_c)
        else:
            mu_a = np.sum(P_c[:, None] * attributions_a[indices], axis=0) / W_c

        components[c] = {
            'mu_e': mu_e,
            'mu_a': mu_a,
            'W_c': W_c,
            'indices': indices,
        }

    # If no valid components, fall back to single component
    if not components:
        return initialize_single_component(embeddings_e, attributions_a, path_probs, metric_a=metric_a)

    return components, assignments


def compute_initial_statistics(
    components: Dict[int, Dict],
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    path_probs: np.ndarray,
    W_total: Optional[float] = None,
) -> Dict[int, Dict[str, float]]:
    """Compute initial variance statistics for components.

    Args:
        components: Component dictionary
        embeddings_e: Semantic embeddings
        attributions_a: Attribution embeddings
        path_probs: Probability weights
        W_total: Total probability mass (computed if None)

    Returns:
        Dictionary of statistics {c: {Var_e, Var_a, n_samples}}
    """
    if W_total is None:
        W_total = np.sum(path_probs)

    statistics = {}

    for c, comp in components.items():
        indices = comp['indices']
        mu_e = comp['mu_e']
        mu_a = comp['mu_a']
        W_c = comp['W_c']

        if not indices or W_c == 0:
            statistics[c] = {
                'n_samples': 0,
                'Var_e': 0.0,
                'Var_a': 0.0,
            }
            continue

        # Get data for this component
        e_c = embeddings_e[indices]
        a_c = attributions_a[indices]
        P_c = path_probs[indices]

        # Compute weighted variances
        diff_e = e_c - mu_e[None, :]
        Var_e = np.sum(P_c * np.sum(diff_e ** 2, axis=1)) / W_c

        diff_a = a_c - mu_a[None, :]
        Var_a = np.sum(P_c * np.sum(diff_a ** 2, axis=1)) / W_c

        statistics[c] = {
            'n_samples': len(indices),
            'Var_e': float(Var_e),
            'Var_a': float(Var_a),
        }

    return statistics


if __name__ == "__main__":
    # Test initialization
    np.random.seed(42)

    n_samples = 100
    d_e = 64
    d_a = 128

    embeddings_e = np.random.randn(n_samples, d_e)
    attributions_a = np.random.randn(n_samples, d_a)
    path_probs = np.random.rand(n_samples)
    path_probs = path_probs / np.sum(path_probs)

    print("Testing single-component initialization...")
    components, assignments = initialize_single_component(
        embeddings_e, attributions_a, path_probs
    )

    print(f"Initialized {len(components)} component(s)")
    for c, comp in components.items():
        print(f"  Component {c}:")
        print(f"    W_c: {comp['W_c']:.4f}")
        print(f"    n_samples: {len(comp['indices'])}")
        print(f"    mu_e shape: {comp['mu_e'].shape}")
        print(f"    mu_a shape: {comp['mu_a'].shape}")

    print("\nComputing statistics...")
    stats = compute_initial_statistics(
        components, embeddings_e, attributions_a, path_probs
    )

    for c, stat in stats.items():
        print(f"  Component {c}:")
        print(f"    Var_e: {stat['Var_e']:.4f}")
        print(f"    Var_a: {stat['Var_a']:.4f}")

    print("\nTesting warm-start initialization...")
    # Simulate previous solution with 3 clusters
    prev_assignments = [1] * 40 + [2] * 35 + [3] * 25
    prev_components = {}
    for c in [1, 2, 3]:
        idx = [i for i, a in enumerate(prev_assignments) if a == c]
        W_c = np.sum(path_probs[idx])
        prev_components[c] = {
            'mu_e': np.sum(path_probs[idx, None] * embeddings_e[idx], axis=0) / W_c,
            'mu_a': np.sum(path_probs[idx, None] * attributions_a[idx], axis=0) / W_c,
            'W_c': W_c,
            'indices': idx,
        }

    components2, assignments2 = initialize_from_previous(
        embeddings_e, attributions_a, path_probs,
        prev_components, prev_assignments
    )

    print(f"Warm-started with {len(components2)} component(s)")
    for c, comp in components2.items():
        print(f"  Component {c}: {len(comp['indices'])} samples, W_c={comp['W_c']:.4f}")

    print("\nInitialization tests passed!")
