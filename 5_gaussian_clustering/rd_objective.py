"""Rate-distortion objective computation for semantic-anchored two-view clustering.

Implements the rate-distortion framework:
    L_RD = H(C) + β_e D^(e) + β_a D^(a)

Where:
    - H(C) = -Σ P̄_c log P̄_c  (entropy / rate)
    - D^(e) = Σ P̄_c Var_c^(e,w)  (semantic distortion)
    - D^(a) = Σ P̄_c Var_c^(a,w)  (attribution distortion)
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_component_masses(
    assignments: List[int],
    path_probs: np.ndarray,
    component_ids: List[int]
) -> Tuple[Dict[int, float], float]:
    """Compute probability mass for each component (vectorized).

    Args:
        assignments: Component assignments c(n) for each sample
        path_probs: Path probabilities P_n
        component_ids: List of active component IDs

    Returns:
        Tuple of (W_c dict, W_total)
    """
    # Convert to numpy array for vectorized operations
    assignments_arr = np.asarray(assignments)

    W_c = {}
    W_total = 0.0

    for c in component_ids:
        # Vectorized mask instead of list comprehension
        mask = assignments_arr == c
        W_c[c] = float(np.sum(path_probs[mask])) if np.any(mask) else 0.0
        W_total += W_c[c]

    return W_c, W_total


def compute_normalized_masses(W_c: Dict[int, float], W_total: float) -> Dict[int, float]:
    """Compute normalized component masses P̄_c = W_c / W_total.

    Args:
        W_c: Component masses
        W_total: Total mass

    Returns:
        Normalized masses P̄_c
    """
    if W_total == 0:
        return {c: 0.0 for c in W_c}

    return {c: W / W_total for c, W in W_c.items()}


def compute_entropy(P_bar: Dict[int, float], eps: float = 1e-10) -> float:
    """Compute entropy (rate) H(C) = -Σ P̄_c log P̄_c.

    Args:
        P_bar: Normalized component masses
        eps: Small constant to avoid log(0)

    Returns:
        Entropy H(C)
    """
    H = 0.0
    for p in P_bar.values():
        if p > eps:
            H -= p * np.log(p)
    return H


def compute_semantic_distortion(
    embeddings_e: np.ndarray,
    assignments: List[int],
    path_probs: np.ndarray,
    components: Dict[int, Dict],
    P_bar: Dict[int, float],
    W_c: Dict[int, float],
) -> Tuple[float, Dict[int, float]]:
    """Compute semantic distortion D^(e) = Σ P̄_c Var_c^(e,w) (vectorized).

    Args:
        embeddings_e: Semantic embeddings (n_samples, d_e)
        assignments: Component assignments
        path_probs: Path probabilities
        components: Component dict with mu_e centers
        P_bar: Normalized component masses
        W_c: Component masses

    Returns:
        Tuple of (total distortion D^(e), per-component Var_c^(e,w))
    """
    # Convert assignments to numpy array once
    assignments_arr = np.asarray(assignments)
    n_samples = len(assignments_arr)

    Var_e = {}
    D_e = 0.0

    for c, comp in components.items():
        if c not in P_bar or P_bar[c] == 0: # Check if active in P_bar
             Var_e[c] = 0.0
             continue

        # Vectorized mask instead of list comprehension
        mask = assignments_arr == c
        if not np.any(mask):
            Var_e[c] = 0.0
            continue

        e_c = embeddings_e[mask]
        mu_e = comp['mu_e']

        # Probability-weighted variance
        diff = e_c - mu_e[None, :]
        sq_dists = np.sum(diff ** 2, axis=1)
        
        P_c = path_probs[mask]
        if W_c.get(c, 0) == 0:
            Var_e[c] = 0.0
        else:
            Var_e[c] = float(np.sum(P_c * sq_dists) / W_c[c])
        D_e += P_bar[c] * Var_e[c]

    return D_e, Var_e


def compute_attribution_distortion(
    attributions_a: np.ndarray,
    assignments: List[int],
    path_probs: np.ndarray,
    components: Dict[int, Dict],
    P_bar: Dict[int, float],
    W_c: Dict[int, float],
    metric: str = "l2",
) -> Tuple[float, Dict[int, float]]:
    """Compute attribution distortion D^(a) = Σ P̄_c Var_c^(a,w) (vectorized).

    Args:
        attributions_a: Attribution embeddings (n_samples, d_a)
        assignments: Component assignments
        path_probs: Path probabilities
        components: Component dict with mu_a centers
        P_bar: Normalized component masses
        W_c: Component masses
        metric: "l2" or "l1"

    Returns:
        Tuple of (total distortion D^(a), per-component Var_c^(a,w))
    """
    # Convert assignments to numpy array once
    assignments_arr = np.asarray(assignments)
    n_samples = len(assignments_arr)

    Var_a = {}
    D_a = 0.0

    for c, comp in components.items():
        if c not in P_bar or P_bar[c] == 0:
             Var_a[c] = 0.0
             continue

        # Vectorized mask instead of list comprehension
        mask = assignments_arr == c
        if not np.any(mask):
            Var_a[c] = 0.0
            continue

        a_c = attributions_a[mask]
        mu_a = comp['mu_a']

        diff = a_c - mu_a[None, :]
        if metric == "l2":
            dists = np.sum(diff ** 2, axis=1)
        elif metric == "l1":
            dists = np.sum(np.abs(diff), axis=1)
        else:
             raise ValueError(f"Unknown metric {metric}")

        # Probability-weighted distortion
        P_c = path_probs[mask]
        if W_c.get(c, 0) == 0:
            Var_a[c] = 0.0
        else:
            Var_a[c] = float(np.sum(P_c * dists) / W_c[c])
        D_a += P_bar[c] * Var_a[c]

    return D_a, Var_a


def compute_rd_objective(
    H: float,
    D_e: float,
    D_a: float,
    beta_e: float,
    beta_a: float
) -> float:
    """Compute the full rate-distortion objective.

    L_RD = H(C) + β_e D^(e) + β_a D^(a)

    Args:
        H: Entropy (rate)
        D_e: Semantic distortion
        D_a: Attribution distortion
        beta_e: Semantic distortion weight
        beta_a: Attribution distortion weight

    Returns:
        Rate-distortion objective L_RD
    """
    return H + beta_e * D_e + beta_a * D_a


def compute_full_rd_statistics(
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    assignments: List[int],
    path_probs: np.ndarray,
    components: Dict[int, Dict],
    beta_e: float,
    beta_a: float,
    metric_a: str = "l2",
) -> Dict:
    """Compute all rate-distortion statistics.

    Args:
        embeddings_e: Semantic embeddings
        attributions_a: Attribution embeddings
        assignments: Component assignments
        path_probs: Path probabilities
        components: Component dictionary
        beta_e: Semantic distortion weight
        beta_a: Attribution distortion weight
        metric_a: Attribution metric ("l2" or "l1")

    Returns:
        Dictionary with all R-D statistics
    """
    component_ids = list(components.keys())

    # Compute masses
    W_c, W_total = compute_component_masses(assignments, path_probs, component_ids)
    P_bar = compute_normalized_masses(W_c, W_total)

    # Compute rate (entropy)
    H = compute_entropy(P_bar)

    # Compute distortions (always probability-weighted)
    D_e, Var_e = compute_semantic_distortion(
        embeddings_e, assignments, path_probs, components, P_bar, W_c,
    )
    D_a, Var_a = compute_attribution_distortion(
        attributions_a, assignments, path_probs, components, P_bar, W_c,
        metric=metric_a,
    )

    # Compute full objective
    L_RD = compute_rd_objective(H, D_e, D_a, beta_e, beta_a)

    return {
        'L_RD': L_RD,
        'H': H,
        'D_e': D_e,
        'D_a': D_a,
        'W_c': W_c,
        'W_total': W_total,
        'P_bar': P_bar,
        'Var_e': Var_e,
        'Var_a': Var_a,
        'beta_e': beta_e,
        'beta_a': beta_a,
        'metric_a': metric_a,
    }


def compute_squared_distances(
    data: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """Compute squared distances from data points to centers.

    Args:
        data: Data points (n_samples, d)
        centers: Center points (n_centers, d)

    Returns:
        Squared distances (n_samples, n_centers)
    """
    # Broadcasting: (n_samples, 1, d) - (n_centers, d) -> (n_samples, n_centers, d)
    diff = data[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sum(diff ** 2, axis=2)

def compute_l1_distances(
    data: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """Compute L1 distances from data points to centers.

    Args:
        data: Data points (n_samples, d)
        centers: Center points (n_centers, d)

    Returns:
        L1 distances (n_samples, n_centers)
    """
    diff = data[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sum(np.abs(diff), axis=2)


def compute_squared_distance_to_center(
    data: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    """Compute squared distances from data points to a single center.

    Args:
        data: Data points (n_samples, d)
        center: Single center point (d,)

    Returns:
        Squared distances (n_samples,)
    """
    diff = data - center
    return np.sum(diff ** 2, axis=1)

def compute_l1_distance_to_center(
    data: np.ndarray,
    center: np.ndarray,
) -> np.ndarray:
    """Compute L1 distances from data points to a single center.

    Args:
        data: Data points (n_samples, d)
        center: Single center point (d,)

    Returns:
        L1 distances (n_samples,)
    """
    diff = data - center
    return np.sum(np.abs(diff), axis=1)


def compute_squared_distance_pairwise(
    data: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """Compute squared distances from data points to corresponding centers.

    Unlike compute_squared_distances which computes all-to-all distances,
    this computes element-wise distances where each data[i] is compared to centers[i].

    Args:
        data: Data points (n_samples, d)
        centers: Center points (n_samples, d) - one center per data point

    Returns:
        Squared distances (n_samples,)
    """
    diff = data - centers
    return np.sum(diff ** 2, axis=1)

def compute_l1_distance_pairwise(
    data: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    """Compute L1 distances from data points to corresponding centers.

    Args:
        data: Data points (n_samples, d)
        centers: Center points (n_samples, d) - one center per data point

    Returns:
        L1 distances (n_samples,)
    """
    diff = data - centers
    return np.sum(np.abs(diff), axis=1)


def compute_component_variance(
    data: np.ndarray,
    center: np.ndarray,
    weights: np.ndarray,
    total_weight: float,
    metric: str = "l2",
) -> float:
    """Compute probability-weighted variance for a component.

    Args:
        data: Data points for this component (n_c, d)
        center: Component center (d,)
        weights: Probability weights (n_c,)
        total_weight: Sum of weights (W_c)
        metric: "l2" or "l1"

    Returns:
        Probability-weighted variance
    """
    if len(data) == 0:
        return 0.0
        
    if metric == "l2":
        dists = compute_squared_distance_to_center(data, center)
    elif metric == "l1":
        dists = compute_l1_distance_to_center(data, center)
    else:
        raise ValueError(f"Unknown metric {metric}")
        
    if total_weight == 0:
        return 0.0
    return float(np.sum(weights * dists) / total_weight)


# Aliases for backward compatibility
compute_mse_distances = compute_squared_distances
compute_mse_distance_to_center = compute_squared_distance_to_center
compute_mse_distance_pairwise = compute_squared_distance_pairwise


def binary_entropy(alpha: float, eps: float = 1e-10) -> float:
    """Compute binary entropy H_binary(α) = -α log α - (1-α) log(1-α).

    Args:
        alpha: Probability value in [0, 1]
        eps: Small constant to avoid log(0)

    Returns:
        Binary entropy
    """
    if alpha < eps or alpha > 1 - eps:
        return 0.0

    return -alpha * np.log(alpha) - (1 - alpha) * np.log(1 - alpha)


if __name__ == "__main__":
    # Test R-D objective computation
    np.random.seed(42)

    n_samples = 100
    d_e = 64
    d_a = 128
    K = 3

    embeddings_e = np.random.randn(n_samples, d_e)
    attributions_a = np.random.randn(n_samples, d_a)
    path_probs = np.random.rand(n_samples)
    path_probs = path_probs / np.sum(path_probs)

    # Create dummy components (probability-weighted initialization)
    assignments = list(np.random.choice([1, 2, 3], size=n_samples))
    components = {}
    for c in [1, 2, 3]:
        indices = [i for i, a in enumerate(assignments) if a == c]
        P_c = path_probs[indices]
        W_c = np.sum(P_c)
        if W_c > 0:
            components[c] = {
                'mu_e': np.sum(P_c[:, None] * embeddings_e[indices], axis=0) / W_c,
                'mu_a': np.sum(P_c[:, None] * attributions_a[indices], axis=0) / W_c,
            }

    # Compute R-D statistics (Standard)
    stats = compute_full_rd_statistics(
        embeddings_e, attributions_a, assignments, path_probs,
        components, beta_e=1.0, beta_a=1.0
    )

    print("Rate-Distortion Statistics:")
    print(f"  L_RD: {stats['L_RD']:.4f}")
    print(f"  H(C): {stats['H']:.4f}")
    print(f"  D^(e): {stats['D_e']:.4f}")
    print(f"  D^(a): {stats['D_a']:.4f}")
    print(f"  W_total: {stats['W_total']:.4f}")
    
    # Compute R-D statistics (L1)
    stats_l1 = compute_full_rd_statistics(
        embeddings_e, attributions_a, assignments, path_probs,
        components, beta_e=1.0, beta_a=1.0, metric_a="l1"
    )
    print("\nRate-Distortion Statistics (L1):")
    print(f"  L_RD: {stats_l1['L_RD']:.4f}")
    print(f"  H(C): {stats_l1['H']:.4f}")
    print(f"  D^(e): {stats_l1['D_e']:.4f}")
    print(f"  D^(a): {stats_l1['D_a']:.4f}")

    print("\nTest passed!")
