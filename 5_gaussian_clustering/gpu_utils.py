"""GPU-accelerated utilities for Rate-Distortion clustering.

Provides optional GPU acceleration using PyTorch for large datasets.
Falls back to NumPy when GPU is unavailable or for small datasets.

Usage:
    from gpu_utils import get_compute_backend
    backend = get_compute_backend(use_gpu=True, n_samples=50000)

    # Use backend functions
    assignments = backend.rd_e_step(embeddings_e, attributions_a, ...)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

# Try to import torch, but make it optional
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class NumPyBackend:
    """NumPy backend for CPU computation."""

    def __init__(self):
        self.name = "numpy"

    def rd_e_step(
        self,
        embeddings_e: np.ndarray,
        attributions_a: np.ndarray,
        centers_e: np.ndarray,
        centers_a: np.ndarray,
        rate_costs: np.ndarray,
        beta_e: float,
        beta_a: float,
    ) -> np.ndarray:
        """Vectorized E-step using NumPy broadcasting.

        Uses raw squared distances (no MSE normalization).

        Args:
            embeddings_e: (N, d_e)
            attributions_a: (N, d_a)
            centers_e: (K, d_e)
            centers_a: (K, d_a)
            rate_costs: (K,)
            beta_e, beta_a: distortion weights

        Returns:
            best_indices: (N,) array of best component indices
        """
        # Compute squared distances using broadcasting
        diff_e = embeddings_e[:, np.newaxis, :] - centers_e[np.newaxis, :, :]
        dist_e_sq = np.sum(diff_e ** 2, axis=2)

        diff_a = attributions_a[:, np.newaxis, :] - centers_a[np.newaxis, :, :]
        dist_a_sq = np.sum(diff_a ** 2, axis=2)

        # Total cost: (N, K)
        total_cost = rate_costs[np.newaxis, :] + beta_e * dist_e_sq + beta_a * dist_a_sq

        # Find best assignment
        return np.argmin(total_cost, axis=1)

    def compute_distortions_batch(
        self,
        embeddings: np.ndarray,
        centers: np.ndarray,
        assignments: np.ndarray,
        path_probs: np.ndarray,
        K: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute distortions for all components in batch.

        Args:
            embeddings: (N, d)
            centers: (K, d)
            assignments: (N,) component indices
            path_probs: (N,)
            K: number of components

        Returns:
            Tuple of (W_c array (K,), Var_c array (K,))
        """
        W_c = np.zeros(K)
        Var_c = np.zeros(K)

        for c in range(K):
            mask = assignments == c
            if not np.any(mask):
                continue

            W_c[c] = np.sum(path_probs[mask])
            if W_c[c] == 0:
                continue

            diff = embeddings[mask] - centers[c]
            sq_dists = np.sum(diff ** 2, axis=1)
            Var_c[c] = np.sum(path_probs[mask] * sq_dists) / W_c[c]

        return W_c, Var_c


class TorchBackend:
    """PyTorch backend for GPU computation."""

    def __init__(self, device: str = 'cuda'):
        self.name = "torch"
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def rd_e_step(
        self,
        embeddings_e: np.ndarray,
        attributions_a: np.ndarray,
        centers_e: np.ndarray,
        centers_a: np.ndarray,
        rate_costs: np.ndarray,
        beta_e: float,
        beta_a: float,
        metric_a: str = "l2",
    ) -> np.ndarray:
        """GPU-accelerated E-step using PyTorch.

        Supports both L2 (squared) and L1 distances.

        Args:
            embeddings_e: (N, d_e)
            attributions_a: (N, d_a)
            centers_e: (K, d_e)
            centers_a: (K, d_a)
            rate_costs: (K,)
            beta_e, beta_a: distortion weights
            metric_a: "l2" or "l1" for attribution distance

        Returns:
            best_indices: (N,) numpy array of best component indices
        """
        # Move to GPU
        e = torch.from_numpy(embeddings_e).float().to(self.device)
        a = torch.from_numpy(attributions_a).float().to(self.device)
        mu_e = torch.from_numpy(centers_e).float().to(self.device)
        mu_a = torch.from_numpy(centers_a).float().to(self.device)
        rates = torch.from_numpy(rate_costs).float().to(self.device)

        # Embeddings always use L2 squared distance
        # cdist returns (N, K) for (N, d) and (K, d)
        dist_e_sq = torch.cdist(e, mu_e, p=2) ** 2
        
        # Attributions use specified metric
        if metric_a == "l1":
            dist_a = torch.cdist(a, mu_a, p=1)  # L1 distance (not squared)
        else:
            dist_a = torch.cdist(a, mu_a, p=2) ** 2  # L2 squared

        # Total cost: (N, K)
        total_cost = rates.unsqueeze(0) + beta_e * dist_e_sq + beta_a * dist_a

        # Find best assignment
        best_indices = torch.argmin(total_cost, dim=1)

        return best_indices.cpu().numpy()

    def compute_distortions_batch(
        self,
        embeddings: np.ndarray,
        centers: np.ndarray,
        assignments: np.ndarray,
        path_probs: np.ndarray,
        K: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated distortion computation.

        Args:
            embeddings: (N, d)
            centers: (K, d)
            assignments: (N,) component indices
            path_probs: (N,)
            K: number of components

        Returns:
            Tuple of (W_c array (K,), Var_c array (K,))
        """
        # Move to GPU
        emb = torch.from_numpy(embeddings).float().to(self.device)
        ctr = torch.from_numpy(centers).float().to(self.device)
        assign = torch.from_numpy(assignments).long().to(self.device)
        probs = torch.from_numpy(path_probs).float().to(self.device)

        W_c = torch.zeros(K, device=self.device)
        Var_c = torch.zeros(K, device=self.device)

        for c in range(K):
            mask = assign == c
            if not torch.any(mask):
                continue

            W_c[c] = torch.sum(probs[mask])
            if W_c[c] == 0:
                continue

            diff = emb[mask] - ctr[c]
            sq_dists = torch.sum(diff ** 2, dim=1)
            Var_c[c] = torch.sum(probs[mask] * sq_dists) / W_c[c]

        return W_c.cpu().numpy(), Var_c.cpu().numpy()


def get_compute_backend(
    use_gpu: bool = True,
    n_samples: int = 0,
    min_samples_for_gpu: int = 50,  # Lowered - large GPUs have negligible transfer overhead
    device: str = 'cuda',
) -> 'NumPyBackend | TorchBackend':
    """Get the appropriate compute backend.

    Args:
        use_gpu: Whether to prefer GPU if available
        n_samples: Number of samples (for deciding if GPU is worth the overhead)
        min_samples_for_gpu: Minimum samples to use GPU (default 50)
        device: CUDA device string

    Returns:
        Backend instance (NumPyBackend or TorchBackend)
    """
    if use_gpu and TORCH_AVAILABLE and n_samples >= min_samples_for_gpu:
        try:
            backend = TorchBackend(device)
            # Test GPU availability
            if torch.cuda.is_available():
                return backend
        except Exception:
            pass

    return NumPyBackend()


# Convenience functions for direct use
def rd_e_step_gpu(
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    component_ids: List[int],
    components: Dict[int, Dict],
    P_bar: Dict[int, float],
    beta_e: float,
    beta_a: float,
    use_gpu: bool = True,
) -> List[int]:
    """GPU-accelerated E-step with automatic backend selection.

    This is a drop-in replacement for rd_e_step that automatically
    uses GPU when beneficial.

    Args:
        Same as em_loop.rd_e_step plus use_gpu flag

    Returns:
        assignments: List of component assignments
    """
    n_samples = embeddings_e.shape[0]
    K = len(component_ids)
    eps = 1e-10

    if K == 0:
        return [0] * n_samples

    # Stack centers and rate costs
    centers_e = np.array([components[c]['mu_e'] for c in component_ids])
    centers_a = np.array([components[c]['mu_a'] for c in component_ids])
    rate_costs = np.array([-np.log(P_bar.get(c, eps) + eps) for c in component_ids])

    # Get backend
    backend = get_compute_backend(use_gpu=use_gpu, n_samples=n_samples)

    # Compute assignments
    best_indices = backend.rd_e_step(
        embeddings_e, attributions_a,
        centers_e, centers_a, rate_costs,
        beta_e, beta_a
    )

    # Map back to component IDs
    assignments = [component_ids[i] for i in best_indices]

    return assignments


if __name__ == "__main__":
    # Test GPU utils
    np.random.seed(42)

    N = 50000
    K = 20
    d_e = 256
    d_a = 8192

    print(f"Testing GPU utils with N={N}, K={K}, d_e={d_e}, d_a={d_a}")

    # Generate test data
    embeddings_e = np.random.randn(N, d_e).astype(np.float32)
    attributions_a = np.random.randn(N, d_a).astype(np.float32)
    centers_e = np.random.randn(K, d_e).astype(np.float32)
    centers_a = np.random.randn(K, d_a).astype(np.float32)
    rate_costs = np.random.randn(K).astype(np.float32)

    beta_e = 15.0
    beta_a = 15.0

    # Test NumPy backend
    print("\nTesting NumPy backend...")
    import time

    numpy_backend = NumPyBackend()
    start = time.time()
    result_numpy = numpy_backend.rd_e_step(
        embeddings_e, attributions_a,
        centers_e, centers_a, rate_costs,
        beta_e, beta_a
    )
    numpy_time = time.time() - start
    print(f"  NumPy time: {numpy_time:.3f}s")

    # Test Torch backend
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print("\nTesting Torch GPU backend...")
        torch_backend = TorchBackend('cuda')

        # Warm up
        _ = torch_backend.rd_e_step(
            embeddings_e[:100], attributions_a[:100],
            centers_e, centers_a, rate_costs,
            beta_e, beta_a
        )

        start = time.time()
        result_torch = torch_backend.rd_e_step(
            embeddings_e, attributions_a,
            centers_e, centers_a, rate_costs,
            beta_e, beta_a
        )
        torch_time = time.time() - start
        print(f"  Torch GPU time: {torch_time:.3f}s")
        print(f"  Speedup: {numpy_time / torch_time:.1f}x")

        # Verify correctness
        match = np.mean(result_numpy == result_torch)
        print(f"  Results match: {match * 100:.1f}%")
    else:
        print("\nTorch GPU not available, skipping GPU test")

    print("\nGPU utils test complete!")
