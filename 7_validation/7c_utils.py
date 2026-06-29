#!/usr/bin/env python3
"""7c_utils.py - Shared utilities for 7c validation modules.

This module contains extracted utility functions to eliminate code duplication across:
- 7c_steering.py
- 7c_hypotheses.py
- 7c_metrics.py
- 7c_graph.py

Organized by category:
1. Constants
2. Steering computation utilities
3. Data transformation utilities
4. Metrics computation utilities
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Union, Optional
from dataclasses import dataclass

import numpy as np
import torch


# =============================================================================
# Constants
# =============================================================================

# Numerical stability
EPSILON_SMALL = 1e-10  # Zero-checking H_c values
EPSILON_TINY = 1e-8    # Division by zero prevention
CLIP_MAX = 50.0        # Log-delta upper bound
CLIP_MIN = -50.0       # Log-delta lower bound

# Defaults
DEFAULT_BATCH_SIZE = 128
DEFAULT_TOP_B = 10
DEFAULT_MAX_CLUSTER_SAMPLES = 30


# =============================================================================
# Steering Computation Utilities
# =============================================================================

def compute_steering_delta(
    current_val: Union[float, torch.Tensor],
    h_c_val: float,
    h_c_norm: float,
    steering_method: str,
    epsilon: float
) -> Union[float, torch.Tensor]:
    """Compute steering delta based on method.

    Supports 5 steering methods:
    - additive: delta = epsilon * (h_c_val / h_c_norm)
    - multiplicative: delta = current_val * epsilon * (h_c_val / h_c_norm)
    - absolute: delta = epsilon - current_val
    - sign: delta = sign(h_c_val) * current_val * epsilon
    - scaling: delta = epsilon * current_val

    Args:
        current_val: Current activation value(s). Can be scalar or tensor.
        h_c_val: H_c value for this feature (semantic graph weight)
        h_c_norm: ||H_c|| normalization factor (L2 norm of all H_c values)
        steering_method: One of "additive", "multiplicative", "absolute", "sign", "scaling"
        epsilon: Steering strength parameter

    Returns:
        Delta scalar(s) to apply. Same type as current_val (float or tensor).

    Raises:
        ValueError: If steering_method is not recognized

    Examples:
        >>> # Additive steering
        >>> compute_steering_delta(2.0, 0.5, 1.0, "additive", 0.1)
        0.05

        >>> # Multiplicative steering with tensor
        >>> vals = torch.tensor([1.0, 2.0])
        >>> compute_steering_delta(vals, 0.5, 1.0, "multiplicative", 0.1)
        tensor([0.05, 0.10])
    """
    if steering_method == "additive":
        # NOTE: Intervention code uses additive steering
        delta_scalar = epsilon * (h_c_val / h_c_norm)
    elif steering_method == "multiplicative":
        # Multiplicative: scales current activation proportional to relative H_c strength
        delta_scalar = current_val * epsilon * (h_c_val / h_c_norm)
    elif steering_method == "absolute":
        # Absolute: force activation to target value (epsilon)
        delta_scalar = epsilon - current_val
    elif steering_method == "sign":
        # Sign: scale current activation by sign of H_c
        delta_scalar = np.sign(h_c_val) * current_val * epsilon
    elif steering_method == "scaling":
        # Scaling: scale current activation by epsilon (simple gain control)
        delta_scalar = epsilon * current_val
    else:
        raise ValueError(f"Unknown steering method: {steering_method}")

    return delta_scalar


def compute_steering_delta_vectorized(
    current_vals: torch.Tensor,
    h_c_vals: torch.Tensor,
    h_c_norms: torch.Tensor,
    steering_method: str,
    epsilon: float
) -> torch.Tensor:
    """Vectorized steering delta computation for batched operations.

    Args:
        current_vals: Current activation values [n_entries]
        h_c_vals: H_c values [n_entries]
        h_c_norms: H_c normalization factors [n_entries]
        steering_method: One of "additive", "multiplicative", "absolute", "sign", "scaling"
        epsilon: Steering strength parameter

    Returns:
        Delta scalars [n_entries]
    """
    if steering_method == "additive":
        return epsilon * (h_c_vals / h_c_norms)
    elif steering_method == "multiplicative":
        return current_vals * epsilon * (h_c_vals / h_c_norms)
    elif steering_method == "absolute":
        return epsilon - current_vals
    elif steering_method == "sign":
        return torch.sign(h_c_vals) * current_vals * epsilon
    elif steering_method == "scaling":
        return epsilon * current_vals
    else:
        raise ValueError(f"Unknown steering method: {steering_method}")


# =============================================================================
# Data Transformation Utilities
# =============================================================================

def build_branches_from_data(
    branches_data: Dict[str, Any],
    assignments: List[int],
) -> List[Dict]:
    """Build branches list from branches.json data and cluster assignments.

    Reconstructs the full branch structure with cluster assignments.

    Args:
        branches_data: Loaded branches.json content with keys:
            - prefix: Prefix text
            - prefix_tokens_with_bos: Prefix token IDs including BOS
            - continuations: List of continuation data
        assignments: Cluster assignment for each branch (ordered)

    Returns:
        List of branch dictionaries with keys:
            - sequence: Full text sequence (prefix + continuation)
            - full_token_ids: Full token ID sequence including BOS
            - continuation_token_ids: Continuation token IDs only
            - cluster_id: Assigned cluster ID
            - branch_id: Sequential branch index
    """
    branches = []
    branch_idx = 0
    prefix_text = branches_data.get("prefix", "")
    prefix_tokens_with_bos = branches_data.get("prefix_tokens_with_bos", [])

    for cont in branches_data.get("continuations", []):
        cluster_id = assignments[branch_idx] if branch_idx < len(assignments) else 0
        full_token_ids = cont.get("full_token_ids")
        if full_token_ids is None:
            full_token_ids = prefix_tokens_with_bos + cont.get("token_ids", [])

        branches.append({
            "sequence": prefix_text + cont.get("text", ""),
            "full_token_ids": full_token_ids,
            "continuation_token_ids": cont.get("token_ids", []),
            "cluster_id": int(cluster_id),
            "branch_id": branch_idx,
        })
        branch_idx += 1

    return branches


# =============================================================================
# Metrics Computation Utilities
# =============================================================================

def compute_spearman_correlation(
    x: np.ndarray,
    y: np.ndarray
) -> float:
    """Compute Spearman rank correlation between x and y.

    Spearman correlation measures monotonic relationship (not just linear).
    It's more robust to outliers and non-linear relationships than Pearson.

    Args:
        x: Independent variable array (e.g., epsilon values)
        y: Dependent variable array (e.g., mean effects)

    Returns:
        Spearman correlation coefficient. Returns 0.0 if:
        - Correlation is NaN
        - Arrays have insufficient variance (all values identical)
        - Arrays have fewer than 3 elements
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Check if we have enough data points
    if len(x) < 3 or len(y) < 3:
        return 0.0

    # Check if all values are identical (no variance)
    if len(np.unique(x)) == 1 or len(np.unique(y)) == 1:
        return 0.0

    # Compute ranks
    def rankdata(arr):
        """Assign ranks to data (1-based, ties get average rank)."""
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(arr) + 1)
        return ranks

    rx = rankdata(x)
    ry = rankdata(y)

    # Pearson correlation of ranks = Spearman correlation
    corr_matrix = np.corrcoef(rx, ry)
    spearman = corr_matrix[0, 1]

    # Handle NaN
    if np.isnan(spearman):
        return 0.0

    return float(spearman)


def compute_correlation_and_r2(
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[float, float]:
    """Compute Pearson correlation and R² between x and y.

    Handles edge cases: NaN values, insufficient variance, and identical values.

    Args:
        x: Independent variable array (e.g., epsilon values)
        y: Dependent variable array (e.g., mean effects)

    Returns:
        (correlation, r_squared) tuple. Returns (0.0, 0.0) if:
        - Correlation is NaN
        - Arrays have insufficient variance (all values identical)
        - Arrays have fewer than 3 elements
    """
    # Check if we have enough data points
    if len(x) < 3 or len(y) < 3:
        return 0.0, 0.0

    # Check if all values are identical (no variance)
    if len(set(x)) == 1 or len(set(y)) == 1:
        return 0.0, 0.0

    # Compute correlation
    corr_matrix = np.corrcoef(x, y)
    corr = corr_matrix[0, 1]

    # Handle NaN
    if np.isnan(corr):
        return 0.0, 0.0

    # Compute R²
    r2 = corr ** 2

    return float(corr), float(r2)


def compute_all_correlations(
    x: np.ndarray,
    y: np.ndarray
) -> Tuple[float, float, float]:
    """Compute Pearson correlation, R², and Spearman correlation.

    Convenience function that returns all correlation metrics at once.

    Args:
        x: Independent variable array (e.g., epsilon values)
        y: Dependent variable array (e.g., mean effects)

    Returns:
        (pearson_corr, r_squared, spearman_corr) tuple.
    """
    pearson, r2 = compute_correlation_and_r2(x, y)
    spearman = compute_spearman_correlation(x, y)
    return pearson, r2, spearman


# =============================================================================
# Batch Processing Utilities (Phase 2)
# =============================================================================

def create_padded_batch(
    batch_token_ids: List[List[int]],
    pad_token_id: int,
    device: torch.device,
    max_seq_len: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create padded batch tensor and attention mask.

    Args:
        batch_token_ids: List of token ID sequences [batch_size, varying_seq_len]
        pad_token_id: Token ID to use for padding
        device: Target device for tensors
        max_seq_len: Fixed padding length. If None, uses max length in batch.
                    Sequences longer than max_seq_len will be truncated.

    Returns:
        (batch_tokens, attention_mask) tuple:
        - batch_tokens: [batch_size, max_len] padded token tensor
        - attention_mask: [batch_size, max_len] (1 for valid, 0 for padding)

    Examples:
        >>> tokens = [[1, 2, 3], [4, 5]]
        >>> batch, mask = create_padded_batch(tokens, pad_token_id=0, device=torch.device('cpu'))
        >>> batch.shape
        torch.Size([2, 3])
        >>> mask.tolist()
        [[1, 1, 1], [1, 1, 0]]
    """
    batch_size = len(batch_token_ids)
    batch_max_len = max(len(tokens) for tokens in batch_token_ids)
    max_len = max_seq_len if max_seq_len is not None else batch_max_len

    # Truncate sequences that exceed max_len
    if batch_max_len > max_len:
        batch_token_ids = [tokens[:max_len] for tokens in batch_token_ids]

    # Create padded tensor and attention mask
    batch_tokens = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)

    # OPTIMIZATION: Convert all token sequences to single tensor then copy
    # This is faster than creating tensors in a loop
    for i, tokens in enumerate(batch_token_ids):
        seq_len = len(tokens)
        if seq_len > 0:
            # Use torch.as_tensor for potential zero-copy from numpy/list
            batch_tokens[i, :seq_len] = torch.as_tensor(tokens, dtype=torch.long, device=device)
            attention_mask[i, :seq_len] = 1

    return batch_tokens, attention_mask


def get_encoder_weights(
    model,
    layer: int,
    feat_ids_tensor: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get encoder weights for given feature IDs.

    Handles both TranscoderSet and CrossLayerTranscoder architectures automatically.

    Args:
        model: ReplacementModel with transcoders
        layer: Layer index
        feat_ids_tensor: Feature IDs to extract [n_features]

    Returns:
        (W_enc_subset, b_enc_subset) tuple:
        - W_enc_subset: [n_features, d_model] encoder weight matrix
        - b_enc_subset: [n_features] encoder bias vector

    Examples:
        >>> # With TranscoderSet
        >>> feat_ids = torch.tensor([0, 5, 10], device=device)
        >>> W_enc, b_enc = get_encoder_weights(model, layer=3, feat_ids_tensor=feat_ids)
        >>> W_enc.shape  # doctest: +SKIP
        torch.Size([3, d_model])
    """
    if hasattr(model.transcoders, "transcoders"):
        # TranscoderSet (Single Layer Transcoders)
        W_enc_full = model.transcoders[layer].W_enc
        b_enc_subset = model.transcoders[layer].b_enc[feat_ids_tensor]
    else:
        # CrossLayerTranscoder
        W_enc_full = model.transcoders._get_encoder_weights(layer)
        b_enc_subset = model.transcoders.b_enc[layer][feat_ids_tensor]

    W_enc_subset = W_enc_full[feat_ids_tensor]

    return W_enc_subset, b_enc_subset


# =============================================================================
# Configuration Utilities (Phase 2)
# =============================================================================

@dataclass
class SweepKey:
    """Type-safe sweep configuration key.

    Format: (method, hc_selection, top_B)
    """
    method: str
    hc_selection: str
    top_B: int

    @classmethod
    def from_string(cls, key_str: str) -> 'SweepKey':
        """Parse sweep key from string format.

        Format:
        - "multiplicative_full_B10"

        Args:
            key_str: Key string to parse

        Returns:
            SweepKey instance

        Examples:
            >>> key = SweepKey.from_string("multiplicative_full_B10")
            >>> key.method, key.top_B
            ('multiplicative', 10)
        """
        parts = key_str.split('_')
        if len(parts) < 3:
            raise ValueError(f"Invalid sweep key format: {key_str}")

        method = parts[0]
        hc_sel = parts[1]

        # Extract top_B (format: B10 or just 10)
        b_part = parts[2]
        top_B = int(b_part[1:]) if b_part.startswith('B') else int(b_part)

        return cls(method, hc_sel, top_B)

    @classmethod
    def from_tuple(cls, key_tuple: Tuple) -> 'SweepKey':
        """Create from tuple format.

        Args:
            key_tuple: (method, hc_sel, top_B)
        """
        if len(key_tuple) >= 3:
            return cls(key_tuple[0], key_tuple[1], key_tuple[2])
        raise ValueError(f"Invalid tuple length: {len(key_tuple)}")

    def to_tuple(self) -> Tuple:
        """Convert to tuple format for dict keys."""
        return (self.method, self.hc_selection, self.top_B)

    def to_string(self) -> str:
        """Convert to string format."""
        return f"{self.method}_{self.hc_selection}_B{self.top_B}"


def validate_and_normalize_sweep_config(
    sweep_dict: Dict[str, Any],
    defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Validate and normalize a single sweep configuration.

    Ensures required fields exist, converts scalars to lists, applies defaults.

    Args:
        sweep_dict: Sweep configuration dictionary
        defaults: Default values to apply for missing keys

    Returns:
        Normalized sweep configuration

    Example:
        >>> sweep = {"steering_method": "multiplicative", "top_B": 10}
        >>> defaults = {"h_c_selections": ["full"], "epsilon_values": [-1.0, 0.0, 1.0]}
        >>> normalized = validate_and_normalize_sweep_config(sweep, defaults)
        >>> normalized["h_c_selections"]
        ['full']
    """
    defaults = defaults or {}
    normalized = sweep_dict.copy()

    # Set name if missing
    normalized.setdefault("name", f"sweep_{normalized.get('steering_method', 'unknown')}")

    # Normalize h_c_selections (handle both old and new key names)
    if "h_c_selections" not in normalized and "hc_selections" in normalized:
        normalized["h_c_selections"] = normalized.pop("hc_selections")
    normalized.setdefault("h_c_selections", defaults.get("h_c_selections", ["full"]))

    # Normalize epsilon_values
    if "epsilon_values" not in normalized and "epsilons" in normalized:
        normalized["epsilon_values"] = normalized.pop("epsilons")
    normalized.setdefault("epsilon_values", defaults.get("epsilon_values", [-1.0, 0.0, 1.0]))

    # Normalize other fields
    normalized.setdefault("top_B", defaults.get("top_B", [10]))
    normalized.setdefault("feature_selection", defaults.get("feature_selection", "magnitude"))

    # Convert scalars to lists
    for key in ["h_c_selections", "top_B", "epsilon_values", "feature_selection"]:
        if key in normalized and not isinstance(normalized[key], list):
            normalized[key] = [normalized[key]]

    return normalized
