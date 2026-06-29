"""Data loading and manipulation utilities."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON serialization.
    
    This handles both values and dictionary keys containing numpy types.
    """
    if isinstance(obj, dict):
        return {
            convert_numpy_types(k): convert_numpy_types(v) 
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def load_cloze_data(cloze_file: Path) -> List[Dict[str, Any]]:
    """Load cloze data from a JSON file.

    Args:
        cloze_file: Path to cloze JSON file

    Returns:
        List of cloze dictionaries
    """
    with open(cloze_file, 'r') as f:
        return json.load(f)


def load_attribution_metadata(metadata_file: Path) -> Dict[str, Any]:
    """Load attribution graph metadata.

    Args:
        metadata_file: Path to attribution_metadata.json

    Returns:
        Metadata dictionary
    """
    with open(metadata_file, 'r') as f:
        return json.load(f)


def save_json(data: Any, output_path: Path, indent: int = 2):
    """Save data to JSON file.
    
    Automatically converts numpy types (including dictionary keys) to
    Python native types for JSON serialization.

    Args:
        data: Data to save
        output_path: Output file path
        indent: JSON indentation
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert numpy types in both keys and values before serialization
    converted_data = convert_numpy_types(data)
    with open(output_path, 'w') as f:
        json.dump(converted_data, f, indent=indent)


def load_json(input_path: Path) -> Any:
    """Load data from JSON file.

    Args:
        input_path: Input file path

    Returns:
        Loaded data
    """
    with open(input_path, 'r') as f:
        return json.load(f)


def save_torch(data: Any, output_path: Path):
    """Save data using torch.save.

    Args:
        data: Data to save
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)


def load_torch(input_path: Path) -> Any:
    """Load data using torch.load.

    Args:
        input_path: Input file path

    Returns:
        Loaded data
    """
    return torch.load(input_path, weights_only=False)


def compute_path_probability(
    token_ids: List[int],
    logprobs: List[float]
) -> float:
    """Compute path probability P_n from token logprobs.

    Args:
        token_ids: List of token IDs in the continuation
        logprobs: List of log probabilities for each token

    Returns:
        Path probability (product of individual token probs)
    """
    return np.exp(np.sum(logprobs))


def normalize_probabilities(
    probs: np.ndarray,
    temperature: float = 1.0
) -> np.ndarray:
    """Normalize and optionally temperature-scale probabilities.

    Args:
        probs: Array of probabilities
        temperature: Temperature parameter (β in P_n^β)

    Returns:
        Normalized probabilities
    """
    if temperature != 1.0:
        probs = np.power(probs, temperature)

    total = np.sum(probs)
    if total > 0:
        return probs / total
    else:
        return probs


def select_random_clozes(
    cloze_data: List[Dict[str, Any]],
    n_samples: int,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """Select random subset of clozes for testing.

    Args:
        cloze_data: Full cloze dataset
        n_samples: Number of samples to select
        seed: Random seed

    Returns:
        Subset of clozes
    """
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(cloze_data), size=min(n_samples, len(cloze_data)), replace=False)
    return [cloze_data[i] for i in indices]


def extract_prefix_from_cloze(cloze: Dict[str, Any]) -> str:
    """Extract prefix text from cloze dictionary.

    Args:
        cloze: Cloze dictionary with 'context' and 'target' fields

    Returns:
        Prefix text (context before target)
    """
    # Assuming cloze has format like: "context [MASK] rest"
    # Extract the part before [MASK] or target
    context = cloze.get('context', '')
    target = cloze.get('target', '')

    # Simple extraction: return context before target position
    if '[MASK]' in context:
        return context.split('[MASK]')[0].strip()
    else:
        return context.strip()


def reconstruct_active_features(
    decoder_locations: Any,
    selected_features: Any,
    activation_matrix: Any = None,
    return_numpy: bool = True,
) -> Any:
    """Reconstruct active_features (N, 3) from decoder_locations and feature info.

    This is the canonical function for reconstructing active_features from the
    Stage 3 prefix_context format. Both Stage 6 and Stage 7 should use this
    function to ensure consistent handling.

    IMPORTANT: The feature IDs come from one of two sources:
    1. If activation_matrix is provided: feat_ids = activation_matrix.indices()[2][selected_features]
       This is needed because selected_features contains INDICES into the sparse tensor,
       not the actual feature IDs.
    2. If activation_matrix is None: we assume selected_features already contains actual feat_ids
       (for backward compatibility with pre-processed data).

    Args:
        decoder_locations: (2, N) array/tensor where row 0 = layer, row 1 = position
        selected_features: (N,) array/tensor. If activation_matrix is provided, these are
            indices into the sparse tensor. Otherwise, assumed to be actual feature IDs.
        activation_matrix: Optional sparse tensor (n_layers, n_pos, d_transcoder).
            If provided, actual feature IDs are extracted from indices()[2].
        return_numpy: If True, return numpy array. If False, return torch tensor.

    Returns:
        active_features: (N, 3) array/tensor where each row is [layer, pos, feat_id]
    """
    # Convert decoder_locations to numpy
    if hasattr(decoder_locations, 'cpu'):
        dec_locs = decoder_locations.cpu().numpy()
    elif hasattr(decoder_locations, '__array__'):
        dec_locs = np.array(decoder_locations)
    else:
        dec_locs = np.array(decoder_locations)

    # Convert selected_features to numpy
    if hasattr(selected_features, 'cpu'):
        sel_feats = selected_features.cpu().numpy()
    elif hasattr(selected_features, '__array__'):
        sel_feats = np.array(selected_features)
    else:
        sel_feats = np.array(selected_features)

    # Validate shapes
    if dec_locs.shape[0] != 2:
        raise ValueError(f"decoder_locations must have shape (2, N), got {dec_locs.shape}")
    if dec_locs.shape[1] != sel_feats.shape[0]:
        raise ValueError(f"Shape mismatch: decoder_locations has {dec_locs.shape[1]} features, "
                        f"selected_features has {sel_feats.shape[0]}")

    N = sel_feats.shape[0]

    # Get actual feature IDs
    if activation_matrix is not None:
        # selected_features contains indices into the sparse tensor
        # Extract actual feature IDs from the sparse tensor indices
        if hasattr(activation_matrix, 'indices'):
            all_feat_ids = activation_matrix.indices()[2]  # (nnz,) of actual feature IDs
            if hasattr(all_feat_ids, 'cpu'):
                all_feat_ids = all_feat_ids.cpu().numpy()
            else:
                all_feat_ids = np.array(all_feat_ids)
            feat_ids = all_feat_ids[sel_feats]
        else:
            # Fallback if activation_matrix doesn't have indices method
            feat_ids = sel_feats
    else:
        # Assume selected_features already contains actual feat_ids
        feat_ids = sel_feats

    # Reconstruct active_features as (N, 3) = [layer, pos, feat_id]
    active_features = np.zeros((N, 3), dtype=np.int64)
    active_features[:, 0] = dec_locs[0, :]  # layer
    active_features[:, 1] = dec_locs[1, :]  # pos
    active_features[:, 2] = feat_ids        # actual feature ID

    if return_numpy:
        return active_features
    else:
        return torch.from_numpy(active_features)


def create_continuation_dataset(
    prefix: str,
    continuations: List[str],
    path_probs: List[float],
    first_tokens: List[int],
    embeddings: Optional[np.ndarray] = None,
    attributions: Optional[np.ndarray] = None,
    logits: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Create a structured dataset for a single prefix.

    Args:
        prefix: The prefix text (x_{1:t-1})
        continuations: List of continuation texts
        path_probs: List of path probabilities P_n
        first_tokens: List of first token IDs s_n
        embeddings: Optional semantic embeddings e_n
        attributions: Optional attribution embeddings a_n
        logits: Optional logit vectors z_n
        valid_mask: Optional boolean mask for valid samples

    Returns:
        Dataset dictionary
    """
    n_samples = len(continuations)

    if valid_mask is None:
        valid_mask = np.ones(n_samples, dtype=bool)

    dataset = {
        'prefix': prefix,
        'n_samples': n_samples,
        'n_valid': np.sum(valid_mask),
        'continuations': continuations,
        'path_probs': path_probs,
        'first_tokens': first_tokens,
        'valid_mask': valid_mask.tolist(),
    }

    if embeddings is not None:
        dataset['embeddings'] = embeddings

    if attributions is not None:
        dataset['attributions'] = attributions

    if logits is not None:
        dataset['logits'] = logits

    return dataset
