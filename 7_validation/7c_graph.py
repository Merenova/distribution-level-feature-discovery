#!/usr/bin/env python3
"""7c_graph.py - Semantic graph and feature selection utilities.

This module contains functions for:
1. Loading attribution context
2. Computing semantic graphs with different strategies
3. Feature selection (by magnitude, distinctiveness)
4. Decoder/Encoder weight precomputation and caching
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch

from utils.data_utils import reconstruct_active_features
from utils.model_backend import get_model_dtype

# Import shared utilities (using same pattern as other modules)
import importlib.util
from pathlib import Path as _Path

_utils_spec = importlib.util.spec_from_file_location("7c_utils", _Path(__file__).parent / "7c_utils.py")
_utils_module = importlib.util.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(_utils_module)
utils = _utils_module


# =============================================================================
# Attribution Context Loading
# =============================================================================

def load_attribution_context(
    attribution_graphs_dir: Path,
    prefix_id: str,
    use_continuation_attribution: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load attribution context (active_features, selected_features) from either format.

    Args:
        attribution_graphs_dir: Directory containing attribution files
        prefix_id: Prefix identifier
        use_continuation_attribution: If True, load from prefix_context.pt (new format).
                                      If False, load from graph.pt (legacy format).

    Returns:
        Tuple of (active_features, selected_features) tensors for feature selection.
        - active_features: (N, 3) tensor mapping feature index to [layer, pos, feat_id]
        - selected_features: (N,) tensor of indices (identity mapping for new format)
    """
    if use_continuation_attribution:
        # New format: load from prefix_context.pt
        context_file = attribution_graphs_dir / f"{prefix_id}_prefix_context.pt"
        context_data = torch.load(context_file, weights_only=False)

        # Use shared utility to reconstruct active_features (N, 3) from
        # decoder_locations (2, N) + selected_features (N,) + activation_matrix
        # NOTE: selected_features contains indices into the sparse tensor, not actual feature IDs.
        #       activation_matrix is needed to extract the actual feature IDs.
        active_features = reconstruct_active_features(
            context_data["decoder_locations"],
            context_data["selected_features"],
            activation_matrix=context_data.get("activation_matrix", None),
            return_numpy=False  # Return as torch tensor
        )

        # selected_features is identity mapping since active_features already contains
        # only the selected features
        n_features = active_features.shape[0]
        selected_features = torch.arange(n_features, dtype=torch.long)

        return active_features, selected_features
    else:
        raise ValueError("Only prefix context format is supported")


# =============================================================================
# Semantic Graph Strategies
# =============================================================================

def compute_semantic_graphs(
    components: Dict[str, Any],
    H_0: np.ndarray,
    logger=None,
    center_Hc: bool = True
) -> Dict[int, np.ndarray]:
    """Compute semantic graphs from clustering components.
    
    Args:
        components: Dict mapping cluster_id -> {mu_a: [...], ...}
        H_0: Global mean attribution vector
        logger: Optional logger
        center_Hc: If True (default), use Delta_H_c = mu_a (centered, H_0 subtracted).
                   If False, use H_c = H_0 + mu_a (non-centered, raw attribution centroid).
    
    Returns:
        Dict mapping cluster_id -> H_c array
    """
    delta_H_c_dict = {}
    for c_str, comp in components.items():
        c = int(c_str)
        if "mu_a" in comp:
            delta_H_c_dict[c] = np.array(comp["mu_a"])

    if not delta_H_c_dict:
        return {}

    semantic_graphs = {}
    for c, delta_H_c in delta_H_c_dict.items():
        if center_Hc:
            # Use centered H_c (mu_a = cluster_centroid - H_0)
            semantic_graphs[c] = delta_H_c
        else:
            # Use non-centered H_c (add H_0 back to get raw centroid)
            if H_0 is not None and len(H_0) > 0:
                semantic_graphs[c] = H_0 + delta_H_c
            else:
                semantic_graphs[c] = delta_H_c

    if logger:
        if center_Hc:
            logger.info("  Using H_c strategy: Delta_H_c (centered, H_0 subtracted)")
        else:
            logger.info("  Using H_c strategy: H_0 + mu_a (non-centered, raw centroid)")

    return semantic_graphs


def build_semantic_graphs_from_clustering(clustering_data: Dict) -> Dict[int, np.ndarray]:
    """Build simple semantic graphs from clustering data (for backward compat).

    Args:
        clustering_data: Clustering result dict with "components" key

    Returns:
        {cluster_id: mu_a_array}
    """
    components = clustering_data.get("components", {})
    graphs = {}
    for c_str, comp in components.items():
        if "mu_a" in comp:
            graphs[int(c_str)] = np.array(comp["mu_a"])
    return graphs


def compute_semantic_graphs_single_sample(
    assignments: List[int],
    aggregated_attributions: np.ndarray,
    random_seed: int = 42,
    logger=None
) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    """Build semantic graphs using a single randomly selected continuation per cluster.

    This is an ablation baseline for H4A_SINGLE hypothesis testing.
    Instead of using the cluster centroid (mu_a), we use a single randomly
    selected continuation's raw attribution as H_c.

    Args:
        assignments: Cluster assignment for each continuation [n_continuations]
        aggregated_attributions: Attribution vectors [n_continuations, n_features]
        random_seed: Fixed seed for reproducibility
        logger: Optional logger instance

    Returns:
        Tuple of:
        - semantic_graphs: {cluster_id: H_c array} where H_c = attr[selected_idx] (raw, no centering)
        - selected_indices: {cluster_id: selected_continuation_idx} for logging/debugging
    """
    rng = np.random.RandomState(random_seed)

    # Group continuation indices by cluster
    cluster_to_indices: Dict[int, List[int]] = {}
    for i, cluster_id in enumerate(assignments):
        cluster_id = int(cluster_id)
        if cluster_id not in cluster_to_indices:
            cluster_to_indices[cluster_id] = []
        cluster_to_indices[cluster_id].append(i)

    semantic_graphs = {}
    selected_indices = {}

    for cluster_id in sorted(cluster_to_indices.keys()):
        indices = cluster_to_indices[cluster_id]
        # Randomly select one continuation from this cluster
        selected_idx = rng.choice(indices)
        selected_indices[cluster_id] = int(selected_idx)

        # Use the raw attribution (no H_0 subtraction)
        H_c = aggregated_attributions[selected_idx].copy()
        semantic_graphs[cluster_id] = H_c

    if logger:
        logger.info(f"  H4A_SINGLE: Built semantic graphs from single samples per cluster")
        logger.info(f"    Selected indices: {selected_indices}")

    return semantic_graphs, selected_indices


def compute_semantic_graphs_medoid(
    assignments: List[int],
    aggregated_attributions: np.ndarray,
    weights: np.ndarray = None,
    logger=None
) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    """Build semantic graphs using L1-medoid per cluster.

    The medoid is the actual data point that minimizes weighted L1 distance
    to all other points in the cluster. This stays on the data manifold,
    unlike component-wise median which creates a "Frankenstein vector".

    Args:
        assignments: Cluster assignment for each continuation [n_continuations]
        aggregated_attributions: Attribution vectors [n_continuations, n_features]
        weights: Optional weights for each point (e.g., path_probs). 
                 If None, uses uniform weights within each cluster.
        logger: Optional logger instance

    Returns:
        Tuple of:
        - semantic_graphs: {cluster_id: H_c array} where H_c = attr[medoid_idx]
        - selected_indices: {cluster_id: medoid_continuation_idx} for logging/debugging
    """
    # Group continuation indices by cluster
    cluster_to_indices: Dict[int, List[int]] = {}
    for i, cluster_id in enumerate(assignments):
        cluster_id = int(cluster_id)
        if cluster_id not in cluster_to_indices:
            cluster_to_indices[cluster_id] = []
        cluster_to_indices[cluster_id].append(i)

    semantic_graphs = {}
    selected_indices = {}

    for cluster_id in sorted(cluster_to_indices.keys()):
        indices = cluster_to_indices[cluster_id]
        n_cluster = len(indices)
        
        if n_cluster == 1:
            # Single point: it's the medoid
            medoid_idx = indices[0]
        else:
            # Get cluster points
            cluster_attrs = aggregated_attributions[indices]  # [n_cluster, n_features]
            
            # Get weights for this cluster (default: uniform)
            if weights is not None:
                cluster_weights = weights[indices]
            else:
                cluster_weights = np.ones(n_cluster)
            
            # Normalize weights within cluster
            cluster_weights = cluster_weights / cluster_weights.sum()
            
            # Compute weighted L1 distance sum for each point as candidate medoid
            # medoid = argmin_i sum_j w_j * ||x_i - x_j||_1
            total_distances = np.zeros(n_cluster)
            for i in range(n_cluster):
                # L1 distance from point i to all others
                l1_dists = np.abs(cluster_attrs - cluster_attrs[i]).sum(axis=1)
                total_distances[i] = np.sum(cluster_weights * l1_dists)
            
            # Find medoid (point with minimum total weighted distance)
            medoid_local_idx = np.argmin(total_distances)
            medoid_idx = indices[medoid_local_idx]
        
        selected_indices[cluster_id] = int(medoid_idx)
        
        # Use the raw attribution (no H_0 subtraction)
        H_c = aggregated_attributions[medoid_idx].copy()
        semantic_graphs[cluster_id] = H_c

    if logger:
        logger.info(f"  MEDOID: Built semantic graphs from L1-medoid per cluster")
        logger.info(f"    Selected indices: {selected_indices}")

    return semantic_graphs, selected_indices


def compute_semantic_graphs_combined_medoid(
    assignments: List[int],
    aggregated_attributions: np.ndarray,
    embeddings: np.ndarray,
    gamma: float,
    weights: np.ndarray = None,
    logger=None
) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
    """Build semantic graphs using combined-distance medoid per cluster.

    The medoid is the actual data point that minimizes the weighted combined
    distance to all other points in the cluster, where combined distance is:
        d_combined = gamma * d_emb_L2 + (1-gamma) * d_attr_L1

    This is consistent with the RD clustering objective that uses both views.

    Args:
        assignments: Cluster assignment for each continuation [n_continuations]
        aggregated_attributions: Attribution vectors [n_continuations, n_features]
        embeddings: Semantic embeddings [n_continuations, d_emb]
        gamma: Weight for embedding distance (1-gamma for attribution distance)
        weights: Optional weights for each point (e.g., path_probs). 
                 If None, uses uniform weights within each cluster.
        logger: Optional logger instance

    Returns:
        Tuple of:
        - semantic_graphs: {cluster_id: H_c array} where H_c = attr[medoid_idx]
        - selected_indices: {cluster_id: medoid_continuation_idx} for logging/debugging
    """
    # Group continuation indices by cluster
    cluster_to_indices: Dict[int, List[int]] = {}
    for i, cluster_id in enumerate(assignments):
        cluster_id = int(cluster_id)
        if cluster_id not in cluster_to_indices:
            cluster_to_indices[cluster_id] = []
        cluster_to_indices[cluster_id].append(i)

    semantic_graphs = {}
    selected_indices = {}

    for cluster_id in sorted(cluster_to_indices.keys()):
        indices = cluster_to_indices[cluster_id]
        n_cluster = len(indices)
        
        if n_cluster == 1:
            # Single point: it's the medoid
            medoid_idx = indices[0]
        else:
            # Get cluster points
            cluster_attrs = aggregated_attributions[indices]  # [n_cluster, n_features]
            cluster_embs = embeddings[indices]  # [n_cluster, d_emb]
            
            # Get weights for this cluster (default: uniform)
            if weights is not None:
                cluster_weights = weights[indices]
            else:
                cluster_weights = np.ones(n_cluster)
            
            # Normalize weights within cluster
            cluster_weights = cluster_weights / cluster_weights.sum()
            
            # Compute weighted combined distance sum for each point as candidate medoid
            # medoid = argmin_i sum_j w_j * [gamma * d_emb(i,j) + (1-gamma) * d_attr(i,j)]
            total_distances = np.zeros(n_cluster)
            for i in range(n_cluster):
                # L2 distance for embeddings
                emb_diffs = cluster_embs - cluster_embs[i]
                l2_dists = np.sqrt(np.sum(emb_diffs ** 2, axis=1))
                
                # L1 distance for attributions
                l1_dists = np.abs(cluster_attrs - cluster_attrs[i]).sum(axis=1)
                
                # Combined distance
                combined_dists = gamma * l2_dists + (1 - gamma) * l1_dists
                total_distances[i] = np.sum(cluster_weights * combined_dists)
            
            # Find medoid (point with minimum total weighted distance)
            medoid_local_idx = np.argmin(total_distances)
            medoid_idx = indices[medoid_local_idx]
        
        selected_indices[cluster_id] = int(medoid_idx)
        
        # Use the raw attribution (no H_0 subtraction)
        H_c = aggregated_attributions[medoid_idx].copy()
        semantic_graphs[cluster_id] = H_c

    if logger:
        logger.info(f"  COMBINED_MEDOID (gamma={gamma}): Built semantic graphs from combined-distance medoid")
        logger.info(f"    Selected indices: {selected_indices}")

    return semantic_graphs, selected_indices


# =============================================================================
# Feature Selection
# =============================================================================

def compute_feature_ranking_scores(
    cluster_id: int,
    semantic_graphs: Dict[int, np.ndarray],
    n_features: int,
    selection_mode: str = "magnitude",
) -> np.ndarray:
    """Compute feature ranking scores for a cluster.

    Ranking is decoupled from the signed H_c values used for steering. This lets
    downstream code rank by distinctiveness while still steering with the
    original signed coefficients.
    """
    H_c = semantic_graphs[cluster_id]
    H_c_features = H_c[:n_features]

    if selection_mode != "distinct":
        return np.abs(H_c_features)

    other_cluster_ids = [c for c in semantic_graphs.keys() if c != cluster_id]
    if not other_cluster_ids:
        return np.abs(H_c_features)

    other_H_c_stack = np.stack(
        [np.abs(semantic_graphs[c][:n_features]) for c in other_cluster_ids],
        axis=0,
    )
    max_other = np.max(other_H_c_stack, axis=0)
    return np.abs(H_c_features) / (max_other + utils.EPSILON_TINY)


def _select_top_feature_indices(
    cluster_id: int,
    semantic_graphs: Dict[int, np.ndarray],
    n_features: int,
    top_B: int,
    selection_mode: str = "magnitude",
) -> List[int]:
    """Return feature indices ordered by the requested ranking mode."""
    rank_scores = compute_feature_ranking_scores(
        cluster_id=cluster_id,
        semantic_graphs=semantic_graphs,
        n_features=n_features,
        selection_mode=selection_mode,
    )
    if top_B <= 0:
        top_B = len(rank_scores)
    top_B = min(top_B, len(rank_scores))
    if top_B <= 0:
        return []
    return [int(idx) for idx in np.argsort(rank_scores)[-top_B:][::-1]]


def select_top_features_by_magnitude(
    H_c: np.ndarray,
    active_features: torch.Tensor,
    selected_features: torch.Tensor,
    top_B: int = 50
) -> List[Tuple[int, int, int, float]]:
    """Select top-B features by |H_c| magnitude.

    H_c has dimension n_attribution_nodes = n_features + n_error + n_token.
    Only the first n_features elements correspond to actual model features.
    selected_features provides the mapping from H_c index to active_features index.
    active_features[selected_features[i]] gives [layer, pos, feat_id] for H_c[i].
    """
    n_features = len(selected_features)
    semantic_graphs = {0: H_c}
    H_c_features = H_c[:n_features]
    top_indices = _select_top_feature_indices(
        cluster_id=0,
        semantic_graphs=semantic_graphs,
        n_features=n_features,
        top_B=top_B,
        selection_mode="magnitude",
    )

    features = []
    for idx in top_indices:
        # Map H_c index -> active_features index via selected_features
        feat_idx = selected_features[idx].item()
        layer, pos, feat_id = active_features[feat_idx].tolist()
        H_c_value = float(H_c_features[idx])
        features.append((int(layer), int(pos), int(feat_id), H_c_value))

    return features


def select_features_with_hc_selection(
    cache_data: Dict,
    top_B: int,
    hc_selection: str = 'full'
) -> Tuple[List[float], List[torch.Tensor], List[int], List[int], List[int]]:
    """Select features based on H_c selection mode and top_B.

    Args:
        cache_data: Dict with keys: h_c_values, decoder_vecs, layers, positions, feat_ids
        top_B: Number of top features to select
        hc_selection: 'full' (all), 'positive' (H_c > 0), 'negative' (H_c < 0)

    Returns:
        (h_c_vals, decoder_vecs, layers, positions, feat_ids) - all filtered
    """
    h_c_vals = cache_data['h_c_values']
    decoder_vecs = cache_data['decoder_vecs']
    layers = cache_data['layers']
    positions = cache_data['positions']
    feat_ids = cache_data['feat_ids']
    rank_scores = cache_data.get('rank_scores')

    # Filter by sign
    if hc_selection == 'positive':
        indices = [i for i, v in enumerate(h_c_vals) if v > 0]
    elif hc_selection == 'negative':
        indices = [i for i, v in enumerate(h_c_vals) if v < 0]
    else:  # full
        indices = list(range(len(h_c_vals)))

    if not indices:
        return [], [], [], [], []

    # Sort by ranking score and take top_B. Fall back to |H_c| for older caches.
    if rank_scores is None:
        sorted_indices = sorted(indices, key=lambda i: abs(h_c_vals[i]), reverse=True)
    else:
        sorted_indices = sorted(indices, key=lambda i: rank_scores[i], reverse=True)
    if top_B > 0:
        sorted_indices = sorted_indices[:top_B]

    return (
        [h_c_vals[i] for i in sorted_indices],
        [decoder_vecs[i] for i in sorted_indices],
        [layers[i] for i in sorted_indices],
        [positions[i] for i in sorted_indices],
        [feat_ids[i] for i in sorted_indices]
    )


def build_cluster_decoder_cache(
    semantic_graphs: Dict[int, np.ndarray],
    global_decoder_cache: Dict[int, torch.Tensor],
    active_features: torch.Tensor,
    selected_features: torch.Tensor,
    max_features: int = 1000,
    selection_mode: str = "magnitude",
) -> Dict[int, Dict]:
    """Build per-cluster decoder cache from caller-provided decoder vectors.

    The global decoder cache maps H_c-local feature indices to decoder vectors
    and is prepared by the caller.

    Args:
        semantic_graphs: {cluster_id: H_c array}
        global_decoder_cache: {h_c_idx: decoder_vector}
        active_features: Tensor mapping feat_idx -> (layer, pos, feat_id)
        selected_features: Tensor mapping H_c index -> active_features index
        max_features: Maximum features per cluster

    Returns:
        {cluster_id: {h_c_values, decoder_vecs, layers, positions, feat_ids, h_c_indices}}
    """
    n_features = len(selected_features)
    cache = {}

    for cluster_id, H_c in semantic_graphs.items():
        H_c_features = H_c[:n_features]

        rank_scores = compute_feature_ranking_scores(
            cluster_id=cluster_id,
            semantic_graphs=semantic_graphs,
            n_features=n_features,
            selection_mode=selection_mode,
        )
        top_indices = _select_top_feature_indices(
            cluster_id=cluster_id,
            semantic_graphs=semantic_graphs,
            n_features=n_features,
            top_B=max_features,
            selection_mode=selection_mode,
        )

        h_c_vals = []
        decoder_vecs = []
        layers = []
        positions = []
        feat_ids = []
        h_c_indices = []
        cache_rank_scores = []

        for h_c_idx in top_indices:
            h_c_val = H_c_features[h_c_idx]
            if abs(h_c_val) < utils.EPSILON_SMALL:
                continue

            # Skip if not in global cache
            if h_c_idx not in global_decoder_cache:
                continue

            feat_idx = selected_features[h_c_idx].item()
            layer, pos, feat_id = active_features[feat_idx].tolist()

            h_c_vals.append(float(h_c_val))
            decoder_vecs.append(global_decoder_cache[h_c_idx])
            layers.append(int(layer))
            positions.append(int(pos))
            feat_ids.append(int(feat_id))
            h_c_indices.append(int(h_c_idx))
            cache_rank_scores.append(float(rank_scores[h_c_idx]))

        if h_c_vals:
            cache[cluster_id] = {
                'h_c_values': h_c_vals,
                'decoder_vecs': decoder_vecs,
                'layers': layers,
                'positions': positions,
                'feat_ids': feat_ids,
                'h_c_indices': h_c_indices,
                'rank_scores': cache_rank_scores,
            }

    return cache


# =============================================================================
# Encoder Weight Precomputation
# =============================================================================

def precompute_cluster_encoder_weights(
    model,
    features_by_cluster: Dict[int, List[Tuple[int, int, int, float]]],
    device: torch.device
) -> Dict[int, Dict[int, Dict[str, torch.Tensor]]]:
    """Pre-compute encoder weights for each cluster's features.

    This enables column-wise activation computation during on-the-fly steering.

    Args:
        model: ReplacementModel with transcoders
        features_by_cluster: {cluster_id: [(layer, pos, feat_id, h_c_val), ...]}
        device: torch device

    Returns:
        {cluster_id: {layer: {'W_enc': [n_feats, d_model], 'b_enc': [n_feats], ...}}}
    """
    cluster_encoders = {}

    for cluster_id, features in features_by_cluster.items():
        cluster_encoders[cluster_id] = preload_encoder_weights_for_cluster(
            model, features, device
        )

    return cluster_encoders


def preload_encoder_weights_for_cluster(
    model,
    features: List[Tuple[int, int, int, float]],
    device: torch.device
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Pre-load encoder weights for column-wise activation computation.

    Only loads the encoder columns for features we need to intervene on.
    This is much more memory efficient than loading the full encoder.

    Args:
        model: ReplacementModel with transcoders
        features: List of (layer, pos, feat_id, h_c_val) tuples
        device: torch device

    Returns:
        {layer: {'W_enc': [n_feats, d_model], 'b_enc': [n_feats], 'feat_ids': [feat_id, ...]}}
    """
    # Group features by layer
    by_layer = {}
    for layer, pos, feat_id, h_c_val in features:
        if layer not in by_layer:
            by_layer[layer] = set()
        by_layer[layer].add(feat_id)

    encoder_cache = {}
    for layer, feat_ids_set in by_layer.items():
        feat_ids_list = sorted(feat_ids_set)
        feat_ids_t = torch.tensor(feat_ids_list, device=device, dtype=torch.long)

        # Load only the needed encoder columns (column-wise) - using shared utility
        W_enc_subset, b_enc_subset = utils.get_encoder_weights(model, layer, feat_ids_t)

        encoder_cache[layer] = {
            'W_enc': W_enc_subset,
            'b_enc': b_enc_subset,
            'feat_ids': feat_ids_list,
            'feat_id_to_idx': {fid: i for i, fid in enumerate(feat_ids_list)}
        }

    return encoder_cache


def precompute_cluster_decoder_vectors(
    model,
    features_by_cluster: Dict[int, List[Tuple[int, int, int, float]]],
    device: torch.device
) -> Dict[int, Dict[int, Tuple[List[int], List[int], torch.Tensor, torch.Tensor]]]:
    """Pre-compute decoder vectors for each cluster's features.

    This avoids redundant calls to _get_decoder_vectors when processing multiple branches
    with the same cluster's steering features.

    Args:
        model: ReplacementModel
        features_by_cluster: {cluster_id: [(layer, pos, feat_id, h_c_val), ...]}
        device: torch device

    Returns:
        {cluster_id: {layer: (positions, feat_ids, decoder_vecs, h_c_values)}}
        where decoder_vecs is [n_features_in_layer, d_model] tensor
    """
    cluster_decoders = {}

    for cluster_id, features in features_by_cluster.items():
        # Group features by layer
        by_layer = {}
        for layer, pos, feat_id, h_c_val in features:
            if layer not in by_layer:
                by_layer[layer] = []
            by_layer[layer].append((pos, feat_id, h_c_val))

        cluster_decoders[cluster_id] = {}
        for layer, items in by_layer.items():
            positions = [x[0] for x in items]
            feat_ids_list = [x[1] for x in items]
            feat_ids_tensor = torch.tensor(feat_ids_list, device=device, dtype=torch.long)
            h_c_vals = torch.tensor(
                [x[2] for x in items],
                device=device,
                dtype=get_model_dtype(model, fallback=torch.float32),
            )

            # Get decoder vectors ONCE per cluster per layer
            decoder_vecs = model.transcoders._get_decoder_vectors(layer, feat_ids_tensor)

            cluster_decoders[cluster_id][layer] = (positions, feat_ids_list, decoder_vecs, h_c_vals)

    return cluster_decoders
