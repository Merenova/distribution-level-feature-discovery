#!/usr/bin/env -S uv run python
"""7c_cluster_analysis.py - Cluster Analysis and Beta-Hierarchical Activation Map.

This script analyzes RD clustering results to:
1. Label each cluster using an LLM based on assigned continuations
2. Extract attribution centroids (mu_a) demeaned by global H_0
3. Build a beta-hierarchical map of top magnitude activations with (layer, token_position, feature_idx)

Usage:
    python 7c_cluster_analysis.py --results-dir /path/to/results [options]
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data_utils import load_json, save_json, reconstruct_active_features
from utils.logging_utils import setup_logger, get_log_path


# =============================================================================
# Constants
# =============================================================================

DEFAULT_TOP_N_FEATURES = 20
DEFAULT_MAX_CONTINUATIONS_PER_CLUSTER = 10
DEFAULT_BETA_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
DEFAULT_GAMMA = 0.5  # Default gamma for hierarchical analysis


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_clustering_data(clustering_file: Path) -> Dict[str, Any]:
    """Load clustering sweep results from JSON file.
    
    Args:
        clustering_file: Path to {prefix_id}_sweep_results.json
        
    Returns:
        Dict with keys: prefix_id, prefix, H_0, sweep_config, grid
    """
    return load_json(clustering_file)


def load_branches_data(branches_file: Path) -> Dict[str, Any]:
    """Load branch sampling data.
    
    Args:
        branches_file: Path to {prefix_id}_branches.json
        
    Returns:
        Dict with keys: prefix_id, prefix, continuations, etc.
    """
    return load_json(branches_file)


def load_attribution_context(attribution_graphs_dir: Path, prefix_id: str) -> Dict[str, Any]:
    """Load attribution context for feature mapping.
    
    Args:
        attribution_graphs_dir: Directory containing attribution files
        prefix_id: Prefix identifier
        
    Returns:
        Dict with decoder_locations, selected_features, activation_matrix
    """
    context_file = attribution_graphs_dir / f"{prefix_id}_prefix_context.pt"
    context_data = torch.load(context_file, weights_only=False)
    return context_data


def get_active_features_mapping(context_data: Dict[str, Any]) -> np.ndarray:
    """Reconstruct active_features (N, 3) array mapping index to (layer, pos, feat_id).
    
    Args:
        context_data: Attribution context data
        
    Returns:
        (N, 3) array where each row is [layer, position, feature_id]
    """
    active_features = reconstruct_active_features(
        context_data["decoder_locations"],
        context_data["selected_features"],
        activation_matrix=context_data.get("activation_matrix", None),
        return_numpy=True
    )
    return active_features


# =============================================================================
# Cluster Labeling Functions (LLM-based)
# =============================================================================

def get_cluster_continuations(
    branches_data: Dict[str, Any],
    assignments: List[int],
    cluster_id: int,
    max_samples: int = DEFAULT_MAX_CONTINUATIONS_PER_CLUSTER
) -> List[str]:
    """Get continuation texts assigned to a specific cluster.
    
    Args:
        branches_data: Branch sampling data
        assignments: Cluster assignments for each continuation
        cluster_id: Target cluster ID
        max_samples: Maximum number of samples to return
        
    Returns:
        List of continuation text strings
    """
    continuations = branches_data.get("continuations", [])
    cluster_texts = []
    
    for i, cont in enumerate(continuations):
        if i < len(assignments) and assignments[i] == cluster_id:
            text = cont.get("text", "")
            if text:
                cluster_texts.append(text)
                if len(cluster_texts) >= max_samples:
                    break
    
    return cluster_texts


def generate_cluster_label_prompt(
    prefix: str,
    continuations: List[str],
    cluster_id: int
) -> str:
    """Generate prompt for LLM cluster labeling.
    
    Args:
        prefix: The prefix/prompt text
        continuations: List of continuation texts in this cluster
        cluster_id: Cluster identifier
        
    Returns:
        Formatted prompt string
    """
    cont_text = "\n".join([f"  {i+1}. {c[:200]}..." if len(c) > 200 else f"  {i+1}. {c}" 
                          for i, c in enumerate(continuations[:10])])
    
    prompt = f"""Analyze these model-generated continuations that have been clustered together based on both semantic similarity and internal activation patterns.

Prefix/Question: "{prefix}"

Continuations in Cluster {cluster_id}:
{cont_text}

Based on these continuations, provide a concise label (3-7 words) that captures what distinguishes this cluster. Consider:
- Common themes, topics, or answers
- Writing style (formal, casual, detailed, concise)
- Specific concepts, entities, or perspectives
- Factual approach vs. speculative content

Respond with ONLY the label, no explanation."""
    
    return prompt


def label_cluster_with_llm(
    prefix: str,
    continuations: List[str],
    cluster_id: int,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini"
) -> str:
    """Generate a descriptive label for a cluster using OpenAI API.
    
    Args:
        prefix: The prefix/prompt text
        continuations: List of continuation texts
        cluster_id: Cluster identifier
        api_key: OpenAI API key (uses env var if not provided)
        model: OpenAI model to use
        
    Returns:
        Generated cluster label string
    """
    if not continuations:
        return f"Empty Cluster {cluster_id}"
    
    try:
        from openai import OpenAI
    except ImportError:
        return f"Cluster {cluster_id} (LLM unavailable)"
    
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return f"Cluster {cluster_id} (no API key)"
    
    try:
        client = OpenAI(api_key=api_key)
        prompt = generate_cluster_label_prompt(prefix, continuations, cluster_id)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing and categorizing text. Provide concise, descriptive labels."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.3
        )
        
        label = response.choices[0].message.content.strip()
        return label if label else f"Cluster {cluster_id}"
        
    except Exception as e:
        return f"Cluster {cluster_id} (error: {str(e)[:30]})"


def label_all_clusters(
    branches_data: Dict[str, Any],
    grid_entry: Dict[str, Any],
    api_key: Optional[str] = None,
    max_samples: int = DEFAULT_MAX_CONTINUATIONS_PER_CLUSTER,
    logger: Optional[logging.Logger] = None
) -> Dict[int, str]:
    """Generate labels for all clusters in a grid entry.
    
    Args:
        branches_data: Branch sampling data
        grid_entry: Single entry from clustering grid
        api_key: OpenAI API key
        max_samples: Max continuations per cluster for labeling
        logger: Optional logger
        
    Returns:
        Dict mapping cluster_id -> label string
    """
    prefix = branches_data.get("prefix", "")
    assignments = grid_entry.get("assignments", [])
    components = grid_entry.get("components", {})
    
    labels = {}
    for cluster_str in components.keys():
        cluster_id = int(cluster_str)
        continuations = get_cluster_continuations(
            branches_data, assignments, cluster_id, max_samples
        )
        
        if logger:
            logger.debug(f"  Labeling cluster {cluster_id} with {len(continuations)} samples")
        
        label = label_cluster_with_llm(prefix, continuations, cluster_id, api_key)
        labels[cluster_id] = label
    
    return labels


# =============================================================================
# Attribution Centroid Extraction
# =============================================================================

def extract_centroid_top_features(
    mu_a: np.ndarray,
    active_features: np.ndarray,
    top_n: int = DEFAULT_TOP_N_FEATURES
) -> List[Dict[str, Any]]:
    """Extract top-N features by magnitude from attribution centroid.
    
    Args:
        mu_a: Attribution centroid vector (demeaned Delta_H_c)
        active_features: (N, 3) array mapping index to (layer, pos, feat_id)
        top_n: Number of top features to extract
        
    Returns:
        List of dicts with keys: layer, position, feature_id, magnitude, value
    """
    n_features = min(len(mu_a), len(active_features))
    mu_a_features = mu_a[:n_features]
    
    # Get top indices by absolute magnitude
    abs_vals = np.abs(mu_a_features)
    top_indices = np.argsort(abs_vals)[-top_n:][::-1]
    
    features = []
    for idx in top_indices:
        if idx < len(active_features):
            layer, pos, feat_id = active_features[idx]
            features.append({
                "index": int(idx),
                "layer": int(layer),
                "position": int(pos),
                "feature_id": int(feat_id),
                "magnitude": float(abs_vals[idx]),
                "value": float(mu_a_features[idx])
            })
    
    return features


def extract_cluster_centroids(
    grid_entry: Dict[str, Any],
    active_features: np.ndarray,
    top_n: int = DEFAULT_TOP_N_FEATURES
) -> Dict[int, Dict[str, Any]]:
    """Extract centroids and top features for all clusters in a grid entry.
    
    Args:
        grid_entry: Single entry from clustering grid
        active_features: Feature mapping array
        top_n: Number of top features per cluster
        
    Returns:
        Dict mapping cluster_id -> {mu_a, top_features, W_c}
    """
    components = grid_entry.get("components", {})
    
    centroids = {}
    for cluster_str, comp in components.items():
        cluster_id = int(cluster_str)
        mu_a = np.array(comp.get("mu_a", []))
        W_c = comp.get("W_c", 0.0)
        
        if len(mu_a) == 0:
            continue
        
        top_features = extract_centroid_top_features(mu_a, active_features, top_n)
        
        centroids[cluster_id] = {
            "mu_a": mu_a.tolist(),
            "top_features": top_features,
            "W_c": W_c,
            "n_features": len(top_features)
        }
    
    return centroids


# =============================================================================
# Beta-Hierarchical Activation Map
# =============================================================================

def build_beta_hierarchical_map(
    clustering_data: Dict[str, Any],
    active_features: np.ndarray,
    beta_values: List[float] = DEFAULT_BETA_VALUES,
    gamma: float = DEFAULT_GAMMA,
    top_n: int = DEFAULT_TOP_N_FEATURES,
    min_cluster_weight: float = 1.0,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Build hierarchical map of activations across beta values.
    
    Args:
        clustering_data: Full clustering sweep data
        active_features: Feature mapping array
        beta_values: List of beta values to include
        gamma: Gamma value to use (fixed for hierarchy)
        top_n: Number of top features per cluster
        min_cluster_weight: Minimum W_c to include cluster (filters outliers)
        logger: Optional logger
        
    Returns:
        Dict with beta hierarchy data structure
    """
    grid = clustering_data.get("grid", [])
    H_0 = np.array(clustering_data.get("H_0", []))
    
    # Group grid entries by beta
    beta_to_entries = {}
    for entry in grid:
        beta = entry.get("beta")
        entry_gamma = entry.get("gamma")
        if beta is not None and entry_gamma is not None:
            # Use closest gamma to target
            if beta not in beta_to_entries:
                beta_to_entries[beta] = []
            beta_to_entries[beta].append((entry_gamma, entry))
    
    # For each beta, select entry with closest gamma to target
    hierarchy = {
        "H_0_norm": float(np.linalg.norm(H_0)) if len(H_0) > 0 else 0.0,
        "target_gamma": gamma,
        "min_cluster_weight": min_cluster_weight,
        "beta_levels": {}
    }
    
    for beta in sorted(beta_values):
        if beta not in beta_to_entries:
            if logger:
                logger.warning(f"Beta {beta} not found in grid")
            continue
        
        # Find entry with closest gamma
        entries = beta_to_entries[beta]
        closest_entry = min(entries, key=lambda x: abs(x[0] - gamma))[1]
        actual_gamma = closest_entry.get("gamma")
        
        if logger:
            logger.debug(f"  Beta {beta}: using gamma={actual_gamma}, K={closest_entry.get('K')}")
        
        # Extract centroids for this beta level
        centroids = extract_cluster_centroids(closest_entry, active_features, top_n)
        
        # Collect top activations, weighted by cluster importance
        # Filter out small clusters (likely outliers) and weight by W_c
        all_activations = []
        weighted_activations = []  # Separate list weighted by cluster size
        
        for cluster_id, centroid_data in centroids.items():
            W_c = centroid_data.get("W_c", 0.0)
            is_significant = W_c >= min_cluster_weight
            
            for feat in centroid_data["top_features"]:
                activation = {
                    "cluster_id": cluster_id,
                    "cluster_weight": W_c,
                    **feat
                }
                all_activations.append(activation)
                
                if is_significant:
                    # Weight magnitude by cluster probability mass
                    weighted_mag = feat["magnitude"] * W_c
                    weighted_activations.append({
                        **activation,
                        "weighted_magnitude": weighted_mag
                    })
        
        # Sort by raw magnitude
        all_activations.sort(key=lambda x: x["magnitude"], reverse=True)
        
        # Sort weighted activations by weighted magnitude
        weighted_activations.sort(key=lambda x: x["weighted_magnitude"], reverse=True)
        
        hierarchy["beta_levels"][beta] = {
            "gamma": actual_gamma,
            "K": closest_entry.get("K"),
            "H": closest_entry.get("H"),
            "D_e": closest_entry.get("D_e"),
            "D_a": closest_entry.get("D_a"),
            "n_clusters": len(centroids),
            "n_significant_clusters": len([c for c, d in centroids.items() if d.get("W_c", 0) >= min_cluster_weight]),
            "clusters": centroids,
            "top_activations": all_activations[:top_n * 2],  # Top 2*N by raw magnitude
            "top_weighted_activations": weighted_activations[:top_n * 2]  # Top 2*N by weighted magnitude
        }
    
    return hierarchy


def get_unique_features_across_betas(
    hierarchy: Dict[str, Any],
    use_weighted: bool = True
) -> List[Tuple[int, int, int]]:
    """Extract unique (layer, position, feature_id) tuples across all beta levels.
    
    Args:
        hierarchy: Beta hierarchy data
        use_weighted: If True, use weighted activations (filters outliers)
        
    Returns:
        List of unique (layer, pos, feat_id) tuples, sorted by total magnitude
    """
    feature_magnitudes = {}  # (layer, pos, feat_id) -> total magnitude
    
    for beta, level_data in hierarchy.get("beta_levels", {}).items():
        # Use weighted activations if available and requested
        activations_key = "top_weighted_activations" if use_weighted else "top_activations"
        activations = level_data.get(activations_key, level_data.get("top_activations", []))
        
        for activation in activations:
            key = (activation["layer"], activation["position"], activation["feature_id"])
            # Use weighted magnitude if available
            mag = activation.get("weighted_magnitude", activation["magnitude"])
            feature_magnitudes[key] = feature_magnitudes.get(key, 0.0) + mag
    
    # Sort by total magnitude
    sorted_features = sorted(feature_magnitudes.items(), key=lambda x: x[1], reverse=True)
    return [f[0] for f in sorted_features]


# =============================================================================
# Output Functions
# =============================================================================

def save_cluster_labels_json(
    labels_by_beta: Dict[float, Dict[int, str]],
    output_file: Path,
    metadata: Optional[Dict[str, Any]] = None
):
    """Save cluster labels to JSON file.
    
    Args:
        labels_by_beta: Dict mapping beta -> {cluster_id -> label}
        output_file: Output file path
        metadata: Optional metadata to include
    """
    output = {
        "metadata": metadata or {},
        "labels_by_beta": {
            str(beta): {str(k): v for k, v in labels.items()}
            for beta, labels in labels_by_beta.items()
        }
    }
    save_json(output, output_file)


def save_activation_map_json(
    hierarchy: Dict[str, Any],
    labels_by_beta: Dict[float, Dict[int, str]],
    output_file: Path,
    metadata: Optional[Dict[str, Any]] = None
):
    """Save full activation map to JSON file.
    
    Args:
        hierarchy: Beta hierarchy data
        labels_by_beta: Cluster labels
        output_file: Output file path
        metadata: Optional metadata
    """
    # Add labels to hierarchy
    for beta_str, level_data in hierarchy.get("beta_levels", {}).items():
        beta = float(beta_str) if isinstance(beta_str, str) else beta_str
        if beta in labels_by_beta:
            for cluster_id, centroid_data in level_data.get("clusters", {}).items():
                cid = int(cluster_id) if isinstance(cluster_id, str) else cluster_id
                if cid in labels_by_beta[beta]:
                    centroid_data["label"] = labels_by_beta[beta][cid]
    
    output = {
        "metadata": metadata or {},
        "hierarchy": hierarchy
    }
    save_json(output, output_file)


def save_activation_map_csv(
    hierarchy: Dict[str, Any],
    labels_by_beta: Dict[float, Dict[int, str]],
    output_file: Path
):
    """Save activation map to CSV file.
    
    Args:
        hierarchy: Beta hierarchy data
        labels_by_beta: Cluster labels
        output_file: Output file path
    """
    rows = []
    
    for beta, level_data in sorted(hierarchy.get("beta_levels", {}).items()):
        beta_float = float(beta) if isinstance(beta, str) else beta
        gamma = level_data.get("gamma", 0.0)
        K = level_data.get("K", 0)
        
        for cluster_id, centroid_data in level_data.get("clusters", {}).items():
            cid = int(cluster_id) if isinstance(cluster_id, str) else cluster_id
            label = labels_by_beta.get(beta_float, {}).get(cid, f"Cluster {cid}")
            W_c = centroid_data.get("W_c", 0.0)
            
            for feat in centroid_data.get("top_features", []):
                rows.append({
                    "beta": beta_float,
                    "gamma": gamma,
                    "K": K,
                    "cluster_id": cid,
                    "cluster_label": label,
                    "cluster_weight": W_c,
                    "layer": feat["layer"],
                    "position": feat["position"],
                    "feature_id": feat["feature_id"],
                    "magnitude": feat["magnitude"],
                    "value": feat["value"],
                    "feature_index": feat["index"]
                })
    
    if rows:
        fieldnames = ["beta", "gamma", "K", "cluster_id", "cluster_label", "cluster_weight",
                      "layer", "position", "feature_id", "magnitude", "value", "feature_index"]
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def create_beta_heatmap(
    hierarchy: Dict[str, Any],
    output_file: Path,
    top_n_features: int = 30,
    figsize: Tuple[int, int] = (16, 12),
    use_weighted: bool = True
):
    """Create heatmap visualization of activations across beta values.
    
    Creates a 2-panel figure:
    1. Main heatmap showing activation values across beta
    2. Bar chart showing K (number of clusters) vs beta
    
    Args:
        hierarchy: Beta hierarchy data
        output_file: Output file path
        top_n_features: Number of features to show
        figsize: Figure size
        use_weighted: If True, use weighted activations (filters outlier clusters)
    """
    # Get unique features across all betas (using weighted to filter outliers)
    unique_features = get_unique_features_across_betas(hierarchy, use_weighted=use_weighted)[:top_n_features]
    
    if not unique_features:
        return
    
    # Build heatmap matrix
    beta_values = sorted([float(b) for b in hierarchy.get("beta_levels", {}).keys()])
    
    if not beta_values:
        return
    
    # Collect K values and metadata for each beta
    K_values = []
    H_values = []
    n_sig_clusters = []
    for beta in beta_values:
        level_data = hierarchy["beta_levels"].get(beta, hierarchy["beta_levels"].get(str(beta), {}))
        K_values.append(level_data.get("K", 0))
        H_values.append(level_data.get("H", 0))
        n_sig_clusters.append(level_data.get("n_significant_clusters", level_data.get("K", 0)))
    
    # Create feature labels
    feature_labels = [f"L{l}_P{p}_F{f}" for l, p, f in unique_features]
    
    # Build magnitude matrix
    matrix = np.zeros((len(unique_features), len(beta_values)))
    
    for j, beta in enumerate(beta_values):
        level_data = hierarchy["beta_levels"].get(beta, hierarchy["beta_levels"].get(str(beta), {}))
        
        # Build lookup for this beta from weighted activations (filters outliers)
        activations_key = "top_weighted_activations" if use_weighted else "top_activations"
        activations = level_data.get(activations_key, level_data.get("top_activations", []))
        
        # Build lookup: for duplicate features, use the one from the LARGEST cluster (by W_c)
        # This ensures we show the most representative value, not just the max magnitude
        activation_lookup = {}  # key -> (value, cluster_weight)
        for activation in activations:
            key = (activation["layer"], activation["position"], activation["feature_id"])
            W_c = activation.get("cluster_weight", 0.0)
            
            if key not in activation_lookup or W_c > activation_lookup[key][1]:
                activation_lookup[key] = (activation["value"], W_c)
        
        for i, feat_key in enumerate(unique_features):
            if feat_key in activation_lookup:
                matrix[i, j] = activation_lookup[feat_key][0]  # Get the value
    
    # Create figure with 2 subplots
    fig, (ax_bar, ax_heat) = plt.subplots(
        2, 1, figsize=figsize, 
        gridspec_kw={'height_ratios': [1, 4]},
        sharex=True
    )
    
    # Top panel: K values bar chart (show both total and significant clusters)
    x_pos = np.arange(len(beta_values))
    width = 0.35
    
    bars1 = ax_bar.bar(x_pos - width/2, K_values, width, label='Total K', color='steelblue', alpha=0.7, edgecolor='navy')
    bars2 = ax_bar.bar(x_pos + width/2, n_sig_clusters, width, label='Significant (W_c≥1)', color='coral', alpha=0.7, edgecolor='darkred')
    
    ax_bar.set_ylabel("# Clusters", fontsize=11)
    ax_bar.set_title("Beta-Hierarchical Activation Map (Weighted by Cluster Size)\nFilters outlier clusters, shows features from significant clusters", fontsize=13)
    ax_bar.legend(loc='upper left', fontsize=9)
    
    # Add K values on bars
    for bar, k in zip(bars1, K_values):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                   str(k), ha='center', va='bottom', fontsize=9)
    for bar, k in zip(bars2, n_sig_clusters):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                   str(k), ha='center', va='bottom', fontsize=9)
    
    ax_bar.set_ylim(0, max(K_values) * 1.2)
    ax_bar.grid(axis='y', alpha=0.3)
    
    # Bottom panel: Heatmap
    vmax = np.abs(matrix).max()
    if vmax < 1e-6:
        vmax = 1.0  # Prevent division by zero for empty matrices
    
    im = ax_heat.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    
    # Set labels
    ax_heat.set_xticks(x_pos)
    ax_heat.set_xticklabels([f"β={b}\n(K={K_values[i]}, sig={n_sig_clusters[i]})" for i, b in enumerate(beta_values)], fontsize=9)
    ax_heat.set_yticks(range(len(feature_labels)))
    ax_heat.set_yticklabels(feature_labels, fontsize=8)
    
    ax_heat.set_xlabel("Beta (Rate-Distortion Trade-off)", fontsize=12)
    ax_heat.set_ylabel("Feature (Layer_Position_FeatureID)", fontsize=10)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Attribution Value (Demeaned Δμ_a)", fontsize=10)
    
    # Add annotation about identical columns
    col_sums = np.abs(matrix).sum(axis=0)
    for i in range(len(beta_values) - 1):
        if np.allclose(matrix[:, i], matrix[:, i+1], rtol=0.01):
            ax_heat.axvline(x=i + 0.5, color='orange', linestyle='--', alpha=0.5, linewidth=2)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also create a per-beta normalized version for better visibility
    norm_output = output_file.parent / (output_file.stem + "_normalized.png")
    create_normalized_heatmap(matrix, beta_values, K_values, n_sig_clusters, feature_labels, norm_output, figsize)


def create_normalized_heatmap(
    matrix: np.ndarray,
    beta_values: List[float],
    K_values: List[int],
    n_sig_clusters: List[int],
    feature_labels: List[str],
    output_file: Path,
    figsize: Tuple[int, int] = (16, 12)
):
    """Create column-normalized heatmap for better visibility of per-beta patterns.
    
    Args:
        matrix: Activation matrix [n_features, n_betas]
        beta_values: List of beta values
        K_values: Number of clusters per beta
        n_sig_clusters: Number of significant clusters per beta
        feature_labels: Feature label strings
        output_file: Output file path
        figsize: Figure size
    """
    # Normalize each column independently
    norm_matrix = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        col_max = np.abs(matrix[:, j]).max()
        if col_max > 1e-6:
            norm_matrix[:, j] = matrix[:, j] / col_max
    
    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] * 0.8))
    
    im = ax.imshow(norm_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Set labels
    ax.set_xticks(range(len(beta_values)))
    ax.set_xticklabels([f"β={b}\n(K={K_values[i]}, sig={n_sig_clusters[i]})" for i, b in enumerate(beta_values)], fontsize=9)
    ax.set_yticks(range(len(feature_labels)))
    ax.set_yticklabels(feature_labels, fontsize=8)
    
    ax.set_xlabel("Beta (Rate-Distortion Trade-off)", fontsize=12)
    ax.set_ylabel("Feature (Layer_Position_FeatureID)", fontsize=10)
    ax.set_title("Beta-Hierarchical Activation Map (Column-Normalized, Weighted)\nFilters outlier clusters, each column scaled to [-1, 1]", fontsize=13)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Attribution Value", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def create_per_cluster_heatmap(
    hierarchy: Dict[str, Any],
    output_file: Path,
    top_n_features: int = 10,
    max_clusters_per_beta: int = 8,
    min_cluster_weight: float = 1.0,
    figsize_per_beta: Tuple[int, int] = (12, 8)
):
    """Create heatmap showing K different clusters for each beta.
    
    For each beta, shows top clusters with their representative features.
    Each cluster gets its own column in the heatmap.
    
    Args:
        hierarchy: Beta hierarchy data
        output_file: Output file path
        top_n_features: Number of features to show per cluster
        max_clusters_per_beta: Maximum clusters to show per beta
        min_cluster_weight: Minimum W_c to include cluster
        figsize_per_beta: Figure size for each beta subplot
    """
    beta_values = sorted([float(b) for b in hierarchy.get("beta_levels", {}).keys()])
    
    if not beta_values:
        return
    
    # Create a figure with subplots for each beta
    n_betas = len(beta_values)
    fig, axes = plt.subplots(1, n_betas, figsize=(figsize_per_beta[0] * n_betas / 2, figsize_per_beta[1]))
    
    if n_betas == 1:
        axes = [axes]
    
    # Color palette for clusters (distinct colors)
    cluster_colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    for ax_idx, beta in enumerate(beta_values):
        ax = axes[ax_idx]
        level_data = hierarchy["beta_levels"].get(beta, hierarchy["beta_levels"].get(str(beta), {}))
        clusters = level_data.get("clusters", {})
        K = level_data.get("K", 0)
        
        # Sort clusters by weight and filter
        sorted_clusters = []
        for cid, cdata in clusters.items():
            W_c = cdata.get("W_c", 0)
            if W_c >= min_cluster_weight:
                sorted_clusters.append((int(cid), W_c, cdata))
        sorted_clusters.sort(key=lambda x: -x[1])  # Sort by weight descending
        sorted_clusters = sorted_clusters[:max_clusters_per_beta]
        
        if not sorted_clusters:
            ax.text(0.5, 0.5, f"No clusters\n(K={K})", ha='center', va='center', fontsize=12)
            ax.set_title(f"β={beta}", fontsize=11)
            ax.axis('off')
            continue
        
        # Collect all unique features across clusters for this beta
        all_features = set()
        for cid, W_c, cdata in sorted_clusters:
            for feat in cdata.get("top_features", [])[:top_n_features]:
                all_features.add((feat["layer"], feat["position"], feat["feature_id"]))
        
        # Sort features by layer, then position
        all_features = sorted(all_features, key=lambda x: (x[0], x[1], x[2]))[:top_n_features * 2]
        
        if not all_features:
            ax.text(0.5, 0.5, f"No features\n(K={K})", ha='center', va='center', fontsize=12)
            ax.set_title(f"β={beta}", fontsize=11)
            ax.axis('off')
            continue
        
        # Build matrix: rows = features, cols = clusters
        n_feats = len(all_features)
        n_clusters = len(sorted_clusters)
        matrix = np.zeros((n_feats, n_clusters))
        
        for j, (cid, W_c, cdata) in enumerate(sorted_clusters):
            # Build feature lookup for this cluster
            feat_lookup = {}
            for feat in cdata.get("top_features", []):
                key = (feat["layer"], feat["position"], feat["feature_id"])
                feat_lookup[key] = feat["value"]
            
            for i, feat_key in enumerate(all_features):
                if feat_key in feat_lookup:
                    matrix[i, j] = feat_lookup[feat_key]
        
        # Plot heatmap
        vmax = np.abs(matrix).max()
        if vmax < 1e-6:
            vmax = 1.0
        
        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        
        # Labels
        cluster_labels = [f"C{cid}\n(W={W_c:.0f})" for cid, W_c, _ in sorted_clusters]
        feature_labels = [f"L{l}_P{p}" for l, p, f in all_features]
        
        ax.set_xticks(range(n_clusters))
        ax.set_xticklabels(cluster_labels, fontsize=8, rotation=45, ha='right')
        ax.set_yticks(range(n_feats))
        ax.set_yticklabels(feature_labels, fontsize=7)
        
        ax.set_title(f"β={beta} (K={K})", fontsize=11, fontweight='bold')
        ax.set_xlabel("Cluster", fontsize=9)
        if ax_idx == 0:
            ax.set_ylabel("Feature (Layer_Position)", fontsize=9)
    
    # Add overall title and colorbar
    fig.suptitle("Per-Cluster Attribution Patterns Across Beta Values\nEach column = one cluster's top features", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Add colorbar to the right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Attribution Value (Δμ_a)", fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def create_cluster_feature_barplot(
    hierarchy: Dict[str, Any],
    output_dir: Path,
    top_n_features: int = 15,
    max_clusters: int = 10,
    min_cluster_weight: float = 1.0
):
    """Create bar plots showing top features for each cluster at each beta.
    
    Creates one PNG per beta showing each cluster's distinctive features.
    
    Args:
        hierarchy: Beta hierarchy data
        output_dir: Output directory for plots
        top_n_features: Number of features per cluster
        max_clusters: Maximum clusters to show
        min_cluster_weight: Minimum W_c to include cluster
    """
    beta_values = sorted([float(b) for b in hierarchy.get("beta_levels", {}).keys()])
    
    for beta in beta_values:
        level_data = hierarchy["beta_levels"].get(beta, hierarchy["beta_levels"].get(str(beta), {}))
        clusters = level_data.get("clusters", {})
        K = level_data.get("K", 0)
        
        # Sort clusters by weight
        sorted_clusters = []
        for cid, cdata in clusters.items():
            W_c = cdata.get("W_c", 0)
            if W_c >= min_cluster_weight:
                sorted_clusters.append((int(cid), W_c, cdata))
        sorted_clusters.sort(key=lambda x: -x[1])
        sorted_clusters = sorted_clusters[:max_clusters]
        
        if not sorted_clusters:
            continue
        
        n_clusters = len(sorted_clusters)
        n_cols = min(3, n_clusters)
        n_rows = (n_clusters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_clusters == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (cid, W_c, cdata) in enumerate(sorted_clusters):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            top_feats = cdata.get("top_features", [])[:top_n_features]
            
            if not top_feats:
                ax.text(0.5, 0.5, "No features", ha='center', va='center')
                ax.set_title(f"Cluster {cid} (W={W_c:.1f})")
                continue
            
            # Create horizontal bar plot
            labels = [f"L{f['layer']}_P{f['position']}_F{f['feature_id']}" for f in top_feats]
            values = [f['value'] for f in top_feats]
            colors = ['coral' if v > 0 else 'steelblue' for v in values]
            
            y_pos = range(len(labels))
            ax.barh(y_pos, values, color=colors, alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=7)
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.set_xlabel("Attribution Value", fontsize=9)
            ax.set_title(f"Cluster {cid} (W={W_c:.1f}, {len(top_feats)} feats)", fontsize=10)
            ax.invert_yaxis()
        
        # Hide empty subplots
        for idx in range(n_clusters, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].axis('off')
        
        fig.suptitle(f"β={beta} — Top Features per Cluster (K={K})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_file = output_dir / f"beta_{beta}_cluster_features.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()


def create_feature_cluster_sankey(
    hierarchy: Dict[str, Any],
    output_file: Path,
    top_n_features: int = 30,
    min_cluster_weight: float = 1.0,
    figsize: Tuple[int, int] = (20, 14)
):
    """Create Sankey-style diagram showing feature-to-cluster assignments across beta.
    
    For each beta, assigns features to clusters by maximum normalized magnitude.
    Shows how features "flow" between clusters as beta changes.
    
    Args:
        hierarchy: Beta hierarchy data
        output_file: Output file path
        top_n_features: Number of top features to track
        min_cluster_weight: Minimum W_c to include cluster
        figsize: Figure size
    """
    beta_values = sorted([float(b) for b in hierarchy.get("beta_levels", {}).keys()])
    
    if len(beta_values) < 2:
        return
    
    # Step 1: Collect all significant features across all betas
    all_features = set()
    for beta in beta_values:
        level_data = hierarchy["beta_levels"].get(beta, hierarchy["beta_levels"].get(str(beta), {}))
        clusters = level_data.get("clusters", {})
        
        for cid, cdata in clusters.items():
            W_c = cdata.get("W_c", 0)
            if W_c < min_cluster_weight:
                continue
            for feat in cdata.get("top_features", [])[:top_n_features]:
                all_features.add((feat["layer"], feat["position"], feat["feature_id"]))
    
    # Limit to top N features by total weighted magnitude
    feature_total_mag = {}
    for feat_key in all_features:
        total = 0
        for beta in beta_values:
            level_data = hierarchy["beta_levels"].get(beta, hierarchy["beta_levels"].get(str(beta), {}))
            for cid, cdata in level_data.get("clusters", {}).items():
                W_c = cdata.get("W_c", 0)
                if W_c < min_cluster_weight:
                    continue
                for feat in cdata.get("top_features", []):
                    if (feat["layer"], feat["position"], feat["feature_id"]) == feat_key:
                        total += abs(feat["value"]) * W_c
        feature_total_mag[feat_key] = total
    
    # Sort and select top features
    sorted_features = sorted(feature_total_mag.items(), key=lambda x: -x[1])
    top_features = [f[0] for f in sorted_features[:top_n_features]]
    
    if not top_features:
        return
    
    # Step 2: For each beta, assign each feature to best cluster
    # feature_assignments[beta][feature_key] = (cluster_id, normalized_mag, color_idx)
    feature_assignments = {}
    cluster_colors_per_beta = {}
    
    for beta in beta_values:
        level_data = hierarchy["beta_levels"].get(beta, hierarchy["beta_levels"].get(str(beta), {}))
        clusters = level_data.get("clusters", {})
        
        # Get significant clusters sorted by weight
        sig_clusters = []
        for cid, cdata in clusters.items():
            W_c = cdata.get("W_c", 0)
            if W_c >= min_cluster_weight:
                sig_clusters.append((int(cid), W_c, cdata))
        sig_clusters.sort(key=lambda x: -x[1])
        
        # Assign color indices to clusters (by rank)
        cluster_to_color = {cid: idx for idx, (cid, _, _) in enumerate(sig_clusters)}
        cluster_colors_per_beta[beta] = cluster_to_color
        
        # Assign each feature to best cluster
        feature_assignments[beta] = {}
        for feat_key in top_features:
            best_cluster = None
            best_mag = 0
            
            for cid, W_c, cdata in sig_clusters:
                for feat in cdata.get("top_features", []):
                    if (feat["layer"], feat["position"], feat["feature_id"]) == feat_key:
                        mag = abs(feat["value"])
                        if mag > best_mag:
                            best_mag = mag
                            best_cluster = cid
            
            if best_cluster is not None:
                color_idx = cluster_to_color.get(best_cluster, 0)
                feature_assignments[beta][feat_key] = (best_cluster, best_mag, color_idx)
            else:
                feature_assignments[beta][feat_key] = (None, 0, -1)
    
    # Step 3: Create visualization
    n_features = len(top_features)
    n_betas = len(beta_values)
    
    # Use a distinctive colormap for clusters
    max_clusters = max(len(cluster_colors_per_beta.get(b, {})) for b in beta_values)
    cluster_cmap = plt.cm.get_cmap('tab20', max(20, max_clusters))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Layout: features on Y-axis, betas on X-axis
    # Draw lines connecting feature positions across betas
    x_positions = np.linspace(0, 1, n_betas)
    y_positions = np.linspace(0, 1, n_features)
    
    # Draw feature paths
    for feat_idx, feat_key in enumerate(top_features):
        y = y_positions[feat_idx]
        
        # Collect points and colors for this feature
        points_x = []
        points_y = []
        colors = []
        
        for beta_idx, beta in enumerate(beta_values):
            x = x_positions[beta_idx]
            assignment = feature_assignments[beta].get(feat_key, (None, 0, -1))
            cluster_id, mag, color_idx = assignment
            
            points_x.append(x)
            points_y.append(y)
            
            if color_idx >= 0:
                colors.append(cluster_cmap(color_idx % 20))
            else:
                colors.append((0.7, 0.7, 0.7, 0.5))  # Gray for unassigned
        
        # Draw segments with changing colors
        for i in range(len(points_x) - 1):
            # Gradient between two colors
            ax.plot([points_x[i], points_x[i+1]], [points_y[i], points_y[i+1]], 
                   color=colors[i], linewidth=2, alpha=0.7)
        
        # Draw markers at each beta
        for i, (x, c) in enumerate(zip(points_x, colors)):
            ax.scatter(x, y, c=[c], s=50, zorder=5, edgecolors='white', linewidth=0.5)
    
    # Add beta labels on X-axis
    for beta_idx, beta in enumerate(beta_values):
        x = x_positions[beta_idx]
        level_data = hierarchy["beta_levels"].get(beta, hierarchy["beta_levels"].get(str(beta), {}))
        K = level_data.get("K", 0)
        n_sig = level_data.get("n_significant_clusters", K)
        ax.text(x, -0.05, f"β={beta}\n(K={K})", ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Add feature labels on Y-axis
    for feat_idx, feat_key in enumerate(top_features):
        y = y_positions[feat_idx]
        label = f"L{feat_key[0]}_P{feat_key[1]}_F{feat_key[2]}"
        ax.text(-0.02, y, label, ha='right', va='center', fontsize=7)
    
    # Add cluster legend (for one representative beta, e.g., the one with most clusters)
    best_beta = max(beta_values, key=lambda b: len(cluster_colors_per_beta.get(b, {})))
    cluster_to_color = cluster_colors_per_beta[best_beta]
    
    # Create legend patches
    from matplotlib.patches import Patch
    legend_patches = []
    sorted_clusters = sorted(cluster_to_color.items(), key=lambda x: x[1])[:15]  # Top 15 clusters
    for cid, color_idx in sorted_clusters:
        level_data = hierarchy["beta_levels"].get(best_beta, hierarchy["beta_levels"].get(str(best_beta), {}))
        W_c = level_data.get("clusters", {}).get(str(cid), {}).get("W_c", 0)
        legend_patches.append(Patch(facecolor=cluster_cmap(color_idx % 20), 
                                    label=f"C{cid} (W={W_c:.0f})"))
    
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.02, 1), 
              fontsize=8, title=f"Clusters at β={best_beta}")
    
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.1, 1.05)
    ax.axis('off')
    ax.set_title("Feature-to-Cluster Assignment Flow Across Beta Values\n"
                 "Each line = one feature, color = assigned cluster (by max magnitude)",
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def create_cluster_sankey_alluvial(
    hierarchy: Dict[str, Any],
    output_file: Path,
    top_n_features: int = 50,
    min_cluster_weight: float = 1.0,
    min_flow_threshold: float = 0.1,
    figsize: Tuple[int, int] = (18, 12)
):
    """Create alluvial/Sankey diagram showing cluster composition changes across beta.
    
    Shows how clusters at one beta "split" or "merge" as beta increases.
    Uses feature overlap to determine cluster relationships.
    
    Args:
        hierarchy: Beta hierarchy data
        output_file: Output file path
        top_n_features: Features per cluster to consider for overlap
        min_cluster_weight: Minimum W_c to include
        min_flow_threshold: Minimum relative flow to show (0-1, fraction of max flow)
        figsize: Figure size
    """
    beta_values = sorted([float(b) for b in hierarchy.get("beta_levels", {}).keys()])
    
    if len(beta_values) < 2:
        return
    
    # Step 1: For each beta, get clusters and their top features
    beta_clusters = {}  # beta -> {cluster_id: set of feature keys}
    beta_cluster_weights = {}  # beta -> {cluster_id: W_c}
    
    for beta in beta_values:
        level_data = hierarchy["beta_levels"].get(beta, hierarchy["beta_levels"].get(str(beta), {}))
        clusters = level_data.get("clusters", {})
        
        beta_clusters[beta] = {}
        beta_cluster_weights[beta] = {}
        
        for cid, cdata in clusters.items():
            W_c = cdata.get("W_c", 0)
            if W_c < min_cluster_weight:
                continue
            
            # Get top features for this cluster
            feat_set = set()
            for feat in cdata.get("top_features", [])[:top_n_features]:
                feat_set.add((feat["layer"], feat["position"], feat["feature_id"]))
            
            beta_clusters[beta][int(cid)] = feat_set
            beta_cluster_weights[beta][int(cid)] = W_c
    
    # Step 2: Compute flow between adjacent beta levels
    # flows[i] = list of (src_cluster, dst_cluster, weight)
    flows = []
    
    for i in range(len(beta_values) - 1):
        beta1, beta2 = beta_values[i], beta_values[i+1]
        clusters1 = beta_clusters[beta1]
        clusters2 = beta_clusters[beta2]
        
        level_flows = []
        for cid1, feats1 in clusters1.items():
            if not feats1:
                continue
            for cid2, feats2 in clusters2.items():
                if not feats2:
                    continue
                # Compute Jaccard overlap
                overlap = len(feats1 & feats2)
                if overlap > 0:
                    # Weight by smaller cluster's weight
                    W1 = beta_cluster_weights[beta1].get(cid1, 0)
                    W2 = beta_cluster_weights[beta2].get(cid2, 0)
                    flow_weight = overlap * min(W1, W2) / max(len(feats1), len(feats2))
                    level_flows.append((cid1, cid2, flow_weight))
        
        flows.append(level_flows)
    
    # Step 3: Create alluvial diagram
    fig, ax = plt.subplots(figsize=figsize)
    
    # Assign consistent colors to clusters across betas using spectral clustering on overlap
    # For simplicity, use rank-based colors per beta
    cluster_cmap = plt.cm.get_cmap('tab20', 20)
    
    # X positions for each beta
    x_positions = np.linspace(0, 1, len(beta_values))
    
    # For each beta, stack clusters vertically
    beta_y_ranges = {}  # beta -> {cluster_id: (y_start, y_end)}
    
    for beta_idx, beta in enumerate(beta_values):
        weights = beta_cluster_weights[beta]
        if not weights:
            beta_y_ranges[beta] = {}
            continue
        
        # Sort by weight
        sorted_clusters = sorted(weights.items(), key=lambda x: -x[1])
        total_weight = sum(w for _, w in sorted_clusters)
        
        y_cursor = 0
        beta_y_ranges[beta] = {}
        
        for cid, W_c in sorted_clusters:
            height = W_c / total_weight if total_weight > 0 else 0
            beta_y_ranges[beta][cid] = (y_cursor, y_cursor + height)
            y_cursor += height + 0.01  # Small gap
    
    # Draw clusters as bars
    for beta_idx, beta in enumerate(beta_values):
        x = x_positions[beta_idx]
        
        for rank, (cid, (y_start, y_end)) in enumerate(beta_y_ranges[beta].items()):
            color = cluster_cmap(rank % 20)
            height = y_end - y_start
            
            # Draw bar
            rect = plt.Rectangle((x - 0.02, y_start), 0.04, height, 
                                  facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8)
            ax.add_patch(rect)
            
            # Add cluster label
            if height > 0.03:
                W_c = beta_cluster_weights[beta].get(cid, 0)
                ax.text(x, y_start + height/2, f"C{cid}", ha='center', va='center', 
                       fontsize=7, fontweight='bold', color='white')
    
    # Draw flows between adjacent betas
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath
    
    for i, level_flows in enumerate(flows):
        beta1, beta2 = beta_values[i], beta_values[i+1]
        x1, x2 = x_positions[i], x_positions[i+1]
        
        # Normalize flows
        if not level_flows:
            continue
        
        max_flow = max(f[2] for f in level_flows)
        if max_flow == 0:
            continue
        
        # Group flows by source cluster for per-cluster pruning
        flows_by_src = {}
        for cid1, cid2, flow_weight in level_flows:
            if cid1 not in flows_by_src:
                flows_by_src[cid1] = []
            flows_by_src[cid1].append((cid2, flow_weight))
        
        # For each source cluster, keep top flows (relative to that cluster's max)
        flows_to_draw = []
        for cid1, outflows in flows_by_src.items():
            if not outflows:
                continue
            
            # Sort by flow weight descending
            outflows.sort(key=lambda x: -x[1])
            cluster_max_flow = outflows[0][1]
            
            if cluster_max_flow == 0:
                continue
            
            # Keep flows above threshold relative to THIS cluster's max
            # Always keep at least the top 1-2 flows per cluster
            kept = 0
            for cid2, flow_weight in outflows:
                relative_to_cluster = flow_weight / cluster_max_flow
                # Keep if above threshold OR if we haven't kept any yet
                if relative_to_cluster >= min_flow_threshold or kept < 2:
                    flows_to_draw.append((cid1, cid2, flow_weight))
                    kept += 1
        
        for cid1, cid2, flow_weight in flows_to_draw:
            if cid1 not in beta_y_ranges[beta1] or cid2 not in beta_y_ranges[beta2]:
                continue
            
            y1_start, y1_end = beta_y_ranges[beta1][cid1]
            y2_start, y2_end = beta_y_ranges[beta2][cid2]
            
            # Flow ribbon from middle of source to middle of dest
            y1 = (y1_start + y1_end) / 2
            y2 = (y2_start + y2_end) / 2
            
            # Width proportional to flow (relative to global max for consistent sizing)
            relative_flow = flow_weight / max_flow
            ribbon_width = 0.02 * relative_flow
            
            # Get color from source cluster
            src_rank = list(beta_y_ranges[beta1].keys()).index(cid1)
            color = cluster_cmap(src_rank % 20)
            
            # Draw curved ribbon using Bezier
            verts = [
                (x1 + 0.02, y1 - ribbon_width/2),
                ((x1 + x2) / 2, y1 - ribbon_width/2),
                ((x1 + x2) / 2, y2 - ribbon_width/2),
                (x2 - 0.02, y2 - ribbon_width/2),
                (x2 - 0.02, y2 + ribbon_width/2),
                ((x1 + x2) / 2, y2 + ribbon_width/2),
                ((x1 + x2) / 2, y1 + ribbon_width/2),
                (x1 + 0.02, y1 + ribbon_width/2),
                (x1 + 0.02, y1 - ribbon_width/2),
            ]
            codes = [MPath.MOVETO] + [MPath.CURVE4] * 3 + [MPath.LINETO] + [MPath.CURVE4] * 3 + [MPath.CLOSEPOLY]
            
            path = MPath(verts, codes)
            patch = PathPatch(path, facecolor=color, edgecolor='none', alpha=0.4)
            ax.add_patch(patch)
    
    # Add beta labels
    for beta_idx, beta in enumerate(beta_values):
        x = x_positions[beta_idx]
        level_data = hierarchy["beta_levels"].get(beta, hierarchy["beta_levels"].get(str(beta), {}))
        K = level_data.get("K", 0)
        n_sig = len(beta_cluster_weights.get(beta, {}))
        ax.text(x, -0.08, f"β={beta}\nK={K} ({n_sig} sig)", 
               ha='center', va='top', fontsize=11, fontweight='bold')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.15, 1.1)
    ax.axis('off')
    ax.set_title("Cluster Flow Across Beta Values (Alluvial Diagram)\n"
                 "Bars = clusters (height ∝ weight), Ribbons = feature overlap between clusters",
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def create_cluster_similarity_heatmap(
    hierarchy: Dict[str, Any],
    output_file: Path,
    metric: str = "spearman",  # "spearman" or "jaccard"
    top_n_features: int = 50,
    min_cluster_weight: float = 1.0
):
    """Create a heatmap showing pairwise cluster similarity across beta values.
    
    Like the Jaccard index heatmap in genomics papers, this shows similarity between
    all clusters linearized across beta values, with clear boundaries between betas.
    
    Args:
        hierarchy: Beta hierarchical map
        output_file: Output PNG path
        metric: "spearman" for Spearman correlation, "jaccard" for Jaccard overlap
        top_n_features: Number of features to use for similarity calculation
        min_cluster_weight: Minimum cluster weight to include
    """
    from scipy.stats import spearmanr
    
    beta_levels = hierarchy.get("beta_levels", {})
    if not beta_levels:
        return
    
    # Collect all clusters across betas
    all_clusters = []  # List of (beta, cluster_id, feature_vector, weight)
    beta_boundaries = [0]  # Track where each beta starts
    
    sorted_betas = sorted([float(b) for b in beta_levels.keys()])
    
    for beta in sorted_betas:
        beta_data = beta_levels.get(beta, beta_levels.get(str(beta), {}))
        clusters = beta_data.get("clusters", {})
        
        # Sort clusters by weight (descending)
        cluster_items = []
        for cid, cdata in clusters.items():
            weight = cdata.get("W_c", 0)
            if weight >= min_cluster_weight:
                cluster_items.append((int(cid), cdata, weight))
        
        cluster_items.sort(key=lambda x: -x[2])  # Sort by weight desc
        
        for cid, cdata, weight in cluster_items:
            features = cdata.get("top_features", [])
            all_clusters.append({
                "beta": beta,
                "cluster_id": cid,
                "features": features,
                "weight": weight,
                "label": f"β{beta}_C{cid}"
            })
        
        beta_boundaries.append(len(all_clusters))
    
    if len(all_clusters) < 2:
        return
    
    n_clusters = len(all_clusters)
    
    # Build feature index mapping (union of all features across clusters)
    all_feature_keys = set()
    for cluster in all_clusters:
        for feat in cluster["features"][:top_n_features]:
            feat_key = (feat.get("layer", 0), feat.get("position", 0), feat.get("feature_id", 0))
            all_feature_keys.add(feat_key)
    
    feat_to_idx = {f: i for i, f in enumerate(sorted(all_feature_keys))}
    n_features = len(feat_to_idx)
    
    # Build feature vectors for each cluster
    cluster_vectors = []
    for cluster in all_clusters:
        vec = np.zeros(n_features)
        for feat in cluster["features"][:top_n_features]:
            feat_key = (feat.get("layer", 0), feat.get("position", 0), feat.get("feature_id", 0))
            if feat_key in feat_to_idx:
                vec[feat_to_idx[feat_key]] = feat.get("value", 0)
        cluster_vectors.append(vec)
    
    cluster_vectors = np.array(cluster_vectors)
    
    # Compute pairwise similarity
    similarity_matrix = np.zeros((n_clusters, n_clusters))
    
    if metric == "spearman":
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Spearman correlation
                    corr, _ = spearmanr(cluster_vectors[i], cluster_vectors[j])
                    similarity_matrix[i, j] = corr if not np.isnan(corr) else 0.0
    else:  # jaccard
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Jaccard: intersection / union of non-zero features
                    set_i = set(np.where(np.abs(cluster_vectors[i]) > 1e-6)[0])
                    set_j = set(np.where(np.abs(cluster_vectors[j]) > 1e-6)[0])
                    
                    if len(set_i | set_j) == 0:
                        similarity_matrix[i, j] = 0.0
                    else:
                        similarity_matrix[i, j] = len(set_i & set_j) / len(set_i | set_j)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create colormap - use blue-yellow like the reference image
    if metric == "spearman":
        cmap = 'RdYlBu_r'  # Red-Yellow-Blue reversed (blue=low, yellow=high)
        vmin, vmax = -1, 1
    else:
        # Custom blue to yellow colormap for Jaccard
        from matplotlib.colors import LinearSegmentedColormap
        colors = [(0.2, 0.2, 0.8), (0.9, 0.9, 0.3)]  # Blue to Yellow
        cmap = LinearSegmentedColormap.from_list('blue_yellow', colors)
        vmin, vmax = 0, 1
    
    # Plot heatmap
    im = ax.imshow(similarity_matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    metric_name = "Spearman Correlation" if metric == "spearman" else "Jaccard Index"
    cbar.set_label(metric_name, fontsize=12)
    
    # Add beta boundaries as red lines
    for i, boundary in enumerate(beta_boundaries[1:-1]):
        ax.axhline(y=boundary - 0.5, color='red', linewidth=2, linestyle='-')
        ax.axvline(x=boundary - 0.5, color='red', linewidth=2, linestyle='-')
    
    # Add beta labels on the sides
    tick_positions = []
    tick_labels = []
    
    for i, beta in enumerate(sorted_betas):
        start = beta_boundaries[i]
        end = beta_boundaries[i + 1]
        if end > start:
            mid = (start + end - 1) / 2
            tick_positions.append(mid)
            n_clusters_in_beta = end - start
            tick_labels.append(f"β={beta}\n(K={n_clusters_in_beta})")
    
    # Set custom ticks
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=10, fontweight='bold')
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=10, fontweight='bold')
    
    # Add minor ticks for individual clusters
    minor_ticks = list(range(n_clusters))
    minor_labels = [c["label"] for c in all_clusters]
    
    ax2 = ax.secondary_xaxis('top')
    ax2.set_xticks(minor_ticks)
    ax2.set_xticklabels(minor_labels, fontsize=6, rotation=90)
    
    ax3 = ax.secondary_yaxis('right')
    ax3.set_yticks(minor_ticks)
    ax3.set_yticklabels(minor_labels, fontsize=6)
    
    # Add annotations for beta blocks
    for i, beta in enumerate(sorted_betas):
        start = beta_boundaries[i]
        end = beta_boundaries[i + 1]
        if end > start:
            # Draw rectangle around diagonal block
            rect = plt.Rectangle((start - 0.5, start - 0.5), end - start, end - start,
                                fill=False, edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
    
    ax.set_title(f"Cluster {metric_name} Across Beta Values\n"
                 f"(Red lines = beta boundaries, Diagonal blocks = within-beta clusters)",
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save triangular version like reference image
    tri_output = output_file.parent / output_file.name.replace('.png', '_triangular.png')
    create_triangular_similarity_heatmap(
        all_clusters, cluster_vectors, sorted_betas, beta_boundaries,
        metric, tri_output
    )


def create_triangular_similarity_heatmap(
    all_clusters: List[Dict],
    cluster_vectors: np.ndarray,
    sorted_betas: List[float],
    beta_boundaries: List[int],
    metric: str,
    output_file: Path
):
    """Create a triangular heatmap ordered by beta (like the reference image)."""
    from scipy.stats import spearmanr
    from matplotlib.colors import LinearSegmentedColormap
    
    n_clusters = len(all_clusters)
    if n_clusters < 3:
        return
    
    # Compute similarity matrix
    if metric == "spearman":
        similarity = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(i, n_clusters):
                if i == j:
                    similarity[i, j] = 1.0
                else:
                    corr, _ = spearmanr(cluster_vectors[i], cluster_vectors[j])
                    corr = 0.0 if np.isnan(corr) else corr
                    similarity[i, j] = corr
                    similarity[j, i] = corr
    else:
        # Jaccard similarity
        similarity = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(i, n_clusters):
                if i == j:
                    similarity[i, j] = 1.0
                else:
                    set_i = set(np.where(np.abs(cluster_vectors[i]) > 1e-6)[0])
                    set_j = set(np.where(np.abs(cluster_vectors[j]) > 1e-6)[0])
                    if len(set_i | set_j) == 0:
                        jac = 0.0
                    else:
                        jac = len(set_i & set_j) / len(set_i | set_j)
                    similarity[i, j] = jac
                    similarity[j, i] = jac
    
    # Create figure - no dendrogram, just triangular heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create triangular mask (upper triangle hidden, show lower triangle)
    mask = np.triu(np.ones_like(similarity, dtype=bool), k=1)
    masked_sim = np.ma.masked_where(mask, similarity)
    
    # Custom colormap (blue to yellow like reference)
    colors = [(0.2, 0.2, 0.7), (0.95, 0.95, 0.4)]  # Blue to Yellow
    cmap = LinearSegmentedColormap.from_list('blue_yellow', colors)
    
    # Plot heatmap
    vmin = 0 if metric == "jaccard" else -1
    im = ax.imshow(masked_sim, cmap=cmap, vmin=vmin, vmax=1, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, aspect=25, pad=0.02)
    metric_name = "Spearman ρ" if metric == "spearman" else "Jaccard Index"
    cbar.set_label(metric_name, fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    
    # Add beta boundary lines (black)
    for boundary in beta_boundaries[1:-1]:
        ax.axhline(y=boundary - 0.5, color='black', linewidth=2, linestyle='-')
        ax.axvline(x=boundary - 0.5, color='black', linewidth=2, linestyle='-')
    
    # Labels on y-axis (left side) - cluster labels colored by beta
    labels = [c["label"] for c in all_clusters]
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels(labels, fontsize=10)
    
    # Labels on x-axis (bottom) - rotated
    ax.set_xticks(range(n_clusters))
    ax.set_xticklabels(labels, fontsize=10, rotation=90)
    
    # Color-code labels by beta - use Dark2 colorblind-friendly palette
    # Dark2 colors: teal, orange, purple, pink, green, yellow, brown, gray
    dark2_colors = [
        '#1b9e77',  # teal
        '#d95f02',  # orange
        '#7570b3',  # purple
        '#e7298a',  # pink
        '#66a61e',  # green
        '#e6ab02',  # yellow/gold
        '#a6761d',  # brown
        '#666666',  # gray
    ]
    beta_to_color = {beta: dark2_colors[i % len(dark2_colors)] for i, beta in enumerate(sorted_betas)}
    
    for i, cluster in enumerate(all_clusters):
        color = beta_to_color.get(cluster["beta"], 'black')
        ax.get_xticklabels()[i].set_color(color)
        ax.get_yticklabels()[i].set_color(color)
    
    # Add beta interval annotations on the side
    for i, beta in enumerate(sorted_betas):
        start = beta_boundaries[i]
        end = beta_boundaries[i + 1]
        if end > start:
            mid_y = (start + end - 1) / 2
            n_in_beta = end - start
            # Add bracket annotation on the left
            ax.annotate(f'β={beta}\n(K={n_in_beta})', 
                       xy=(-0.5, mid_y), xytext=(-3, mid_y),
                       fontsize=14, fontweight='bold',
                       ha='right', va='center',
                       color=beta_to_color[beta],
                       annotation_clip=False)
    
    # Add legend for beta colors (upper right, inside the empty triangle area)
    legend_handles = [plt.Line2D([0], [0], marker='s', color='w', 
                                  markerfacecolor=beta_to_color[b], markersize=14,
                                  label=f'β={b}') 
                      for b in sorted_betas]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=12,
             title="Beta Values", title_fontsize=14, framealpha=0.9)
    
    ax.set_title(f"Cluster {metric_name} (Ordered by β)\n"
                f"Black lines = beta boundaries",
                fontsize=16, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Processing Function
# =============================================================================

def process_prefix(
    prefix_id: str,
    clustering_dir: Path,
    samples_dir: Path,
    attribution_graphs_dir: Path,
    output_dir: Path,
    beta_values: List[float] = DEFAULT_BETA_VALUES,
    gamma: float = DEFAULT_GAMMA,
    top_n: int = DEFAULT_TOP_N_FEATURES,
    max_samples_per_cluster: int = DEFAULT_MAX_CONTINUATIONS_PER_CLUSTER,
    api_key: Optional[str] = None,
    skip_llm: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Process a single prefix to generate cluster analysis.
    
    Args:
        prefix_id: Prefix identifier
        clustering_dir: Directory with clustering results
        samples_dir: Directory with branch samples
        attribution_graphs_dir: Directory with attribution context
        output_dir: Output directory
        beta_values: Beta values to include
        gamma: Target gamma value
        top_n: Number of top features
        max_samples_per_cluster: Max continuations for LLM labeling
        api_key: OpenAI API key
        skip_llm: Skip LLM labeling
        logger: Optional logger
        
    Returns:
        Dict with processing results
    """
    if logger:
        logger.info(f"Processing prefix: {prefix_id}")
    
    # Load data
    clustering_file = clustering_dir / f"{prefix_id}_sweep_results.json"
    branches_file = samples_dir / f"{prefix_id}_branches.json"
    
    if not clustering_file.exists():
        if logger:
            logger.warning(f"Clustering file not found: {clustering_file}")
        return {"error": "clustering_file_not_found"}
    
    if not branches_file.exists():
        if logger:
            logger.warning(f"Branches file not found: {branches_file}")
        return {"error": "branches_file_not_found"}
    
    clustering_data = load_clustering_data(clustering_file)
    branches_data = load_branches_data(branches_file)
    
    # Load attribution context for feature mapping
    context_data = load_attribution_context(attribution_graphs_dir, prefix_id)
    active_features = get_active_features_mapping(context_data)
    
    if logger:
        logger.info(f"  Loaded {len(active_features)} features, {len(clustering_data.get('grid', []))} grid entries")
    
    # Build beta hierarchy
    hierarchy = build_beta_hierarchical_map(
        clustering_data, active_features, 
        beta_values=beta_values, 
        gamma=gamma, 
        top_n=top_n,
        min_cluster_weight=1.0,  # Filter outlier clusters with W_c < 1.0
        logger=logger
    )
    
    # Generate cluster labels
    labels_by_beta = {}
    grid = clustering_data.get("grid", [])
    
    for beta in beta_values:
        # Find grid entry for this beta (closest gamma)
        matching_entries = [e for e in grid if e.get("beta") == beta]
        if not matching_entries:
            continue
        
        closest_entry = min(matching_entries, key=lambda x: abs(x.get("gamma", 0) - gamma))
        
        if skip_llm:
            # Generate simple labels without LLM
            labels = {int(c): f"Cluster {c}" for c in closest_entry.get("components", {}).keys()}
        else:
            if logger:
                logger.info(f"  Generating LLM labels for beta={beta}")
            labels = label_all_clusters(
                branches_data, closest_entry, api_key, max_samples_per_cluster, logger
            )
        
        labels_by_beta[beta] = labels
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare metadata
    metadata = {
        "prefix_id": prefix_id,
        "prefix": branches_data.get("prefix", "")[:200],
        "beta_values": beta_values,
        "target_gamma": gamma,
        "top_n_features": top_n,
        "n_continuations": len(branches_data.get("continuations", [])),
        "H_0_dim": len(clustering_data.get("H_0", []))
    }
    
    # Save outputs
    labels_file = output_dir / f"{prefix_id}_cluster_labels.json"
    activation_map_file = output_dir / f"{prefix_id}_activation_map.json"
    csv_file = output_dir / f"{prefix_id}_activation_map.csv"
    heatmap_file = output_dir / f"{prefix_id}_beta_heatmap.png"
    per_cluster_file = output_dir / f"{prefix_id}_per_cluster_heatmap.png"
    
    save_cluster_labels_json(labels_by_beta, labels_file, metadata)
    save_activation_map_json(hierarchy, labels_by_beta, activation_map_file, metadata)
    save_activation_map_csv(hierarchy, labels_by_beta, csv_file)
    create_beta_heatmap(hierarchy, heatmap_file, top_n_features=min(top_n * 2, 50))
    
    # Create per-cluster visualization (shows K different clusters for each beta)
    create_per_cluster_heatmap(
        hierarchy, per_cluster_file,
        top_n_features=top_n,
        max_clusters_per_beta=8,
        min_cluster_weight=1.0
    )
    
    # Create per-beta cluster feature bar plots
    create_cluster_feature_barplot(
        hierarchy, output_dir,
        top_n_features=top_n,
        max_clusters=10,
        min_cluster_weight=1.0
    )
    
    # Create Sankey/alluvial diagrams showing feature-cluster flow across beta
    sankey_file = output_dir / f"{prefix_id}_feature_cluster_flow.png"
    create_feature_cluster_sankey(
        hierarchy, sankey_file,
        top_n_features=min(top_n * 2, 40),
        min_cluster_weight=1.0
    )
    
    alluvial_file = output_dir / f"{prefix_id}_cluster_alluvial.png"
    create_cluster_sankey_alluvial(
        hierarchy, alluvial_file,
        top_n_features=top_n,
        min_cluster_weight=1.0,
        min_flow_threshold=0.15  # Prune flows below 15% of max
    )
    
    # Create Jaccard/Spearman similarity heatmaps
    spearman_file = output_dir / f"{prefix_id}_cluster_spearman.png"
    create_cluster_similarity_heatmap(
        hierarchy, spearman_file,
        metric="spearman",
        top_n_features=top_n,
        min_cluster_weight=1.0
    )
    
    jaccard_file = output_dir / f"{prefix_id}_cluster_jaccard.png"
    create_cluster_similarity_heatmap(
        hierarchy, jaccard_file,
        metric="jaccard",
        top_n_features=top_n,
        min_cluster_weight=1.0
    )
    
    if logger:
        logger.info(f"  Saved outputs to {output_dir}")
    
    return {
        "prefix_id": prefix_id,
        "n_beta_levels": len(hierarchy.get("beta_levels", {})),
        "n_clusters_total": sum(
            len(level.get("clusters", {})) 
            for level in hierarchy.get("beta_levels", {}).values()
        ),
        "output_files": [str(f) for f in [labels_file, activation_map_file, csv_file, heatmap_file]]
    }


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cluster Analysis and Beta-Hierarchical Activation Map"
    )
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Base results directory (contains 2_branch_sampling, 3_attribution_graphs, 5_clustering)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: {results-dir}/7_validation/cluster_analysis)")
    parser.add_argument("--prefix-id", type=str, default=None,
                        help="Process only a specific prefix")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of prefixes to process")
    parser.add_argument("--beta-values", type=float, nargs="+", default=DEFAULT_BETA_VALUES,
                        help="Beta values to include in hierarchy")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA,
                        help="Target gamma value for hierarchy")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N_FEATURES,
                        help="Number of top features per cluster")
    parser.add_argument("--max-continuations", type=int, default=DEFAULT_MAX_CONTINUATIONS_PER_CLUSTER,
                        help="Max continuations per cluster for LLM labeling")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM-based cluster labeling")
    parser.add_argument("--log-dir", type=Path, default=None,
                        help="Log directory")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce logging verbosity")
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = get_log_path("7c_cluster_analysis", args.log_dir)
    log_level = logging.WARNING if args.quiet else logging.INFO
    logger = setup_logger("cluster_analysis", log_file=log_file, level=log_level)
    
    logger.info("=" * 60)
    logger.info("CLUSTER ANALYSIS AND BETA-HIERARCHICAL ACTIVATION MAP")
    logger.info("=" * 60)
    
    # Setup directories
    results_dir = args.results_dir
    clustering_dir = results_dir / "5_clustering"
    samples_dir = results_dir / "2_branch_sampling"
    attribution_graphs_dir = results_dir / "3_attribution_graphs"
    
    output_dir = args.output_dir or (results_dir / "7_validation" / "cluster_analysis")
    
    # Validate directories
    for d, name in [(clustering_dir, "clustering"), (samples_dir, "samples"), 
                    (attribution_graphs_dir, "attribution_graphs")]:
        if not d.exists():
            logger.error(f"{name} directory not found: {d}")
            sys.exit(1)
    
    # Find prefixes to process
    clustering_files = sorted(clustering_dir.glob("*_sweep_results.json"))
    prefix_ids = [f.stem.replace("_sweep_results", "") for f in clustering_files]
    
    if args.prefix_id:
        if args.prefix_id in prefix_ids:
            prefix_ids = [args.prefix_id]
        else:
            logger.error(f"Prefix {args.prefix_id} not found")
            sys.exit(1)
    
    if args.max_samples and len(prefix_ids) > args.max_samples:
        prefix_ids = prefix_ids[:args.max_samples]
    
    logger.info(f"Processing {len(prefix_ids)} prefixes")
    logger.info(f"Beta values: {args.beta_values}")
    logger.info(f"Target gamma: {args.gamma}")
    logger.info(f"Top-N features: {args.top_n}")
    logger.info(f"LLM labeling: {'disabled' if args.skip_llm else 'enabled'}")
    logger.info(f"Output directory: {output_dir}")
    
    # Process prefixes
    results = []
    for prefix_id in tqdm(prefix_ids, desc="Processing prefixes"):
        try:
            result = process_prefix(
                prefix_id=prefix_id,
                clustering_dir=clustering_dir,
                samples_dir=samples_dir,
                attribution_graphs_dir=attribution_graphs_dir,
                output_dir=output_dir,
                beta_values=args.beta_values,
                gamma=args.gamma,
                top_n=args.top_n,
                max_samples_per_cluster=args.max_continuations,
                api_key=None,
                skip_llm=args.skip_llm,
                logger=logger
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {prefix_id}: {e}")
            results.append({"prefix_id": prefix_id, "error": str(e)})
    
    # Save summary
    summary = {
        "n_prefixes": len(prefix_ids),
        "n_successful": len([r for r in results if "error" not in r]),
        "beta_values": args.beta_values,
        "gamma": args.gamma,
        "results": results
    }
    save_json(summary, output_dir / "analysis_summary.json")
    
    logger.info("\n" + "=" * 60)
    logger.info("CLUSTER ANALYSIS COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Processed: {summary['n_successful']}/{summary['n_prefixes']} prefixes")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
