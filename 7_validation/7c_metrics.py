#!/usr/bin/env python3
"""7c_metrics.py - Metrics computation and aggregation.

This module contains functions for:
1. Log probability computation
2. Steering metrics computation (dose-response, win rate, correlations)
3. Aggregation and summary output
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

# Import shared utilities (using same pattern as other modules)
import importlib.util
from pathlib import Path as _Path

_utils_spec = importlib.util.spec_from_file_location("7c_utils", _Path(__file__).parent / "7c_utils.py")
_utils_module = importlib.util.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(_utils_module)

# Imported utility functions
compute_correlation_and_r2 = _utils_module.compute_correlation_and_r2
compute_spearman_correlation = _utils_module.compute_spearman_correlation
SweepKey = _utils_module.SweepKey


# =============================================================================
# Log Probability Computation
# =============================================================================

def compute_continuation_log_prob_batched(
    logits: torch.Tensor,
    batch_cont_info: List[Tuple[List[int], int]],
) -> List[float]:
    """Compute continuation log probabilities for a batch.

    Args:
        logits: Batched logits [batch_size, max_seq_len, vocab_size]
        batch_cont_info: List of (continuation_token_ids, continuation_start) tuples
    Returns:
        List of log probabilities, one per batch item
    """
    log_probs_list = []
    seq_len = logits.shape[1]

    for batch_idx, (cont_ids, cont_start) in enumerate(batch_cont_info):
        if not cont_ids:
            log_probs_list.append(0.0)
            continue

        valid_len = min(len(cont_ids), seq_len - cont_start)

        if valid_len <= 0:
            log_probs_list.append(0.0)
            continue

        # Vectorized: gather all positions at once
        positions = torch.arange(cont_start, cont_start + valid_len, device=logits.device)
        cont_logits = logits[batch_idx, positions, :].float()  # [valid_len, vocab_size]
        log_probs = F.log_softmax(cont_logits, dim=-1)  # [valid_len, vocab_size]

        # Select log probs for target tokens
        token_ids = torch.tensor(cont_ids[:valid_len], device=logits.device, dtype=torch.long)
        selected_log_probs = log_probs[torch.arange(valid_len, device=logits.device), token_ids]

        log_P = float(selected_log_probs.sum().item())
        log_probs_list.append(log_P)

    return log_probs_list


def compute_mean_target_logit_and_prob_batched(
    logits: torch.Tensor,
    batch_cont_info: List[Tuple[List[int], int]],
) -> List[Tuple[float, float]]:
    """Compute (mean target logit, mean target prob) for each continuation in a batch.

    Teacher-forced: uses the provided target continuation tokens at their positions.

    Args:
        logits: Batched logits [batch_size, max_seq_len, vocab_size]
        batch_cont_info: List of (continuation_token_ids, continuation_start) tuples

    Returns:
        List[(mean_target_logit, mean_target_prob)] one per batch item.
    """
    out: List[Tuple[float, float]] = []
    seq_len = logits.shape[1]

    for batch_idx, (cont_ids, cont_start) in enumerate(batch_cont_info):
        if not cont_ids:
            out.append((0.0, 0.0))
            continue

        valid_len = min(len(cont_ids), seq_len - cont_start)
        if valid_len <= 0:
            out.append((0.0, 0.0))
            continue

        positions = torch.arange(cont_start, cont_start + valid_len, device=logits.device)
        cont_logits = logits[batch_idx, positions, :].float()  # [valid_len, vocab_size]
        token_ids = torch.tensor(cont_ids[:valid_len], device=logits.device, dtype=torch.long)

        target_logits = cont_logits[torch.arange(valid_len, device=logits.device), token_ids]  # [valid_len]
        mean_target_logit = float(target_logits.mean().item())

        log_probs = F.log_softmax(cont_logits, dim=-1)
        target_logprobs = log_probs[torch.arange(valid_len, device=logits.device), token_ids]  # [valid_len]
        mean_target_prob = float(torch.exp(target_logprobs).mean().item())

        out.append((mean_target_logit, mean_target_prob))

    return out


def compute_delta_curve_metrics(
    steered: Dict[int, Dict[float, List[float]]],
    original: Dict[int, Dict[float, List[float]]],
    epsilons: List[float],
    metric_prefix: str,
) -> Dict[str, Any]:
    """Generic cluster-level metrics for a scalar signal per continuation (already token-averaged).

    For each cluster, each epsilon:
      - mean_*_original: mean over continuations
      - mean_*_steered:  mean over continuations
      - *_diff: steered - original

    Also computes slope/corr/r2/spearman of *_diff vs epsilon per cluster.
    """
    cluster_ids = sorted(steered.keys())
    per_cluster = {}

    def _safe_slope(x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 2 or y.size < 2:
            return 0.0
        if float(np.std(x)) == 0.0:
            return 0.0
        # least-squares slope
        return float(np.polyfit(x, y, 1)[0])

    for c in cluster_ids:
        mean_original = []
        mean_steered = []
        diffs = []

        for eps in epsilons:
            orig = original[c].get(eps, [])
            st = steered[c].get(eps, [])
            if not orig or not st:
                m_orig = 0.0
                m_st = 0.0
            else:
                m_orig = float(np.mean(orig))
                m_st = float(np.mean(st))
            mean_original.append(m_orig)
            mean_steered.append(m_st)
            diffs.append(m_st - m_orig)

        X = np.asarray(epsilons, dtype=np.float64)
        Y = np.asarray(diffs, dtype=np.float64)
        corr, r2 = compute_correlation_and_r2(X, Y)
        spearman = compute_spearman_correlation(X, Y)
        slope = _safe_slope(X, Y)

        per_cluster[c] = {
            f"mean_{metric_prefix}_original": {eps: mean_original[i] for i, eps in enumerate(epsilons)},
            f"mean_{metric_prefix}_steered": {eps: mean_steered[i] for i, eps in enumerate(epsilons)},
            f"{metric_prefix}_diff": {eps: diffs[i] for i, eps in enumerate(epsilons)},
            f"{metric_prefix}_slope": slope,
            f"{metric_prefix}_r2": r2,
            f"{metric_prefix}_corr": corr,
            f"{metric_prefix}_spearman": spearman,
            "n_samples": len(original[c].get(epsilons[0], [])) if epsilons else 0,
        }

    return {f"per_cluster_{metric_prefix}": per_cluster}


def compute_h_c_norm(h_c_vals: List[float]) -> float:
    """Compute ||H_c|| norm from H_c values.

    Args:
        h_c_vals: List of H_c values

    Returns:
        L2 norm, minimum 1.0 to avoid division by zero
    """
    if not h_c_vals:
        return 1.0
    norm = np.linalg.norm(h_c_vals)
    return norm if norm > 0 else 1.0


# =============================================================================
# Steering Metrics Computation
# =============================================================================

def compute_centered_logit_metrics(
    centered_logits_steered: Dict[int, Dict[float, List[float]]],
    centered_logits_original: Dict[int, Dict[float, List[float]]],
    epsilons: List[float]
) -> Dict[str, Any]:
    """Compute cluster-level centered logit metrics (both mean and sum aggregation).

    For each cluster:
    - mean_centered_logit = mean across continuations of (mean across tokens of centered logit)
    - sum_centered_logit = sum across continuations of (mean across tokens of centered logit)

    Uses absolute difference (not relative) for correlation with epsilon.

    Args:
        centered_logits_steered: {cluster_id: {epsilon: [mean_centered_logit per continuation]}}
        centered_logits_original: {cluster_id: {epsilon: [mean_centered_logit per continuation]}}
        epsilons: List of epsilon values

    Returns:
        Dict with per-cluster centered logit metrics:
        - mean_centered_logit_*: Mean aggregation across continuations
        - sum_centered_logit_*: Sum aggregation across continuations
        - centered_logit_diff: steered - original (absolute difference)
    """
    cluster_ids = sorted(centered_logits_steered.keys())
    per_cluster_logit_stats = {}

    for c in cluster_ids:
        mean_original = []
        mean_steered = []
        mean_diffs = []
        sum_original = []
        sum_steered = []
        sum_diffs = []

        for eps in epsilons:
            logits_orig = centered_logits_original[c].get(eps, [])
            logits_steer = centered_logits_steered[c].get(eps, [])

            if not logits_orig or not logits_steer:
                mean_original.append(0.0)
                mean_steered.append(0.0)
                mean_diffs.append(0.0)
                sum_original.append(0.0)
                sum_steered.append(0.0)
                sum_diffs.append(0.0)
                continue

            # Mean across continuations (each entry is already mean across tokens)
            m_orig = float(np.mean(logits_orig))
            m_steer = float(np.mean(logits_steer))
            mean_diff = m_steer - m_orig

            # Sum across continuations (each entry is already mean across tokens)
            s_orig = float(np.sum(logits_orig))
            s_steer = float(np.sum(logits_steer))
            sum_diff = s_steer - s_orig

            mean_original.append(m_orig)
            mean_steered.append(m_steer)
            mean_diffs.append(mean_diff)
            sum_original.append(s_orig)
            sum_steered.append(s_steer)
            sum_diffs.append(sum_diff)

        # Compute correlation for centered logit diff vs epsilon (using absolute diff)
        X = np.array(epsilons)
        
        # Mean-based correlations
        Y_mean = np.array(mean_diffs)
        mean_logit_corr, mean_logit_r2 = compute_correlation_and_r2(X, Y_mean)
        mean_logit_spearman = compute_spearman_correlation(X, Y_mean)
        
        # Sum-based correlations
        Y_sum = np.array(sum_diffs)
        sum_logit_corr, sum_logit_r2 = compute_correlation_and_r2(X, Y_sum)
        sum_logit_spearman = compute_spearman_correlation(X, Y_sum)

        per_cluster_logit_stats[c] = {
            # Mean-based metrics (original)
            'mean_centered_logit_original': {eps: mean_original[i] for i, eps in enumerate(epsilons)},
            'mean_centered_logit_steered': {eps: mean_steered[i] for i, eps in enumerate(epsilons)},
            'centered_logit_diff': {eps: mean_diffs[i] for i, eps in enumerate(epsilons)},
            'centered_logit_r2': mean_logit_r2,
            'centered_logit_corr': mean_logit_corr,
            'centered_logit_spearman': mean_logit_spearman,
            # Sum-based metrics (new)
            'sum_centered_logit_original': {eps: sum_original[i] for i, eps in enumerate(epsilons)},
            'sum_centered_logit_steered': {eps: sum_steered[i] for i, eps in enumerate(epsilons)},
            'sum_centered_logit_diff': {eps: sum_diffs[i] for i, eps in enumerate(epsilons)},
            'sum_centered_logit_r2': sum_logit_r2,
            'sum_centered_logit_corr': sum_logit_corr,
            'sum_centered_logit_spearman': sum_logit_spearman,
            'n_samples': len(logits_orig) if logits_orig else 0
        }

    # Add alias: demeaned_logit == centered_logit (same quantity, different naming)
    per_cluster_demeaned = {}
    for c, stats in per_cluster_logit_stats.items():
        per_cluster_demeaned[c] = {
            'mean_demeaned_logit_original': stats['mean_centered_logit_original'],
            'mean_demeaned_logit_steered': stats['mean_centered_logit_steered'],
            'demeaned_logit_diff': stats['centered_logit_diff'],
            'demeaned_logit_r2': stats['centered_logit_r2'],
            'demeaned_logit_corr': stats['centered_logit_corr'],
            'demeaned_logit_spearman': stats['centered_logit_spearman'],
            # Sum-based aliases
            'sum_demeaned_logit_original': stats['sum_centered_logit_original'],
            'sum_demeaned_logit_steered': stats['sum_centered_logit_steered'],
            'sum_demeaned_logit_diff': stats['sum_centered_logit_diff'],
            'sum_demeaned_logit_r2': stats['sum_centered_logit_r2'],
            'sum_demeaned_logit_corr': stats['sum_centered_logit_corr'],
            'sum_demeaned_logit_spearman': stats['sum_centered_logit_spearman'],
            'n_samples': stats.get('n_samples', 0),
        }

    return {
        'per_cluster_logit': per_cluster_logit_stats,
        'per_cluster_demeaned_logit': per_cluster_demeaned,
    }


def compute_cluster_mass_metrics(
    log_probs_steered: Dict[int, Dict[float, List[float]]],
    log_probs_original: Dict[int, Dict[float, List[float]]],
    epsilons: List[float]
) -> Dict[str, Any]:
    """Compute cluster-level mass metrics (less sensitive to outliers).

    Cluster mass = sum of probabilities (exp(log_P)) across all continuations in cluster.
    This is more robust than mean of relative changes which can be skewed by outliers.

    NOTE: In practice, these masses can be extremely tiny (e.g., ~1e-200) because each
    continuation probability is a product over many tokens. To make the metric numerically
    stable and interpretable, we also compute:
        - cluster_log_mass_* = log(sum(exp(log_P))) using stable log-sum-exp
        - delta_log_mass = log_mass_steered - log_mass_original
      and correlate delta_log_mass with epsilon.

    Args:
        log_probs_steered: {cluster_id: {epsilon: [log_P_steered]}}
        log_probs_original: {cluster_id: {epsilon: [log_P_original]}}
        epsilons: List of epsilon values

    Returns:
        Dict with per-cluster mass metrics:
        - cluster_mass_original: Sum of P(continuation) in cluster (baseline)
        - cluster_mass_steered: Sum of P(continuation) in cluster (steered)
        - cluster_relative_diff: (mass_steered - mass_original) / mass_original
        - cluster_win_rate: 1 if mass_steered > mass_original, 0 otherwise
        - cluster_log_mass_original: log(sum(exp(log_P_original))) (stable)
        - cluster_log_mass_steered: log(sum(exp(log_P_steered))) (stable)
        - delta_log_mass: cluster_log_mass_steered - cluster_log_mass_original
    """
    cluster_ids = sorted(log_probs_steered.keys())
    per_cluster_mass_stats = {}

    def _logsumexp(vals: List[float]) -> float:
        """Stable log(sum(exp(vals))) for Python floats."""
        if not vals:
            return 0.0
        v = np.asarray(vals, dtype=np.float64)
        if v.size == 0:
            return 0.0
        m = float(np.max(v))
        # if all -inf, return -inf (but shouldn't happen with finite log probs)
        if not np.isfinite(m):
            return -np.inf
        return float(m + np.log(np.sum(np.exp(v - m))))

    for c in cluster_ids:
        mass_original = []
        mass_steered = []
        relative_diffs = []
        win_rates = []
        log_mass_original = []
        log_mass_steered = []
        delta_log_mass = []

        for eps in epsilons:
            log_P_orig = log_probs_original[c].get(eps, [])
            log_P_steer = log_probs_steered[c].get(eps, [])

            if not log_P_orig or not log_P_steer:
                mass_original.append(0.0)
                mass_steered.append(0.0)
                relative_diffs.append(0.0)
                win_rates.append(0.0)
                log_mass_original.append(0.0)
                log_mass_steered.append(0.0)
                delta_log_mass.append(0.0)
                continue

            # Cluster mass = sum of probabilities
            m_orig = float(np.sum(np.exp(log_P_orig)))
            m_steer = float(np.sum(np.exp(log_P_steer)))

            # Stable log-mass (log-sum-exp)
            lm_orig = _logsumexp(log_P_orig)
            lm_steer = _logsumexp(log_P_steer)
            dlm = lm_steer - lm_orig

            # Relative difference
            if m_orig != 0:
                rel_diff = (m_steer - m_orig) / m_orig
            else:
                rel_diff = 0.0

            # Win rate (binary)
            win = 1.0 if m_steer > m_orig else 0.0

            mass_original.append(m_orig)
            mass_steered.append(m_steer)
            relative_diffs.append(rel_diff)
            win_rates.append(win)
            log_mass_original.append(lm_orig)
            log_mass_steered.append(lm_steer)
            delta_log_mass.append(dlm)

        # Compute correlation for cluster mass relative diff
        X = np.array(epsilons)
        Y = np.array(relative_diffs)
        mass_corr, mass_r2 = compute_correlation_and_r2(X, Y)
        mass_spearman = compute_spearman_correlation(X, Y)

        # Compute correlation for delta_log_mass (stable)
        Y_log = np.array(delta_log_mass)
        log_mass_corr, log_mass_r2 = compute_correlation_and_r2(X, Y_log)
        log_mass_spearman = compute_spearman_correlation(X, Y_log)

        # Compute correlation for cluster mass win rate
        Y_win = np.array(win_rates)
        mass_win_corr, mass_win_r2 = compute_correlation_and_r2(X, Y_win)
        mass_win_spearman = compute_spearman_correlation(X, Y_win)

        per_cluster_mass_stats[c] = {
            'cluster_mass_original': {eps: mass_original[i] for i, eps in enumerate(epsilons)},
            'cluster_mass_steered': {eps: mass_steered[i] for i, eps in enumerate(epsilons)},
            'cluster_relative_diff': {eps: relative_diffs[i] for i, eps in enumerate(epsilons)},
            'cluster_win_rate': {eps: win_rates[i] for i, eps in enumerate(epsilons)},
            'cluster_mass_r2': mass_r2,
            'cluster_mass_corr': mass_corr,
            'cluster_mass_spearman': mass_spearman,
            'cluster_log_mass_original': {eps: log_mass_original[i] for i, eps in enumerate(epsilons)},
            'cluster_log_mass_steered': {eps: log_mass_steered[i] for i, eps in enumerate(epsilons)},
            'delta_log_mass': {eps: delta_log_mass[i] for i, eps in enumerate(epsilons)},
            'delta_log_mass_r2': log_mass_r2,
            'delta_log_mass_corr': log_mass_corr,
            'delta_log_mass_spearman': log_mass_spearman,
            'cluster_mass_win_r2': mass_win_r2,
            'cluster_mass_win_corr': mass_win_corr,
            'cluster_mass_win_spearman': mass_win_spearman,
            'n_samples': len(log_P_orig) if log_P_orig else 0
        }

    return {'per_cluster_mass': per_cluster_mass_stats}


def compute_steering_metrics(
    dose_response: Dict[int, Dict[float, List[float]]],
    log_deltas: Dict[int, Dict[float, List[float]]],
    epsilons: List[float],
    log_probs_steered: Dict[int, Dict[float, List[float]]] = None,
    log_probs_original: Dict[int, Dict[float, List[float]]] = None,
    centered_logits_steered: Dict[int, Dict[float, List[float]]] = None,
    centered_logits_original: Dict[int, Dict[float, List[float]]] = None,
    target_logits_steered: Dict[int, Dict[float, List[float]]] = None,
    target_logits_original: Dict[int, Dict[float, List[float]]] = None,
    target_probs_steered: Dict[int, Dict[float, List[float]]] = None,
    target_probs_original: Dict[int, Dict[float, List[float]]] = None,
) -> Dict[str, Any]:
    """Compute all metrics for a sweep configuration.

    Args:
        dose_response: {cluster_id: {epsilon: [rel_changes]}}
        log_deltas: {cluster_id: {epsilon: [log_deltas]}}
        epsilons: List of epsilon values
        log_probs_steered: {cluster_id: {epsilon: [log_P_steered]}} (optional, for cluster mass metrics)
        log_probs_original: {cluster_id: {epsilon: [log_P_original]}} (optional, for cluster mass metrics)
        centered_logits_steered: {cluster_id: {epsilon: [mean_centered_logit]}} (optional, for centered logit metrics)
        centered_logits_original: {cluster_id: {epsilon: [mean_centered_logit]}} (optional, for centered logit metrics)

    Returns:
        Dict with per-cluster and aggregated metrics:
        - per_cluster: {cluster_id: {mean_delta_prob, std_delta_prob, win_rate, r2, corr, ...}}
        - per_cluster_mass: {cluster_id: {cluster_mass_*, cluster_relative_diff, ...}} (if log probs provided)
        - per_cluster_logit: {cluster_id: {centered_logit_*, ...}} (if centered logits provided)
        - mean_r2, mean_corr, mean_win_r2, mean_win_corr
        - mean_logit_r2, mean_logit_corr (if centered logits provided)
        - n_clusters_with_effect
    """
    cluster_ids = sorted(dose_response.keys())

    per_cluster_stats = {}
    all_r2 = []
    all_corr = []
    all_spearman = []
    all_win_r2 = []
    all_win_corr = []
    all_win_spearman = []

    for c in cluster_ids:
        means = []
        stds = []
        win_rates = []

        for eps in epsilons:
            vals = dose_response[c].get(eps, [])
            log_vals = log_deltas[c].get(eps, [])

            means.append(np.mean(vals) if vals else 0.0)
            stds.append(np.std(vals) if vals else 0.0)

            # Win-rate: fraction of positive log_deltas
            win_rate = np.mean([1.0 if v > 0 else 0.0 for v in log_vals]) if log_vals else 0.0
            win_rates.append(win_rate)

        # Compute correlation and R^2 for dose-response (mean rel_change)
        X = np.array(epsilons)
        Y = np.array(means)
        corr, r2 = compute_correlation_and_r2(X, Y)
        spearman = compute_spearman_correlation(X, Y)

        # Compute correlation and R^2 for win-rate
        X_win = np.array(epsilons)
        Y_win = np.array(win_rates)
        win_corr, win_r2 = compute_correlation_and_r2(X_win, Y_win)
        win_spearman = compute_spearman_correlation(X_win, Y_win)

        per_cluster_stats[c] = {
            'mean_delta_prob': {eps: means[i] for i, eps in enumerate(epsilons)},
            'std_delta_prob': {eps: stds[i] for i, eps in enumerate(epsilons)},
            'win_rate': {eps: win_rates[i] for i, eps in enumerate(epsilons)},
            'dose_response_r2': r2,
            'dose_response_corr': corr,
            'dose_response_spearman': spearman,
            'win_rate_r2': win_r2,
            'win_rate_corr': win_corr,
            'win_rate_spearman': win_spearman,
            'n_samples': len(dose_response[c][epsilons[0]]) if epsilons else 0
        }

        if r2 > 0:
            all_r2.append(r2)
        if corr != 0:
            all_corr.append(corr)
        if spearman != 0:
            all_spearman.append(spearman)
        if win_r2 > 0:
            all_win_r2.append(win_r2)
        if win_corr != 0:
            all_win_corr.append(win_corr)
        if win_spearman != 0:
            all_win_spearman.append(win_spearman)

    result = {
        'per_cluster': per_cluster_stats,
        'mean_r2': float(np.mean(all_r2)) if all_r2 else 0.0,
        'mean_corr': float(np.mean(all_corr)) if all_corr else 0.0,
        'mean_spearman': float(np.mean(all_spearman)) if all_spearman else 0.0,
        'mean_win_r2': float(np.mean(all_win_r2)) if all_win_r2 else 0.0,
        'mean_win_corr': float(np.mean(all_win_corr)) if all_win_corr else 0.0,
        'mean_win_spearman': float(np.mean(all_win_spearman)) if all_win_spearman else 0.0,
        'n_clusters_with_effect': len(all_r2)
    }

    # Add cluster mass metrics if log probabilities are provided
    if log_probs_steered is not None and log_probs_original is not None:
        mass_metrics = compute_cluster_mass_metrics(log_probs_steered, log_probs_original, epsilons)
        result.update(mass_metrics)

    # Add centered logit metrics if centered logits are provided
    if centered_logits_steered is not None and centered_logits_original is not None:
        logit_metrics = compute_centered_logit_metrics(centered_logits_steered, centered_logits_original, epsilons)
        result.update(logit_metrics)

        # Aggregate mean_logit_r2, mean_logit_corr, and mean_logit_spearman across clusters
        # Also add logit metrics to per_cluster for consistency
        all_logit_r2 = []
        all_logit_corr = []
        all_logit_spearman = []
        all_sum_logit_r2 = []
        all_sum_logit_corr = []
        all_sum_logit_spearman = []
        for c, stats in logit_metrics.get('per_cluster_logit', {}).items():
            if 'centered_logit_r2' in stats:
                all_logit_r2.append(stats['centered_logit_r2'])
                # Add to per_cluster as well
                if c in result['per_cluster']:
                    result['per_cluster'][c]['centered_logit_r2'] = stats['centered_logit_r2']
            if 'centered_logit_corr' in stats:
                all_logit_corr.append(stats['centered_logit_corr'])
                # Add to per_cluster as well
                if c in result['per_cluster']:
                    result['per_cluster'][c]['centered_logit_corr'] = stats['centered_logit_corr']
            if 'centered_logit_spearman' in stats:
                all_logit_spearman.append(stats['centered_logit_spearman'])
                # Add to per_cluster as well
                if c in result['per_cluster']:
                    result['per_cluster'][c]['centered_logit_spearman'] = stats['centered_logit_spearman']
            # Sum-based metrics
            if 'sum_centered_logit_r2' in stats:
                all_sum_logit_r2.append(stats['sum_centered_logit_r2'])
                if c in result['per_cluster']:
                    result['per_cluster'][c]['sum_centered_logit_r2'] = stats['sum_centered_logit_r2']
            if 'sum_centered_logit_corr' in stats:
                all_sum_logit_corr.append(stats['sum_centered_logit_corr'])
                if c in result['per_cluster']:
                    result['per_cluster'][c]['sum_centered_logit_corr'] = stats['sum_centered_logit_corr']
            if 'sum_centered_logit_spearman' in stats:
                all_sum_logit_spearman.append(stats['sum_centered_logit_spearman'])
                if c in result['per_cluster']:
                    result['per_cluster'][c]['sum_centered_logit_spearman'] = stats['sum_centered_logit_spearman']

        result['mean_logit_r2'] = float(np.mean(all_logit_r2)) if all_logit_r2 else 0.0
        result['mean_logit_corr'] = float(np.mean(all_logit_corr)) if all_logit_corr else 0.0
        result['mean_logit_spearman'] = float(np.mean(all_logit_spearman)) if all_logit_spearman else 0.0
        result['mean_sum_logit_r2'] = float(np.mean(all_sum_logit_r2)) if all_sum_logit_r2 else 0.0
        result['mean_sum_logit_corr'] = float(np.mean(all_sum_logit_corr)) if all_sum_logit_corr else 0.0
        result['mean_sum_logit_spearman'] = float(np.mean(all_sum_logit_spearman)) if all_sum_logit_spearman else 0.0

        # Also aggregate alias demeaned logit
        all_demeaned_r2 = []
        all_demeaned_corr = []
        all_demeaned_spearman = []
        for c, stats in logit_metrics.get('per_cluster_demeaned_logit', {}).items():
            all_demeaned_r2.append(stats.get('demeaned_logit_r2', 0.0))
            all_demeaned_corr.append(stats.get('demeaned_logit_corr', 0.0))
            all_demeaned_spearman.append(stats.get('demeaned_logit_spearman', 0.0))
            if c in result['per_cluster']:
                result['per_cluster'][c]['demeaned_logit_r2'] = stats.get('demeaned_logit_r2', 0.0)
                result['per_cluster'][c]['demeaned_logit_corr'] = stats.get('demeaned_logit_corr', 0.0)
                result['per_cluster'][c]['demeaned_logit_spearman'] = stats.get('demeaned_logit_spearman', 0.0)
                # Sum-based aliases
                result['per_cluster'][c]['sum_demeaned_logit_r2'] = stats.get('sum_demeaned_logit_r2', 0.0)
                result['per_cluster'][c]['sum_demeaned_logit_corr'] = stats.get('sum_demeaned_logit_corr', 0.0)
                result['per_cluster'][c]['sum_demeaned_logit_spearman'] = stats.get('sum_demeaned_logit_spearman', 0.0)
        result['mean_demeaned_logit_r2'] = float(np.mean(all_demeaned_r2)) if all_demeaned_r2 else 0.0
        result['mean_demeaned_logit_corr'] = float(np.mean(all_demeaned_corr)) if all_demeaned_corr else 0.0
        result['mean_demeaned_logit_spearman'] = float(np.mean(all_demeaned_spearman)) if all_demeaned_spearman else 0.0

    # Add cluster-level target logit/prob diffs and slope/corr vs epsilon (main scores)
    if target_logits_steered is not None and target_logits_original is not None:
        logit_curve = compute_delta_curve_metrics(
            target_logits_steered, target_logits_original, epsilons, metric_prefix="target_logit"
        )
        result.update(logit_curve)
    if target_probs_steered is not None and target_probs_original is not None:
        prob_curve = compute_delta_curve_metrics(
            target_probs_steered, target_probs_original, epsilons, metric_prefix="target_prob"
        )
        result.update(prob_curve)

    return result


# =============================================================================
# Aggregation and Output
# =============================================================================

def aggregate_results_across_prefixes(
    all_results: Dict[str, Dict],
    sweep_keys: List[Tuple]
) -> Dict[str, Any]:
    """Aggregate metrics across prefixes for summary table.

    Args:
        all_results: {prefix_id: {key: {mean_r2, mean_corr, ...}}}
        sweep_keys: List of tuples to aggregate. Format: (method, hc_sel, top_B)

    Returns:
        {tuple_key: {r2: [], corr: [], spearman: [], win_r2: [], win_corr: [], win_spearman: [], logit_r2: [], logit_corr: [], logit_spearman: []}}
    """
    aggregated = {key: {
        'r2': [], 'corr': [], 'spearman': [],
        'win_r2': [], 'win_corr': [], 'win_spearman': [],
        'logit_r2': [], 'logit_corr': [], 'logit_spearman': []
    } for key in sweep_keys}

    for prefix_id, prefix_results in all_results.items():
        results = prefix_results.get('results', {})
        for result_key, res in results.items():
            if 'mean_r2' not in res:
                continue

            # Parse key using SweepKey utility
            try:
                sweep_key = SweepKey.from_string(result_key)
                agg_key = sweep_key.to_tuple()

                if agg_key in aggregated:
                    aggregated[agg_key]['r2'].append(res.get('mean_r2', 0.0))
                    aggregated[agg_key]['corr'].append(res.get('mean_corr', 0.0))
                    aggregated[agg_key]['spearman'].append(res.get('mean_spearman', 0.0))
                    aggregated[agg_key]['win_r2'].append(res.get('mean_win_r2', 0.0))
                    aggregated[agg_key]['win_corr'].append(res.get('mean_win_corr', 0.0))
                    aggregated[agg_key]['win_spearman'].append(res.get('mean_win_spearman', 0.0))
                    aggregated[agg_key]['logit_r2'].append(res.get('mean_logit_r2', 0.0))
                    aggregated[agg_key]['logit_corr'].append(res.get('mean_logit_corr', 0.0))
                    aggregated[agg_key]['logit_spearman'].append(res.get('mean_logit_spearman', 0.0))
            except (ValueError, IndexError):
                # Skip malformed keys
                continue

    return aggregated


def format_summary_table(aggregated: Dict) -> str:
    """Format aggregated results as a summary table string.

    Args:
        aggregated: Output from aggregate_results_across_prefixes
                   Keys are 3-tuples: (method, hc_sel, top_B)

    Returns:
        Formatted table string
    """
    lines = []
    if not aggregated:
        return "No results to aggregate."

    # Check if logit metrics are present
    sample_metrics = next(iter(aggregated.values()), {})
    has_logit = 'logit_r2' in sample_metrics and sample_metrics.get('logit_r2')

    # Header with Spearman
    if has_logit:
        lines.append(f"\n{'Method':<15} {'H_c Sel':<10} {'Top-B':<6} {'Corr':>8} {'Spear':>8} {'WinCorr':>8} {'WinSpear':>8} {'LogCorr':>8} {'LogSpear':>8} {'N':>6}")
        lines.append("-" * 115)
    else:
        lines.append(f"\n{'Method':<15} {'H_c Sel':<10} {'Top-B':<6} {'Corr':>8} {'Spear':>8} {'WinCorr':>8} {'WinSpear':>8} {'N':>6}")
        lines.append("-" * 85)

    # Collect rows with their metrics
    rows = []
    for key, metrics in aggregated.items():
        if metrics['r2']:
            sweep_key = SweepKey.from_tuple(key)
            mean_corr = np.mean(metrics['corr'])
            mean_spearman = np.mean(metrics.get('spearman', [])) if metrics.get('spearman') else 0.0
            mean_win_corr = np.mean(metrics['win_corr'])
            mean_win_spearman = np.mean(metrics.get('win_spearman', [])) if metrics.get('win_spearman') else 0.0
            mean_logit_corr = np.mean(metrics.get('logit_corr', [])) if metrics.get('logit_corr') else 0.0
            mean_logit_spearman = np.mean(metrics.get('logit_spearman', [])) if metrics.get('logit_spearman') else 0.0
            n = len(metrics['r2'])
            rows.append((sweep_key, mean_corr, mean_spearman, mean_win_corr, mean_win_spearman, mean_logit_corr, mean_logit_spearman, n))

    # Sort by Spearman (most meaningful metric for non-linear relationships)
    rows.sort(key=lambda x: x[2], reverse=True)  # x[2] is mean_spearman

    for sweep_key, mean_corr, mean_spearman, mean_win_corr, mean_win_spearman, mean_logit_corr, mean_logit_spearman, n in rows:
        if has_logit:
            lines.append(f"{sweep_key.method:<15} {sweep_key.hc_selection:<10} {sweep_key.top_B:<6} {mean_corr:>+8.4f} {mean_spearman:>+8.4f} {mean_win_corr:>+8.4f} {mean_win_spearman:>+8.4f} {mean_logit_corr:>+8.4f} {mean_logit_spearman:>+8.4f} {n:>6}")
        else:
            lines.append(f"{sweep_key.method:<15} {sweep_key.hc_selection:<10} {sweep_key.top_B:<6} {mean_corr:>+8.4f} {mean_spearman:>+8.4f} {mean_win_corr:>+8.4f} {mean_win_spearman:>+8.4f} {n:>6}")

    return "\n".join(lines)


def save_aggregated_summary(
    aggregated: Dict,
    prefixes: List[str],
    sweeps: List[Dict],
    output_path: Path
):
    """Save aggregated summary to JSON file.

    Args:
        aggregated: Output from aggregate_results_across_prefixes
                   Keys can be 3-tuple or 4-tuple
        prefixes: List of prefix IDs
        sweeps: Sweep configurations
        output_path: Path to save JSON file
    """
    if not aggregated:
        return

    summary_rows = []
    for key, metrics in aggregated.items():
        if metrics['r2']:
            sweep_key = SweepKey.from_tuple(key)
            row = {
                'steering_method': sweep_key.method,
                'h_c_selection': sweep_key.hc_selection,
                'top_B': sweep_key.top_B,
                'mean_r2': float(np.mean(metrics['r2'])),
                'mean_corr': float(np.mean(metrics['corr'])),
                'mean_spearman': float(np.mean(metrics.get('spearman', [0.0]))) if metrics.get('spearman') else 0.0,
                'mean_win_r2': float(np.mean(metrics['win_r2'])),
                'mean_win_corr': float(np.mean(metrics['win_corr'])),
                'mean_win_spearman': float(np.mean(metrics.get('win_spearman', [0.0]))) if metrics.get('win_spearman') else 0.0,
                'n_prefixes': len(metrics['r2'])
            }
            # Add logit metrics if available
            if metrics.get('logit_r2'):
                row['mean_logit_r2'] = float(np.mean(metrics['logit_r2']))
            if metrics.get('logit_corr'):
                row['mean_logit_corr'] = float(np.mean(metrics['logit_corr']))
            if metrics.get('logit_spearman'):
                row['mean_logit_spearman'] = float(np.mean(metrics['logit_spearman']))
            summary_rows.append(row)

    # Sort by Spearman (most meaningful for non-linear relationships)
    summary_rows.sort(key=lambda x: x.get('mean_spearman', 0.0), reverse=True)

    summary = {
        'prefixes': prefixes,
        'config': {'sweeps': sweeps},
        'results': summary_rows
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)


def generate_sweep_key(method: str, hc_sel: str, top_B: int) -> str:
    """Generate a consistent key string for sweep configuration."""
    return f"{method}_{hc_sel}_B{top_B}"

