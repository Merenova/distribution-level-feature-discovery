#!/usr/bin/env python3
"""7c_metrics.py - Metrics computation and aggregation.

This module contains functions for:
1. Log probability computation
2. Steering metrics computation (demeaned-logit correlations)
3. Aggregation and summary output
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

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


def compute_per_token_logit_values_batched(
    logits: torch.Tensor,
    batch_cont_info: List[Tuple[List[int], int]],
) -> List[Dict[str, Any]]:
    """Extract raw target logits, probabilities, and demeaned logits per sequence."""
    values: List[Dict[str, Any]] = []
    seq_len = logits.shape[1]

    for batch_idx, (cont_ids, cont_start) in enumerate(batch_cont_info):
        if not cont_ids:
            values.append({
                "raw_target_logits": [],
                "target_probs": [],
                "demeaned_logits": [],
                "mean_raw_target_logit": 0.0,
                "mean_target_prob": 0.0,
                "mean_demeaned_logit": 0.0,
            })
            continue

        valid_len = min(len(cont_ids), seq_len - cont_start)
        if valid_len <= 0:
            values.append({
                "raw_target_logits": [],
                "target_probs": [],
                "demeaned_logits": [],
                "mean_raw_target_logit": 0.0,
                "mean_target_prob": 0.0,
                "mean_demeaned_logit": 0.0,
            })
            continue

        positions = torch.arange(cont_start, cont_start + valid_len, device=logits.device)
        cont_logits = logits[batch_idx, positions, :].float()
        token_ids = torch.tensor(cont_ids[:valid_len], device=logits.device, dtype=torch.long)

        target_logits = cont_logits[torch.arange(valid_len, device=logits.device), token_ids]
        target_probs = torch.exp(
            F.log_softmax(cont_logits, dim=-1)[torch.arange(valid_len, device=logits.device), token_ids]
        )
        demeaned_logits = target_logits - cont_logits.mean(dim=-1)

        values.append({
            "raw_target_logits": [float(v) for v in target_logits.detach().cpu().tolist()],
            "target_probs": [float(v) for v in target_probs.detach().cpu().tolist()],
            "demeaned_logits": [float(v) for v in demeaned_logits.detach().cpu().tolist()],
            "mean_raw_target_logit": float(target_logits.mean().item()),
            "mean_target_prob": float(target_probs.mean().item()),
            "mean_demeaned_logit": float(demeaned_logits.mean().item()),
        })

    return values


# =============================================================================
# Steering Metrics Computation
# =============================================================================

def compute_centered_logit_metrics(
    centered_logits_steered: Dict[int, Dict[float, List[float]]],
    centered_logits_original: Dict[int, Dict[float, List[float]]],
    epsilons: List[float]
) -> Dict[str, Any]:
    """Compute cluster-level demeaned logit correlations.

    For each cluster:
    - mean_centered_logit = mean across continuations of (mean across tokens of centered logit)

    Uses absolute difference (steered - original) for correlation with epsilon.

    Args:
        centered_logits_steered: {cluster_id: {epsilon: [mean_centered_logit per continuation]}}
        centered_logits_original: {cluster_id: {epsilon: [mean_centered_logit per continuation]}}
        epsilons: List of epsilon values

    Returns:
        Dict with per-cluster centered/demeaned logit metrics.
    """
    cluster_ids = sorted(centered_logits_steered.keys())
    per_cluster_logit_stats = {}
    per_cluster_demeaned = {}

    for c in cluster_ids:
        mean_original = []
        mean_steered = []
        mean_diffs = []
        n_samples = 0

        for eps in epsilons:
            logits_orig = centered_logits_original.get(c, {}).get(eps, [])
            logits_steer = centered_logits_steered.get(c, {}).get(eps, [])
            n_samples = max(n_samples, len(logits_orig), len(logits_steer))

            if not logits_orig or not logits_steer:
                mean_original.append(0.0)
                mean_steered.append(0.0)
                mean_diffs.append(0.0)
                continue

            m_orig = float(np.mean(logits_orig))
            m_steer = float(np.mean(logits_steer))
            mean_diff = m_steer - m_orig

            mean_original.append(m_orig)
            mean_steered.append(m_steer)
            mean_diffs.append(mean_diff)

        X = np.array(epsilons)
        Y_mean = np.array(mean_diffs)
        mean_logit_corr, _ = compute_correlation_and_r2(X, Y_mean)
        mean_logit_spearman = compute_spearman_correlation(X, Y_mean)

        per_cluster_logit_stats[c] = {
            'mean_centered_logit_original': {eps: mean_original[i] for i, eps in enumerate(epsilons)},
            'mean_centered_logit_steered': {eps: mean_steered[i] for i, eps in enumerate(epsilons)},
            'centered_logit_diff': {eps: mean_diffs[i] for i, eps in enumerate(epsilons)},
            'centered_logit_corr': mean_logit_corr,
            'centered_logit_spearman': mean_logit_spearman,
            'n_samples': n_samples,
        }
        per_cluster_demeaned[c] = {
            'mean_demeaned_logit_original': per_cluster_logit_stats[c]['mean_centered_logit_original'],
            'mean_demeaned_logit_steered': per_cluster_logit_stats[c]['mean_centered_logit_steered'],
            'demeaned_logit_diff': per_cluster_logit_stats[c]['centered_logit_diff'],
            'demeaned_logit_corr': mean_logit_corr,
            'demeaned_logit_spearman': mean_logit_spearman,
            'n_samples': n_samples,
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
    centered_logits_steered: Dict[int, Dict[float, List[float]]],
    centered_logits_original: Dict[int, Dict[float, List[float]]],
    epsilons: List[float],
) -> Dict[str, Any]:
    """Compute paper-facing steering metrics for a sweep configuration.

    Args:
        centered_logits_steered: {cluster_id: {epsilon: [mean_demeaned_logit]}}
        centered_logits_original: {cluster_id: {epsilon: [mean_demeaned_logit]}}
        epsilons: List of epsilon values

    Returns:
        Dict with per-cluster centered/demeaned logit correlations and means.
    """
    result = compute_centered_logit_metrics(
        centered_logits_steered,
        centered_logits_original,
        epsilons,
    )

    per_cluster_logit = result.get('per_cluster_logit', {})
    logit_corr = [
        stats.get('centered_logit_corr', 0.0)
        for stats in per_cluster_logit.values()
    ]
    logit_spearman = [
        stats.get('centered_logit_spearman', 0.0)
        for stats in per_cluster_logit.values()
    ]

    per_cluster_demeaned = result.get('per_cluster_demeaned_logit', {})
    demeaned_corr = [
        stats.get('demeaned_logit_corr', 0.0)
        for stats in per_cluster_demeaned.values()
    ]
    demeaned_spearman = [
        stats.get('demeaned_logit_spearman', 0.0)
        for stats in per_cluster_demeaned.values()
    ]

    result.update({
        'mean_logit_corr': float(np.mean(logit_corr)) if logit_corr else 0.0,
        'mean_logit_spearman': float(np.mean(logit_spearman)) if logit_spearman else 0.0,
        'mean_demeaned_logit_corr': float(np.mean(demeaned_corr)) if demeaned_corr else 0.0,
        'mean_demeaned_logit_spearman': float(np.mean(demeaned_spearman)) if demeaned_spearman else 0.0,
    })
    return result


# =============================================================================
# Aggregation and Output
# =============================================================================

PAPER_CORRELATION_METRICS = (
    "centered_logit_spearman_mean",
    "centered_logit_corr_mean",
)

def aggregate_results_across_prefixes(
    all_results: Dict[str, Dict],
    sweep_keys: List[Tuple]
) -> Dict[str, Any]:
    """Aggregate metrics across prefixes for summary table.

    Args:
        all_results: {prefix_id: {key: {mean_logit_corr, mean_logit_spearman}}}
        sweep_keys: List of tuples to aggregate. Format: (method, hc_sel, top_B)

    Returns:
        {tuple_key: {logit_corr: [], logit_spearman: []}}
    """
    aggregated = {key: {
        'logit_corr': [],
        'logit_spearman': [],
    } for key in sweep_keys}

    for prefix_id, prefix_results in all_results.items():
        results = prefix_results.get('results', {})
        for result_key, res in results.items():
            if 'mean_logit_corr' not in res and 'mean_logit_spearman' not in res:
                continue

            # Parse key using SweepKey utility
            try:
                sweep_key = SweepKey.from_string(result_key)
                agg_key = sweep_key.to_tuple()

                if agg_key in aggregated:
                    if 'mean_logit_corr' in res:
                        aggregated[agg_key]['logit_corr'].append(res['mean_logit_corr'])
                    if 'mean_logit_spearman' in res:
                        aggregated[agg_key]['logit_spearman'].append(res['mean_logit_spearman'])
            except (ValueError, IndexError):
                # Skip malformed keys
                continue

    return aggregated


def build_paper_correlation_summary_rows(aggregated: Dict) -> List[Dict[str, Any]]:
    """Build paper-facing rows from existing prefix correlation aggregates.

    The aggregation contract is intentionally unchanged: callers still use
    aggregate_results_across_prefixes(), which stores per-prefix centered-logit
    Pearson/Spearman means under ``logit_corr`` and ``logit_spearman``. This
    helper only renames and filters those existing aggregates for reporting.
    """
    rows: List[Dict[str, Any]] = []
    for key, metric_values in aggregated.items():
        logit_corr = metric_values.get("logit_corr", [])
        logit_spearman = metric_values.get("logit_spearman", [])
        n_prefixes = max(
            len(logit_corr),
            len(logit_spearman),
        )
        if n_prefixes == 0:
            continue

        sweep_key = SweepKey.from_tuple(key)
        rows.append({
            "steering_method": sweep_key.method,
            "h_c_selection": sweep_key.hc_selection,
            "top_B": sweep_key.top_B,
            "centered_logit_spearman_mean": (
                float(np.mean(logit_spearman)) if logit_spearman else 0.0
            ),
            "centered_logit_corr_mean": (
                float(np.mean(logit_corr)) if logit_corr else 0.0
            ),
            "n_prefixes": n_prefixes,
        })

    rows.sort(
        key=lambda row: (
            row["centered_logit_spearman_mean"],
            row["centered_logit_corr_mean"],
        ),
        reverse=True,
    )
    return rows


def format_summary_table(aggregated: Dict) -> str:
    """Format paper-facing centered-logit correlation summary table.

    Args:
        aggregated: Output from aggregate_results_across_prefixes
                   Keys are 3-tuples: (method, hc_sel, top_B)

    Returns:
        Formatted table string
    """
    lines = []
    if not aggregated:
        return "No results to aggregate."

    rows = build_paper_correlation_summary_rows(aggregated)
    if not rows:
        return "No centered-logit correlation results to aggregate."

    lines.append(f"\n{'Method':<15} {'H_c Sel':<10} {'Top-B':<6} {'rho_s':>8} {'rho':>8} {'N':>6}")
    lines.append("-" * 62)

    for row in rows:
        lines.append(
            f"{row['steering_method']:<15} {row['h_c_selection']:<10} "
            f"{row['top_B']:<6} {row['centered_logit_spearman_mean']:>+8.4f} "
            f"{row['centered_logit_corr_mean']:>+8.4f} {row['n_prefixes']:>6}"
        )

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

    summary = {
        'prefixes': prefixes,
        'config': {'sweeps': sweeps},
        'metrics': list(PAPER_CORRELATION_METRICS),
        'results': build_paper_correlation_summary_rows(aggregated)
    }

    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)


def generate_sweep_key(method: str, hc_sel: str, top_B: int) -> str:
    """Generate a consistent key string for sweep configuration."""
    return f"{method}_{hc_sel}_B{top_B}"
