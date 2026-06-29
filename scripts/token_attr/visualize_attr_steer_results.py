#!/usr/bin/env -S uv run python
"""
Visualize outputs produced by scripts/token_attribution_magnitude_by_position.py

Expected files (from run_attr_steer_all.sh):
  - position_stats.json
  - samples.json
  - steer.json

This script produces:
  1) Position-wise L1 attribution magnitude summary (mean + p10/p90 band; var).
  2) Per-sample token/word L1 magnitude traces (first N samples).
  3) Steering dose-response curves (mean Δ target logit/prob vs epsilon).
  4) Steering heatmaps (epsilon x token position) for Δ target logit/prob (first N samples).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _maybe(path: Path) -> Optional[Path]:
    return path if path.exists() else None


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _plot_position_stats(position_rows: List[Dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not position_rows:
        return

    pos = np.array([r["pos"] for r in position_rows], dtype=np.int64)
    mean = np.array([r.get("mean", np.nan) for r in position_rows], dtype=np.float64)
    p10 = np.array([r.get("p10", np.nan) for r in position_rows], dtype=np.float64)
    p90 = np.array([r.get("p90", np.nan) for r in position_rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 4))
    l_mean, = ax.plot(pos, mean, label="mean L1", linewidth=2)
    band = ax.fill_between(pos, p10, p90, alpha=0.2, label="p10–p90")
    
    # Find and mark max and min of mean
    valid_mask = ~np.isnan(mean)
    if valid_mask.any():
        max_idx = np.nanargmax(mean)
        min_idx = np.nanargmin(mean)
        max_pos, max_val = int(pos[max_idx]), float(mean[max_idx])
        min_pos, min_val = int(pos[min_idx]), float(mean[min_idx])
        
        # Mark max point
        ax.scatter([max_pos], [max_val], color="red", s=80, zorder=5, marker="^")
        ax.annotate(
            f"max: pos={max_pos}, val={max_val:.4f}",
            xy=(max_pos, max_val),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            color="red",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="red", alpha=0.8),
        )
        
        # Mark min point
        ax.scatter([min_pos], [min_val], color="blue", s=80, zorder=5, marker="v")
        ax.annotate(
            f"min: pos={min_pos}, val={min_val:.4f}",
            xy=(min_pos, min_val),
            xytext=(10, -20),
            textcoords="offset points",
            fontsize=9,
            color="blue",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="blue", alpha=0.8),
        )
    
    ax.set_title("Token attribution magnitude vs position (L1)")
    ax.set_xlabel("Token position in continuation")
    ax.set_ylabel("L1 magnitude (sum(abs(attr)))")
    ax.grid(True, alpha=0.3)
    # Single legend: mean + band
    handles = [l_mean, band]
    labels = ["mean", "p10–p90"]
    ax.legend(handles, labels, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_sample_traces(samples: List[Dict[str, Any]], out_dir: Path, max_samples: int = 10) -> None:
    import matplotlib.pyplot as plt

    if not samples:
        return

    for idx, s in enumerate(samples[:max_samples]):
        token_l1 = s.get("token_attr_l1") or []
        word_l1 = s.get("word_attr_l1") or []
        token_strs = s.get("continuation_token_strs") or []
        prefix_id = s.get("prefix_id", "unknown")
        cont_idx = s.get("cont_idx", -1)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(np.arange(len(token_l1)), token_l1, linewidth=1.5)
        ax.set_title(f"Token L1 attribution magnitude (prefix={prefix_id}, cont={cont_idx})")
        ax.set_xlabel("Token position")
        ax.set_ylabel("L1 magnitude")
        ax.grid(True, alpha=0.3)

        # Optional: annotate a few token strings (avoid clutter)
        if token_strs and len(token_strs) == len(token_l1) and len(token_l1) <= 80:
            for i, tok in enumerate(token_strs):
                if i % 8 == 0:
                    ax.text(i, token_l1[i], tok, fontsize=7, rotation=45, alpha=0.7)

        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{idx:02d}_token_l1.png", dpi=200)
        plt.close(fig)

        # Word-level plot
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.bar(np.arange(len(word_l1)), word_l1, width=0.9)
        ax.set_title(f"Word L1 attribution magnitude (prefix={prefix_id}, cont={cont_idx})")
        ax.set_xlabel("Word index")
        ax.set_ylabel("L1 magnitude (grouped)")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{idx:02d}_word_l1.png", dpi=200)
        plt.close(fig)


def _extract_epsilons(per_epsilon: Dict[str, Any]) -> List[float]:
    eps = []
    for k in per_epsilon.keys():
        try:
            eps.append(float(k))
        except Exception:
            continue
    return sorted(eps)


def _extract_base_probs(sample: Dict[str, Any]) -> Optional[Dict[str, List[float]]]:
    """Extract base probabilities from steering at eps=0."""
    steering = sample.get("steering", {})
    pe = steering.get("per_epsilon", {})
    eps0_data = pe.get("0.0") or pe.get("0")
    if not eps0_data:
        return None
    token_probs = eps0_data.get("token_target_probs", [])
    word_probs = eps0_data.get("word_target_probs", [])
    if not token_probs and not word_probs:
        return None
    return {"token": token_probs, "word": word_probs}


def _compute_delta_logprob(probs: List[float], delta_probs: List[float]) -> List[float]:
    """Compute delta log probability: log(p_steered) - log(p_original).
    
    p_steered = probs (the probability after steering at this epsilon)
    p_original = probs - delta_probs (probability at epsilon=0)
    delta_logprob = log(p_steered) - log(p_original)
    """
    result = []
    for p, dp in zip(probs, delta_probs):
        p_orig = p - dp
        # Clamp to avoid log(0)
        p_steered = max(p, 1e-12)
        p_orig = max(p_orig, 1e-12)
        result.append(float(np.log(p_steered) - np.log(p_orig)))
    return result


def _get_metric_with_logprob(
    pe_data: Dict[str, Any],
    metric: str,
    base_probs: Optional[Dict[str, List[float]]] = None,
) -> List[float]:
    """Get metric value, computing log prob variants on-the-fly if needed.
    
    Args:
        pe_data: per_epsilon data dict for a specific epsilon
        metric: metric name to retrieve
        base_probs: optional dict with 'token' and 'word' base probabilities
                    (used for random baseline where target_probs is not available)
    """
    # Direct metric
    if metric in pe_data:
        return pe_data.get(metric, [])
    
    # Computed log prob metrics
    if metric == "token_delta_target_logprobs":
        probs = pe_data.get("token_target_probs", [])
        delta_probs = pe_data.get("token_delta_target_probs", [])
        # If probs not available but we have base_probs, compute from delta
        if not probs and base_probs and "token" in base_probs and delta_probs:
            bp = base_probs["token"]
            L = min(len(bp), len(delta_probs))
            probs = [bp[i] + delta_probs[i] for i in range(L)]
            delta_probs = delta_probs[:L]
        if probs and delta_probs and len(probs) == len(delta_probs):
            return _compute_delta_logprob(probs, delta_probs)
    elif metric == "word_delta_target_logprobs":
        probs = pe_data.get("word_target_probs", [])
        delta_probs = pe_data.get("word_delta_target_probs", [])
        # If probs not available but we have base_probs, compute from delta
        if not probs and base_probs and "word" in base_probs and delta_probs:
            bp = base_probs["word"]
            L = min(len(bp), len(delta_probs))
            probs = [bp[i] + delta_probs[i] for i in range(L)]
            delta_probs = delta_probs[:L]
        if probs and delta_probs and len(probs) == len(delta_probs):
            return _compute_delta_logprob(probs, delta_probs)
    
    return []


def _mean_over_tokens(xs: List[float]) -> float:
    if not xs:
        return float("nan")
    return float(np.mean(np.asarray(xs, dtype=np.float64)))


def _plot_steering_dose_response(
    steer_samples: List[Dict[str, Any]],
    out_path: Path,
    max_samples: int = 30,
    metric: str = "token_delta_demeaned_target_logits",
) -> None:
    import matplotlib.pyplot as plt

    if not steer_samples:
        return

    # Build per-sample curve: mean(delta over tokens) vs eps
    curves: List[Tuple[np.ndarray, np.ndarray]] = []
    for s in steer_samples[:max_samples]:
        pe = (s.get("steering", {}) or {}).get("per_epsilon", {}) or {}
        eps = _extract_epsilons(pe)
        if not eps:
            continue
        y = []
        for e in eps:
            arr = _get_metric_with_logprob(pe.get(str(e), {}), metric)
            y.append(_mean_over_tokens(arr))
        curves.append((np.asarray(eps, dtype=np.float64), np.asarray(y, dtype=np.float64)))

    if not curves:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for x, y in curves:
        ax.plot(x, y, color="tab:blue", alpha=0.25, linewidth=1.0)

    # Aggregate mean curve across samples (align by eps)
    all_eps = sorted({float(v) for x, _ in curves for v in x.tolist()})
    mean_y = []
    for e in all_eps:
        vals = []
        for x, y in curves:
            # find exact eps match
            m = np.where(np.isclose(x, e))[0]
            if m.size:
                vals.append(float(y[m[0]]))
        mean_y.append(np.nanmean(vals) if vals else np.nan)

    ax.plot(all_eps, mean_y, color="black", linewidth=2.0, label="mean across samples")
    ax.axhline(0, color="gray", linewidth=1, alpha=0.5)
    ax.set_title(f"Dose-response: mean({metric}) vs epsilon")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("mean over continuation tokens")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_heatmap_eps_by_pos(
    steer_samples: List[Dict[str, Any]],
    out_dir: Path,
    max_samples: int = 10,
    metric: str = "token_delta_demeaned_target_logits",
    vmax: Optional[float] = None,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not steer_samples:
        return

    for idx, s in enumerate(steer_samples[:max_samples]):
        pe = (s.get("steering", {}) or {}).get("per_epsilon", {}) or {}
        eps = _extract_epsilons(pe)
        if not eps:
            continue

        rows = []
        for e in eps:
            rows.append(_get_metric_with_logprob(pe.get(str(e), {}), metric))

        # Ragged -> pad with NaN
        max_len = max((len(r) for r in rows), default=0)
        if max_len == 0:
            continue
        mat = np.full((len(rows), max_len), np.nan, dtype=np.float64)
        for i, r in enumerate(rows):
            rr = np.asarray(r, dtype=np.float64)
            mat[i, : rr.size] = rr

        prefix_id = s.get("prefix_id", "unknown")
        cont_idx = s.get("cont_idx", -1)

        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(
            mat,
            ax=ax,
            cmap="coolwarm",
            center=0.0,
            vmin=-vmax if vmax else None,
            vmax=vmax,
            cbar_kws={"label": metric},
        )
        ax.set_title(f"Heatmap: {metric} (prefix={prefix_id}, cont={cont_idx})")
        ax.set_xlabel("Token position")
        ax.set_ylabel("epsilon index")
        ax.set_yticks(np.arange(len(eps)) + 0.5)
        ax.set_yticklabels([str(e) for e in eps], rotation=0)
        fig.tight_layout()
        fig.savefig(out_dir / f"heatmap_{idx:02d}_{metric}.png", dpi=200)
        plt.close(fig)


def _plot_aggregate_heatmap_eps_by_pos(
    steer_samples: List[Dict[str, Any]],
    out_path: Path,
    metric: str,
    cut_mode: str = "minmax",
    vmax: Optional[float] = None,
    source: str = "steering",
) -> None:
    """Aggregate heatmap over all samples.

    cut_mode:
      - minmax: truncate positions to min(max_len_per_sample) (no missing values after truncation)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not steer_samples:
        return

    # Collect per-sample epsilon->array
    per_sample = []
    eps_all = None
    need_base_probs = "logprob" in metric and source != "steering"
    for s in steer_samples:
        pe = (s.get(source, {}) or {}).get("per_epsilon", {}) or {}
        eps = _extract_epsilons(pe)
        if not eps:
            continue
        if eps_all is None:
            eps_all = eps
        else:
            # require same eps set to aggregate cleanly
            if len(eps) != len(eps_all) or any(not np.isclose(eps[i], eps_all[i]) for i in range(len(eps))):
                continue
        base_probs = _extract_base_probs(s) if need_base_probs else None
        rows = [_get_metric_with_logprob(pe.get(str(e), {}), metric, base_probs) for e in eps_all]
        per_sample.append(rows)

    if not per_sample or eps_all is None:
        return

    # Determine truncation length
    if cut_mode == "minmax":
        maxlens = []
        for rows in per_sample:
            maxlens.append(max((len(r) for r in rows), default=0))
        L = int(min(maxlens)) if maxlens else 0
    elif cut_mode == "max":
        maxlens = []
        for rows in per_sample:
            maxlens.append(max((len(r) for r in rows), default=0))
        L = int(max(maxlens)) if maxlens else 0
    else:
        raise ValueError(f"Unknown cut_mode: {cut_mode}")

    if L <= 0:
        return

    # Build mean matrix [n_eps, L]
    mats = []
    for rows in per_sample:
        mat = np.full((len(eps_all), L), np.nan, dtype=np.float64)
        for i, r in enumerate(rows):
            rr = np.asarray(r, dtype=np.float64)
            n = min(len(rr), L)
            mat[i, :n] = rr[:n]
        mats.append(mat)
    mean_mat = np.nanmean(np.stack(mats, axis=0), axis=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        mean_mat,
        ax=ax,
        cmap="coolwarm",
        center=0.0,
        vmin=-vmax if vmax else None,
        vmax=vmax,
        cbar_kws={"label": f"mean({metric}) across samples"},
    )
    ax.set_title(f"Aggregate heatmap (cut={cut_mode}, L={L}): {metric}")
    ax.set_xlabel("Position")
    ax.set_ylabel("epsilon")
    ax.set_yticks(np.arange(len(eps_all)) + 0.5)
    ax.set_yticklabels([str(e) for e in eps_all], rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _compute_aggregate_matrix_eps_by_pos(
    steer_samples: List[Dict[str, Any]],
    metric: str,
    cut_mode: str = "minmax",
    source: str = "steering",
) -> Tuple[Optional[np.ndarray], Optional[List[float]], int]:
    """Compute mean matrix [n_eps, L] for a given metric/source without plotting."""
    per_sample = []
    eps_all = None
    need_base_probs = "logprob" in metric and source != "steering"
    for s in steer_samples:
        pe = (s.get(source, {}) or {}).get("per_epsilon", {}) or {}
        eps = _extract_epsilons(pe)
        if not eps:
            continue
        if eps_all is None:
            eps_all = eps
        else:
            if len(eps) != len(eps_all) or any(not np.isclose(eps[i], eps_all[i]) for i in range(len(eps))):
                continue
        base_probs = _extract_base_probs(s) if need_base_probs else None
        rows = [_get_metric_with_logprob(pe.get(str(e), {}), metric, base_probs) for e in eps_all]
        per_sample.append(rows)

    if not per_sample or eps_all is None:
        return None, None, 0

    if cut_mode == "minmax":
        maxlens = [max((len(r) for r in rows), default=0) for rows in per_sample]
        L = int(min(maxlens)) if maxlens else 0
    elif cut_mode == "max":
        maxlens = [max((len(r) for r in rows), default=0) for rows in per_sample]
        L = int(max(maxlens)) if maxlens else 0
    else:
        raise ValueError(f"Unknown cut_mode: {cut_mode}")

    if L <= 0:
        return None, eps_all, 0

    mats = []
    for rows in per_sample:
        mat = np.full((len(eps_all), L), np.nan, dtype=np.float64)
        for i, r in enumerate(rows):
            rr = np.asarray(r, dtype=np.float64)
            n = min(len(rr), L)
            mat[i, :n] = rr[:n]
        mats.append(mat)
    mean_mat = np.nanmean(np.stack(mats, axis=0), axis=0)
    return mean_mat, eps_all, L


def _compute_aggregate_diff_matrix_eps_by_pos(
    steer_samples: List[Dict[str, Any]],
    metric: str,
    cut_mode: str = "minmax",
    source_a: str = "steering",
    source_b: str = "random_baseline",
) -> Tuple[Optional[np.ndarray], Optional[List[float]], int]:
    """Compute mean(A-B) matrix [n_eps, L] for a given metric across samples."""
    per_sample_a = []
    per_sample_b = []
    eps_all = None
    need_base_probs = "logprob" in metric

    for s in steer_samples:
        pe_a = (s.get(source_a, {}) or {}).get("per_epsilon", {}) or {}
        pe_b = (s.get(source_b, {}) or {}).get("per_epsilon", {}) or {}
        eps_a = _extract_epsilons(pe_a)
        eps_b = _extract_epsilons(pe_b)
        if not eps_a or not eps_b:
            continue
        if len(eps_a) != len(eps_b) or any(not np.isclose(eps_a[i], eps_b[i]) for i in range(len(eps_a))):
            continue

        if eps_all is None:
            eps_all = eps_a
        else:
            if len(eps_a) != len(eps_all) or any(not np.isclose(eps_a[i], eps_all[i]) for i in range(len(eps_a))):
                continue

        base_probs = _extract_base_probs(s) if need_base_probs else None
        base_probs_a = base_probs if source_a != "steering" else None
        base_probs_b = base_probs if source_b != "steering" else None
        rows_a = [_get_metric_with_logprob(pe_a.get(str(e), {}), metric, base_probs_a) for e in eps_all]
        rows_b = [_get_metric_with_logprob(pe_b.get(str(e), {}), metric, base_probs_b) for e in eps_all]
        per_sample_a.append(rows_a)
        per_sample_b.append(rows_b)

    if not per_sample_a or eps_all is None:
        return None, None, 0

    if cut_mode == "minmax":
        maxlens = []
        for rows_a, rows_b in zip(per_sample_a, per_sample_b):
            max_a = max((len(r) for r in rows_a), default=0)
            max_b = max((len(r) for r in rows_b), default=0)
            maxlens.append(min(max_a, max_b))
        L = int(min(maxlens)) if maxlens else 0
    elif cut_mode == "max":
        maxlens = []
        for rows_a, rows_b in zip(per_sample_a, per_sample_b):
            max_a = max((len(r) for r in rows_a), default=0)
            max_b = max((len(r) for r in rows_b), default=0)
            maxlens.append(max(max_a, max_b))
        L = int(max(maxlens)) if maxlens else 0
    else:
        raise ValueError(f"Unknown cut_mode: {cut_mode}")

    if L <= 0:
        return None, eps_all, 0

    mats = []
    for rows_a, rows_b in zip(per_sample_a, per_sample_b):
        ma = np.full((len(eps_all), L), np.nan, dtype=np.float64)
        mb = np.full((len(eps_all), L), np.nan, dtype=np.float64)
        for i, (ra, rb) in enumerate(zip(rows_a, rows_b)):
            aa = np.asarray(ra, dtype=np.float64)
            bb = np.asarray(rb, dtype=np.float64)
            na, nb = min(len(aa), L), min(len(bb), L)
            ma[i, :na] = aa[:na]
            mb[i, :nb] = bb[:nb]
        mats.append(ma - mb)
    mean_diff = np.nanmean(np.stack(mats, axis=0), axis=0)
    return mean_diff, eps_all, L


def _plot_matrix_heatmap(
    mat: np.ndarray,
    eps_all: List[float],
    out_path: Path,
    title: str,
    cbar_label: str,
    vmax: Optional[float],
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if mat.size == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    # Symmetric scaling if vmax provided
    sns.heatmap(
        mat,
        ax=ax,
        cmap="coolwarm",
        center=0.0,
        vmin=-vmax if vmax else None,
        vmax=vmax,
        cbar_kws={"label": cbar_label},
    )
    # Styling for aggregate heatmaps (requested: no title, standardized labels)
    ax.set_title("")
    ax.set_xlabel("Position")
    ax.set_ylabel(r"steering strength ($\epsilon$)")
    ax.set_yticks(np.arange(len(eps_all)) + 0.5)
    ax.set_yticklabels([str(e) for e in eps_all], rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_aggregate_diff_heatmap_eps_by_pos(
    steer_samples: List[Dict[str, Any]],
    out_path: Path,
    metric: str,
    cut_mode: str = "minmax",
    vmax: Optional[float] = None,
    source_a: str = "steering",
    source_b: str = "random_baseline",
) -> None:
    """Aggregate heatmap of mean( A - B ) across samples, epsilon x position.

    Truncation uses min(max_len) across samples (and both sources), so no missing values after truncation.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if not steer_samples:
        return

    per_sample_a = []
    per_sample_b = []
    eps_all = None
    need_base_probs = "logprob" in metric

    for s in steer_samples:
        pe_a = (s.get(source_a, {}) or {}).get("per_epsilon", {}) or {}
        pe_b = (s.get(source_b, {}) or {}).get("per_epsilon", {}) or {}
        eps_a = _extract_epsilons(pe_a)
        eps_b = _extract_epsilons(pe_b)
        if not eps_a or not eps_b:
            continue

        # require same eps set
        if len(eps_a) != len(eps_b) or any(not np.isclose(eps_a[i], eps_b[i]) for i in range(len(eps_a))):
            continue

        if eps_all is None:
            eps_all = eps_a
        else:
            if len(eps_a) != len(eps_all) or any(not np.isclose(eps_a[i], eps_all[i]) for i in range(len(eps_a))):
                continue

        base_probs = _extract_base_probs(s) if need_base_probs else None
        base_probs_a = base_probs if source_a != "steering" else None
        base_probs_b = base_probs if source_b != "steering" else None
        rows_a = [_get_metric_with_logprob(pe_a.get(str(e), {}), metric, base_probs_a) for e in eps_all]
        rows_b = [_get_metric_with_logprob(pe_b.get(str(e), {}), metric, base_probs_b) for e in eps_all]
        per_sample_a.append(rows_a)
        per_sample_b.append(rows_b)

    if not per_sample_a or eps_all is None:
        return

    # Determine truncation length
    if cut_mode == "minmax":
        maxlens = []
        for rows_a, rows_b in zip(per_sample_a, per_sample_b):
            max_a = max((len(r) for r in rows_a), default=0)
            max_b = max((len(r) for r in rows_b), default=0)
            maxlens.append(min(max_a, max_b))
        L = int(min(maxlens)) if maxlens else 0
    elif cut_mode == "max":
        maxlens = []
        for rows_a, rows_b in zip(per_sample_a, per_sample_b):
            max_a = max((len(r) for r in rows_a), default=0)
            max_b = max((len(r) for r in rows_b), default=0)
            maxlens.append(max(max_a, max_b))
        L = int(max(maxlens)) if maxlens else 0
    else:
        raise ValueError(f"Unknown cut_mode: {cut_mode}")

    if L <= 0:
        return

    mats = []
    for rows_a, rows_b in zip(per_sample_a, per_sample_b):
        ma = np.full((len(eps_all), L), np.nan, dtype=np.float64)
        mb = np.full((len(eps_all), L), np.nan, dtype=np.float64)
        for i, (ra, rb) in enumerate(zip(rows_a, rows_b)):
            aa = np.asarray(ra, dtype=np.float64)
            bb = np.asarray(rb, dtype=np.float64)
            na, nb = min(len(aa), L), min(len(bb), L)
            ma[i, :na] = aa[:na]
            mb[i, :nb] = bb[:nb]
        mats.append(ma - mb)

    mean_diff = np.nanmean(np.stack(mats, axis=0), axis=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        mean_diff,
        ax=ax,
        cmap="coolwarm",
        center=0.0,
        vmin=-vmax if vmax else None,
        vmax=vmax,
        cbar_kws={"label": f"mean({source_a} - {source_b}) {metric}"},
    )
    ax.set_title(f"Aggregate diff heatmap (cut={cut_mode}, L={L}): {metric} ({source_a} - {source_b})")
    ax.set_xlabel("Position")
    ax.set_ylabel("epsilon")
    ax.set_yticks(np.arange(len(eps_all)) + 0.5)
    ax.set_yticklabels([str(e) for e in eps_all], rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_dose_response_diff(
    steer_samples: List[Dict[str, Any]],
    out_path: Path,
    metric: str,
    max_samples: int = 200,
    source_a: str = "steering",
    source_b: str = "random_baseline",
) -> None:
    """Plot mean over positions of (A - B) vs epsilon (per-sample thin lines + mean thick line)."""
    import matplotlib.pyplot as plt

    need_base_probs = "logprob" in metric
    curves: List[Tuple[np.ndarray, np.ndarray]] = []
    for s in steer_samples[:max_samples]:
        pe_a = (s.get(source_a, {}) or {}).get("per_epsilon", {}) or {}
        pe_b = (s.get(source_b, {}) or {}).get("per_epsilon", {}) or {}
        eps_a = _extract_epsilons(pe_a)
        eps_b = _extract_epsilons(pe_b)
        if not eps_a or not eps_b:
            continue
        if len(eps_a) != len(eps_b) or any(not np.isclose(eps_a[i], eps_b[i]) for i in range(len(eps_a))):
            continue

        base_probs = _extract_base_probs(s) if need_base_probs else None
        base_probs_a = base_probs if source_a != "steering" else None
        base_probs_b = base_probs if source_b != "steering" else None
        y = []
        for e in eps_a:
            a_arr = _get_metric_with_logprob(pe_a.get(str(e), {}), metric, base_probs_a)
            b_arr = _get_metric_with_logprob(pe_b.get(str(e), {}), metric, base_probs_b)
            # align by truncating to min length
            L = min(len(a_arr), len(b_arr))
            if L <= 0:
                y.append(float("nan"))
            else:
                da = np.asarray(a_arr[:L], dtype=np.float64)
                db = np.asarray(b_arr[:L], dtype=np.float64)
                y.append(float(np.mean(da - db)))
        curves.append((np.asarray(eps_a, dtype=np.float64), np.asarray(y, dtype=np.float64)))

    if not curves:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for x, y in curves:
        ax.plot(x, y, color="tab:purple", alpha=0.2, linewidth=1.0)

    all_eps = sorted({float(v) for x, _ in curves for v in x.tolist()})
    mean_y = []
    for e in all_eps:
        vals = []
        for x, y in curves:
            m = np.where(np.isclose(x, e))[0]
            if m.size:
                vals.append(float(y[m[0]]))
        mean_y.append(np.nanmean(vals) if vals else np.nan)

    ax.plot(all_eps, mean_y, color="black", linewidth=2.0, label="mean across samples")
    ax.axhline(0, color="gray", linewidth=1, alpha=0.5)
    ax.set_title(f"Dose-response: mean({source_a} - {source_b}) {metric} vs epsilon")
    ax.set_xlabel("epsilon")
    ax.set_ylabel("mean over positions")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing position_stats.json / samples.json / steer.json",
    )
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--max-samples", type=int, default=10, help="How many sample traces/heatmaps to render")
    ap.add_argument("--max-steer-curves", type=int, default=30, help="How many steering curves to overlay")
    ap.add_argument("--heatmap-vmax", type=float, default=None, help="Optional symmetric clamp for heatmap colors")
    ap.add_argument("--logprob-vmax", type=float, default=None, help="Optional symmetric clamp for logprob heatmap colors (default: auto)")
    ap.add_argument(
        "--cut-mode",
        type=str,
        default="max",
        choices=["minmax", "max"],
        help="Position truncation mode: 'minmax' = min(max_len) across samples (no NaN), 'max' = max across all (pool available samples)",
    )
    args = ap.parse_args()

    in_dir = args.input_dir
    out_dir = args.output_dir or (in_dir / "viz")
    _ensure_dir(out_dir)

    pos_path = _maybe(in_dir / "position_stats.json")
    sample_path = _maybe(in_dir / "samples.json")
    steer_path = _maybe(in_dir / "steer.json")

    # Prefer stats from position_stats.json; otherwise accept from samples/steer payload.
    position_rows: List[Dict[str, Any]] = []
    if pos_path:
        pos_payload = _load_json(pos_path)
        position_rows = pos_payload.get("position_stats", []) or []

    if not position_rows and sample_path:
        sample_payload = _load_json(sample_path)
        position_rows = sample_payload.get("position_stats", []) or []

    if not position_rows and steer_path:
        steer_payload = _load_json(steer_path)
        position_rows = steer_payload.get("position_stats", []) or []

    if position_rows:
        _plot_position_stats(position_rows, out_dir / "position_l1_stats.png")

    # Samples
    samples: List[Dict[str, Any]] = []
    if sample_path:
        sample_payload = _load_json(sample_path)
        samples = sample_payload.get("samples", []) or []
    if samples:
        _plot_sample_traces(samples, out_dir, max_samples=args.max_samples)

    # Steering
    steer_samples: List[Dict[str, Any]] = []
    if steer_path:
        steer_payload = _load_json(steer_path)
        steer_samples = steer_payload.get("samples", []) or []

    if steer_samples:
        _plot_steering_dose_response(
            steer_samples, out_dir / "dose_response_token_dlogit_demeaned.png", max_samples=args.max_steer_curves, metric="token_delta_demeaned_target_logits"
        )
        _plot_steering_dose_response(
            steer_samples, out_dir / "dose_response_token_dprob.png", max_samples=args.max_steer_curves, metric="token_delta_target_probs"
        )
        _plot_steering_dose_response(
            steer_samples, out_dir / "dose_response_token_dlogprob.png", max_samples=args.max_steer_curves, metric="token_delta_target_logprobs"
        )
        _plot_steering_dose_response(
            steer_samples, out_dir / "dose_response_word_dlogit_demeaned.png", max_samples=args.max_steer_curves, metric="word_delta_demeaned_target_logits"
        )
        _plot_steering_dose_response(
            steer_samples, out_dir / "dose_response_word_dprob.png", max_samples=args.max_steer_curves, metric="word_delta_target_probs"
        )
        _plot_steering_dose_response(
            steer_samples, out_dir / "dose_response_word_dlogprob.png", max_samples=args.max_steer_curves, metric="word_delta_target_logprobs"
        )

        _plot_heatmap_eps_by_pos(
            steer_samples,
            out_dir,
            max_samples=args.max_samples,
            metric="token_delta_demeaned_target_logits",
            vmax=args.heatmap_vmax,
        )
        _plot_heatmap_eps_by_pos(
            steer_samples,
            out_dir,
            max_samples=args.max_samples,
            metric="token_delta_target_probs",
            vmax=args.heatmap_vmax,
        )

        # Aggregate heatmaps across ALL samples, with shared symmetric scales:
        # - shared across token/word for dlogit and for dprob
        # - shared across steering/random/diff within each family (if random exists)
        have_random = any("random_baseline" in s for s in steer_samples)

        # Compute matrices (steering)
        tok_dlogit, eps1, _ = _compute_aggregate_matrix_eps_by_pos(
            steer_samples, metric="token_delta_demeaned_target_logits", cut_mode=args.cut_mode, source="steering"
        )
        word_dlogit, eps2, _ = _compute_aggregate_matrix_eps_by_pos(
            steer_samples, metric="word_delta_demeaned_target_logits", cut_mode=args.cut_mode, source="steering"
        )
        tok_dprob, eps3, _ = _compute_aggregate_matrix_eps_by_pos(
            steer_samples, metric="token_delta_target_probs", cut_mode=args.cut_mode, source="steering"
        )
        word_dprob, eps4, _ = _compute_aggregate_matrix_eps_by_pos(
            steer_samples, metric="word_delta_target_probs", cut_mode=args.cut_mode, source="steering"
        )
        tok_dlogprob, eps5, _ = _compute_aggregate_matrix_eps_by_pos(
            steer_samples, metric="token_delta_target_logprobs", cut_mode=args.cut_mode, source="steering"
        )
        word_dlogprob, eps6, _ = _compute_aggregate_matrix_eps_by_pos(
            steer_samples, metric="word_delta_target_logprobs", cut_mode=args.cut_mode, source="steering"
        )

        # Compute matrices (random + diff) if present
        tok_dlogit_r = word_dlogit_r = tok_dprob_r = word_dprob_r = None
        tok_dlogit_diff = word_dlogit_diff = tok_dprob_diff = word_dprob_diff = None
        tok_dlogprob_r = word_dlogprob_r = tok_dlogprob_diff = word_dlogprob_diff = None
        if have_random:
            tok_dlogit_r, _, _ = _compute_aggregate_matrix_eps_by_pos(
                steer_samples, metric="token_delta_demeaned_target_logits", cut_mode=args.cut_mode, source="random_baseline"
            )
            word_dlogit_r, _, _ = _compute_aggregate_matrix_eps_by_pos(
                steer_samples, metric="word_delta_demeaned_target_logits", cut_mode=args.cut_mode, source="random_baseline"
            )
            tok_dprob_r, _, _ = _compute_aggregate_matrix_eps_by_pos(
                steer_samples, metric="token_delta_target_probs", cut_mode=args.cut_mode, source="random_baseline"
            )
            word_dprob_r, _, _ = _compute_aggregate_matrix_eps_by_pos(
                steer_samples, metric="word_delta_target_probs", cut_mode=args.cut_mode, source="random_baseline"
            )
            tok_dlogit_diff, _, _ = _compute_aggregate_diff_matrix_eps_by_pos(
                steer_samples, metric="token_delta_demeaned_target_logits", cut_mode=args.cut_mode, source_a="steering", source_b="random_baseline"
            )
            word_dlogit_diff, _, _ = _compute_aggregate_diff_matrix_eps_by_pos(
                steer_samples, metric="word_delta_demeaned_target_logits", cut_mode=args.cut_mode, source_a="steering", source_b="random_baseline"
            )
            tok_dprob_diff, _, _ = _compute_aggregate_diff_matrix_eps_by_pos(
                steer_samples, metric="token_delta_target_probs", cut_mode=args.cut_mode, source_a="steering", source_b="random_baseline"
            )
            word_dprob_diff, _, _ = _compute_aggregate_diff_matrix_eps_by_pos(
                steer_samples, metric="word_delta_target_probs", cut_mode=args.cut_mode, source_a="steering", source_b="random_baseline"
            )
            tok_dlogprob_r, _, _ = _compute_aggregate_matrix_eps_by_pos(
                steer_samples, metric="token_delta_target_logprobs", cut_mode=args.cut_mode, source="random_baseline"
            )
            word_dlogprob_r, _, _ = _compute_aggregate_matrix_eps_by_pos(
                steer_samples, metric="word_delta_target_logprobs", cut_mode=args.cut_mode, source="random_baseline"
            )
            tok_dlogprob_diff, _, _ = _compute_aggregate_diff_matrix_eps_by_pos(
                steer_samples, metric="token_delta_target_logprobs", cut_mode=args.cut_mode, source_a="steering", source_b="random_baseline"
            )
            word_dlogprob_diff, _, _ = _compute_aggregate_diff_matrix_eps_by_pos(
                steer_samples, metric="word_delta_target_logprobs", cut_mode=args.cut_mode, source_a="steering", source_b="random_baseline"
            )

        def _vmax_from(mats: List[Optional[np.ndarray]]) -> Optional[float]:
            vals = []
            for m in mats:
                if m is None:
                    continue
                if m.size == 0:
                    continue
                mn = float(np.nanmin(m))
                mx = float(np.nanmax(m))
                vals.append(max(abs(mn), abs(mx)))
            if not vals:
                return None
            return float(max(vals))

        # Shared vlims per family
        vmax_dlogit = _vmax_from([tok_dlogit, word_dlogit, tok_dlogit_r, word_dlogit_r, tok_dlogit_diff, word_dlogit_diff])
        vmax_dprob = _vmax_from([tok_dprob, word_dprob, tok_dprob_r, word_dprob_r, tok_dprob_diff, word_dprob_diff])
        vmax_dlogprob = args.logprob_vmax if args.logprob_vmax else _vmax_from([tok_dlogprob, word_dlogprob, tok_dlogprob_r, word_dlogprob_r, tok_dlogprob_diff, word_dlogprob_diff])

        # Plot steering aggregate
        if tok_dlogit is not None and eps1 is not None:
            _plot_matrix_heatmap(
                tok_dlogit, eps1, out_dir / "aggregate_heatmap_token_dlogit_demeaned.png",
                title="Aggregate heatmap: token demeaned Δlogit (steering)",
                cbar_label=r"mean($\Delta$ logit)",
                vmax=vmax_dlogit,
            )
        if word_dlogit is not None and eps2 is not None:
            _plot_matrix_heatmap(
                word_dlogit, eps2, out_dir / "aggregate_heatmap_word_dlogit_demeaned.png",
                title="Aggregate heatmap: word demeaned Δlogit (steering)",
                cbar_label=r"mean($\Delta$ logit)",
                vmax=vmax_dlogit,
            )
        if tok_dprob is not None and eps3 is not None:
            _plot_matrix_heatmap(
                tok_dprob, eps3, out_dir / "aggregate_heatmap_token_dprob.png",
                title="Aggregate heatmap: token Δprob (steering)",
                cbar_label=r"mean($\Delta$ prob)",
                vmax=vmax_dprob,
            )
        if word_dprob is not None and eps4 is not None:
            _plot_matrix_heatmap(
                word_dprob, eps4, out_dir / "aggregate_heatmap_word_dprob.png",
                title="Aggregate heatmap: word Δprob (steering)",
                cbar_label=r"mean($\Delta$ prob)",
                vmax=vmax_dprob,
            )
        if tok_dlogprob is not None and eps5 is not None:
            _plot_matrix_heatmap(
                tok_dlogprob, eps5, out_dir / "aggregate_heatmap_token_dlogprob.png",
                title="Aggregate heatmap: token Δlogprob (steering)",
                cbar_label=r"mean($\Delta$ log prob)",
                vmax=vmax_dlogprob,
            )
        if word_dlogprob is not None and eps6 is not None:
            _plot_matrix_heatmap(
                word_dlogprob, eps6, out_dir / "aggregate_heatmap_word_dlogprob.png",
                title="Aggregate heatmap: word Δlogprob (steering)",
                cbar_label=r"mean($\Delta$ log prob)",
                vmax=vmax_dlogprob,
            )

        # Random-feature baseline (if present)
        if have_random:
            if tok_dlogit_r is not None and eps1 is not None:
                _plot_matrix_heatmap(
                    tok_dlogit_r, eps1, out_dir / "aggregate_heatmap_token_dlogit_demeaned_random.png",
                    title="Aggregate heatmap: token demeaned Δlogit (random baseline)",
                    cbar_label=r"mean($\Delta$ logit)",
                    vmax=vmax_dlogit,
                )
            if word_dlogit_r is not None and eps2 is not None:
                _plot_matrix_heatmap(
                    word_dlogit_r, eps2, out_dir / "aggregate_heatmap_word_dlogit_demeaned_random.png",
                    title="Aggregate heatmap: word demeaned Δlogit (random baseline)",
                    cbar_label=r"mean($\Delta$ logit)",
                    vmax=vmax_dlogit,
                )
            if tok_dprob_r is not None and eps3 is not None:
                _plot_matrix_heatmap(
                    tok_dprob_r, eps3, out_dir / "aggregate_heatmap_token_dprob_random.png",
                    title="Aggregate heatmap: token Δprob (random baseline)",
                    cbar_label=r"mean($\Delta$ prob)",
                    vmax=vmax_dprob,
                )
            if word_dprob_r is not None and eps4 is not None:
                _plot_matrix_heatmap(
                    word_dprob_r, eps4, out_dir / "aggregate_heatmap_word_dprob_random.png",
                    title="Aggregate heatmap: word Δprob (random baseline)",
                    cbar_label=r"mean($\Delta$ prob)",
                    vmax=vmax_dprob,
                )
            if tok_dlogprob_r is not None and eps5 is not None:
                _plot_matrix_heatmap(
                    tok_dlogprob_r, eps5, out_dir / "aggregate_heatmap_token_dlogprob_random.png",
                    title="Aggregate heatmap: token Δlogprob (random baseline)",
                    cbar_label=r"mean($\Delta$ log prob)",
                    vmax=vmax_dlogprob,
                )
            if word_dlogprob_r is not None and eps6 is not None:
                _plot_matrix_heatmap(
                    word_dlogprob_r, eps6, out_dir / "aggregate_heatmap_word_dlogprob_random.png",
                    title="Aggregate heatmap: word Δlogprob (random baseline)",
                    cbar_label=r"mean($\Delta$ log prob)",
                    vmax=vmax_dlogprob,
                )

            # Differences: steering - random_baseline
            if tok_dlogit_diff is not None and eps1 is not None:
                _plot_matrix_heatmap(
                    tok_dlogit_diff, eps1, out_dir / "aggregate_diff_heatmap_token_dlogit_demeaned_minus_random.png",
                    title="Aggregate diff heatmap: token demeaned Δlogit (steering - random)",
                    cbar_label=r"mean($\Delta$ logit)",
                    vmax=vmax_dlogit,
                )
            if word_dlogit_diff is not None and eps2 is not None:
                _plot_matrix_heatmap(
                    word_dlogit_diff, eps2, out_dir / "aggregate_diff_heatmap_word_dlogit_demeaned_minus_random.png",
                    title="Aggregate diff heatmap: word demeaned Δlogit (steering - random)",
                    cbar_label=r"mean($\Delta$ logit)",
                    vmax=vmax_dlogit,
                )
            if tok_dprob_diff is not None and eps3 is not None:
                _plot_matrix_heatmap(
                    tok_dprob_diff, eps3, out_dir / "aggregate_diff_heatmap_token_dprob_minus_random.png",
                    title="Aggregate diff heatmap: token Δprob (steering - random)",
                    cbar_label=r"mean($\Delta$ prob)",
                    vmax=vmax_dprob,
                )
            if word_dprob_diff is not None and eps4 is not None:
                _plot_matrix_heatmap(
                    word_dprob_diff, eps4, out_dir / "aggregate_diff_heatmap_word_dprob_minus_random.png",
                    title="Aggregate diff heatmap: word Δprob (steering - random)",
                    cbar_label=r"mean($\Delta$ prob)",
                    vmax=vmax_dprob,
                )
            if tok_dlogprob_diff is not None and eps5 is not None:
                _plot_matrix_heatmap(
                    tok_dlogprob_diff, eps5, out_dir / "aggregate_diff_heatmap_token_dlogprob_minus_random.png",
                    title="Aggregate diff heatmap: token Δlogprob (steering - random)",
                    cbar_label=r"mean($\Delta$ log prob)",
                    vmax=vmax_dlogprob,
                )
            if word_dlogprob_diff is not None and eps6 is not None:
                _plot_matrix_heatmap(
                    word_dlogprob_diff, eps6, out_dir / "aggregate_diff_heatmap_word_dlogprob_minus_random.png",
                    title="Aggregate diff heatmap: word Δlogprob (steering - random)",
                    cbar_label=r"mean($\Delta$ log prob)",
                    vmax=vmax_dlogprob,
                )

            _plot_dose_response_diff(
                steer_samples,
                out_dir / "dose_response_diff_token_dlogit_demeaned_minus_random.png",
                metric="token_delta_demeaned_target_logits",
                max_samples=args.max_steer_curves,
            )
            _plot_dose_response_diff(
                steer_samples,
                out_dir / "dose_response_diff_token_dprob_minus_random.png",
                metric="token_delta_target_probs",
                max_samples=args.max_steer_curves,
            )
            _plot_dose_response_diff(
                steer_samples,
                out_dir / "dose_response_diff_token_dlogprob_minus_random.png",
                metric="token_delta_target_logprobs",
                max_samples=args.max_steer_curves,
            )
            _plot_dose_response_diff(
                steer_samples,
                out_dir / "dose_response_diff_word_dlogit_demeaned_minus_random.png",
                metric="word_delta_demeaned_target_logits",
                max_samples=args.max_steer_curves,
            )
            _plot_dose_response_diff(
                steer_samples,
                out_dir / "dose_response_diff_word_dprob_minus_random.png",
                metric="word_delta_target_probs",
                max_samples=args.max_steer_curves,
            )
            _plot_dose_response_diff(
                steer_samples,
                out_dir / "dose_response_diff_word_dlogprob_minus_random.png",
                metric="word_delta_target_logprobs",
                max_samples=args.max_steer_curves,
            )

    # Write a tiny index.html that links to generated images (nice for browsing)
    imgs = sorted([p.name for p in out_dir.glob("*.png")])
    if imgs:
        lines = ["<html><body><h2>Attr/Steer Visualizations</h2>"]
        for name in imgs:
            lines.append(f"<div><h3>{name}</h3><img src='{name}' style='max-width: 1200px; width: 100%;'/></div>")
        lines.append("</body></html>")
        (out_dir / "index.html").write_text("\n".join(lines))

    print(str(out_dir))


if __name__ == "__main__":
    main()


