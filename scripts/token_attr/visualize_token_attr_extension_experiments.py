#!/usr/bin/env -S uv run python
"""Visualize token attribution extension experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


MODE_LABELS = {
    "full_sum": "Full Sum",
    "first_token": "First Token",
    "last_token": "Last Token",
    "first5_sum": "First 5 Sum",
}


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _maybe(path: Path) -> Optional[Path]:
    return path if path.exists() else None


def _mean_or_nan(values: Iterable[float]) -> float:
    xs = [float(v) for v in values if v is not None and not np.isnan(float(v))]
    if not xs:
        return float("nan")
    return float(np.mean(np.asarray(xs, dtype=np.float64)))


def _pair_summary_to_matrix(
    rows: Sequence[Dict[str, Any]],
    metric: str,
    diagonal_value: Optional[float] = None,
) -> Optional[np.ndarray]:
    if not rows:
        return None
    max_pos = 0
    for row in rows:
        max_pos = max(max_pos, int(row["i"]), int(row["j"]))
    mat = np.full((max_pos + 1, max_pos + 1), np.nan, dtype=np.float64)
    for row in rows:
        val = row.get(metric)
        if val is None:
            continue
        i = int(row["i"])
        j = int(row["j"])
        mat[i, j] = float(val)
        mat[j, i] = float(val)
    if diagonal_value is not None:
        np.fill_diagonal(mat, diagonal_value)
    return mat


def _plot_pair_heatmap(
    rows: Sequence[Dict[str, Any]],
    metric: str,
    out_path: Path,
    title: str,
    center: Optional[float] = None,
    lower_triangle: bool = False,
    diagonal_value: Optional[float] = None,
    cbar_label: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    mat = _pair_summary_to_matrix(rows, metric, diagonal_value=diagonal_value)
    if mat is None:
        return
    mask = None
    if lower_triangle:
        mask = np.triu(np.ones_like(mat, dtype=bool), k=0 if diagonal_value is None else 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        mat,
        ax=ax,
        cmap="coolwarm" if center is not None else "viridis",
        center=center,
        mask=mask,
        square=True,
        cbar_kws={"label": cbar_label or metric},
    )
    if title:
        ax.set_title(title)
    ax.set_xlabel("Token position")
    ax.set_ylabel("Token position")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_gap_summary(
    rows: Sequence[Dict[str, Any]],
    real_metric: str,
    random_metric: str,
    diff_metric: str,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return

    gaps = np.array([int(row["gap"]) for row in rows], dtype=np.int64)
    real = np.array([row.get(real_metric, np.nan) for row in rows], dtype=np.float64)
    rand = np.array([row.get(random_metric, np.nan) for row in rows], dtype=np.float64)
    diff = np.array([row.get(diff_metric, np.nan) for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(gaps, real, label="real", linewidth=2)
    ax.plot(gaps, rand, label="random", linewidth=2)
    ax.plot(gaps, diff, label="real-random", linewidth=2)
    ax.axhline(0, color="gray", linewidth=1, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Position gap")
    ax.set_ylabel("Mean value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_span_ratio_masses(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    pos = np.array([int(row["target_pos"]) for row in rows], dtype=np.int64)
    cached = np.array([row.get("prefix_only_cached_l1_mean", np.nan) for row in rows], dtype=np.float64)
    prefix = np.array([row.get("dynamic_prefix_l1_mean", np.nan) for row in rows], dtype=np.float64)
    history = np.array([row.get("dynamic_history_l1_mean", np.nan) for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(pos, cached, label="cached prefix-only", linewidth=2)
    ax.plot(pos, prefix, label="dynamic prefix span", linewidth=2)
    ax.plot(pos, history, label="dynamic history span", linewidth=2)
    ax.set_title("Attribution L1 Mass by Target Position")
    ax.set_xlabel("Target token position")
    ax.set_ylabel("Mean L1 mass")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_span_ratio_ratio(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    pos = np.array([int(row["target_pos"]) for row in rows], dtype=np.int64)
    ratio = np.array([row.get("prefix_to_history_ratio_mean", np.nan) for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(pos, ratio, linewidth=2)
    ax.set_title("Prefix / History L1 Ratio by Target Position")
    ax.set_xlabel("Target token position")
    ax.set_ylabel("Mean prefix-to-history ratio")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_span_ratio_fractions(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    pos = np.array([int(row["target_pos"]) for row in rows], dtype=np.int64)
    prefix_frac = np.array([row.get("prefix_fraction_mean", np.nan) for row in rows], dtype=np.float64)
    history_frac = np.array([row.get("history_fraction_mean", np.nan) for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(pos, prefix_frac, label="prefix fraction", linewidth=2)
    ax.plot(pos, history_frac, label="history fraction", linewidth=2)
    ax.set_title("Prefix / History Fraction of Total Attribution Mass")
    ax.set_xlabel("Target token position")
    ax.set_ylabel("Mean fraction of total L1")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _extract_epsilons(per_epsilon: Dict[str, Any]) -> List[float]:
    vals = []
    for k in per_epsilon.keys():
        try:
            vals.append(float(k))
        except Exception:
            continue
    return sorted(vals)


def _mean_array(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    return float(np.mean(np.asarray(xs, dtype=np.float64)))


def _compute_mode_curve(
    steer_samples: Sequence[Dict[str, Any]],
    mode: str,
    metric: str,
    source: str = "steering",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    curves = []
    for sample in steer_samples:
        mode_payload = (sample.get("span_modes", {}) or {}).get(mode, {}) or {}
        pe = (mode_payload.get(source, {}) or {}).get("per_epsilon", {}) or {}
        eps = _extract_epsilons(pe)
        if not eps:
            continue
        ys = []
        for eps_val in eps:
            arr = (pe.get(str(float(eps_val))) or pe.get(str(eps_val)) or {}).get(metric, [])
            ys.append(_mean_array(arr))
        curves.append((np.asarray(eps, dtype=np.float64), np.asarray(ys, dtype=np.float64)))

    if not curves:
        return None, None

    all_eps = sorted({float(v) for x, _ in curves for v in x.tolist()})
    mean_y = []
    for eps_val in all_eps:
        vals = []
        for x, y in curves:
            idx = np.where(np.isclose(x, eps_val))[0]
            if idx.size:
                vals.append(float(y[idx[0]]))
        mean_y.append(_mean_or_nan(vals))
    return np.asarray(all_eps, dtype=np.float64), np.asarray(mean_y, dtype=np.float64)


def _compute_mode_diff_curve(
    steer_samples: Sequence[Dict[str, Any]],
    mode: str,
    metric: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    curves = []
    for sample in steer_samples:
        mode_payload = (sample.get("span_modes", {}) or {}).get(mode, {}) or {}
        pe_a = (mode_payload.get("steering", {}) or {}).get("per_epsilon", {}) or {}
        pe_b = (mode_payload.get("random_baseline", {}) or {}).get("per_epsilon", {}) or {}
        eps_a = _extract_epsilons(pe_a)
        eps_b = _extract_epsilons(pe_b)
        if not eps_a or not eps_b or len(eps_a) != len(eps_b):
            continue
        ys = []
        for eps_val in eps_a:
            key = str(float(eps_val))
            a_arr = (pe_a.get(key) or pe_a.get(str(eps_val)) or {}).get(metric, [])
            b_arr = (pe_b.get(key) or pe_b.get(str(eps_val)) or {}).get(metric, [])
            L = min(len(a_arr), len(b_arr))
            if L <= 0:
                ys.append(float("nan"))
            else:
                aa = np.asarray(a_arr[:L], dtype=np.float64)
                bb = np.asarray(b_arr[:L], dtype=np.float64)
                ys.append(float(np.mean(aa - bb)))
        curves.append((np.asarray(eps_a, dtype=np.float64), np.asarray(ys, dtype=np.float64)))

    if not curves:
        return None, None

    all_eps = sorted({float(v) for x, _ in curves for v in x.tolist()})
    mean_y = []
    for eps_val in all_eps:
        vals = []
        for x, y in curves:
            idx = np.where(np.isclose(x, eps_val))[0]
            if idx.size:
                vals.append(float(y[idx[0]]))
        mean_y.append(_mean_or_nan(vals))
    return np.asarray(all_eps, dtype=np.float64), np.asarray(mean_y, dtype=np.float64)


def _plot_mode_comparison(
    steer_samples: Sequence[Dict[str, Any]],
    modes: Sequence[str],
    metric: str,
    out_path: Path,
    title: str,
    diff: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    plotted = False
    for mode in modes:
        if diff:
            xs, ys = _compute_mode_diff_curve(steer_samples, mode, metric)
        else:
            xs, ys = _compute_mode_curve(steer_samples, mode, metric)
        if xs is None or ys is None:
            continue
        plotted = True
        ax.plot(xs, ys, linewidth=2, label=MODE_LABELS.get(mode, mode))

    if not plotted:
        plt.close(fig)
        return

    ax.axhline(0, color="gray", linewidth=1, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("epsilon")
    ax.set_ylabel("Mean over continuation positions")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _compute_mode_matrix(
    steer_samples: Sequence[Dict[str, Any]],
    mode: str,
    metric: str,
    source: str = "steering",
) -> Tuple[Optional[np.ndarray], Optional[List[float]], int]:
    per_sample = []
    eps_all = None
    for sample in steer_samples:
        mode_payload = (sample.get("span_modes", {}) or {}).get(mode, {}) or {}
        pe = (mode_payload.get(source, {}) or {}).get("per_epsilon", {}) or {}
        eps = _extract_epsilons(pe)
        if not eps:
            continue
        if eps_all is None:
            eps_all = eps
        elif len(eps) != len(eps_all) or any(not np.isclose(eps[i], eps_all[i]) for i in range(len(eps))):
            continue
        rows = []
        for eps_val in eps_all:
            rows.append((pe.get(str(float(eps_val))) or pe.get(str(eps_val)) or {}).get(metric, []))
        per_sample.append(rows)

    if not per_sample or eps_all is None:
        return None, None, 0

    L = min(max((len(r) for r in rows), default=0) for rows in per_sample)
    if L <= 0:
        return None, eps_all, 0

    mats = []
    for rows in per_sample:
        mat = np.full((len(eps_all), L), np.nan, dtype=np.float64)
        for i, row in enumerate(rows):
            arr = np.asarray(row[:L], dtype=np.float64)
            mat[i, : arr.size] = arr
        mats.append(mat)
    mean_mat = np.nanmean(np.stack(mats, axis=0), axis=0)
    return mean_mat, eps_all, L


def _compute_mode_diff_matrix(
    steer_samples: Sequence[Dict[str, Any]],
    mode: str,
    metric: str,
) -> Tuple[Optional[np.ndarray], Optional[List[float]], int]:
    per_sample = []
    eps_all = None
    for sample in steer_samples:
        mode_payload = (sample.get("span_modes", {}) or {}).get(mode, {}) or {}
        pe_a = (mode_payload.get("steering", {}) or {}).get("per_epsilon", {}) or {}
        pe_b = (mode_payload.get("random_baseline", {}) or {}).get("per_epsilon", {}) or {}
        eps_a = _extract_epsilons(pe_a)
        eps_b = _extract_epsilons(pe_b)
        if not eps_a or not eps_b or len(eps_a) != len(eps_b):
            continue
        if eps_all is None:
            eps_all = eps_a
        elif len(eps_a) != len(eps_all) or any(not np.isclose(eps_a[i], eps_all[i]) for i in range(len(eps_a))):
            continue
        rows = []
        for eps_val in eps_all:
            key = str(float(eps_val))
            rows.append(
                (
                    (pe_a.get(key) or pe_a.get(str(eps_val)) or {}).get(metric, []),
                    (pe_b.get(key) or pe_b.get(str(eps_val)) or {}).get(metric, []),
                )
            )
        per_sample.append(rows)

    if not per_sample or eps_all is None:
        return None, None, 0

    L = min(min(len(a), len(b)) for rows in per_sample for a, b in rows)
    if L <= 0:
        return None, eps_all, 0

    mats = []
    for rows in per_sample:
        mat = np.full((len(eps_all), L), np.nan, dtype=np.float64)
        for i, (a, b) in enumerate(rows):
            aa = np.asarray(a[:L], dtype=np.float64)
            bb = np.asarray(b[:L], dtype=np.float64)
            mat[i, :L] = aa - bb
        mats.append(mat)
    mean_mat = np.nanmean(np.stack(mats, axis=0), axis=0)
    return mean_mat, eps_all, L


def _plot_matrix_heatmap(
    mat: np.ndarray,
    eps_all: Sequence[float],
    out_path: Path,
    title: str,
    center: Optional[float] = None,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    if mat.size == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        mat,
        ax=ax,
        cmap="coolwarm" if center is not None else "viridis",
        center=center,
        cbar_kws={"label": title},
    )
    ax.set_title(title)
    ax.set_xlabel("Continuation token position")
    ax.set_ylabel("epsilon")
    ax.set_yticks(np.arange(len(eps_all)) + 0.5)
    ax.set_yticklabels([str(eps) for eps in eps_all], rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    in_dir = args.input_dir
    out_dir = args.output_dir or (in_dir / "viz")
    _ensure_dir(out_dir)

    consistency_path = _maybe(in_dir / "consistency.json")
    span_ratio_path = _maybe(in_dir / "span_ratio.json")
    span_steer_path = _maybe(in_dir / "span_steer.json")

    if consistency_path:
        payload = _load_json(consistency_path)
        pair_rows = payload.get("pair_summary", []) or []
        gap_rows = payload.get("gap_summary", []) or []
        _plot_pair_heatmap(pair_rows, "real_l1_mean", out_dir / "consistency_l1_real.png", "Consistency L1 Distance", center=None)
        _plot_pair_heatmap(
            pair_rows,
            "real_minus_random_l1_mean",
            out_dir / "consistency_l1_minus_random.png",
            "Consistency L1 Distance Minus Random",
            center=0.0,
        )
        _plot_pair_heatmap(pair_rows, "real_jaccard_mean", out_dir / "consistency_jaccard_real.png", "Consistency Jaccard Top-100", center=None)
        _plot_pair_heatmap(
            pair_rows,
            "real_jaccard_mean",
            out_dir / "consistency_jaccard_real_lower.png",
            "",
            center=None,
            lower_triangle=True,
            cbar_label="Top-100 Jaccard",
        )
        _plot_pair_heatmap(
            pair_rows,
            "real_minus_random_jaccard_mean",
            out_dir / "consistency_jaccard_minus_random.png",
            "Consistency Jaccard Minus Random",
            center=0.0,
        )
        _plot_gap_summary(
            gap_rows,
            "real_l1_mean",
            "random_l1_mean",
            "real_minus_random_l1_mean",
            out_dir / "consistency_gap_l1.png",
            "Consistency L1 by Position Gap",
        )
        _plot_gap_summary(
            gap_rows,
            "real_jaccard_mean",
            "random_jaccard_mean",
            "real_minus_random_jaccard_mean",
            out_dir / "consistency_gap_jaccard.png",
            "Consistency Jaccard by Position Gap",
        )

    if span_ratio_path:
        payload = _load_json(span_ratio_path)
        rows = payload.get("position_summary", []) or []
        _plot_span_ratio_masses(rows, out_dir / "span_ratio_masses.png")
        _plot_span_ratio_ratio(rows, out_dir / "span_ratio_prefix_history_ratio.png")
        _plot_span_ratio_fractions(rows, out_dir / "span_ratio_fractions.png")

    if span_steer_path:
        payload = _load_json(span_steer_path)
        samples = payload.get("samples", []) or []
        modes = payload.get("span_modes", []) or []

        _plot_mode_comparison(
            samples,
            modes,
            "token_delta_target_probs",
            out_dir / "span_mode_compare_token_dprob.png",
            "Span-Mode Steering Comparison: Token ΔProb",
            diff=False,
        )
        _plot_mode_comparison(
            samples,
            modes,
            "token_delta_demeaned_target_logits",
            out_dir / "span_mode_compare_token_dlogit_demeaned.png",
            "Span-Mode Steering Comparison: Token ΔDemeaned Logit",
            diff=False,
        )
        _plot_mode_comparison(
            samples,
            modes,
            "token_delta_target_probs",
            out_dir / "span_mode_compare_token_dprob_minus_random.png",
            "Span-Mode Steering Minus Random: Token ΔProb",
            diff=True,
        )
        _plot_mode_comparison(
            samples,
            modes,
            "token_delta_demeaned_target_logits",
            out_dir / "span_mode_compare_token_dlogit_demeaned_minus_random.png",
            "Span-Mode Steering Minus Random: Token ΔDemeaned Logit",
            diff=True,
        )

        for mode in modes:
            label = MODE_LABELS.get(mode, mode)
            mat, eps_all, _ = _compute_mode_matrix(samples, mode, "token_delta_target_probs", source="steering")
            if mat is not None and eps_all is not None:
                _plot_matrix_heatmap(
                    mat,
                    eps_all,
                    out_dir / f"aggregate_heatmap_{mode}_token_dprob.png",
                    f"{label}: Token ΔProb",
                    center=0.0,
                )
            mat, eps_all, _ = _compute_mode_matrix(samples, mode, "token_delta_demeaned_target_logits", source="steering")
            if mat is not None and eps_all is not None:
                _plot_matrix_heatmap(
                    mat,
                    eps_all,
                    out_dir / f"aggregate_heatmap_{mode}_token_dlogit_demeaned.png",
                    f"{label}: Token ΔDemeaned Logit",
                    center=0.0,
                )
            diff_mat, eps_all, _ = _compute_mode_diff_matrix(samples, mode, "token_delta_target_probs")
            if diff_mat is not None and eps_all is not None:
                _plot_matrix_heatmap(
                    diff_mat,
                    eps_all,
                    out_dir / f"aggregate_diff_heatmap_{mode}_token_dprob_minus_random.png",
                    f"{label}: Token ΔProb Minus Random",
                    center=0.0,
                )
            diff_mat, eps_all, _ = _compute_mode_diff_matrix(samples, mode, "token_delta_demeaned_target_logits")
            if diff_mat is not None and eps_all is not None:
                _plot_matrix_heatmap(
                    diff_mat,
                    eps_all,
                    out_dir / f"aggregate_diff_heatmap_{mode}_token_dlogit_demeaned_minus_random.png",
                    f"{label}: Token ΔDemeaned Logit Minus Random",
                    center=0.0,
                )

    imgs = sorted([p.name for p in out_dir.glob("*.png")])
    if imgs:
        lines = ["<html><body><h2>Token Attribution Extension Visualizations</h2>"]
        for name in imgs:
            lines.append(f"<div><h3>{name}</h3><img src='{name}' style='max-width: 1200px; width: 100%;'/></div>")
        lines.append("</body></html>")
        (out_dir / "index.html").write_text("\n".join(lines))

    print(str(out_dir))


if __name__ == "__main__":
    main()
