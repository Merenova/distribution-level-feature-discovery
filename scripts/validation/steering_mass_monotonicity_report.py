#!/usr/bin/env python3
"""
Restricted steering mass monotonicity report (old 7c outputs).

Reads Stage 7c (old) JSON outputs and computes, for sign_full sweeps:
  - Δlog_mass = log(M_steered) - log(M_original) per (prefix, clustering_key, sweep_key, cluster_id, epsilon)
  - Restricts to eps>0 by default
  - Reports monotonicity (non-decreasing) and Pearson corr/slope vs epsilon

This is meant to debug whether "target cluster mass increases with ε>0" holds.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


def _safe_float(x):
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _slope(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2:
        return None
    # least squares slope
    x0 = x - x.mean()
    denom = float((x0 * x0).sum())
    if denom == 0.0:
        return None
    return float((x0 * (y - y.mean())).sum() / denom)


def _is_monotone_non_decreasing(x: np.ndarray, y: np.ndarray) -> bool:
    # assumes x sorted
    if y.size < 2:
        return True
    diffs = np.diff(y)
    return bool(np.all(diffs >= -1e-12))


def extract_sign_full_series(
    data: Dict[str, Any],
    eps_filter: List[float] | None,
) -> List[Dict[str, Any]]:
    """Return list of rows with fields needed for reporting."""
    pid = data.get("prefix_id")
    if not pid:
        return []

    out = []
    runs = data.get("clustering_runs", {}) or {}
    for clustering_key, run in runs.items():
        results = (run.get("results") or {})
        for sweep_key, sweep in results.items():
            # Restrict to sign_full only
            if not str(sweep_key).startswith("sign_full_"):
                continue

            pcm = sweep.get("per_cluster_mass", {}) or {}
            for cluster_id, stats in pcm.items():
                m0d = stats.get("cluster_mass_original", {}) or {}
                msd = stats.get("cluster_mass_steered", {}) or {}
                # intersection of eps keys
                for eps_str in set(m0d.keys()) & set(msd.keys()):
                    eps = _safe_float(eps_str)
                    if eps is None:
                        continue
                    if eps_filter is not None and eps not in eps_filter:
                        continue
                    m0 = _safe_float(m0d.get(eps_str))
                    ms = _safe_float(msd.get(eps_str))
                    if m0 is None or ms is None:
                        continue
                    if m0 <= 0.0 or ms <= 0.0:
                        continue
                    dl = math.log(ms) - math.log(m0)
                    out.append(
                        {
                            "prefix_id": pid,
                            "clustering_key": clustering_key,
                            "sweep_key": sweep_key,
                            "cluster_id": str(cluster_id),
                            "epsilon": eps,
                            "delta_log_mass": dl,
                            "mass_original": m0,
                            "mass_steered": ms,
                        }
                    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-glob",
        required=True,
        help="Glob for cloze_*_sweep_results.json files (quote it).",
    )
    ap.add_argument(
        "--eps-positive-only",
        action="store_true",
        help="If set, restrict to eps>0.",
    )
    ap.add_argument(
        "--eps",
        type=float,
        nargs="*",
        default=None,
        help="Optional explicit epsilon list to include (exact match).",
    )
    ap.add_argument("--out", required=True, help="Output JSON path.")
    args = ap.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        raise SystemExit(f"No files matched: {args.input_glob}")

    eps_filter = None
    if args.eps is not None and len(args.eps) > 0:
        eps_filter = [float(e) for e in args.eps]

    all_rows: List[Dict[str, Any]] = []
    for fp in files:
        with open(fp, "r") as f:
            data = json.load(f)
        rows = extract_sign_full_series(data, eps_filter=eps_filter)
        all_rows.extend(rows)

    # optional eps>0 restriction
    if args.eps_positive_only:
        all_rows = [r for r in all_rows if r["epsilon"] > 0]

    # group into "series": (prefix_id, clustering_key, sweep_key, cluster_id)
    series_map: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for r in all_rows:
        key = (r["prefix_id"], r["clustering_key"], r["sweep_key"], r["cluster_id"])
        series_map.setdefault(key, []).append(r)

    series_summaries = []
    for (pid, ck, sk, cid), rows in series_map.items():
        rows = sorted(rows, key=lambda x: x["epsilon"])
        x = np.asarray([rr["epsilon"] for rr in rows], dtype=np.float64)
        y = np.asarray([rr["delta_log_mass"] for rr in rows], dtype=np.float64)

        corr = _pearson_corr(x, y)
        slope = _slope(x, y)
        mono = _is_monotone_non_decreasing(x, y)

        series_summaries.append(
            {
                "prefix_id": pid,
                "clustering_key": ck,
                "sweep_key": sk,
                "cluster_id": cid,
                "n_points": int(x.size),
                "epsilons": [float(v) for v in x.tolist()],
                "delta_log_mass": [float(v) for v in y.tolist()],
                "corr_eps_delta_log_mass": corr,
                "slope_eps_delta_log_mass": slope,
                "monotone_non_decreasing": mono,
            }
        )

    # overall aggregates
    corrs = np.asarray([s["corr_eps_delta_log_mass"] for s in series_summaries if s["corr_eps_delta_log_mass"] is not None], dtype=np.float64)
    slopes = np.asarray([s["slope_eps_delta_log_mass"] for s in series_summaries if s["slope_eps_delta_log_mass"] is not None], dtype=np.float64)
    mono_rate = float(np.mean([1.0 if s["monotone_non_decreasing"] else 0.0 for s in series_summaries])) if series_summaries else float("nan")

    payload = {
        "input_glob": args.input_glob,
        "eps_positive_only": bool(args.eps_positive_only),
        "eps_filter": eps_filter,
        "n_files": len(files),
        "n_rows": len(all_rows),
        "n_series": len(series_summaries),
        "monotone_rate": mono_rate,
        "corr_summary": {
            "n": int(corrs.size),
            "mean": float(corrs.mean()) if corrs.size else None,
            "median": float(np.median(corrs)) if corrs.size else None,
            "p10": float(np.quantile(corrs, 0.1)) if corrs.size else None,
            "p90": float(np.quantile(corrs, 0.9)) if corrs.size else None,
        },
        "slope_summary": {
            "n": int(slopes.size),
            "mean": float(slopes.mean()) if slopes.size else None,
            "median": float(np.median(slopes)) if slopes.size else None,
            "p10": float(np.quantile(slopes, 0.1)) if slopes.size else None,
            "p90": float(np.quantile(slopes, 0.9)) if slopes.size else None,
        },
        "series": series_summaries,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(args.out)


if __name__ == "__main__":
    main()


