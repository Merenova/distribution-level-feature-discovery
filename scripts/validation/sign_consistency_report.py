#!/usr/bin/env python3
"""
Sign-consistency diagnostic for clustering prototypes (mu_a) vs per-continuation attributions.

For a given prefix_id and clustering config (beta,gamma), we have:
  - Stage 3: aggregated_attributions: [n_cont, n_sources] from *_prefix_context.pt
  - Stage 5: assignments: [n_cont] and components[cluster_id].mu_a: [d_a] from cloze_*_sweep_results.json

We compute, for each cluster k and feature-dimension f among top-|mu_a| dims:
  SignConsistency_k[f] = mean( sign(a_n[f]) == sign(mu_a_k[f]) ) over n in cluster k

Notes/assumptions:
  - We align by index: column f in aggregated_attributions corresponds to dimension f in mu_a.
  - We optionally restrict to the first n_prefix_features dims (typically 4096) since Stage 7 uses
    the first n_features dims when selecting steering features.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def _find_grid_entry(clustering_sweep: Dict[str, Any], beta: float, gamma: float) -> Dict[str, Any] | None:
    grid = clustering_sweep.get("grid", []) or []
    for e in grid:
        try:
            b = float(e.get("beta"))
            g = float(e.get("gamma"))
        except Exception:
            continue
        if abs(b - beta) < 1e-9 and abs(g - gamma) < 1e-9:
            return e
    return None


def _sign(x: np.ndarray, eps: float) -> np.ndarray:
    # ternary sign: -1, 0, +1 with deadzone eps
    out = np.zeros_like(x, dtype=np.int8)
    out[x > eps] = 1
    out[x < -eps] = -1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-ids", nargs="+", required=True, help="Prefix IDs like cloze_0105")
    ap.add_argument("--attribution-dir", required=True, help="Directory with *_prefix_context.pt")
    ap.add_argument("--clustering-dir", required=True, help="Directory with cloze_*_sweep_results.json (Stage 5)")
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--topB", type=int, default=50, help="Top-|mu_a| dims per cluster to check")
    ap.add_argument("--sign-eps", type=float, default=0.0, help="Deadzone for sign() on a_n[f]")
    ap.add_argument("--restrict-prefix-features", action="store_true", help="Restrict dims to [0:n_prefix_features)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    attr_dir = Path(args.attribution_dir)
    clus_dir = Path(args.clustering_dir)

    all_prefix_rows = []

    for pid in args.prefix_ids:
        pt_path = attr_dir / f"{pid}_prefix_context.pt"
        clus_path = clus_dir / f"{pid}_sweep_results.json"
        if not pt_path.exists() or not clus_path.exists():
            continue

        d = torch.load(pt_path, map_location="cpu", weights_only=False)
        A = d.get("aggregated_attributions")
        if not torch.is_tensor(A) or A.ndim != 2:
            continue
        A = A.to(torch.float32).numpy()  # [n_cont, n_sources]

        n_cont, n_sources = A.shape
        n_prefix_features = int(d.get("n_prefix_features", 0) or 0)

        clustering_sweep = json.loads(clus_path.read_text())
        entry = _find_grid_entry(clustering_sweep, args.beta, args.gamma)
        if entry is None:
            continue

        assignments = entry.get("assignments", [])
        components = entry.get("components", {}) or {}
        if not assignments or not components:
            continue

        n_assign = len(assignments)
        n_use = min(n_cont, n_assign)
        if n_use <= 0:
            continue

        A_use = A[:n_use, :]
        assignments = assignments[:n_use]

        # determine usable dims for comparing with mu_a
        # mu_a length can differ slightly; use min across mu_a vectors and A columns
        mu_lengths = []
        for comp in components.values():
            mu = comp.get("mu_a")
            if isinstance(mu, list):
                mu_lengths.append(len(mu))
        if not mu_lengths:
            continue
        d_a = min(mu_lengths)
        max_dim = min(d_a, n_sources)
        if args.restrict_prefix_features and n_prefix_features > 0:
            max_dim = min(max_dim, n_prefix_features)

        # precompute sign(A) once
        signA = _sign(A_use[:, :max_dim], eps=args.sign_eps)  # [n_use, max_dim]

        # compute per-cluster sign consistency
        clusters = sorted([int(k) for k in components.keys() if str(k).isdigit()])
        per_cluster = {}

        for k in clusters:
            comp = components.get(str(k), {})
            mu = comp.get("mu_a")
            if not isinstance(mu, list) or len(mu) < max_dim:
                continue
            mu = np.asarray(mu[:max_dim], dtype=np.float32)
            mu_sign = _sign(mu, eps=0.0)  # do not deadzone the prototype

            idx = np.where(np.asarray(assignments, dtype=np.int64) == k)[0]
            if idx.size == 0:
                continue

            # pick topB dims by |mu|
            topB = min(args.topB, max_dim)
            top_dims = np.argsort(np.abs(mu))[-topB:][::-1]

            # for each dim, fraction of matches among nonzero prototype sign
            dim_scores = []
            for f in top_dims:
                if mu_sign[f] == 0:
                    continue
                s = signA[idx, f]
                # ignore ambiguous zeros in A (if sign_eps > 0)
                valid = s != 0
                if valid.sum() == 0:
                    continue
                agree = np.mean((s[valid] == mu_sign[f]).astype(np.float32))
                dim_scores.append((int(f), float(agree), float(mu[f])))

            # aggregate
            if dim_scores:
                mean_agree = float(np.mean([x[1] for x in dim_scores]))
            else:
                mean_agree = float("nan")

            per_cluster[str(k)] = {
                "n_items": int(idx.size),
                "mean_sign_consistency_topB": mean_agree,
                "dims": dim_scores[: min(50, len(dim_scores))],  # cap for size
            }

        all_prefix_rows.append(
            {
                "prefix_id": pid,
                "beta": args.beta,
                "gamma": args.gamma,
                "n_continuations_used": int(n_use),
                "n_sources": int(n_sources),
                "n_prefix_features": int(n_prefix_features),
                "max_dim_used": int(max_dim),
                "topB": int(args.topB),
                "sign_eps": float(args.sign_eps),
                "restrict_prefix_features": bool(args.restrict_prefix_features),
                "per_cluster": per_cluster,
            }
        )

    # overall summary
    all_means = []
    for row in all_prefix_rows:
        for c, stats in row["per_cluster"].items():
            v = stats.get("mean_sign_consistency_topB")
            if v is not None and math.isfinite(v):
                all_means.append(v)
    all_means_np = np.asarray(all_means, dtype=np.float32)
    summary = {
        "n_prefixes": len(all_prefix_rows),
        "n_cluster_entries": int(all_means_np.size),
        "mean_of_cluster_means": float(all_means_np.mean()) if all_means_np.size else None,
        "median_of_cluster_means": float(np.median(all_means_np)) if all_means_np.size else None,
        "p10": float(np.quantile(all_means_np, 0.1)) if all_means_np.size else None,
        "p90": float(np.quantile(all_means_np, 0.9)) if all_means_np.size else None,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "rows": all_prefix_rows}, indent=2))
    print(str(out_path))


if __name__ == "__main__":
    main()



