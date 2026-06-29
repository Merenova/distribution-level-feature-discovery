#!/usr/bin/env python3
"""
Check whether sign(Delta_H_c) differs between:
- weighted-median (current: clustering attribution_metric="l1") stored in 6_semantic_graphs/*.pt
- probability-weighted mean recomputed from raw per-continuation attributions + path probabilities

This script scans a semantic-graphs directory (Stage 6 output) and, for each
{prefix,beta,gamma} file:
1) loads stored Delta_H_c (per-cluster) from the .pt
2) loads matching clustering assignments from Stage 5 sweep results
3) loads raw per-continuation attributions from Stage 3 prefix_context.pt
4) loads per-continuation probabilities from Stage 2 branches.json
5) recomputes centered attributions (a_n - H_0), then weighted-mean Delta_H_c per cluster
6) compares signs on top-|Delta_H_c| features (configurable topB list)

Outputs a JSON report and prints a short summary.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


PT_RE = re.compile(
    r"^(?P<prefix_id>cloze_\d+)_beta(?P<beta>[0-9.]+)_gamma(?P<gamma>[0-9.]+)_semantic_graphs\.pt$"
)


def _safe_sign(x: float, eps: float) -> int:
    if not np.isfinite(x) or abs(x) < eps:
        return 0
    return 1 if x > 0 else -1


def _weighted_mean(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    wsum = float(np.sum(weights))
    if wsum <= 0:
        return np.zeros((data.shape[1],), dtype=np.float64)
    return (weights[:, None] * data).sum(axis=0) / wsum


def load_path_probs(branches_path: Path) -> np.ndarray:
    branches = json.loads(branches_path.read_text())
    probs = [float(c.get("probability", 0.0)) for c in branches.get("continuations", [])]
    return np.asarray(probs, dtype=np.float64)


def load_attributions(prefix_context_path: Path) -> np.ndarray:
    data = torch.load(prefix_context_path, map_location="cpu", weights_only=False)
    a = data["aggregated_attributions"]
    if isinstance(a, torch.Tensor):
        a = a.detach().float().cpu().numpy()
    return np.asarray(a, dtype=np.float64)


def load_sweep_entry(sweep_path: Path, beta: float, gamma: float) -> Dict[str, Any]:
    obj = json.loads(sweep_path.read_text())
    grid = obj.get("grid", [])
    for entry in grid:
        if float(entry.get("beta")) == float(beta) and float(entry.get("gamma")) == float(gamma):
            return entry
    raise KeyError(f"No grid entry for beta={beta} gamma={gamma} in {sweep_path}")


def check_file(
    pt_path: Path,
    clustering_dir: Path,
    branches_dir: Path,
    attribution_dir: Path,
    topBs: List[int],
    eps: float,
    max_examples_per_cluster: int,
) -> Dict[str, Any]:
    m = PT_RE.match(pt_path.name)
    if not m:
        raise ValueError(f"Unexpected filename: {pt_path.name}")
    prefix_id = m.group("prefix_id")
    beta = float(m.group("beta"))
    gamma = float(m.group("gamma"))

    # Stage 6 output (stored weighted-median Delta_H_c since attribution_metric=l1)
    pt = torch.load(pt_path, map_location="cpu", weights_only=False)
    H_0 = np.asarray(pt["H_0"], dtype=np.float64)
    active_features = np.asarray(pt["active_features"], dtype=np.int64)
    n_features = int(pt["n_features"])
    semantic_graphs: Dict[int, np.ndarray] = {int(k): np.asarray(v, dtype=np.float64) for k, v in pt["semantic_graphs"].items()}

    # Stage 5 assignments
    sweep_path = clustering_dir / f"{prefix_id}_sweep_results.json"
    sweep_entry = load_sweep_entry(sweep_path, beta=beta, gamma=gamma)
    assignments = np.asarray(sweep_entry["assignments"], dtype=np.int64)

    # Stage 3 attributions + Stage 2 weights
    attr_path = attribution_dir / f"{prefix_id}_prefix_context.pt"
    branches_path = branches_dir / f"{prefix_id}_branches.json"
    attributions = load_attributions(attr_path)  # [N, d]
    probs = load_path_probs(branches_path)       # [N]

    # Align lengths defensively (some pipelines can drop/skip items)
    N = min(len(assignments), attributions.shape[0], probs.shape[0])
    assignments = assignments[:N]
    attributions = attributions[:N]
    probs = probs[:N]

    centered = attributions - H_0[None, :]

    per_cluster = {}
    any_disagree = False

    for c, delta_med in semantic_graphs.items():
        mask = assignments == int(c)
        if not np.any(mask):
            per_cluster[str(c)] = {"n_items": 0, "topB": {}}
            continue

        mean_vec = _weighted_mean(centered[mask], probs[mask])

        # Compare on top-|median| features (steering selection uses |H_c| ranking)
        delta_med_feat = delta_med[:n_features]
        abs_med = np.abs(delta_med_feat)

        cluster_out = {"n_items": int(mask.sum()), "topB": {}}
        for topB in topBs:
            topB = int(topB)
            if topB <= 0:
                continue
            idxs = np.argsort(abs_med)[-topB:][::-1]
            disagreements = []
            for idx in idxs:
                vm = float(delta_med_feat[idx])
                vw = float(mean_vec[idx])
                sm = _safe_sign(vm, eps)
                sw = _safe_sign(vw, eps)
                if sm != 0 and sw != 0 and sm != sw:
                    layer, pos, feat_id = active_features[idx].tolist()
                    disagreements.append(
                        {
                            "idx": int(idx),
                            "layer": int(layer),
                            "pos": int(pos),
                            "feat_id": int(feat_id),
                            "median_val": vm,
                            "wmean_val": vw,
                            "median_sign": sm,
                            "wmean_sign": sw,
                        }
                    )
                    if len(disagreements) >= max_examples_per_cluster:
                        break

            if disagreements:
                any_disagree = True
            cluster_out["topB"][str(topB)] = {
                "n_checked": int(len(idxs)),
                "n_disagree": int(len(disagreements)),
                "examples": disagreements,
            }

        per_cluster[str(c)] = cluster_out

    return {
        "file": str(pt_path),
        "prefix_id": prefix_id,
        "beta": beta,
        "gamma": gamma,
        "n_samples_aligned": int(N),
        "n_clusters": int(len(semantic_graphs)),
        "any_disagree": bool(any_disagree),
        "per_cluster": per_cluster,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--semantic-dir", type=Path, required=True)
    ap.add_argument("--clustering-dir", type=Path, required=True)
    ap.add_argument("--branches-dir", type=Path, required=True)
    ap.add_argument("--attribution-dir", type=Path, required=True)
    ap.add_argument("--topB", type=str, default="5,10")
    ap.add_argument("--eps", type=float, default=1e-9)
    ap.add_argument("--max-examples-per-cluster", type=int, default=25)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    topBs = [int(x) for x in args.topB.split(",") if x.strip()]

    pt_files = sorted(args.semantic_dir.glob("*_semantic_graphs.pt"))
    rows = []
    n_any = 0
    for pt_path in pt_files:
        if not PT_RE.match(pt_path.name):
            continue
        row = check_file(
            pt_path=pt_path,
            clustering_dir=args.clustering_dir,
            branches_dir=args.branches_dir,
            attribution_dir=args.attribution_dir,
            topBs=topBs,
            eps=float(args.eps),
            max_examples_per_cluster=int(args.max_examples_per_cluster),
        )
        rows.append(row)
        if row["any_disagree"]:
            n_any += 1

    summary = {
        "semantic_dir": str(args.semantic_dir),
        "n_files": int(len(rows)),
        "n_files_with_any_disagreement": int(n_any),
        "topBs": topBs,
        "eps": float(args.eps),
    }

    report = {"summary": summary, "rows": rows}
    out_path = args.out
    if out_path is None:
        out_path = args.semantic_dir / "median_vs_weighted_mean_sign_report.json"
    out_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()


