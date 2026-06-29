#!/usr/bin/env python3
"""Select qualitative reasoning cases from Stage 7 outputs.

The selector groups reasoning pair prefixes by (dataset, example, target step j)
and scores groups where multiple source steps i<j are available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PREFIX_RE = re.compile(
    r"^(?P<example>.+)_step_(?P<step>\d+)_src_(?P<src>\d+)_tgt_(?P<tgt>\d+)$"
)


@dataclass(frozen=True)
class PrefixInfo:
    dataset: str
    lane: str
    prefix_id: str
    example_id: str
    source_step: int
    target_step: int


def parse_prefix_id(prefix_id: str, dataset: str, lane: str) -> PrefixInfo:
    match = PREFIX_RE.match(prefix_id)
    if not match:
        raise ValueError(f"cannot parse reasoning prefix id: {prefix_id}")
    raw_example = match.group("example")
    example_id = raw_example
    dataset_prefix = f"{dataset}_"
    if example_id.startswith(dataset_prefix):
        example_id = example_id[len(dataset_prefix) :]
    return PrefixInfo(
        dataset=dataset,
        lane=lane,
        prefix_id=prefix_id,
        example_id=example_id,
        source_step=int(match.group("src")),
        target_step=int(match.group("tgt")),
    )


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_prefix_result(path: Path, dataset: str, lane: str) -> dict[str, Any] | None:
    prefix_id = path.name.removesuffix("_sweep_results.json")
    try:
        info = parse_prefix_id(prefix_id, dataset=dataset, lane=lane)
    except ValueError:
        return None

    data = load_json(path)
    clustering_runs = data.get("clustering_runs", {})
    run = clustering_runs.get("beta0.75_gamma0.7")
    if run is None and clustering_runs:
        run = next(iter(clustering_runs.values()))
    if not isinstance(run, dict):
        return None

    results = run.get("results", {})
    result = results.get("sign_full_B5")
    if result is None and results:
        result = next(iter(results.values()))
    if not isinstance(result, dict):
        return None

    cluster_rows = []
    for cluster_id, stats in (result.get("per_cluster_logit") or {}).items():
        if not isinstance(stats, dict):
            continue
        rho_s = finite_float(stats.get("centered_logit_spearman"))
        rho = finite_float(stats.get("centered_logit_corr"))
        n_samples = int(stats.get("n_samples") or 0)
        if rho_s is None or rho is None or n_samples <= 0:
            continue
        cluster_rows.append(
            {
                "cluster_id": str(cluster_id),
                "n_samples": n_samples,
                "centered_logit_spearman": rho_s,
                "centered_logit_corr": rho,
            }
        )

    if not cluster_rows:
        return None

    rho_s_values = [row["centered_logit_spearman"] for row in cluster_rows]
    rho_values = [row["centered_logit_corr"] for row in cluster_rows]
    return {
        "dataset": info.dataset,
        "lane": info.lane,
        "prefix_id": info.prefix_id,
        "example_id": info.example_id,
        "source_step": info.source_step,
        "target_step": info.target_step,
        "beta": float(run.get("beta", 0.75)),
        "gamma": float(run.get("gamma", 0.7)),
        "K": int(run.get("K") or run.get("n_clusters") or len(cluster_rows)),
        "n_clusters_with_metrics": len(cluster_rows),
        "mean_rho_s": mean(rho_s_values),
        "mean_abs_rho_s": mean(abs(x) for x in rho_s_values),
        "max_abs_rho_s": max(abs(x) for x in rho_s_values),
        "mean_rho": mean(rho_values),
        "mean_abs_rho": mean(abs(x) for x in rho_values),
        "max_abs_rho": max(abs(x) for x in rho_values),
        "cluster_rows": cluster_rows,
    }


def collect_prefix_rows(gsm8k_dir: Path, math500_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [
        ("gsm8k", "qwen3_0_6b_gsm8k", gsm8k_dir),
        ("math500", "qwen3_0_6b_math500", math500_dir),
    ]
    for dataset, lane, result_dir in specs:
        for path in sorted(result_dir.glob("*_sweep_results.json")):
            row = summarize_prefix_result(path, dataset=dataset, lane=lane)
            if row is not None:
                rows.append(row)
    return rows


def readability_bonus(mean_k: float, target_step: int) -> float:
    k_bonus = 1.0 if 2 <= mean_k <= 6 else 0.65 if mean_k <= 10 else 0.25
    target_bonus = 1.0 if 2 <= target_step <= 10 else 0.6
    return 0.7 * k_bonus + 0.3 * target_bonus


def score_groups(prefix_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in prefix_rows:
        key = (row["dataset"], row["example_id"], int(row["target_step"]))
        grouped.setdefault(key, []).append(row)

    candidates: list[dict[str, Any]] = []
    for (dataset, example_id, target_step), rows in grouped.items():
        rows = sorted(rows, key=lambda r: int(r["source_step"]))
        source_steps = sorted({int(r["source_step"]) for r in rows})
        if len(source_steps) < 2:
            continue

        mean_source_rhos = [float(r["mean_rho_s"]) for r in rows]
        mean_k = mean(float(r["K"]) for r in rows)
        max_abs = max(float(r["max_abs_rho_s"]) for r in rows)
        mean_abs = mean(float(r["mean_abs_rho_s"]) for r in rows)
        std_sources = pstdev(mean_source_rhos) if len(mean_source_rhos) > 1 else 0.0
        source_norm = min(len(source_steps) / 5.0, 1.0)
        read_bonus = readability_bonus(mean_k=mean_k, target_step=int(target_step))
        sign_diverse = any(x < 0 for x in mean_source_rhos) and any(x > 0 for x in mean_source_rhos)
        sign_bonus = 0.12 if sign_diverse else 0.0
        score = (
            0.35 * max_abs
            + 0.25 * std_sources
            + 0.20 * source_norm
            + 0.20 * read_bonus
            + sign_bonus
        )

        candidates.append(
            {
                "dataset": dataset,
                "lane": rows[0]["lane"],
                "example_id": example_id,
                "target_step": int(target_step),
                "source_steps_available": " ".join(str(x) for x in source_steps),
                "n_stage7_prefixes": len(rows),
                "max_abs_rho_s": max_abs,
                "mean_abs_rho_s": mean_abs,
                "std_rho_s_across_sources": std_sources,
                "mean_n_clusters": mean_k,
                "readability_bonus": read_bonus,
                "sign_diverse": sign_diverse,
                "score": score,
                "prefix_ids": " ".join(r["prefix_id"] for r in rows),
                "prefixes": rows,
            }
        )

    return sorted(candidates, key=lambda r: float(r["score"]), reverse=True)


def write_candidate_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "dataset",
        "example_id",
        "target_step",
        "source_steps_available",
        "n_stage7_prefixes",
        "max_abs_rho_s",
        "mean_abs_rho_s",
        "std_rho_s_across_sources",
        "mean_n_clusters",
        "readability_bonus",
        "sign_diverse",
        "score",
        "prefix_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(candidates, start=1):
            out = {key: row.get(key, "") for key in fieldnames}
            out["rank"] = rank
            for key in [
                "max_abs_rho_s",
                "mean_abs_rho_s",
                "std_rho_s_across_sources",
                "mean_n_clusters",
                "readability_bonus",
                "score",
            ]:
                out[key] = f"{float(out[key]):.6f}"
            writer.writerow(out)


def compact_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row["dataset"],
        "lane": row["lane"],
        "example_id": row["example_id"],
        "target_step": row["target_step"],
        "source_steps_available": row["source_steps_available"],
        "n_stage7_prefixes": row["n_stage7_prefixes"],
        "score": row["score"],
        "max_abs_rho_s": row["max_abs_rho_s"],
        "std_rho_s_across_sources": row["std_rho_s_across_sources"],
        "mean_n_clusters": row["mean_n_clusters"],
        "prefixes": [
            {
                "prefix_id": prefix["prefix_id"],
                "source_step": prefix["source_step"],
                "target_step": prefix["target_step"],
                "K": prefix["K"],
                "mean_rho_s": prefix["mean_rho_s"],
                "mean_abs_rho_s": prefix["mean_abs_rho_s"],
                "max_abs_rho_s": prefix["max_abs_rho_s"],
                "mean_rho": prefix["mean_rho"],
                "cluster_rows": prefix["cluster_rows"],
            }
            for prefix in row["prefixes"]
        ],
    }


def write_pull_list(path: Path, selected: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    for case in selected:
        lane = case["lane"]
        for prefix in case["prefixes"]:
            prefix_id = prefix["prefix_id"]
            lines.append(
                f"experiments/reasoning_runs/{lane}/2_reasoning_pair_samples/{prefix_id}_branches.json"
            )
            lines.append(
                f"experiments/reasoning_runs/{lane}/5_gaussian_clustering/{prefix_id}_sweep_results.json"
            )
    path.write_text("\n".join(sorted(set(lines))) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsm8k-dir", type=Path, required=True)
    parser.add_argument("--math500-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    prefix_rows = collect_prefix_rows(args.gsm8k_dir, args.math500_dir)
    candidates = score_groups(prefix_rows)
    selected = [compact_case(row) for row in candidates[: args.top_n]]

    args.output_root.mkdir(parents=True, exist_ok=True)
    write_candidate_csv(args.output_root / "candidate_scores.csv", candidates)
    (args.output_root / "selected_cases.json").write_text(
        json.dumps({"selected_cases": selected}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_pull_list(args.output_root / "selected_case_inputs_files.txt", selected)

    print(f"Loaded prefix rows: {len(prefix_rows)}")
    print(f"Candidate groups: {len(candidates)}")
    print(f"Selected cases: {len(selected)}")
    print(f"Wrote: {args.output_root / 'candidate_scores.csv'}")
    print(f"Wrote: {args.output_root / 'selected_cases.json'}")
    print(f"Wrote: {args.output_root / 'selected_case_inputs_files.txt'}")


if __name__ == "__main__":
    main()
