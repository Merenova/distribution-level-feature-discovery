#!/usr/bin/env python3
"""Aggregate compact Stage-7 steering results into one paper-facing CSV."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


METHOD_ORDER = {"rd": 0, "km_sem": 1, "single": 2}


def load_result_files(result_dir: Path) -> Iterable[Dict[str, Any]]:
    for path in sorted(result_dir.glob("*_sweep_results.json")):
        with path.open() as handle:
            yield json.load(handle)


def infer_method(result_dir: Path, data: Dict[str, Any]) -> str:
    method = data.get("method")
    if method:
        return str(method)
    return result_dir.name


def _curve_value(curve: Dict[str, Any], epsilon: Any) -> float:
    for key in (epsilon, str(epsilon), str(float(epsilon))):
        if key in curve:
            return float(curve[key])
    return 0.0


def per_prefix_rows(result_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for data in load_result_files(result_dir):
        method = infer_method(result_dir, data)
        prefix_id = data.get("prefix_id")

        for clustering_key, run in data.get("clustering_runs", {}).items():
            beta = float(run["beta"])
            gamma = float(run["gamma"])
            for sweep_key, result in run.get("results", {}).items():
                per_cluster = result.get("per_cluster", {})
                corr_values = []
                spearman_values = []
                sum_curves = defaultdict(float)
                epsilon_values = result.get("epsilon_values", [])

                for stats in per_cluster.values():
                    if "centered_logit_corr" in stats:
                        corr_values.append(float(stats["centered_logit_corr"]))
                    if "centered_logit_spearman" in stats:
                        spearman_values.append(float(stats["centered_logit_spearman"]))
                    for epsilon in epsilon_values:
                        sum_curves[epsilon] += _curve_value(stats.get("sum_centered_logit_diff", {}), epsilon)

                row = {
                    "prefix_id": prefix_id,
                    "method": method,
                    "run_key": clustering_key,
                    "sweep_key": sweep_key,
                    "beta": beta,
                    "gamma": gamma,
                    "h_c_selection": result.get("hc_selection", "full"),
                    "top_B": int(result.get("top_B", 5)),
                    "centered_logit_corr": float(np.mean(corr_values)) if corr_values else 0.0,
                    "centered_logit_spearman": float(np.mean(spearman_values)) if spearman_values else 0.0,
                }
                for epsilon in epsilon_values:
                    key = f"sum_diff_eps{epsilon}"
                    row[key] = float(result.get(key, sum_curves[epsilon]))
                rows.append(row)

    return rows


def aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    epsilon_keys = sorted({key for row in rows for key in row if key.startswith("sum_diff_eps")})

    for row in rows:
        key = (
            row["method"],
            row["beta"],
            row["gamma"],
            row["h_c_selection"],
            row["top_B"],
        )
        grouped[key].append(row)

    summary_rows: List[Dict[str, Any]] = []
    for key, bucket in grouped.items():
        method, beta, gamma, h_c_selection, top_B = key
        summary = {
            "method": method,
            "beta": beta,
            "gamma": gamma,
            "h_c_selection": h_c_selection,
            "top_B": top_B,
            "n_prefixes": len(bucket),
            "centered_logit_corr_mean": float(np.mean([row["centered_logit_corr"] for row in bucket])),
            "centered_logit_spearman_mean": float(np.mean([row["centered_logit_spearman"] for row in bucket])),
        }
        for epsilon_key in epsilon_keys:
            values = [row.get(epsilon_key, 0.0) for row in bucket]
            summary[f"{epsilon_key}_prefix_mean"] = float(np.mean(values))
        summary_rows.append(summary)

    summary_rows.sort(
        key=lambda row: (
            METHOD_ORDER.get(row["method"], 99),
            row["h_c_selection"],
            row["top_B"],
            row["beta"],
            row["gamma"],
        )
    )
    return summary_rows


def write_csv(rows: List[Dict[str, Any]], output_csv: Path) -> None:
    if not rows:
        raise ValueError("No rows to write")

    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate compact Stage-7 steering results")
    parser.add_argument("result_dirs", nargs="+", type=Path, help="Method result directories such as results/7_validation/rd")
    parser.add_argument("--output-csv", type=Path, required=True, help="Path to the aggregated CSV")
    args = parser.parse_args()

    per_prefix: List[Dict[str, Any]] = []
    for result_dir in args.result_dirs:
        per_prefix.extend(per_prefix_rows(result_dir))

    summary_rows = aggregate_rows(per_prefix)
    write_csv(summary_rows, args.output_csv)
    print(f"Wrote {len(summary_rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
