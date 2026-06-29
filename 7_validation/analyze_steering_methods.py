#!/usr/bin/env python3
"""Compare Stage 7 steering methods using paper-facing logit correlations.

The paper-facing Stage 7 metrics are Pearson and Spearman correlations between
steering strength and demeaned/centered logit change. This analyzer keeps the
export surface restricted to those two metrics.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


PAPER_METRICS = [
    "centered_logit_spearman",
    "centered_logit_corr",
]


def load_results_from_dir(result_dir: Path) -> Dict[str, Any]:
    """Load all per-prefix sweep result files from a method directory."""
    results = {}
    result_files = sorted(result_dir.glob("*_sweep_results.json"))
    if not result_files:
        result_files = sorted(result_dir.glob("H4a*/**/*_sweep_results.json"))

    for fpath in result_files:
        prefix_id = fpath.stem.replace("_sweep_results", "")
        try:
            with open(fpath) as f:
                results[prefix_id] = json.load(f)
        except Exception as exc:
            print(f"Warning: Failed to load {fpath}: {exc}")
    return results


def _parse_run_key(run_key: str) -> Dict[str, Any]:
    config: Dict[str, Any] = {"run_key": run_key}
    for part in run_key.split("_"):
        try:
            if part.startswith("beta"):
                config["beta"] = float(part.replace("beta", ""))
            elif part.startswith("gamma"):
                config["gamma"] = float(part.replace("gamma", ""))
        except ValueError:
            pass
    return config


def _parse_sweep_key(sweep_key: str) -> Dict[str, Any]:
    parts = sweep_key.split("_")
    return {
        "sweep_key": sweep_key,
        "h_c_selection": parts[1] if len(parts) > 1 else "full",
        "top_B": int(parts[2].replace("B", "")) if len(parts) > 2 else 5,
    }


def extract_metrics_from_result(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract per-cluster paper metrics from a single prefix result."""
    rows = []
    for run_key, run_data in data.get("clustering_runs", {}).items():
        config = _parse_run_key(run_key)
        for sweep_key, result in run_data.get("results", {}).items():
            sweep = _parse_sweep_key(sweep_key)
            for cluster_id, cluster_stats in result.get("per_cluster_logit", {}).items():
                n_samples = cluster_stats.get("n_samples", 0)
                if n_samples == 0:
                    continue
                rows.append({
                    **config,
                    **sweep,
                    "cluster_id": cluster_id,
                    "n_samples": n_samples,
                    "centered_logit_spearman": cluster_stats.get("centered_logit_spearman", np.nan),
                    "centered_logit_corr": cluster_stats.get("centered_logit_corr", np.nan),
                })
    return rows


def _finite_values(rows: List[Dict[str, Any]], metric: str) -> List[float]:
    values = [row.get(metric) for row in rows if row.get(metric) is not None]
    return [float(value) for value in values if not np.isnan(value)]


def summarize_metric(rows: List[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    values = _finite_values(rows, metric)
    if not values:
        return {"mean": None, "std": None, "median": None, "n": 0}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "n": len(values),
    }


def _format_summary_float(value: Any) -> str:
    if value is None:
        return f"{'nan':>10}"
    return f"{float(value):>10.4f}"


def compare_methods(method_metrics: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    return {
        metric: {
            method: summarize_metric(rows, metric)
            for method, rows in method_metrics.items()
        }
        for metric in PAPER_METRICS
    }


def print_comparison_table(comparisons: Dict[str, Dict[str, Any]], method_names: List[str]) -> None:
    for metric, summaries in comparisons.items():
        print(f"\n{'=' * 70}")
        print(f"Metric: {metric}")
        print("=" * 70)
        print(f"\n{'Method':<40} {'Mean':>10} {'Std':>10} {'Median':>10} {'N':>6}")
        print("-" * 76)
        for method in method_names:
            summary = summaries.get(method, {})
            print(
                f"{method:<40} {_format_summary_float(summary.get('mean'))} "
                f"{_format_summary_float(summary.get('std'))} "
                f"{_format_summary_float(summary.get('median'))} "
                f"{summary.get('n', 0):>6}"
            )


def export_to_csv(
    method_metrics: Dict[str, List[Dict[str, Any]]],
    method_names: List[str],
    output_path: str,
) -> None:
    """Export detailed and factorial summary CSVs with paper metrics only."""
    rows = []
    for method in method_names:
        for metric_row in method_metrics.get(method, []):
            rows.append({
                "method": method,
                "prefix_id": metric_row.get("prefix_id", ""),
                "run_key": metric_row.get("run_key", ""),
                "beta": metric_row.get("beta", ""),
                "gamma": metric_row.get("gamma", ""),
                "sweep_key": metric_row.get("sweep_key", ""),
                "h_c_selection": metric_row.get("h_c_selection", ""),
                "top_B": metric_row.get("top_B", ""),
                "cluster_id": metric_row.get("cluster_id", ""),
                "n_samples": metric_row.get("n_samples", ""),
                "centered_logit_spearman": metric_row.get("centered_logit_spearman", ""),
                "centered_logit_corr": metric_row.get("centered_logit_corr", ""),
            })

    if rows:
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary_rows = []
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for method in method_names:
        for row in method_metrics.get(method, []):
            grouped[(
                method,
                row.get("beta", ""),
                row.get("gamma", ""),
                row.get("h_c_selection", ""),
                row.get("top_B", ""),
            )].append(row)

    for (method, beta, gamma, h_c_selection, top_B), group_rows in sorted(grouped.items()):
        summary_rows.append({
            "method": method,
            "beta": beta,
            "gamma": gamma,
            "h_c_selection": h_c_selection,
            "top_B": top_B,
            "n_clusters": len(group_rows),
            "n_prefixes": len({row.get("prefix_id", "") for row in group_rows}),
            "centered_logit_spearman_mean": summarize_metric(group_rows, "centered_logit_spearman")["mean"],
            "centered_logit_corr_mean": summarize_metric(group_rows, "centered_logit_corr")["mean"],
        })

    summary_path = output_path.replace(".csv", "_summary.csv")
    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Summary CSV exported to: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Stage 7 steering methods with paper-facing logit correlations."
    )
    parser.add_argument("result_dirs", nargs="+", help="Method result directories to compare")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON file")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV path")
    parser.add_argument(
        "--intersection-only",
        action="store_true",
        help="Only compare prefixes present in all methods",
    )
    args = parser.parse_args()

    method_results = {}
    method_metrics = {}
    for dir_path in args.result_dirs:
        path = Path(dir_path)
        if not path.exists():
            print(f"Warning: Directory not found: {path}")
            continue

        method_name = path.name
        if method_name.startswith("H4a"):
            method_name = method_name.replace("H4a_", "").replace("H4a", "default")

        print(f"Loading results from: {path} (method: {method_name})")
        results = load_results_from_dir(path)
        print(f"  Loaded {len(results)} prefix files")

        rows = []
        for prefix_id, data in results.items():
            prefix_rows = extract_metrics_from_result(data)
            for row in prefix_rows:
                row["prefix_id"] = prefix_id
            rows.extend(prefix_rows)
        print(f"  Extracted {len(rows)} cluster-level metric rows")

        method_results[method_name] = results
        method_metrics[method_name] = rows

    if not method_metrics:
        print("No results loaded.")
        return

    method_names = list(method_metrics.keys())
    if args.intersection_only:
        prefix_sets = [
            {row.get("prefix_id", "") for row in method_metrics[method]}
            for method in method_names
        ]
        common_prefixes = set.intersection(*prefix_sets) if prefix_sets else set()
        print(f"\n--intersection-only: Filtering to {len(common_prefixes)} common prefixes")
        for method in method_names:
            before = len(method_metrics[method])
            method_metrics[method] = [
                row for row in method_metrics[method]
                if row.get("prefix_id", "") in common_prefixes
            ]
            print(f"  {method}: {before} -> {len(method_metrics[method])} rows")

    print(f"\nComparing methods: {method_names}")
    comparisons = compare_methods(method_metrics)
    print_comparison_table(comparisons, method_names)

    if args.output:
        output_data = {
            "methods": method_names,
            "metrics": [
                "centered_logit_spearman_mean",
                "centered_logit_corr_mean",
            ],
            "n_prefixes": {method: len(method_results.get(method, {})) for method in method_names},
            "n_clusters": {method: len(method_metrics.get(method, [])) for method in method_names},
            "comparisons": comparisons,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    if args.csv:
        export_to_csv(method_metrics, method_names, args.csv)
        print(f"\nCSV exported to: {args.csv}")


if __name__ == "__main__":
    main()
