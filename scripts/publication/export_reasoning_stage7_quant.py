#!/usr/bin/env python3
"""Export cluster-level Stage 7 reasoning steering metrics.

This script is intentionally dependency-free so it can run in the lightweight
analysis environment used for the reasoning sweeps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_GSM8K_DIR = (
    "experiments/reasoning_runs/qwen3_0_6b_gsm8k/"
    "7_validation/7c_combined_medoid/H4a_combined_medoid"
)
DEFAULT_MATH500_DIR = (
    "experiments/reasoning_runs/qwen3_0_6b_math500/"
    "7_validation/7c_combined_medoid/H4a_combined_medoid"
)
DEFAULT_OUTPUT_ROOT = "experiments/reasoning_runs/analysis/qwen3_0_6b/quantitative"
DEFAULT_RUN_KEY = "beta0.75_gamma0.7"
DEFAULT_SWEEP_KEY = "sign_full_B5"

METHOD = "combined_medoid"
H_C_SELECTION = "full"
TOP_B = 5

CLUSTER_CSV = "reasoning_stage7_cluster_metrics.csv"
PAIR_CSV = "reasoning_stage7_pair_metrics.csv"
SUMMARY_CSV = "reasoning_stage7_summary.csv"
STEP_PAIR_SUMMARY_CSV = "reasoning_stage7_step_pair_summary.csv"
SUMMARY_JSON = "reasoning_steering_summary.json"
LATEX_TABLE = "reasoning_steering_table.tex"
STEP_PAIR_LATEX_TABLE = "reasoning_step_pair_table.tex"

PREFIX_RE = re.compile(
    r"^(?P<body>.+)_step_(?P<step>\d+)_src_(?P<src>\d+)_tgt_(?P<tgt>\d+)$"
)
RUN_KEY_RE = re.compile(r"beta(?P<beta>-?\d+(?:\.\d+)?)_gamma(?P<gamma>-?\d+(?:\.\d+)?)")


class ExportError(RuntimeError):
    """Raised when input files cannot be exported consistently."""


@dataclass(frozen=True)
class ParsedPrefix:
    dataset: str
    prefix_id: str
    example_id: str
    source_step: int
    target_step: int


@dataclass(frozen=True)
class InputSpec:
    dataset: str
    result_dir: Path


@dataclass(frozen=True)
class FileExtraction:
    rows: List[Dict[str, Any]]
    prefix_id: str
    skipped_reason: Optional[str] = None


def _strip_sweep_suffix(name: str) -> str:
    filename = Path(name).name
    if filename.endswith(".json"):
        filename = filename[: -len(".json")]
    if filename.endswith("_sweep_results"):
        filename = filename[: -len("_sweep_results")]
    return filename


def parse_prefix_id(raw_name: str, dataset: str) -> ParsedPrefix:
    """Parse dataset/example/source/target from a Stage 7 result filename.

    Handles ids such as:
    - gsm8k_gsm8k_test_00096_step_05_src_02_tgt_05
    - math500_test_prealgebra_1840.json_step_05_src_03_tgt_05
    """

    prefix_id = _strip_sweep_suffix(raw_name)
    match = PREFIX_RE.match(prefix_id)
    if not match:
        raise ExportError(f"Could not parse prefix id from {raw_name!r}")

    body = match.group("body")
    dataset_upper = dataset.upper()
    example_id = body
    if dataset_upper == "GSM8K" and body.startswith("gsm8k_"):
        example_id = body[len("gsm8k_") :]
    elif dataset_upper == "MATH500":
        example_id = body

    return ParsedPrefix(
        dataset=dataset_upper,
        prefix_id=prefix_id,
        example_id=example_id,
        source_step=int(match.group("src")),
        target_step=int(match.group("tgt")),
    )


def parse_run_key(run_key: str) -> tuple[float, float]:
    match = RUN_KEY_RE.fullmatch(run_key)
    if not match:
        raise ExportError(f"Could not parse beta/gamma from run key {run_key!r}")
    return float(match.group("beta")), float(match.group("gamma"))


def parse_result_prefix(data: Dict[str, Any], path: Path, dataset: str) -> ParsedPrefix:
    errors: List[str] = []
    for raw_name in (data.get("prefix_id"), path.stem):
        if not raw_name:
            continue
        try:
            return parse_prefix_id(str(raw_name), dataset)
        except ExportError as exc:
            errors.append(str(exc))
    detail = "; ".join(errors) if errors else "no prefix_id candidate found"
    raise ExportError(f"Could not parse prefix id for {path}: {detail}")


def finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.12g}"
    return str(value)


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ExportError(f"JSON parse failure in {path}: {exc}") from exc
    except OSError as exc:
        raise ExportError(f"Could not read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ExportError(f"Expected JSON object in {path}")
    return data


def iter_result_files(result_dir: Path) -> List[Path]:
    if not result_dir.exists():
        raise ExportError(f"Result directory does not exist: {result_dir}")
    if not result_dir.is_dir():
        raise ExportError(f"Result path is not a directory: {result_dir}")
    return sorted(result_dir.rglob("*_sweep_results.json"))


def extract_cluster_rows(
    path: Path,
    dataset: str,
    run_key: str,
    sweep_key: str,
) -> FileExtraction:
    data = load_json(path)
    parsed = parse_result_prefix(data, path, dataset)

    clustering_runs = data.get("clustering_runs")
    if not isinstance(clustering_runs, dict) or run_key not in clustering_runs:
        raise ExportError(f"Missing clustering_runs[{run_key!r}] in {path}")

    run = clustering_runs[run_key]
    if not isinstance(run, dict):
        raise ExportError(f"Expected dict for clustering run {run_key!r} in {path}")

    results = run.get("results")
    if not isinstance(results, dict) or sweep_key not in results:
        raise ExportError(f"Missing results[{sweep_key!r}] in {path}")

    result = results[sweep_key]
    if not isinstance(result, dict):
        raise ExportError(f"Expected dict for sweep result {sweep_key!r} in {path}")

    if result.get("error"):
        return FileExtraction(
            rows=[],
            prefix_id=parsed.prefix_id,
            skipped_reason=str(result.get("error")),
        )

    per_cluster = result.get("per_cluster_logit")
    if not isinstance(per_cluster, dict):
        raise ExportError(f"Missing per_cluster_logit dict in {path}")

    beta = finite_float(run.get("beta"))
    gamma = finite_float(run.get("gamma"))
    if beta is None or gamma is None:
        beta, gamma = parse_run_key(run_key)

    rows: List[Dict[str, Any]] = []
    for cluster_id in sorted(per_cluster, key=_cluster_sort_key):
        stats = per_cluster[cluster_id]
        if not isinstance(stats, dict):
            raise ExportError(f"Expected dict for cluster {cluster_id!r} in {path}")
        rows.append(
            {
                "dataset": parsed.dataset,
                "prefix_id": parsed.prefix_id,
                "example_id": parsed.example_id,
                "source_step": parsed.source_step,
                "target_step": parsed.target_step,
                "cluster_id": cluster_id,
                "beta": beta,
                "gamma": gamma,
                "method": METHOD,
                "h_c_selection": H_C_SELECTION,
                "top_B": TOP_B,
                "n_samples": stats.get("n_samples"),
                "centered_logit_spearman": finite_float(
                    stats.get("centered_logit_spearman")
                ),
                "centered_logit_corr": finite_float(stats.get("centered_logit_corr")),
            }
        )
    if not rows:
        return FileExtraction(
            rows=[],
            prefix_id=parsed.prefix_id,
            skipped_reason="empty_per_cluster_logit",
        )
    return FileExtraction(rows=rows, prefix_id=parsed.prefix_id)


def _cluster_sort_key(cluster_id: Any) -> tuple[int, Any]:
    text = str(cluster_id)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def _metric_values(rows: Sequence[Dict[str, Any]], metric: str) -> List[float]:
    values: List[float] = []
    for row in rows:
        value = finite_float(row.get(metric))
        if value is not None:
            values.append(value)
    return values


def _summarize_values(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None, "median": None}
    if len(values) == 1:
        std = 0.0
    else:
        std = statistics.pstdev(values)
    return {
        "mean": statistics.fmean(values),
        "std": std,
        "median": statistics.median(values),
    }


def _mean(values: Sequence[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def summarize_rows(
    rows: Sequence[Dict[str, Any]],
    dataset: str,
    n_prefixes: int,
    n_skipped_files: int,
) -> Dict[str, Any]:
    subset = [row for row in rows if dataset == "All" or row["dataset"] == dataset]
    spearman = _summarize_values(_metric_values(subset, "centered_logit_spearman"))
    pearson = _summarize_values(_metric_values(subset, "centered_logit_corr"))
    beta_values = sorted({row["beta"] for row in subset})
    gamma_values = sorted({row["gamma"] for row in subset})

    return {
        "dataset": dataset,
        "method": METHOD,
        "beta": beta_values[0] if len(beta_values) == 1 else None,
        "gamma": gamma_values[0] if len(gamma_values) == 1 else None,
        "h_c_selection": H_C_SELECTION,
        "top_B": TOP_B,
        "n_prefixes": n_prefixes,
        "n_metric_prefixes": len({row["prefix_id"] for row in subset}),
        "n_skipped_files": n_skipped_files,
        "n_clusters": len(subset),
        "centered_logit_spearman_mean": spearman["mean"],
        "centered_logit_spearman_std": spearman["std"],
        "centered_logit_spearman_median": spearman["median"],
        "centered_logit_corr_mean": pearson["mean"],
        "centered_logit_corr_std": pearson["std"],
        "centered_logit_corr_median": pearson["median"],
    }


def build_pair_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["prefix_id"]), []).append(row)

    pair_rows: List[Dict[str, Any]] = []
    for prefix_id, group in sorted(grouped.items()):
        first = group[0]
        spearman_values = _metric_values(group, "centered_logit_spearman")
        pearson_values = _metric_values(group, "centered_logit_corr")
        n_samples_values = [
            int(value)
            for value in (row.get("n_samples") for row in group)
            if str(value).isdigit()
        ]
        pair_rows.append(
            {
                "dataset": first["dataset"],
                "prefix_id": prefix_id,
                "example_id": first["example_id"],
                "source_step": first["source_step"],
                "target_step": first["target_step"],
                "beta": first["beta"],
                "gamma": first["gamma"],
                "method": first["method"],
                "h_c_selection": first["h_c_selection"],
                "top_B": first["top_B"],
                "n_clusters": len(group),
                "n_samples_total": sum(n_samples_values),
                "centered_logit_spearman_mean": _mean(spearman_values),
                "centered_logit_spearman_abs_mean": _mean(
                    [abs(value) for value in spearman_values]
                ),
                "centered_logit_spearman_max_abs": (
                    max(abs(value) for value in spearman_values)
                    if spearman_values
                    else None
                ),
                "centered_logit_corr_mean": _mean(pearson_values),
                "centered_logit_corr_abs_mean": _mean(
                    [abs(value) for value in pearson_values]
                ),
                "centered_logit_corr_max_abs": (
                    max(abs(value) for value in pearson_values)
                    if pearson_values
                    else None
                ),
            }
        )
    return pair_rows


def summarize_step_pairs(pair_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[str, int, int], List[Dict[str, Any]]] = {}
    for row in pair_rows:
        source_step = int(row["source_step"])
        target_step = int(row["target_step"])
        groups.setdefault((str(row["dataset"]), source_step, target_step), []).append(row)
        groups.setdefault(("All", source_step, target_step), []).append(row)

    summaries: List[Dict[str, Any]] = []
    for (dataset, source_step, target_step), group in sorted(
        groups.items(), key=lambda item: (item[0][0] == "All", item[0][0], item[0][2], item[0][1])
    ):
        rho_s = _summarize_values(
            _metric_values(group, "centered_logit_spearman_mean")
        )
        rho = _summarize_values(_metric_values(group, "centered_logit_corr_mean"))
        abs_rho_s = _summarize_values(
            _metric_values(group, "centered_logit_spearman_abs_mean")
        )
        summaries.append(
            {
                "dataset": dataset,
                "source_step": source_step,
                "target_step": target_step,
                "n_pairs": len(group),
                "n_examples": len({row["example_id"] for row in group}),
                "n_clusters": sum(int(row.get("n_clusters") or 0) for row in group),
                "centered_logit_spearman_mean": rho_s["mean"],
                "centered_logit_spearman_std": rho_s["std"],
                "centered_logit_spearman_median": rho_s["median"],
                "centered_logit_spearman_abs_mean": abs_rho_s["mean"],
                "centered_logit_corr_mean": rho["mean"],
                "centered_logit_corr_std": rho["std"],
                "centered_logit_corr_median": rho["median"],
            }
        )
    return summaries


def write_cluster_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "prefix_id",
        "example_id",
        "source_step",
        "target_step",
        "cluster_id",
        "beta",
        "gamma",
        "method",
        "h_c_selection",
        "top_B",
        "n_samples",
        "centered_logit_spearman",
        "centered_logit_corr",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def write_pair_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "prefix_id",
        "example_id",
        "source_step",
        "target_step",
        "beta",
        "gamma",
        "method",
        "h_c_selection",
        "top_B",
        "n_clusters",
        "n_samples_total",
        "centered_logit_spearman_mean",
        "centered_logit_spearman_abs_mean",
        "centered_logit_spearman_max_abs",
        "centered_logit_corr_mean",
        "centered_logit_corr_abs_mean",
        "centered_logit_corr_max_abs",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def write_summary_csv(path: Path, summaries: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "method",
        "beta",
        "gamma",
        "h_c_selection",
        "top_B",
        "n_prefixes",
        "n_metric_prefixes",
        "n_skipped_files",
        "n_clusters",
        "centered_logit_spearman_mean",
        "centered_logit_spearman_std",
        "centered_logit_spearman_median",
        "centered_logit_corr_mean",
        "centered_logit_corr_std",
        "centered_logit_corr_median",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def write_step_pair_summary_csv(path: Path, summaries: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "source_step",
        "target_step",
        "n_pairs",
        "n_examples",
        "n_clusters",
        "centered_logit_spearman_mean",
        "centered_logit_spearman_std",
        "centered_logit_spearman_median",
        "centered_logit_spearman_abs_mean",
        "centered_logit_corr_mean",
        "centered_logit_corr_std",
        "centered_logit_corr_median",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


def write_summary_json(
    path: Path,
    summaries: Sequence[Dict[str, Any]],
    pair_summaries: Sequence[Dict[str, Any]],
    inputs: Sequence[InputSpec],
    rows: Sequence[Dict[str, Any]],
    pair_rows: Sequence[Dict[str, Any]],
    file_counts: Dict[str, int],
    skipped_counts: Dict[str, Dict[str, int]],
    run_key: str,
    sweep_key: str,
) -> None:
    payload = {
        "run_key": run_key,
        "sweep_key": sweep_key,
        "method": METHOD,
        "h_c_selection": H_C_SELECTION,
        "top_B": TOP_B,
        "metrics": {
            "rho_s": "centered_logit_spearman",
            "rho": "centered_logit_corr",
        },
        "inputs": [
            {"dataset": spec.dataset, "result_dir": str(spec.result_dir)}
            for spec in inputs
        ],
        "file_counts": file_counts,
        "skipped_result_counts": skipped_counts,
        "n_cluster_rows": len(rows),
        "n_pair_rows": len(pair_rows),
        "summaries": list(summaries),
        "step_pair_summaries": list(pair_summaries),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def latex_number(value: Any, precision: int) -> str:
    if value is None:
        return "--"
    value_float = finite_float(value)
    if value_float is None:
        return "--"
    return f"{value_float:.{precision}f}"


def write_latex_table(
    path: Path,
    summaries: Sequence[Dict[str, Any]],
    precision: int,
) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Dataset & $n_{\text{pair}}$ & $n_{\text{valid}}$ & $n_{\text{cluster}}$ & $\rho_s$ & $\rho$ \\",
        r"\midrule",
    ]
    for summary in summaries:
        lines.append(
            " & ".join(
                [
                    str(summary["dataset"]),
                    str(summary["n_prefixes"]),
                    str(summary["n_metric_prefixes"]),
                    str(summary["n_clusters"]),
                    latex_number(
                        summary.get("centered_logit_spearman_mean"), precision
                    ),
                    latex_number(summary.get("centered_logit_corr_mean"), precision),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Reasoning Stage 7 centered-logit steering correlations for Qwen3-0.6B, combined medoid, $\beta=0.75$, $\gamma=0.7$, full $h_c$, and top-$B=5$. Correlations are averaged over valid target-step clusters.}",
            r"\label{tab:reasoning-steering-qwen3-0-6b}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_step_pair_latex_table(
    path: Path,
    pair_summaries: Sequence[Dict[str, Any]],
    precision: int,
) -> None:
    all_rows = [row for row in pair_summaries if row["dataset"] == "All"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$i$ & $j$ & $n_{\text{pair}}$ & $n_{\text{cluster}}$ & $\rho_s$ & $\rho$ \\",
        r"\midrule",
    ]
    for row in sorted(all_rows, key=lambda item: (int(item["target_step"]), int(item["source_step"]))):
        lines.append(
            " & ".join(
                [
                    str(row["source_step"]),
                    str(row["target_step"]),
                    str(row["n_pairs"]),
                    str(row["n_clusters"]),
                    latex_number(row.get("centered_logit_spearman_mean"), precision),
                    latex_number(row.get("centered_logit_corr_mean"), precision),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Reasoning Stage 7 centered-logit steering correlations grouped by explicit reasoning step pairs $(i,j)$ with $i<j$. Values average the per-pair cluster means across GSM8K and MATH500.}",
            r"\label{tab:reasoning-step-pair-steering-qwen3-0-6b}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def collect_rows(
    inputs: Sequence[InputSpec],
    run_key: str,
    sweep_key: str,
) -> tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, Dict[str, int]]]:
    rows: List[Dict[str, Any]] = []
    file_counts: Dict[str, int] = {}
    skipped_counts: Dict[str, Dict[str, int]] = {}
    for spec in inputs:
        files = iter_result_files(spec.result_dir)
        file_counts[spec.dataset] = len(files)
        skipped_counts[spec.dataset] = {}
        if not files:
            raise ExportError(f"No *_sweep_results.json files found in {spec.result_dir}")
        for path in files:
            extracted = extract_cluster_rows(path, spec.dataset, run_key, sweep_key)
            if extracted.skipped_reason:
                counts = skipped_counts[spec.dataset]
                counts[extracted.skipped_reason] = (
                    counts.get(extracted.skipped_reason, 0) + 1
                )
            rows.extend(extracted.rows)
    return rows, file_counts, skipped_counts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export Qwen3-0.6B reasoning Stage 7 combined-medoid cluster metrics "
            "to CSV, JSON, and a compact LaTeX table."
        )
    )
    parser.add_argument("--gsm8k-dir", default=DEFAULT_GSM8K_DIR)
    parser.add_argument("--math500-dir", default=DEFAULT_MATH500_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-key", default=DEFAULT_RUN_KEY)
    parser.add_argument("--sweep-key", default=DEFAULT_SWEEP_KEY)
    parser.add_argument("--latex-precision", type=int, default=2)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    inputs = [
        InputSpec("GSM8K", Path(args.gsm8k_dir)),
        InputSpec("MATH500", Path(args.math500_dir)),
    ]

    try:
        rows, file_counts, skipped_counts = collect_rows(
            inputs, args.run_key, args.sweep_key
        )
        pair_rows = build_pair_rows(rows)
        pair_summaries = summarize_step_pairs(pair_rows)
        datasets = [spec.dataset for spec in inputs]
        summaries = [
            summarize_rows(
                rows,
                dataset,
                file_counts.get(dataset, 0),
                sum(skipped_counts.get(dataset, {}).values()),
            )
            for dataset in datasets
        ]
        summaries.append(
            summarize_rows(
                rows,
                "All",
                sum(file_counts.values()),
                sum(
                    count
                    for dataset_counts in skipped_counts.values()
                    for count in dataset_counts.values()
                ),
            )
        )

        output_root.mkdir(parents=True, exist_ok=True)
        write_cluster_csv(output_root / CLUSTER_CSV, rows)
        write_pair_csv(output_root / PAIR_CSV, pair_rows)
        write_summary_csv(output_root / SUMMARY_CSV, summaries)
        write_step_pair_summary_csv(
            output_root / STEP_PAIR_SUMMARY_CSV,
            pair_summaries,
        )
        write_summary_json(
            output_root / SUMMARY_JSON,
            summaries,
            pair_summaries,
            inputs,
            rows,
            pair_rows,
            file_counts,
            skipped_counts,
            args.run_key,
            args.sweep_key,
        )
        write_latex_table(output_root / LATEX_TABLE, summaries, args.latex_precision)
        write_step_pair_latex_table(
            output_root / STEP_PAIR_LATEX_TABLE,
            pair_summaries,
            args.latex_precision,
        )
    except ExportError as exc:
        parser.exit(1, f"error: {exc}\n")

    print("Export complete")
    for spec in inputs:
        summary = next(row for row in summaries if row["dataset"] == spec.dataset)
        print(
            f"{spec.dataset}: files={file_counts.get(spec.dataset, 0)} "
            f"n_prefixes={summary['n_prefixes']} n_clusters={summary['n_clusters']} "
            f"skipped_files={summary['n_skipped_files']} "
            f"rho_s={latex_number(summary['centered_logit_spearman_mean'], 4)} "
            f"rho={latex_number(summary['centered_logit_corr_mean'], 4)}"
        )
    all_summary = summaries[-1]
    print(
        f"All: n_prefixes={all_summary['n_prefixes']} "
        f"n_clusters={all_summary['n_clusters']} "
        f"skipped_files={all_summary['n_skipped_files']} "
        f"rho_s={latex_number(all_summary['centered_logit_spearman_mean'], 4)} "
        f"rho={latex_number(all_summary['centered_logit_corr_mean'], 4)}"
    )
    print(f"Wrote {output_root / CLUSTER_CSV}")
    print(f"Wrote {output_root / PAIR_CSV}")
    print(f"Wrote {output_root / SUMMARY_CSV}")
    print(f"Wrote {output_root / STEP_PAIR_SUMMARY_CSV}")
    print(f"Wrote {output_root / SUMMARY_JSON}")
    print(f"Wrote {output_root / LATEX_TABLE}")
    print(f"Wrote {output_root / STEP_PAIR_LATEX_TABLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
