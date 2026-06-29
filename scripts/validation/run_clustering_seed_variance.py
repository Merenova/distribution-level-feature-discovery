#!/usr/bin/env -S uv run python
"""Analyze end-to-end clustering variance across split random seeds."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "5_gaussian_clustering"))

from cluster import load_prefix_data, run_clustering
from em_loop import check_convergence
from utils.data_utils import convert_numpy_types, load_json, save_json
from utils.logging_utils import setup_logger


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "AmbigQA_Qwen3-8B" / "results"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "beta_gamma_scaled_config.json"
NUMERIC_METRICS = ["H", "D_e", "D_a", "L_RD", "K", "n_iterations"]
DELTA_METRICS = ["H", "D_e", "D_a", "L_RD", "K"]


def parse_seed_list(seed_text: str) -> List[int]:
    """Parse a comma-separated list of integer seeds."""
    seeds: List[int] = []
    for chunk in seed_text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        seeds.append(int(chunk))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def compute_beta_weights(
    beta: float,
    gamma: float,
    normalize_dims: bool,
    d_e: int,
    d_a: int,
) -> tuple[float, float]:
    """Compute semantic and attribution beta weights."""
    if normalize_dims:
        return gamma * beta / (d_e ** 0.5), (1.0 - gamma) * beta / d_a
    return gamma * beta, (1.0 - gamma) * beta


def get_sweep_entries(sweep_payload: Dict) -> List[Dict]:
    """Return the list of per-config sweep entries from a saved sweep payload."""
    if isinstance(sweep_payload.get("grid"), list):
        return sweep_payload["grid"]
    if isinstance(sweep_payload.get("sweep_results"), list):
        return sweep_payload["sweep_results"]
    raise KeyError("Expected sweep payload to contain either 'grid' or 'sweep_results'.")


def extract_matching_sweep_entry(
    sweep_payload: Dict,
    beta: float,
    gamma: float,
    atol: float = 1e-9,
) -> Optional[Dict]:
    """Find the saved sweep entry for a specific (beta, gamma) pair."""
    for entry in get_sweep_entries(sweep_payload):
        entry_beta = entry.get("beta")
        entry_gamma = entry.get("gamma")
        if entry_beta is None or entry_gamma is None:
            continue
        if math.isclose(float(entry_beta), float(beta), abs_tol=atol) and math.isclose(
            float(entry_gamma), float(gamma), abs_tol=atol
        ):
            return entry
    return None


def summarize_metric_values(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    """Summarize numeric values with null dispersion when fewer than 2 points exist."""
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "variance": None,
            "min": None,
            "max": None,
            "range": None,
        }

    summary: Dict[str, Optional[float]] = {
        "count": len(valid),
        "mean": float(np.mean(valid)),
        "std": None,
        "variance": None,
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "range": float(np.max(valid) - np.min(valid)),
    }
    if len(valid) >= 2:
        variance = float(np.var(valid))
        summary["variance"] = variance
        summary["std"] = float(np.sqrt(variance))
    return summary


def summarize_distribution(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    """Summarize a list of optional values for high-level reporting."""
    valid = [float(value) for value in values if value is not None]
    if not valid:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(valid),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
    }


def derive_converged_flag(
    history: Optional[Dict],
    convergence_threshold: float,
    reported_converged: Optional[bool],
) -> Optional[bool]:
    """Compute a canonical convergence flag from the final objective history."""
    if history and isinstance(history.get("L_RD"), list):
        lrd_history = history["L_RD"]
        if len(lrd_history) >= 2:
            return bool(
                check_convergence(
                    float(lrd_history[-2]),
                    float(lrd_history[-1]),
                    float(convergence_threshold),
                )
            )
    if reported_converged is None:
        return None
    return bool(reported_converged)


def build_success_record(
    *,
    prefix_id: str,
    prefix: Optional[str],
    seed: int,
    status: str,
    source: str,
    beta: float,
    gamma: float,
    beta_e: float,
    beta_a: float,
    pooling: str,
    attribution_metric: str,
    normalize_dims: bool,
    max_iterations: int,
    convergence_threshold: float,
    K: int,
    H: float,
    D_e: float,
    D_a: float,
    L_RD: float,
    n_iterations: int,
    converged: Optional[bool],
    reported_converged: Optional[bool],
    n_samples: Optional[int],
    sweep_file: Optional[Path] = None,
    error: Optional[str] = None,
) -> Dict:
    """Build a JSON-serializable per-prefix, per-seed record."""
    return {
        "prefix_id": prefix_id,
        "prefix": prefix,
        "seed": int(seed),
        "status": status,
        "source": source,
        "beta": float(beta),
        "gamma": float(gamma),
        "beta_e": float(beta_e),
        "beta_a": float(beta_a),
        "pooling": pooling,
        "attribution_metric": attribution_metric,
        "normalize_dims": bool(normalize_dims),
        "max_iterations": int(max_iterations),
        "convergence_threshold": float(convergence_threshold),
        "K": int(K),
        "H": float(H),
        "D_e": float(D_e),
        "D_a": float(D_a),
        "L_RD": float(L_RD),
        "n_iterations": int(n_iterations),
        "converged": None if converged is None else bool(converged),
        "reported_converged": None if reported_converged is None else bool(reported_converged),
        "n_samples": None if n_samples is None else int(n_samples),
        "sweep_file": None if sweep_file is None else str(sweep_file),
        "error": error,
    }


def build_failure_record(
    *,
    prefix_id: str,
    prefix: Optional[str],
    seed: int,
    source: str,
    beta: float,
    gamma: float,
    pooling: str,
    attribution_metric: str,
    normalize_dims: bool,
    max_iterations: int,
    convergence_threshold: float,
    error: str,
) -> Dict:
    """Build a failed per-prefix, per-seed record."""
    return {
        "prefix_id": prefix_id,
        "prefix": prefix,
        "seed": int(seed),
        "status": "failed",
        "source": source,
        "beta": float(beta),
        "gamma": float(gamma),
        "beta_e": None,
        "beta_a": None,
        "pooling": pooling,
        "attribution_metric": attribution_metric,
        "normalize_dims": bool(normalize_dims),
        "max_iterations": int(max_iterations),
        "convergence_threshold": float(convergence_threshold),
        "K": None,
        "H": None,
        "D_e": None,
        "D_a": None,
        "L_RD": None,
        "n_iterations": None,
        "converged": None,
        "reported_converged": None,
        "n_samples": None,
        "sweep_file": None,
        "error": error,
    }


def record_from_sweep_entry(
    *,
    prefix_id: str,
    prefix: Optional[str],
    seed: int,
    source: str,
    entry: Dict,
    config_fields: Dict,
    sweep_file: Path,
) -> Dict:
    """Convert a saved sweep entry into a raw output record."""
    assignments = entry.get("assignments", [])
    return build_success_record(
        prefix_id=prefix_id,
        prefix=prefix,
        seed=seed,
        status="success",
        source=source,
        beta=float(entry["beta"]),
        gamma=float(entry["gamma"]),
        beta_e=float(entry["beta_e"]),
        beta_a=float(entry["beta_a"]),
        pooling=config_fields["pooling"],
        attribution_metric=config_fields["attribution_metric"],
        normalize_dims=config_fields["normalize_dims"],
        max_iterations=config_fields["max_iterations"],
        convergence_threshold=config_fields["convergence_threshold"],
        K=int(entry["K"]),
        H=float(entry["H"]),
        D_e=float(entry["D_e"]),
        D_a=float(entry["D_a"]),
        L_RD=float(entry["L_RD"]),
        n_iterations=int(entry["n_iterations"]),
        converged=derive_converged_flag(
            history=entry.get("history"),
            convergence_threshold=config_fields["convergence_threshold"],
            reported_converged=entry.get("converged"),
        ),
        reported_converged=entry.get("converged"),
        n_samples=len(assignments) if assignments else None,
        sweep_file=sweep_file,
    )


def record_from_clustering_result(
    *,
    data: Dict,
    seed: int,
    source: str,
    result: Dict,
    beta: float,
    gamma: float,
    beta_e: float,
    beta_a: float,
    config_fields: Dict,
) -> Dict:
    """Convert a live clustering result into a raw output record."""
    rd_stats = result["rd_stats"]
    return build_success_record(
        prefix_id=data["prefix_id"],
        prefix=data.get("prefix"),
        seed=seed,
        status="success",
        source=source,
        beta=beta,
        gamma=gamma,
        beta_e=beta_e,
        beta_a=beta_a,
        pooling=config_fields["pooling"],
        attribution_metric=config_fields["attribution_metric"],
        normalize_dims=config_fields["normalize_dims"],
        max_iterations=config_fields["max_iterations"],
        convergence_threshold=config_fields["convergence_threshold"],
        K=len(result["components"]),
        H=float(rd_stats["H"]),
        D_e=float(rd_stats["D_e"]),
        D_a=float(rd_stats["D_a"]),
        L_RD=float(rd_stats["L_RD"]),
        n_iterations=int(result["n_iterations"]),
        converged=derive_converged_flag(
            history=result.get("history"),
            convergence_threshold=config_fields["convergence_threshold"],
            reported_converged=result.get("converged"),
        ),
        reported_converged=result.get("converged"),
        n_samples=int(data["n_samples"]),
    )


def append_jsonl(path: Path, record: Dict) -> None:
    """Append one JSON record to a JSONL file."""
    with path.open("a") as handle:
        handle.write(json.dumps(convert_numpy_types(record)) + "\n")


def build_per_prefix_summary_row(
    prefix_id: str,
    records: Sequence[Dict],
    baseline_seed: int,
) -> Dict:
    """Aggregate successful seed runs for one prefix into one CSV row."""
    sorted_records = sorted(records, key=lambda item: int(item["seed"]))
    successful_records = [record for record in sorted_records if record["status"] == "success"]
    failed_records = [record for record in sorted_records if record["status"] != "success"]

    baseline_record = next(
        (record for record in successful_records if int(record["seed"]) == int(baseline_seed)),
        None,
    )

    row: Dict[str, Optional[object]] = {
        "prefix_id": prefix_id,
        "prefix": successful_records[0].get("prefix") if successful_records else records[0].get("prefix"),
        "successful_seed_count": len(successful_records),
        "failed_seed_count": len(failed_records),
        "successful_seeds": ",".join(str(record["seed"]) for record in successful_records),
        "failed_seeds": ",".join(str(record["seed"]) for record in failed_records),
        "baseline_seed_source": None if baseline_record is None else baseline_record["source"],
        "all_requested_seeds_successful": len(failed_records) == 0,
        "converged_seed_count": sum(1 for record in successful_records if record["converged"]),
        "converged_seed_rate": (
            sum(1 for record in successful_records if record["converged"]) / len(successful_records)
            if successful_records
            else None
        ),
    }

    for metric in NUMERIC_METRICS:
        metric_summary = summarize_metric_values([record.get(metric) for record in successful_records])
        row[f"{metric}_mean"] = metric_summary["mean"]
        row[f"{metric}_std"] = metric_summary["std"]
        row[f"{metric}_variance"] = metric_summary["variance"]
        row[f"{metric}_min"] = metric_summary["min"]
        row[f"{metric}_max"] = metric_summary["max"]
        row[f"{metric}_range"] = metric_summary["range"]

    return row


def write_per_prefix_summary_csv(path: Path, rows: Sequence[Dict]) -> None:
    """Write flattened per-prefix summaries as CSV."""
    fieldnames = [
        "prefix_id",
        "prefix",
        "successful_seed_count",
        "failed_seed_count",
        "successful_seeds",
        "failed_seeds",
        "baseline_seed_source",
        "all_requested_seeds_successful",
        "converged_seed_count",
        "converged_seed_rate",
    ]
    for metric in NUMERIC_METRICS:
        fieldnames.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_variance",
                f"{metric}_min",
                f"{metric}_max",
                f"{metric}_range",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(convert_numpy_types(row))


def build_delta_to_baseline_summary(
    records_by_prefix: Dict[str, List[Dict]],
    seeds: Sequence[int],
    baseline_seed: int,
) -> Dict:
    """Summarize per-seed deltas relative to the baseline seed."""
    summaries: Dict[str, Dict] = {}
    if baseline_seed not in set(int(seed) for seed in seeds):
        return summaries

    for seed in seeds:
        if int(seed) == int(baseline_seed):
            continue

        per_metric_deltas: Dict[str, List[float]] = {metric: [] for metric in DELTA_METRICS}
        prefix_pairs = 0
        for prefix_records in records_by_prefix.values():
            record_map = {
                int(record["seed"]): record
                for record in prefix_records
                if record["status"] == "success"
            }
            baseline_record = record_map.get(int(baseline_seed))
            comparison_record = record_map.get(int(seed))
            if baseline_record is None or comparison_record is None:
                continue
            prefix_pairs += 1
            for metric in DELTA_METRICS:
                per_metric_deltas[metric].append(
                    float(comparison_record[metric]) - float(baseline_record[metric])
                )

        metric_summaries = {}
        for metric, deltas in per_metric_deltas.items():
            if not deltas:
                metric_summaries[metric] = {
                    "count": 0,
                    "mean_signed_delta": None,
                    "mean_absolute_delta": None,
                    "min_signed_delta": None,
                    "max_signed_delta": None,
                }
                continue
            metric_summaries[metric] = {
                "count": len(deltas),
                "mean_signed_delta": float(np.mean(deltas)),
                "mean_absolute_delta": float(np.mean(np.abs(deltas))),
                "min_signed_delta": float(np.min(deltas)),
                "max_signed_delta": float(np.max(deltas)),
            }

        summaries[str(seed)] = {
            "prefix_pairs": prefix_pairs,
            "metrics": metric_summaries,
        }

    return summaries


def build_summary(
    *,
    records: Sequence[Dict],
    per_prefix_rows: Sequence[Dict],
    args: argparse.Namespace,
    clustering_config: Dict,
    processed_prefix_ids: Sequence[str],
) -> Dict:
    """Build the top-level summary JSON."""
    records_by_prefix: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        records_by_prefix[str(record["prefix_id"])].append(record)

    success_records = [record for record in records if record["status"] == "success"]
    failure_records = [record for record in records if record["status"] != "success"]

    per_seed_summary: Dict[str, Dict] = {}
    for seed in args.seeds:
        seed_records = [record for record in records if int(record["seed"]) == int(seed)]
        seed_success = [record for record in seed_records if record["status"] == "success"]
        seed_failures = [record for record in seed_records if record["status"] != "success"]
        per_seed_summary[str(seed)] = {
            "total_prefixes": len(seed_records),
            "successful_prefixes": len(seed_success),
            "failed_prefixes": len(seed_failures),
            "converged_count": sum(1 for record in seed_success if record["converged"]),
            "converged_rate": (
                sum(1 for record in seed_success if record["converged"]) / len(seed_success)
                if seed_success
                else None
            ),
            "metrics": {
                metric: summarize_metric_values([record.get(metric) for record in seed_success])
                for metric in NUMERIC_METRICS
            },
        }

    per_prefix_variability = {}
    for metric in NUMERIC_METRICS:
        per_prefix_variability[metric] = {
            "std": summarize_distribution([row.get(f"{metric}_std") for row in per_prefix_rows]),
            "variance": summarize_distribution(
                [row.get(f"{metric}_variance") for row in per_prefix_rows]
            ),
            "range": summarize_distribution([row.get(f"{metric}_range") for row in per_prefix_rows]),
        }

    baseline_source_counter = Counter(
        row.get("baseline_seed_source")
        for row in per_prefix_rows
        if row.get("baseline_seed_source") is not None
    )

    return {
        "requested_seeds": list(args.seeds),
        "baseline_seed": int(args.baseline_seed),
        "processed_prefixes": len(processed_prefix_ids),
        "processed_prefix_ids": list(processed_prefix_ids),
        "successful_records": len(success_records),
        "failed_records": len(failure_records),
        "prefixes_with_complete_seed_set": sum(
            1 for row in per_prefix_rows if bool(row["all_requested_seeds_successful"])
        ),
        "prefixes_with_failures": sum(1 for row in per_prefix_rows if int(row["failed_seed_count"]) > 0),
        "baseline_source_counts": dict(baseline_source_counter),
        "config": {
            "results_dir": str(args.results_dir),
            "config_path": str(args.config),
            "output_dir": str(args.output_dir),
            "beta": float(args.beta),
            "gamma": float(args.gamma),
            "pooling": clustering_config["pooling"],
            "attribution_metric": clustering_config["attribution_metric"],
            "normalize_dims": bool(clustering_config["normalize_dims"]),
            "K_max": int(clustering_config["K_max"]),
            "max_iterations": int(clustering_config["max_iterations"]),
            "convergence_threshold": float(clustering_config["convergence_threshold"]),
        },
        "per_seed_summary": per_seed_summary,
        "per_prefix_variability": per_prefix_variability,
        "delta_to_baseline": build_delta_to_baseline_summary(
            records_by_prefix=records_by_prefix,
            seeds=args.seeds,
            baseline_seed=args.baseline_seed,
        ),
        "failures": [
            {
                "prefix_id": record["prefix_id"],
                "seed": int(record["seed"]),
                "source": record["source"],
                "error": record["error"],
            }
            for record in failure_records
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end clustering seed variance analysis.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46,47,48,49,50,51")
    parser.add_argument("--baseline-seed", type=int, default=42)
    parser.add_argument("--max-prefixes", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config_data = load_json(args.config)
    clustering = config_data.get("clustering", {})
    clustering_config = {
        "K_max": clustering.get("K_max", 30),
        "max_iterations": clustering.get("max_iterations", 30),
        "convergence_threshold": clustering.get("convergence_threshold", 1e-3),
        "pooling": clustering.get("pooling", "mean"),
        "attribution_metric": clustering.get("attribution_metric", "l1"),
        "normalize_dims": clustering.get("normalize_dims", False),
    }

    args.seeds = parse_seed_list(args.seeds)
    if args.output_dir is None:
        args.output_dir = (
            args.results_dir
            / "5_clustering_seed_variance"
            / f"beta{args.beta}_gamma{args.gamma}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        "clustering_seed_variance",
        log_file=args.output_dir / "seed_variance.log",
        level=logging.WARNING if args.quiet else logging.INFO,
    )

    raw_records_path = args.output_dir / "prefix_seed_stats.jsonl"
    raw_records_path.write_text("")

    embeddings_dir = args.results_dir / "4_feature_extraction" / "embeddings"
    attribution_graphs_dir = args.results_dir / "3_attribution_graphs"
    samples_dir = args.results_dir / "2_branch_sampling"
    clustering_dir = args.results_dir / "5_clustering"

    embedding_meta_files = sorted(embeddings_dir.glob("*_embeddings_meta.json"))
    if not embedding_meta_files:
        raise FileNotFoundError(f"No embedding metadata files found in {embeddings_dir}")
    if args.max_prefixes is not None:
        embedding_meta_files = embedding_meta_files[: args.max_prefixes]

    processed_prefix_ids: List[str] = []
    all_records: List[Dict] = []

    for index, meta_file in enumerate(embedding_meta_files, start=1):
        prefix_id = meta_file.stem.replace("_embeddings_meta", "")
        logger.info("Processing prefix %s (%d/%d)", prefix_id, index, len(embedding_meta_files))
        processed_prefix_ids.append(prefix_id)

        prefix_records: List[Dict] = []
        baseline_entry = None
        baseline_sweep_file = clustering_dir / f"{prefix_id}_sweep_results.json"
        if args.baseline_seed in set(args.seeds) and baseline_sweep_file.exists():
            try:
                sweep_payload = load_json(baseline_sweep_file)
                baseline_entry = extract_matching_sweep_entry(
                    sweep_payload=sweep_payload,
                    beta=args.beta,
                    gamma=args.gamma,
                )
                if baseline_entry is not None:
                    prefix_records.append(
                        record_from_sweep_entry(
                            prefix_id=prefix_id,
                            prefix=sweep_payload.get("prefix"),
                            seed=args.baseline_seed,
                            source=f"existing_seed{args.baseline_seed}",
                            entry=baseline_entry,
                            config_fields=clustering_config,
                            sweep_file=baseline_sweep_file,
                        )
                    )
                else:
                    logger.warning(
                        "Missing stored config for %s at beta=%s gamma=%s; rerunning baseline seed",
                        prefix_id,
                        args.beta,
                        args.gamma,
                    )
            except Exception as exc:
                logger.warning(
                    "Failed reading stored baseline for %s (%s); rerunning baseline seed",
                    prefix_id,
                    exc,
                )
                baseline_entry = None

        seeds_to_rerun = [
            seed
            for seed in args.seeds
            if int(seed) != int(args.baseline_seed) or baseline_entry is None
        ]

        data = None
        beta_e = None
        beta_a = None
        if seeds_to_rerun:
            try:
                data = load_prefix_data(
                    prefix_id=prefix_id,
                    embeddings_dir=embeddings_dir,
                    attribution_graphs_dir=attribution_graphs_dir,
                    samples_dir=samples_dir,
                    logger=logger,
                    pooling=clustering_config["pooling"],
                    metric_a=clustering_config["attribution_metric"],
                )
                beta_e, beta_a = compute_beta_weights(
                    beta=args.beta,
                    gamma=args.gamma,
                    normalize_dims=clustering_config["normalize_dims"],
                    d_e=data["embeddings_e"].shape[1],
                    d_a=data["attributions_a"].shape[1],
                )
            except Exception as exc:
                error_message = f"{type(exc).__name__}: {exc}"
                logger.exception("Failed loading data for prefix %s", prefix_id)
                for seed in seeds_to_rerun:
                    failure_record = build_failure_record(
                        prefix_id=prefix_id,
                        prefix=None,
                        seed=seed,
                        source="rerun",
                        beta=args.beta,
                        gamma=args.gamma,
                        pooling=clustering_config["pooling"],
                        attribution_metric=clustering_config["attribution_metric"],
                        normalize_dims=clustering_config["normalize_dims"],
                        max_iterations=clustering_config["max_iterations"],
                        convergence_threshold=clustering_config["convergence_threshold"],
                        error=error_message,
                    )
                    prefix_records.append(failure_record)

        if baseline_entry is not None and beta_e is not None and beta_a is not None:
            if not math.isclose(float(baseline_entry["beta_e"]), float(beta_e), abs_tol=1e-9):
                logger.warning(
                    "Stored beta_e mismatch for %s: stored=%s live=%s",
                    prefix_id,
                    baseline_entry["beta_e"],
                    beta_e,
                )
            if not math.isclose(float(baseline_entry["beta_a"]), float(beta_a), abs_tol=1e-9):
                logger.warning(
                    "Stored beta_a mismatch for %s: stored=%s live=%s",
                    prefix_id,
                    baseline_entry["beta_a"],
                    beta_a,
                )

        if data is not None:
            for seed in seeds_to_rerun:
                try:
                    result = run_clustering(
                        data=data,
                        beta_e=beta_e,
                        beta_a=beta_a,
                        K_max=clustering_config["K_max"],
                        max_iterations=clustering_config["max_iterations"],
                        convergence_threshold=clustering_config["convergence_threshold"],
                        logger=logger,
                        metric_a=clustering_config["attribution_metric"],
                        split_random_seed=seed,
                    )
                    source = "rerun"
                    prefix_records.append(
                        record_from_clustering_result(
                            data=data,
                            seed=seed,
                            source=source,
                            result=result,
                            beta=args.beta,
                            gamma=args.gamma,
                            beta_e=beta_e,
                            beta_a=beta_a,
                            config_fields=clustering_config,
                        )
                    )
                except Exception as exc:
                    error_message = f"{type(exc).__name__}: {exc}"
                    logger.exception(
                        "Failed clustering prefix %s with split_random_seed=%d",
                        prefix_id,
                        seed,
                    )
                    prefix_records.append(
                        build_failure_record(
                            prefix_id=prefix_id,
                            prefix=data.get("prefix"),
                            seed=seed,
                            source="rerun",
                            beta=args.beta,
                            gamma=args.gamma,
                            pooling=clustering_config["pooling"],
                            attribution_metric=clustering_config["attribution_metric"],
                            normalize_dims=clustering_config["normalize_dims"],
                            max_iterations=clustering_config["max_iterations"],
                            convergence_threshold=clustering_config["convergence_threshold"],
                            error=error_message,
                        )
                    )

        prefix_records.sort(key=lambda record: int(record["seed"]))
        for record in prefix_records:
            all_records.append(record)
            append_jsonl(raw_records_path, record)

    grouped_records: Dict[str, List[Dict]] = defaultdict(list)
    for record in all_records:
        grouped_records[str(record["prefix_id"])].append(record)
    per_prefix_rows = [
        build_per_prefix_summary_row(
            prefix_id=prefix_id,
            records=grouped_records[prefix_id],
            baseline_seed=args.baseline_seed,
        )
        for prefix_id in sorted(grouped_records.keys())
    ]

    write_per_prefix_summary_csv(args.output_dir / "per_prefix_summary.csv", per_prefix_rows)
    summary = build_summary(
        records=all_records,
        per_prefix_rows=per_prefix_rows,
        args=args,
        clustering_config=clustering_config,
        processed_prefix_ids=processed_prefix_ids,
    )
    save_json(summary, args.output_dir / "summary.json")

    logger.info(
        "Seed variance analysis complete: %d prefixes, %d records, %d failures",
        len(processed_prefix_ids),
        len(all_records),
        sum(1 for record in all_records if record["status"] != "success"),
    )


if __name__ == "__main__":
    main()
