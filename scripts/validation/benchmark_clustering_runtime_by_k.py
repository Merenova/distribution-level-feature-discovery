#!/usr/bin/env -S uv run python
"""Benchmark clustering runtime breakdown for natural exact-K outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "5_gaussian_clustering"))

from cluster import load_prefix_data, run_clustering
from scripts.run_clustering_seed_variance import (
    compute_beta_weights,
    derive_converged_flag,
    get_sweep_entries,
)
from utils.data_utils import convert_numpy_types, load_json, save_json
from utils.logging_utils import setup_logger


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "AmbigQA_Qwen3-8B" / "results"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "beta_gamma_scaled_config.json"
TIMING_METRICS = [
    "total_seconds",
    "em_seconds_total",
    "split_seconds_total",
    "junk_seconds_total",
    "residual_seconds_total",
]
SHARE_METRICS = ["em_share", "split_share", "junk_share", "residual_share"]
COUNT_METRICS = ["n_iterations", "splits_done_total", "junks_done_total"]
SUMMARY_METRICS = TIMING_METRICS + SHARE_METRICS + COUNT_METRICS


def parse_int_list(text: str) -> List[int]:
    """Parse a comma-separated list of integers."""
    values: List[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def load_clustering_config(config_path: Path) -> Dict[str, object]:
    """Load the clustering config fields needed for reruns."""
    config_data = load_json(config_path)
    clustering = config_data.get("clustering", {})
    return {
        "K_max": clustering.get("K_max", 30),
        "max_iterations": clustering.get("max_iterations", 30),
        "convergence_threshold": clustering.get("convergence_threshold", 1e-3),
        "pooling": clustering.get("pooling", "mean"),
        "attribution_metric": clustering.get("attribution_metric", "l1"),
        "normalize_dims": clustering.get("normalize_dims", False),
    }


def summarize_values(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    """Summarize a sequence with mean/median/min/max."""
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


def append_jsonl(path: Path, record: Dict) -> None:
    """Append one JSON object to a JSONL file."""
    with path.open("a") as handle:
        handle.write(json.dumps(convert_numpy_types(record)) + "\n")


def extract_converged_exact_k_candidates(
    *,
    prefix_id: str,
    prefix: Optional[str],
    sweep_file: Path,
    sweep_payload: Dict,
    target_ks: Sequence[int],
    convergence_threshold: float,
) -> List[Dict]:
    """Extract converged exact-K candidates from one saved sweep payload."""
    target_k_set = {int(k) for k in target_ks}
    candidates: List[Dict] = []

    for entry in get_sweep_entries(sweep_payload):
        if entry.get("K") is None or entry.get("beta") is None or entry.get("gamma") is None:
            continue

        final_k = int(entry["K"])
        if final_k not in target_k_set:
            continue

        canonical_converged = derive_converged_flag(
            history=entry.get("history"),
            convergence_threshold=convergence_threshold,
            reported_converged=entry.get("converged"),
        )
        if not canonical_converged:
            continue

        candidates.append(
            {
                "prefix_id": prefix_id,
                "prefix": prefix,
                "target_k": final_k,
                "saved_final_k": final_k,
                "beta": float(entry["beta"]),
                "gamma": float(entry["gamma"]),
                "saved_beta_e": float(entry["beta_e"]),
                "saved_beta_a": float(entry["beta_a"]),
                "saved_n_iterations": int(entry["n_iterations"]),
                "saved_converged": bool(canonical_converged),
                "saved_reported_converged": (
                    None if entry.get("converged") is None else bool(entry.get("converged"))
                ),
                "sweep_file": str(sweep_file),
            }
        )

    return candidates


def collect_runtime_candidates(
    *,
    clustering_dir: Path,
    target_ks: Sequence[int],
    convergence_threshold: float,
) -> Iterable[Dict]:
    """Yield converged natural exact-K sweep entries across all prefixes."""
    for sweep_file in sorted(clustering_dir.glob("*_sweep_results.json")):
        payload = load_json(sweep_file)
        prefix_id = payload.get("prefix_id") or sweep_file.name.replace("_sweep_results.json", "")
        prefix = payload.get("prefix")
        for candidate in extract_converged_exact_k_candidates(
            prefix_id=prefix_id,
            prefix=prefix,
            sweep_file=sweep_file,
            sweep_payload=payload,
            target_ks=target_ks,
            convergence_threshold=convergence_threshold,
        ):
            yield candidate


def materialize_runtime_candidates(
    *,
    clustering_dir: Path,
    target_ks: Sequence[int],
    convergence_threshold: float,
) -> List[Dict]:
    """Collect all converged natural exact-K sweep entries across all prefixes."""
    return list(
        collect_runtime_candidates(
            clustering_dir=clustering_dir,
            target_ks=target_ks,
            convergence_threshold=convergence_threshold,
        )
    )


def select_benchmark_cases(
    *,
    candidates: Sequence[Dict],
    target_ks: Sequence[int],
    sample_per_k: int,
    selection_seed: int,
) -> Tuple[Dict[int, List[Dict]], Dict[str, Dict]]:
    """Select at most one case per prefix per K, then sample per K deterministically."""
    grouped: Dict[int, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for candidate in candidates:
        grouped[int(candidate["target_k"])][str(candidate["prefix_id"])].append(dict(candidate))

    rng = np.random.RandomState(selection_seed)
    selected_by_k: Dict[int, List[Dict]] = {}
    selection_summary: Dict[str, Dict] = {}

    for target_k in [int(k) for k in target_ks]:
        per_prefix = grouped.get(target_k, {})
        representatives: List[Dict] = []
        for prefix_id in sorted(per_prefix.keys()):
            options = sorted(
                per_prefix[prefix_id],
                key=lambda item: (
                    float(item["beta"]),
                    float(item["gamma"]),
                    str(item["sweep_file"]),
                ),
            )
            choice_index = int(rng.randint(len(options)))
            representatives.append(options[choice_index])

        candidate_count = sum(len(options) for options in per_prefix.values())
        if len(representatives) <= sample_per_k:
            selected_cases = list(representatives)
        else:
            chosen_indices = sorted(
                int(index)
                for index in rng.choice(
                    len(representatives),
                    size=sample_per_k,
                    replace=False,
                )
            )
            selected_cases = [representatives[index] for index in chosen_indices]

        selected_cases = sorted(
            selected_cases,
            key=lambda item: (
                str(item["prefix_id"]),
                float(item["beta"]),
                float(item["gamma"]),
            ),
        )
        selected_by_k[target_k] = selected_cases
        selection_summary[str(target_k)] = {
            "candidate_entries": int(candidate_count),
            "unique_prefix_candidates": int(len(representatives)),
            "selected_cases": int(len(selected_cases)),
        }

    return selected_by_k, selection_summary


def select_first_cases(
    *,
    candidates: Iterable[Dict],
    target_ks: Sequence[int],
    max_total_cases: int,
) -> Tuple[Dict[int, List[Dict]], Dict[str, Dict]]:
    """Select the first qualifying unique (prefix, K) cases in encounter order."""
    selected_by_k: Dict[int, List[Dict]] = {int(k): [] for k in target_ks}
    seen_prefix_keys = set()
    candidate_entries_seen: Counter = Counter()
    unique_prefix_candidates_seen: Dict[int, set[str]] = {
        int(k): set() for k in target_ks
    }

    total_selected = 0
    for candidate in candidates:
        target_k = int(candidate["target_k"])
        prefix_id = str(candidate["prefix_id"])
        candidate_entries_seen[target_k] += 1
        unique_prefix_candidates_seen[target_k].add(prefix_id)

        dedupe_key = (target_k, prefix_id)
        if dedupe_key in seen_prefix_keys:
            continue

        seen_prefix_keys.add(dedupe_key)
        selected_by_k[target_k].append(dict(candidate))
        total_selected += 1
        if total_selected >= max_total_cases:
            break

    selection_summary = {
        str(target_k): {
            "candidate_entries": int(candidate_entries_seen.get(int(target_k), 0)),
            "unique_prefix_candidates": int(len(unique_prefix_candidates_seen[int(target_k)])),
            "selected_cases": int(len(selected_by_k[int(target_k)])),
        }
        for target_k in target_ks
    }
    return selected_by_k, selection_summary


def compute_runtime_shares(profile: Dict) -> Dict[str, Optional[float]]:
    """Compute runtime shares from a runtime profile."""
    total_seconds = float(profile.get("total_seconds", 0.0))
    if total_seconds <= 0:
        return {metric: None for metric in SHARE_METRICS}
    return {
        "em_share": float(profile.get("em_seconds_total", 0.0)) / total_seconds,
        "split_share": float(profile.get("split_seconds_total", 0.0)) / total_seconds,
        "junk_share": float(profile.get("junk_seconds_total", 0.0)) / total_seconds,
        "residual_share": float(profile.get("residual_seconds_total", 0.0)) / total_seconds,
    }


def build_success_record(
    *,
    case: Dict,
    result: Dict,
    beta_e: float,
    beta_a: float,
    clustering_config: Dict[str, object],
    split_random_seed: int,
) -> Dict:
    """Build one successful benchmark record."""
    runtime_profile = dict(result.get("runtime_profile", {}))
    shares = compute_runtime_shares(runtime_profile)
    converged = derive_converged_flag(
        history=result.get("history"),
        convergence_threshold=float(clustering_config["convergence_threshold"]),
        reported_converged=result.get("converged"),
    )
    final_k = int(len(result["components"]))

    return {
        "prefix_id": case["prefix_id"],
        "prefix": case.get("prefix"),
        "target_k": int(case["target_k"]),
        "final_k": final_k,
        "k_matches_target": final_k == int(case["target_k"]),
        "eligible_for_bucket": bool(converged and final_k == int(case["target_k"])),
        "status": "success",
        "beta": float(case["beta"]),
        "gamma": float(case["gamma"]),
        "beta_e": float(beta_e),
        "beta_a": float(beta_a),
        "pooling": clustering_config["pooling"],
        "attribution_metric": clustering_config["attribution_metric"],
        "normalize_dims": bool(clustering_config["normalize_dims"]),
        "K_max": int(clustering_config["K_max"]),
        "max_iterations": int(clustering_config["max_iterations"]),
        "convergence_threshold": float(clustering_config["convergence_threshold"]),
        "split_random_seed": int(split_random_seed),
        "saved_final_k": int(case["saved_final_k"]),
        "saved_n_iterations": int(case["saved_n_iterations"]),
        "saved_converged": bool(case["saved_converged"]),
        "saved_reported_converged": case.get("saved_reported_converged"),
        "reported_converged": None if result.get("converged") is None else bool(result["converged"]),
        "converged": None if converged is None else bool(converged),
        "n_iterations": int(result["n_iterations"]),
        "total_seconds": float(runtime_profile.get("total_seconds", 0.0)),
        "em_seconds_total": float(runtime_profile.get("em_seconds_total", 0.0)),
        "split_seconds_total": float(runtime_profile.get("split_seconds_total", 0.0)),
        "junk_seconds_total": float(runtime_profile.get("junk_seconds_total", 0.0)),
        "residual_seconds_total": float(runtime_profile.get("residual_seconds_total", 0.0)),
        "em_share": shares["em_share"],
        "split_share": shares["split_share"],
        "junk_share": shares["junk_share"],
        "residual_share": shares["residual_share"],
        "splits_done_total": int(runtime_profile.get("splits_done_total", 0)),
        "junks_done_total": int(runtime_profile.get("junks_done_total", 0)),
        "sweep_file": case["sweep_file"],
        "runtime_profile": runtime_profile,
        "error": None,
    }


def build_failure_record(
    *,
    case: Dict,
    clustering_config: Dict[str, object],
    split_random_seed: int,
    error: str,
) -> Dict:
    """Build one failed benchmark record."""
    return {
        "prefix_id": case["prefix_id"],
        "prefix": case.get("prefix"),
        "target_k": int(case["target_k"]),
        "final_k": None,
        "k_matches_target": None,
        "eligible_for_bucket": False,
        "status": "failed",
        "beta": float(case["beta"]),
        "gamma": float(case["gamma"]),
        "beta_e": None,
        "beta_a": None,
        "pooling": clustering_config["pooling"],
        "attribution_metric": clustering_config["attribution_metric"],
        "normalize_dims": bool(clustering_config["normalize_dims"]),
        "K_max": int(clustering_config["K_max"]),
        "max_iterations": int(clustering_config["max_iterations"]),
        "convergence_threshold": float(clustering_config["convergence_threshold"]),
        "split_random_seed": int(split_random_seed),
        "saved_final_k": int(case["saved_final_k"]),
        "saved_n_iterations": int(case["saved_n_iterations"]),
        "saved_converged": bool(case["saved_converged"]),
        "saved_reported_converged": case.get("saved_reported_converged"),
        "reported_converged": None,
        "converged": None,
        "n_iterations": None,
        "total_seconds": None,
        "em_seconds_total": None,
        "split_seconds_total": None,
        "junk_seconds_total": None,
        "residual_seconds_total": None,
        "em_share": None,
        "split_share": None,
        "junk_share": None,
        "residual_share": None,
        "splits_done_total": None,
        "junks_done_total": None,
        "sweep_file": case["sweep_file"],
        "runtime_profile": None,
        "error": error,
    }


def build_selected_cases_payload(
    *,
    args: argparse.Namespace,
    selection_summary: Dict[str, Dict],
    selected_by_k: Dict[int, List[Dict]],
) -> Dict:
    """Build the saved selection manifest."""
    return {
        "requested_target_ks": [int(k) for k in args.target_ks],
        "sample_per_k": int(args.sample_per_k),
        "selection_seed": int(args.selection_seed),
        "selection_mode": args.selection_mode,
        "max_total_cases": args.max_total_cases,
        "split_random_seed": int(args.split_random_seed),
        "buckets": {
            str(target_k): {
                **selection_summary.get(str(target_k), {}),
                "cases": selected_by_k.get(target_k, []),
            }
            for target_k in args.target_ks
        },
    }


def build_summary(
    *,
    records: Sequence[Dict],
    args: argparse.Namespace,
    clustering_config: Dict[str, object],
    selection_summary: Dict[str, Dict],
) -> Dict:
    """Build the top-level summary JSON."""
    summary_by_k: Dict[str, Dict] = {}

    for target_k in args.target_ks:
        bucket_records = [record for record in records if int(record["target_k"]) == int(target_k)]
        success_records = [record for record in bucket_records if record["status"] == "success"]
        failure_records = [record for record in bucket_records if record["status"] != "success"]
        converged_records = [record for record in success_records if record.get("converged")]
        eligible_records = [record for record in success_records if record.get("eligible_for_bucket")]

        summary_by_k[str(target_k)] = {
            "selection": selection_summary.get(str(target_k), {}),
            "completed_records": len(bucket_records),
            "successful_records": len(success_records),
            "failed_records": len(failure_records),
            "converged_records": len(converged_records),
            "eligible_records": len(eligible_records),
            "k_match_rate": (
                sum(1 for record in success_records if record.get("k_matches_target")) / len(success_records)
                if success_records
                else None
            ),
            "converged_rate": (
                len(converged_records) / len(success_records)
                if success_records
                else None
            ),
            "eligible_rate": (
                len(eligible_records) / len(success_records)
                if success_records
                else None
            ),
            "final_k_distribution": dict(
                sorted(
                    Counter(str(record["final_k"]) for record in success_records if record["final_k"] is not None).items(),
                    key=lambda item: int(item[0]),
                )
            ),
            "metrics": {
                metric: summarize_values([record.get(metric) for record in eligible_records])
                for metric in SUMMARY_METRICS
            },
        }

    return {
        "config": {
            "results_dir": str(args.results_dir),
            "config_path": str(args.config),
            "output_dir": str(args.output_dir),
            "target_ks": [int(k) for k in args.target_ks],
            "sample_per_k": int(args.sample_per_k),
            "selection_seed": int(args.selection_seed),
            "selection_mode": args.selection_mode,
            "max_total_cases": args.max_total_cases,
            "split_random_seed": int(args.split_random_seed),
            "pooling": clustering_config["pooling"],
            "attribution_metric": clustering_config["attribution_metric"],
            "normalize_dims": bool(clustering_config["normalize_dims"]),
            "K_max": int(clustering_config["K_max"]),
            "max_iterations": int(clustering_config["max_iterations"]),
            "convergence_threshold": float(clustering_config["convergence_threshold"]),
        },
        "overall": {
            "total_records": len(records),
            "successful_records": sum(1 for record in records if record["status"] == "success"),
            "failed_records": sum(1 for record in records if record["status"] != "success"),
            "eligible_records": sum(1 for record in records if record.get("eligible_for_bucket")),
        },
        "per_k": summary_by_k,
        "failures": [
            {
                "prefix_id": record["prefix_id"],
                "target_k": int(record["target_k"]),
                "beta": float(record["beta"]),
                "gamma": float(record["gamma"]),
                "error": record["error"],
            }
            for record in records
            if record["status"] != "success"
        ],
    }


def write_summary_csv(path: Path, summary: Dict, target_ks: Sequence[int]) -> None:
    """Write one flat summary row per K bucket."""
    fieldnames = [
        "target_k",
        "candidate_entries",
        "unique_prefix_candidates",
        "selected_cases",
        "completed_records",
        "successful_records",
        "failed_records",
        "converged_records",
        "eligible_records",
        "k_match_rate",
        "converged_rate",
        "eligible_rate",
    ]
    for metric in SUMMARY_METRICS:
        fieldnames.extend(
            [
                f"{metric}_mean",
                f"{metric}_median",
                f"{metric}_min",
                f"{metric}_max",
            ]
        )

    rows: List[Dict] = []
    for target_k in target_ks:
        bucket = summary["per_k"][str(target_k)]
        selection = bucket.get("selection", {})
        row = {
            "target_k": int(target_k),
            "candidate_entries": selection.get("candidate_entries"),
            "unique_prefix_candidates": selection.get("unique_prefix_candidates"),
            "selected_cases": selection.get("selected_cases"),
            "completed_records": bucket.get("completed_records"),
            "successful_records": bucket.get("successful_records"),
            "failed_records": bucket.get("failed_records"),
            "converged_records": bucket.get("converged_records"),
            "eligible_records": bucket.get("eligible_records"),
            "k_match_rate": bucket.get("k_match_rate"),
            "converged_rate": bucket.get("converged_rate"),
            "eligible_rate": bucket.get("eligible_rate"),
        }
        for metric in SUMMARY_METRICS:
            metric_summary = bucket["metrics"][metric]
            row[f"{metric}_mean"] = metric_summary["mean"]
            row[f"{metric}_median"] = metric_summary["median"]
            row[f"{metric}_min"] = metric_summary["min"]
            row[f"{metric}_max"] = metric_summary["max"]
        rows.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(convert_numpy_types(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark clustering runtime by final exact K.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--k-values", type=str, default="3,5,10")
    parser.add_argument("--sample-per-k", type=int, default=20)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--selection-mode", type=str, choices=["random", "first"], default="random")
    parser.add_argument("--max-total-cases", type=int, default=None)
    parser.add_argument("--split-random-seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    args.target_ks = parse_int_list(args.k_values)
    clustering_config = load_clustering_config(args.config)
    if args.output_dir is None:
        args.output_dir = args.results_dir / "5_clustering_runtime_by_k"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        "clustering_runtime_by_k",
        log_file=args.output_dir / "runtime_benchmark.log",
        level=logging.WARNING if args.quiet else logging.INFO,
    )

    embeddings_dir = args.results_dir / "4_feature_extraction" / "embeddings"
    attribution_graphs_dir = args.results_dir / "3_attribution_graphs"
    samples_dir = args.results_dir / "2_branch_sampling"
    clustering_dir = args.results_dir / "5_clustering"
    if not clustering_dir.exists():
        raise FileNotFoundError(f"Clustering results directory not found: {clustering_dir}")

    if args.selection_mode == "first" and args.max_total_cases is not None:
        logger.info(
            "Selecting first %d qualifying cases in scan order",
            args.max_total_cases,
        )
        selected_by_k, selection_summary = select_first_cases(
            candidates=collect_runtime_candidates(
                clustering_dir=clustering_dir,
                target_ks=args.target_ks,
                convergence_threshold=float(clustering_config["convergence_threshold"]),
            ),
            target_ks=args.target_ks,
            max_total_cases=int(args.max_total_cases),
        )
    else:
        candidates = materialize_runtime_candidates(
            clustering_dir=clustering_dir,
            target_ks=args.target_ks,
            convergence_threshold=float(clustering_config["convergence_threshold"]),
        )
        logger.info("Collected %d converged exact-K candidates", len(candidates))
        selected_by_k, selection_summary = select_benchmark_cases(
            candidates=candidates,
            target_ks=args.target_ks,
            sample_per_k=args.sample_per_k,
            selection_seed=args.selection_seed,
        )
    save_json(
        build_selected_cases_payload(
            args=args,
            selection_summary=selection_summary,
            selected_by_k=selected_by_k,
        ),
        args.output_dir / "selected_cases.json",
    )

    raw_records_path = args.output_dir / "runtime_records.jsonl"
    raw_records_path.write_text("")

    data_cache: Dict[str, Dict] = {}
    records: List[Dict] = []

    for target_k in args.target_ks:
        cases = selected_by_k.get(int(target_k), [])
        logger.info(
            "Benchmarking K=%d with %d selected cases",
            target_k,
            len(cases),
        )
        for case in cases:
            prefix_id = str(case["prefix_id"])
            try:
                if prefix_id not in data_cache:
                    data_cache[prefix_id] = load_prefix_data(
                        prefix_id=prefix_id,
                        embeddings_dir=embeddings_dir,
                        attribution_graphs_dir=attribution_graphs_dir,
                        samples_dir=samples_dir,
                        logger=logger,
                        pooling=str(clustering_config["pooling"]),
                        metric_a=str(clustering_config["attribution_metric"]),
                    )
                data = data_cache[prefix_id]
                beta_e, beta_a = compute_beta_weights(
                    beta=float(case["beta"]),
                    gamma=float(case["gamma"]),
                    normalize_dims=bool(clustering_config["normalize_dims"]),
                    d_e=int(data["embeddings_e"].shape[1]),
                    d_a=int(data["attributions_a"].shape[1]),
                )
                if not math.isclose(float(case["saved_beta_e"]), float(beta_e), abs_tol=1e-9):
                    logger.warning(
                        "beta_e mismatch for %s K=%s beta=%s gamma=%s: saved=%s live=%s",
                        prefix_id,
                        target_k,
                        case["beta"],
                        case["gamma"],
                        case["saved_beta_e"],
                        beta_e,
                    )
                if not math.isclose(float(case["saved_beta_a"]), float(beta_a), abs_tol=1e-9):
                    logger.warning(
                        "beta_a mismatch for %s K=%s beta=%s gamma=%s: saved=%s live=%s",
                        prefix_id,
                        target_k,
                        case["beta"],
                        case["gamma"],
                        case["saved_beta_a"],
                        beta_a,
                    )

                result = run_clustering(
                    data=data,
                    beta_e=beta_e,
                    beta_a=beta_a,
                    K_max=int(clustering_config["K_max"]),
                    max_iterations=int(clustering_config["max_iterations"]),
                    convergence_threshold=float(clustering_config["convergence_threshold"]),
                    logger=logger,
                    metric_a=str(clustering_config["attribution_metric"]),
                    split_random_seed=int(args.split_random_seed),
                    collect_runtime_profile=True,
                )
                record = build_success_record(
                    case=case,
                    result=result,
                    beta_e=beta_e,
                    beta_a=beta_a,
                    clustering_config=clustering_config,
                    split_random_seed=int(args.split_random_seed),
                )
            except Exception as exc:
                logger.exception(
                    "Failed benchmarking prefix %s target K=%s beta=%s gamma=%s",
                    prefix_id,
                    target_k,
                    case["beta"],
                    case["gamma"],
                )
                record = build_failure_record(
                    case=case,
                    clustering_config=clustering_config,
                    split_random_seed=int(args.split_random_seed),
                    error=f"{type(exc).__name__}: {exc}",
                )

            records.append(record)
            append_jsonl(raw_records_path, record)

    summary = build_summary(
        records=records,
        args=args,
        clustering_config=clustering_config,
        selection_summary=selection_summary,
    )
    save_json(summary, args.output_dir / "summary.json")
    write_summary_csv(args.output_dir / "summary.csv", summary, args.target_ks)

    logger.info(
        "Runtime benchmark complete: %d records, %d eligible",
        len(records),
        sum(1 for record in records if record.get("eligible_for_bucket")),
    )


if __name__ == "__main__":
    main()
