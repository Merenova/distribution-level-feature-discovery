#!/usr/bin/env -S uv run python
"""Run split-seed stability analysis for AmbigQA Qwen3-8B clustering."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "5_gaussian_clustering"))

from adaptive_control import run_split_trial
from cluster import load_prefix_data, run_clustering
from split_stability import (
    build_child_memberships,
    compute_label_invariant_split_jaccard,
    summarize_values,
)
from utils.data_utils import convert_numpy_types, load_json, save_json
from utils.logging_utils import setup_logger


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "AmbigQA_Qwen3-8B" / "results"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "beta_gamma_scaled_config.json"


def parse_seed_list(seed_text: str) -> List[int]:
    """Parse a comma-separated seed list."""
    seeds = []
    for chunk in seed_text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        seeds.append(int(chunk))
    if not seeds:
        raise ValueError("At least one replay seed is required.")
    return seeds


def compute_beta_weights(
    beta: float,
    gamma: float,
    normalize_dims: bool,
    d_e: int,
    d_a: int,
) -> tuple[float, float]:
    """Compute the semantic and attribution beta weights."""
    if normalize_dims:
        return gamma * beta / (d_e ** 0.5), (1.0 - gamma) * beta / d_a
    return gamma * beta, (1.0 - gamma) * beta


def serialize_split_trial(
    trial: Dict,
    parent_global_indices: List[int],
    seed: int,
) -> Dict:
    """Convert a split trial result into a JSON-friendly record."""
    child_memberships = None
    child_sizes = None
    child_local_indices = trial.get("child_local_indices")
    if child_local_indices and len(child_local_indices) == 2:
        child_memberships = build_child_memberships(parent_global_indices, child_local_indices)
        child_sizes = [len(child) for child in child_memberships]

    return {
        "seed": int(seed),
        "accepted": bool(trial.get("accepted", False)),
        "failure_reason": trial.get("failure_reason"),
        "alpha": trial.get("alpha"),
        "delta_D_e": trial.get("delta_D_e"),
        "delta_D_a": trial.get("delta_D_a"),
        "rate_cost": trial.get("rate_cost"),
        "distortion_benefit": trial.get("distortion_benefit"),
        "W1": trial.get("W1"),
        "W2": trial.get("W2"),
        "var1_e": trial.get("var1_e"),
        "var2_e": trial.get("var2_e"),
        "var1_a": trial.get("var1_a"),
        "var2_a": trial.get("var2_a"),
        "child_global_indices": child_memberships,
        "child_sizes": child_sizes,
    }


class SplitStudyCollector:
    """Collect and summarize accepted split events during clustering."""

    def __init__(
        self,
        max_events: int,
        replay_seeds: List[int],
        baseline_seed: int,
        events_path: Path,
        logger: logging.Logger,
    ) -> None:
        self.max_events = max_events
        self.replay_seeds = replay_seeds
        self.baseline_seed = baseline_seed
        self.events_path = events_path
        self.logger = logger
        self.records: List[Dict] = []

    def is_full(self) -> bool:
        return len(self.records) >= self.max_events

    def observe(self, event: Dict) -> None:
        if self.is_full():
            return

        event_id = len(self.records) + 1
        parent_global_indices = [int(idx) for idx in event["parent_global_indices"].tolist()]

        baseline_live = serialize_split_trial(
            event["trial"],
            parent_global_indices=parent_global_indices,
            seed=int(event["split_random_seed"]),
        )
        baseline_live["child_component_ids"] = [
            int(comp_id) for comp_id in event.get("child_component_ids", [])
        ]

        replay_records = []
        baseline_vs_seed_jaccard: Dict[str, Optional[float]] = {}
        baseline_children = baseline_live.get("child_global_indices")

        for seed in self.replay_seeds:
            replay_trial = run_split_trial(
                event["parent_embeddings_e"],
                event["parent_attributions_a"],
                event["parent_path_probs"],
                event["beta_e"],
                event["beta_a"],
                event["parent_var_e"],
                event["parent_var_a"],
                event["parent_P_bar"],
                metric_a=event["metric_a"],
                split_random_seed=seed,
            )
            replay_record = serialize_split_trial(
                replay_trial,
                parent_global_indices=parent_global_indices,
                seed=seed,
            )
            if baseline_live["accepted"] and replay_record["accepted"]:
                jaccard = compute_label_invariant_split_jaccard(
                    baseline_children,
                    replay_record["child_global_indices"],
                )
                baseline_vs_seed_jaccard[str(seed)] = float(jaccard)
                replay_record["label_invariant_jaccard_vs_live_baseline"] = float(jaccard)
            else:
                baseline_vs_seed_jaccard[str(seed)] = None
                replay_record["label_invariant_jaccard_vs_live_baseline"] = None
            replay_records.append(replay_record)

        pairwise_matrix: List[List[Optional[float]]] = []
        pairwise_pairs = []
        for i, replay_i in enumerate(replay_records):
            row: List[Optional[float]] = []
            for j, replay_j in enumerate(replay_records):
                if not replay_i["accepted"] or not replay_j["accepted"]:
                    row.append(None)
                    continue

                score = compute_label_invariant_split_jaccard(
                    replay_i["child_global_indices"],
                    replay_j["child_global_indices"],
                )
                row.append(float(score))
                if i < j:
                    pairwise_pairs.append(
                        {
                            "seed_i": int(replay_i["seed"]),
                            "seed_j": int(replay_j["seed"]),
                            "jaccard": float(score),
                        }
                    )
            pairwise_matrix.append(row)

        accepted_pairwise_summary = summarize_values(
            [pair["jaccard"] for pair in pairwise_pairs]
        )
        accepted_replay_count = sum(1 for replay in replay_records if replay["accepted"])

        record = {
            "event_id": event_id,
            "prefix_id": event["prefix_id"],
            "prefix": event["prefix"],
            "iteration": int(event["iteration"]),
            "parent_component_id": int(event["parent_component_id"]),
            "parent_n_samples": len(parent_global_indices),
            "parent_mass": float(event["parent_mass"]),
            "parent_p_bar": float(event["parent_P_bar"]),
            "parent_var_e": float(event["parent_var_e"]),
            "parent_var_a": float(event["parent_var_a"]),
            "parent_global_indices": parent_global_indices,
            "baseline_live": baseline_live,
            "replays": replay_records,
            "baseline_vs_seed_jaccard": baseline_vs_seed_jaccard,
            "pairwise_jaccard_matrix": pairwise_matrix,
            "pairwise_jaccard_pairs": pairwise_pairs,
            "accepted_replay_count": accepted_replay_count,
            "accepted_replay_rate": accepted_replay_count / len(replay_records),
            "accepted_pairwise_jaccard": accepted_pairwise_summary,
        }

        self.records.append(record)
        with self.events_path.open("a") as handle:
            handle.write(json.dumps(convert_numpy_types(record)) + "\n")

        self.logger.info(
            "Recorded split event %d/%d: %s iter=%d parent=%d accepted_replays=%d/%d",
            event_id,
            self.max_events,
            event["prefix_id"],
            int(event["iteration"]),
            int(event["parent_component_id"]),
            accepted_replay_count,
            len(replay_records),
        )


def build_summary(
    collector: SplitStudyCollector,
    args: argparse.Namespace,
    clustering_config: Dict,
    processed_prefix_ids: List[str],
    errors: List[Dict],
    beta_e: Optional[float],
    beta_a: Optional[float],
) -> Dict:
    """Aggregate overall study statistics."""
    records = collector.records
    pairwise_scores = [
        pair["jaccard"]
        for record in records
        for pair in record.get("pairwise_jaccard_pairs", [])
    ]
    per_event_mean_scores = [
        record["accepted_pairwise_jaccard"]["mean"]
        for record in records
        if record["accepted_pairwise_jaccard"]["mean"] is not None
    ]

    acceptance_by_seed = {}
    baseline_vs_seed_summary = {}
    for seed in collector.replay_seeds:
        seed_records = [
            replay
            for record in records
            for replay in record.get("replays", [])
            if replay["seed"] == seed
        ]
        accepted_count = sum(1 for replay in seed_records if replay["accepted"])
        acceptance_by_seed[str(seed)] = {
            "accepted": accepted_count,
            "total": len(seed_records),
            "rate": (accepted_count / len(seed_records)) if seed_records else None,
        }
        baseline_vs_seed_summary[str(seed)] = summarize_values(
            [record["baseline_vs_seed_jaccard"].get(str(seed)) for record in records]
        )

    total_replays = len(records) * len(collector.replay_seeds)
    total_accepted_replays = sum(
        1
        for record in records
        for replay in record.get("replays", [])
        if replay["accepted"]
    )

    return {
        "requested_events": args.max_events,
        "recorded_events": len(records),
        "processed_prefixes": len(processed_prefix_ids),
        "processed_prefix_ids": processed_prefix_ids,
        "stopped_after_reaching_max_events": collector.is_full(),
        "errors": errors,
        "config": {
            "results_dir": str(args.results_dir),
            "config_path": str(args.config),
            "beta": float(args.beta),
            "gamma": float(args.gamma),
            "beta_e": beta_e,
            "beta_a": beta_a,
            "baseline_split_seed": int(args.split_random_seed),
            "replay_seeds": collector.replay_seeds,
            "max_iterations": int(clustering_config["max_iterations"]),
            "convergence_threshold": float(clustering_config["convergence_threshold"]),
            "pooling": clustering_config["pooling"],
            "attribution_metric": clustering_config["attribution_metric"],
            "normalize_dims": bool(clustering_config["normalize_dims"]),
        },
        "acceptance": {
            "total_replays": total_replays,
            "accepted_replays": total_accepted_replays,
            "overall_rate": (total_accepted_replays / total_replays) if total_replays else None,
            "by_seed": acceptance_by_seed,
        },
        "jaccard": {
            "accepted_pairwise": summarize_values(pairwise_scores),
            "per_event_mean": summarize_values(per_event_mean_scores),
            "baseline_vs_seed": baseline_vs_seed_summary,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run split seed stability study.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--max-prefixes", type=int, default=None)
    parser.add_argument("--split-random-seed", type=int, default=42)
    parser.add_argument(
        "--replay-seeds",
        type=str,
        default="42,43,44,45,46,47,48,49,50,51",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config_data = load_json(args.config)
    clustering = config_data.get("clustering", {})
    clustering_config = {
        "max_iterations": clustering.get("max_iterations", 30),
        "convergence_threshold": clustering.get("convergence_threshold", 1e-3),
        "pooling": clustering.get("pooling", "mean"),
        "attribution_metric": clustering.get("attribution_metric", "l1"),
        "normalize_dims": clustering.get("normalize_dims", False),
    }

    args.replay_seeds = parse_seed_list(args.replay_seeds)
    if args.output_dir is None:
        args.output_dir = (
            args.results_dir
            / "5_clustering_split_seed_stability"
            / f"beta{args.beta}_gamma{args.gamma}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        "split_seed_stability",
        log_file=args.output_dir / "study.log",
        level=logging.WARNING if args.quiet else logging.INFO,
    )
    events_path = args.output_dir / "events.jsonl"
    events_path.write_text("")

    embeddings_dir = args.results_dir / "4_feature_extraction" / "embeddings"
    attribution_graphs_dir = args.results_dir / "3_attribution_graphs"
    samples_dir = args.results_dir / "2_branch_sampling"
    embedding_meta_files = sorted(embeddings_dir.glob("*_embeddings_meta.json"))
    if not embedding_meta_files:
        raise FileNotFoundError(f"No embedding metadata files found in {embeddings_dir}")

    collector = SplitStudyCollector(
        max_events=args.max_events,
        replay_seeds=args.replay_seeds,
        baseline_seed=args.split_random_seed,
        events_path=events_path,
        logger=logger,
    )

    processed_prefix_ids: List[str] = []
    errors: List[Dict] = []
    beta_e = None
    beta_a = None

    for meta_index, meta_file in enumerate(embedding_meta_files, start=1):
        if collector.is_full():
            break
        if args.max_prefixes is not None and meta_index > args.max_prefixes:
            break

        prefix_id = meta_file.stem.replace("_embeddings_meta", "")
        logger.info("Processing prefix %s (%d/%d)", prefix_id, meta_index, len(embedding_meta_files))

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

            if beta_e is None or beta_a is None:
                beta_e, beta_a = compute_beta_weights(
                    args.beta,
                    args.gamma,
                    clustering_config["normalize_dims"],
                    d_e=data["embeddings_e"].shape[1],
                    d_a=data["attributions_a"].shape[1],
                )

            run_clustering(
                data=data,
                beta_e=beta_e,
                beta_a=beta_a,
                K_max=clustering.get("K_max", 30),
                max_iterations=clustering_config["max_iterations"],
                convergence_threshold=clustering_config["convergence_threshold"],
                logger=logger,
                metric_a=clustering_config["attribution_metric"],
                split_random_seed=args.split_random_seed,
                split_event_observer=collector.observe,
            )
            processed_prefix_ids.append(prefix_id)
        except Exception as exc:
            logger.exception("Failed processing prefix %s", prefix_id)
            errors.append({"prefix_id": prefix_id, "error": f"{type(exc).__name__}: {exc}"})

    summary = build_summary(
        collector=collector,
        args=args,
        clustering_config=clustering_config,
        processed_prefix_ids=processed_prefix_ids,
        errors=errors,
        beta_e=beta_e,
        beta_a=beta_a,
    )
    save_json(summary, args.output_dir / "summary.json")

    logger.info(
        "Split seed study complete: recorded %d/%d events across %d prefixes",
        len(collector.records),
        args.max_events,
        len(processed_prefix_ids),
    )


if __name__ == "__main__":
    main()
