#!/usr/bin/env -S uv run python
"""Build a deterministic H4C clustering manifest balanced across K buckets."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data_utils import load_json, save_json


def _is_valid_grid_entry(entry: Dict) -> bool:
    return bool(entry.get("components")) and bool(entry.get("assignments")) and "error" not in entry


def build_manifest(clustering_dir: Path, k_values: List[int], per_k: int) -> List[Dict]:
    buckets = {int(k): [] for k in k_values}
    target_k = set(buckets.keys())

    sweep_files = sorted(clustering_dir.glob("*_sweep_results.json"))
    for sweep_file in sweep_files:
        prefix_id = sweep_file.stem.replace("_sweep_results", "")
        clustering_sweep = load_json(sweep_file)

        for entry in clustering_sweep.get("grid", []):
            if not _is_valid_grid_entry(entry):
                continue

            K = int(entry.get("K", len(entry.get("components", {}))))
            if K not in target_k:
                continue
            if len(buckets[K]) >= per_k:
                continue

            beta = entry.get("beta")
            gamma = entry.get("gamma")
            if beta is None or gamma is None:
                continue

            buckets[K].append(
                {
                    "prefix_id": prefix_id,
                    "beta": float(beta),
                    "gamma": float(gamma),
                    "K": K,
                }
            )

            if all(len(bucket) >= per_k for bucket in buckets.values()):
                manifest = []
                for k in k_values:
                    manifest.extend(buckets[int(k)])
                return manifest

    manifest = []
    for k in k_values:
        manifest.extend(buckets[int(k)])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced H4C clustering manifest")
    parser.add_argument("--clustering-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k-values", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--per-k", type=int, default=10)
    args = parser.parse_args()

    manifest = build_manifest(args.clustering_dir, args.k_values, args.per_k)
    counts = {int(k): 0 for k in args.k_values}
    for entry in manifest:
        counts[int(entry["K"])] = counts.get(int(entry["K"]), 0) + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(manifest, args.output)

    expected_total = len(args.k_values) * args.per_k
    actual_total = len(manifest)
    print(json.dumps(
        {
            "output": str(args.output),
            "expected_total": expected_total,
            "actual_total": actual_total,
            "counts_by_K": counts,
        },
        indent=2,
    ))

    if any(counts.get(int(k), 0) < args.per_k for k in args.k_values):
        raise SystemExit("Insufficient clusterings to satisfy requested manifest")


if __name__ == "__main__":
    main()
