#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config import load_config


COMMON_REQUIRED_FILES = {
    ".gitignore",
    ".python-version",
    "README.md",
    "pyproject.toml",
    "uv.lock",
    "utils/__init__.py",
    "utils/config.py",
    "utils/data_utils.py",
    "utils/logging_utils.py",
    "utils/memory_utils.py",
    "scripts/run_paper_pipeline.sh",
    "scripts/run_integrity_compare.sh",
    "scripts/compare_integrity_artifacts.py",
    "scripts/audit/necessity_inventory.py",
    "inputs/prefixes.example.json",
    "5_gaussian_clustering/README.md",
    "5_gaussian_clustering/__init__.py",
    "5_gaussian_clustering/adaptive_control.py",
    "5_gaussian_clustering/cluster.py",
    "5_gaussian_clustering/em_loop.py",
    "5_gaussian_clustering/gpu_utils.py",
    "5_gaussian_clustering/initialize.py",
    "5_gaussian_clustering/rd_objective.py",
    "5_gaussian_clustering/sweep_utils.py",
    "6_semantic_graphs/extract_graphs.py",
    "7_validation/7c_baseline_combined_medoid.py",
    "7_validation/7c_baseline_kmeans.py",
    "7_validation/7c_baseline_single.py",
    "7_validation/7c_graph.py",
    "7_validation/7c_hypotheses.py",
    "7_validation/7c_metrics.py",
    "7_validation/7c_steering.py",
    "7_validation/7c_utils.py",
    "7_validation/analyze_steering_methods.py",
    "7_validation/km_sem.py",
    "7_validation/rd_medoid.py",
    "7_validation/single.py",
}

ADAPTER_FILES = {
    "ambigqa": {
        "0_preprocess/prepare_ambigqa_questions.py",
        "1_data_preparation/format_ambigqa_questions.py",
        "2_branch_sampling/sample_branches.py",
        "3_attribution_graphs/compute_continuation_attribution.py",
        "4_feature_extraction/compute_embeddings.py",
    }
}

GENERATED_PREFIXES = (
    ".git/",
    ".venv/",
    ".pytest_cache/",
    "results/",
    "logs/",
    "output/",
    "tmp/",
)

GENERATED_SUFFIXES = (
    ".pyc",
    ".egg-info",
)


def git_tracked_files(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return sorted(line for line in result.stdout.splitlines() if line)


def is_generated_path(path: str) -> bool:
    return (
        any(path.startswith(prefix) for prefix in GENERATED_PREFIXES)
        or any(path.endswith(suffix) for suffix in GENERATED_SUFFIXES)
        or "/__pycache__/" in path
    )


def resolve_config_path(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def relative_config_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def config_dependency_paths(root: Path, config_path: Path, seen: set[Path] | None = None) -> list[Path]:
    resolved_path = resolve_config_path(root, config_path).resolve()
    seen_paths = set() if seen is None else set(seen)
    if resolved_path in seen_paths:
        cycle = " -> ".join(str(path) for path in [*seen_paths, resolved_path])
        raise ValueError(f"Config extends cycle detected: {cycle}")
    seen_paths.add(resolved_path)

    with resolved_path.open("r", encoding="utf-8") as file:
        raw_config = json.load(file)
    if not isinstance(raw_config, dict):
        raise ValueError(f"Config file must contain a JSON object: {resolved_path}")

    extends = raw_config.get("extends", [])
    if isinstance(extends, str):
        extend_paths = [extends]
    elif isinstance(extends, list):
        extend_paths = extends
    else:
        raise ValueError(f"Config extends must be a string or list: {resolved_path}")

    dependencies = [resolved_path]
    for extend_path in extend_paths:
        if not isinstance(extend_path, str):
            raise ValueError(f"Config extends entries must be strings: {resolved_path}")
        dependencies.extend(config_dependency_paths(root, resolved_path.parent / extend_path, seen_paths))
    return dependencies


def build_inventory(root: Path, config_paths: list[Path]) -> dict[str, Any]:
    required = set(COMMON_REQUIRED_FILES)
    required_configs: list[str] = []
    unsupported_presets: list[dict[str, str]] = []

    for config_path in config_paths:
        resolved_config = resolve_config_path(root, config_path)
        rel_config = relative_config_path(root, resolved_config)
        config = load_config(resolved_config)
        for dependency_path in config_dependency_paths(root, resolved_config):
            required.add(relative_config_path(root, dependency_path))
        required_configs.append(rel_config)

        adapter = str(config.get("data", {}).get("adapter", "ambigqa"))
        adapter_files = ADAPTER_FILES.get(adapter)
        if adapter_files is None:
            unsupported_presets.append({"config": rel_config, "adapter": adapter})
            continue
        required.update(adapter_files)

    tracked = git_tracked_files(root)
    unclassified = [
        path
        for path in tracked
        if path not in required
        and not path.startswith("docs/")
        and not path.startswith("tests/")
        and not path.startswith("circuit-tracer/")
        and not is_generated_path(path)
    ]

    return {
        "presets": required_configs,
        "required_files": sorted(required),
        "unsupported_presets": unsupported_presets,
        "unclassified_tracked_files": sorted(unclassified),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Explain which files are required by retained reproduction presets")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        dest="configs",
        default=[],
        help="Config or preset to include. May be passed multiple times.",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    configs = args.configs or [args.root / "configs/default.json"]
    inventory = build_inventory(args.root, configs)
    text = json.dumps(inventory, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")
    print(text)
    return 1 if inventory["unsupported_presets"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
