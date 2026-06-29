#!/usr/bin/env -S uv run python
"""Render a compact dashboard for MMLU pipeline progress."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


BAR_WIDTH = 19


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return "--"
    total_seconds = max(0, int(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def format_count(value: int) -> str:
    return f"{value:,}"


def render_bar(done: int, total: int, status: str) -> tuple[str, int]:
    if total <= 0:
        pct = 100 if status in {"completed", "skipped"} else 0
    elif status in {"completed", "skipped"}:
        pct = 100
    else:
        pct = max(0, min(100, int(round((done / total) * 100))))

    filled = int(round((pct / 100) * BAR_WIDTH))
    filled = max(0, min(BAR_WIDTH, filled))
    return f"[{'█' * filled}{'░' * (BAR_WIDTH - filled)}]", pct


def count_stage_outputs(output_dir: Path, count_spec: dict[str, Any]) -> int:
    mode = count_spec.get("mode")
    if mode == "file":
        rel_path = count_spec.get("path")
        if not rel_path:
            return 0
        return 1 if (output_dir / rel_path).exists() else 0

    if mode == "comparison_manifest":
        manifest_rel = count_spec.get("manifest_path")
        result_dir_rel = count_spec.get("result_dir")
        if not manifest_rel or not result_dir_rel:
            return 0
        try:
            from multiview_comparison.stage7_distribution import count_completed_manifest_entries
        except Exception:
            return 0
        return count_completed_manifest_entries(
            output_dir / manifest_rel,
            output_dir / result_dir_rel,
            hypothesis=str(count_spec.get("hypothesis", "H4A")),
            required_result_keys=count_spec.get("required_result_keys") or [],
        )

    if mode == "comparison_prefix_list":
        assignment_rel = count_spec.get("assignment_path")
        result_dir_rel = count_spec.get("result_dir")
        if not assignment_rel or not result_dir_rel:
            return 0
        try:
            from multiview_comparison.stage7_distribution import count_completed_prefix_outputs
        except Exception:
            return 0
        return count_completed_prefix_outputs(
            output_dir / assignment_rel,
            output_dir / result_dir_rel,
        )

    root = output_dir / count_spec.get("root", "")
    if not root.exists():
        return 0

    pattern = count_spec.get("pattern", "*")
    if count_spec.get("recursive", False):
        iterator = root.rglob(pattern)
    else:
        iterator = root.glob(pattern)
    return sum(1 for path in iterator if path.is_file())


def compute_eta(done: int, total: int, elapsed_seconds: float) -> float | None:
    if total <= 0 or done <= 0 or elapsed_seconds <= 0 or done >= total:
        return None
    rate = done / elapsed_seconds
    if rate <= 0:
        return None
    return (total - done) / rate


def render_model(output_dir: Path, now: float, progress_subdir: str) -> list[str]:
    progress_dir = output_dir / progress_subdir
    state = load_json(progress_dir / "state.json") or {}
    meta = load_json(progress_dir / "meta.json") or {}

    model_tag = state.get("model_tag") or output_dir.name
    gpu_id = state.get("gpu_id")
    state_name = state.get("state", "pending")
    started_at = state.get("started_at")
    ended_at = state.get("ended_at")

    lines = []
    header = f"{model_tag}"
    if gpu_id not in (None, ""):
        header += f"   GPU: {gpu_id}"
    header += f"   State: {state_name}"
    lines.append(header)

    stages = meta.get("stages", [])
    if not stages:
        elapsed = format_duration((ended_at or now) - started_at) if started_at else "0m 00s"
        lines.append(f"Overall: [░░░░░░░░░░░░░░░░░░░]   0%   Elapsed: {elapsed}   ETA: --   Files: 0/0")
        lines.append("Waiting for progress metadata...")
        return lines

    stage_snapshots = []
    for stage in stages:
        stage_state = load_json(progress_dir / "stages" / f"{stage['key']}.json") or {}
        status = stage_state.get("status", "pending")
        total_units = int(stage.get("total_units", 0))
        observed_done = count_stage_outputs(output_dir, stage.get("count", {}))
        done_units = total_units if status in {"completed", "skipped"} else min(observed_done, total_units)

        start_epoch = stage_state.get("start_epoch")
        end_epoch = stage_state.get("end_epoch")
        if start_epoch is None:
            elapsed_seconds = 0
        else:
            elapsed_seconds = (end_epoch or now) - start_epoch

        eta_seconds = None
        if status == "running":
            eta_seconds = compute_eta(done_units, total_units, elapsed_seconds)

        stage_snapshots.append(
            {
                "stage": stage,
                "status": status,
                "done_units": done_units,
                "display_done": observed_done if status == "running" else done_units,
                "total_units": total_units,
                "elapsed_seconds": elapsed_seconds,
                "eta_seconds": eta_seconds,
            }
        )

    overall_total = sum(snapshot["total_units"] for snapshot in stage_snapshots)
    overall_done = sum(snapshot["done_units"] for snapshot in stage_snapshots)
    overall_elapsed = ((ended_at or now) - started_at) if started_at else 0
    overall_eta = None
    if state_name not in {"completed", "failed"}:
        overall_eta = compute_eta(overall_done, overall_total, overall_elapsed)

    overall_bar, overall_pct = render_bar(overall_done, overall_total, "completed" if state_name == "completed" else "running")
    overall_line = (
        f"Overall: {overall_bar} {overall_pct:>3}%   "
        f"Elapsed: {format_duration(overall_elapsed)}   "
        f"ETA: {format_duration(overall_eta)}   "
        f"Files: {format_count(overall_done)}/{format_count(overall_total)}"
    )
    lines.append(overall_line)
    lines.append("")

    total_stage_count = len(stage_snapshots)
    for index, snapshot in enumerate(stage_snapshots, start=1):
        stage = snapshot["stage"]
        status = snapshot["status"]
        done_units = snapshot["done_units"]
        display_done = snapshot["display_done"]
        total_units = snapshot["total_units"]
        bar, pct = render_bar(done_units, total_units, status)
        line = (
            f"[{index}/{total_stage_count}] {stage['name']:<18} "
            f"{bar} {pct:>3}%   "
            f"Time: {format_duration(snapshot['elapsed_seconds'])}"
        )
        if status == "running":
            line += f"   ETA: {format_duration(snapshot['eta_seconds'])}"
        line += f"   Files: {format_count(display_done)}/{format_count(total_units)}"
        if status == "running":
            line += "   < active"
        elif status == "failed":
            line += "   < failed"
        elif status == "skipped":
            line += "   < skipped"
        lines.append(line)

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Render MMLU pipeline progress")
    parser.add_argument("--output-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--progress-subdir", default="progress")
    args = parser.parse_args()

    now = time.time()
    all_lines = []
    for output_dir in args.output_dirs:
        all_lines.extend(render_model(output_dir, now, args.progress_subdir))
        all_lines.append("")

    if all_lines:
        while all_lines and not all_lines[-1]:
            all_lines.pop()
    print("\n".join(all_lines))


if __name__ == "__main__":
    main()
