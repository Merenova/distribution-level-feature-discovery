#!/usr/bin/env -S uv run python
"""Canonical paper runner for the single-continuation steering baseline."""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper Single steering with one continuation per RD cluster")
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--attribution-graphs-dir", type=Path, required=True)
    parser.add_argument("--clustering-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prefix-id", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("7c_baseline_single.py")),
        "--samples-dir", str(args.samples_dir),
        "--attribution-graphs-dir", str(args.attribution_graphs_dir),
        "--clustering-dir", str(args.clustering_dir),
        "--output-dir", str(args.output_dir),
        "--config", str(args.config),
    ]
    if args.prefix_id:
        cmd.extend(["--prefix-id", args.prefix_id])
    if args.quiet:
        cmd.append("--quiet")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
