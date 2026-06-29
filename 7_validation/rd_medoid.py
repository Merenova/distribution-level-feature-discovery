#!/usr/bin/env -S uv run python
"""Canonical paper runner for RD medoid steering."""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper RD steering with combined-distance medoids")
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--attribution-graphs-dir", type=Path, required=True)
    parser.add_argument("--clustering-dir", type=Path, required=True)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prefix-id", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--transcoder", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-cluster-samples", type=int, default=None)
    parser.add_argument("--max-batch-size", type=int, default=None)
    parser.add_argument("--cross-prefix-batching", action="store_true")
    parser.add_argument("--prefix-batch-size", type=int, default=None)
    parser.add_argument("--use-weights", action="store_true")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--K-clamp", type=int, default=None)
    parser.add_argument("--beta-values", type=float, nargs="+", default=None)
    parser.add_argument("--gamma-values", type=float, nargs="+", default=None)
    parser.add_argument("--clustering-manifest", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(Path(__file__).with_name("7c_baseline_combined_medoid.py")),
        "--samples-dir", str(args.samples_dir),
        "--attribution-graphs-dir", str(args.attribution_graphs_dir),
        "--clustering-dir", str(args.clustering_dir),
        "--embeddings-dir", str(args.embeddings_dir),
        "--output-dir", str(args.output_dir),
        "--config", str(args.config),
    ]
    if args.prefix_id:
        cmd.extend(["--prefix-id", args.prefix_id])
    if args.model:
        cmd.extend(["--model", args.model])
    if args.transcoder:
        cmd.extend(["--transcoder", args.transcoder])
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.max_cluster_samples is not None:
        cmd.extend(["--max-cluster-samples", str(args.max_cluster_samples)])
    if args.max_batch_size is not None:
        cmd.extend(["--max-batch-size", str(args.max_batch_size)])
    if args.cross_prefix_batching:
        cmd.append("--cross-prefix-batching")
    if args.prefix_batch_size is not None:
        cmd.extend(["--prefix-batch-size", str(args.prefix_batch_size)])
    if args.use_weights:
        cmd.append("--use-weights")
    if args.log_dir:
        cmd.extend(["--log-dir", str(args.log_dir)])
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.K_clamp is not None:
        cmd.extend(["--K-clamp", str(args.K_clamp)])
    if args.beta_values:
        cmd.extend(["--beta-values", *[str(value) for value in args.beta_values]])
    if args.gamma_values:
        cmd.extend(["--gamma-values", *[str(value) for value in args.gamma_values]])
    if args.clustering_manifest:
        cmd.extend(["--clustering-manifest", str(args.clustering_manifest)])
    if args.quiet:
        cmd.append("--quiet")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
