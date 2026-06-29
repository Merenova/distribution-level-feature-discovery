#!/usr/bin/env bash
# Resume Qwen3-0.6B reasoning runs from existing Stage 4 embeddings and run Stage 5 only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/reasoning_runs}"
CONFIG_REL="${CONFIG_REL:-configs/reasoning_qwen3_small.yaml}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/reasoning_qwen3_0_6b_stage5_single_$(date +%Y%m%d_%H%M%S)}"
PREFIX_WORKERS="${PREFIX_WORKERS:-4}"
DRY_RUN="false"

usage() {
  cat <<'EOF'
Usage: bash scripts/remote/run_reasoning_qwen3_0_6b_stage5_single_remote.sh [options]

Options:
  --config PATH       Config path relative to repo root
  --output-root DIR   Sweep output root
  --log-dir DIR       Directory for per-lane logs
  --workers N         Prefix workers per lane (default: 4)
  --dry-run           Print commands without running
  --help              Show this help

This resumes only Stage 5 for:
  qwen3_0_6b:gsm8k
  qwen3_0_6b:math500

It expects existing:
  2_reasoning_pair_samples/
  3_attribution_graphs/
  4_feature_extraction/embeddings/
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_REL="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --workers)
      PREFIX_WORKERS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "$LOG_DIR"

if ! config_exports="$(
  uv run python - "$CONFIG_REL" "$OUTPUT_ROOT" <<'PY'
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
output_root = Path(sys.argv[2])

with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

clustering = dict(config.get("clustering", {}))
for key in ("beta_values", "gamma_values", "sweeps"):
    clustering.pop(key, None)

if "beta" not in clustering or "gamma" not in clustering:
    raise SystemExit("clustering.beta and clustering.gamma are required for single-config Stage 5")

runtime = {"clustering": clustering}
for lane in ("qwen3_0_6b_gsm8k", "qwen3_0_6b_math500"):
    path = output_root / lane / "runtime_clustering_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(runtime, handle, indent=2)
        handle.write("\n")

def emit(name: str, value: object) -> None:
    print(f"{name}={shlex.quote(str(value))}")

def bool_text(value: object) -> str:
    return "true" if bool(value) else "false"

emit("BETA", clustering["beta"])
emit("GAMMA", clustering["gamma"])
emit("POOLING", clustering.get("pooling", "mean"))
emit("ATTRIBUTION_METRIC", clustering.get("attribution_metric", "l1"))
emit("NORMALIZE_DIMS", bool_text(clustering.get("normalize_dims", False)))
emit("SAVE_INTERMEDIATE", bool_text(clustering.get("save_intermediate", False)))
emit("SKIP_EXISTING", bool_text(clustering.get("skip_existing", True)))
PY
)"; then
  echo "Error: failed to prepare single-config runtime clustering config" >&2
  exit 1
fi
eval "$config_exports"

run_lane() {
  local lane="$1"
  local gpu="$2"
  local root="$OUTPUT_ROOT/$lane"
  local log_file="$LOG_DIR/${lane}.log"

  local clustering_cmd=(
    uv run python 5_gaussian_clustering/cluster.py
    --samples-dir "$root/2_reasoning_pair_samples"
    --attribution-graphs-dir "$root/3_attribution_graphs"
    --embeddings-dir "$root/4_feature_extraction/embeddings"
    --output-dir "$root/5_gaussian_clustering"
    --config "$root/runtime_clustering_config.json"
    --pooling "$POOLING"
    --attribution-metric "$ATTRIBUTION_METRIC"
    --n-workers "$PREFIX_WORKERS"
  )

  if [[ "$SAVE_INTERMEDIATE" == "true" ]]; then
    clustering_cmd+=(--save-intermediate)
  fi
  if [[ "$NORMALIZE_DIMS" == "true" ]]; then
    clustering_cmd+=(--normalize-dims)
  fi
  if [[ "$SKIP_EXISTING" == "true" ]]; then
    clustering_cmd+=(--skip-existing)
  fi

  {
    echo "=== $lane Stage 5 single-config on CUDA_VISIBLE_DEVICES=$gpu ==="
    echo "Beta: $BETA"
    echo "Gamma: $GAMMA"
    echo "Prefix workers: $PREFIX_WORKERS"
    printf 'Command: CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    printf '%q ' "${clustering_cmd[@]}"
    printf '\n'

    if [[ "$DRY_RUN" != "true" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" "${clustering_cmd[@]}"
    fi
  } >"$log_file" 2>&1
}

run_lane qwen3_0_6b_gsm8k 0 &
pid_gsm8k="$!"
run_lane qwen3_0_6b_math500 1 &
pid_math500="$!"

status=0
wait "$pid_gsm8k" || status=1
wait "$pid_math500" || status=1

echo "Logs: $LOG_DIR"
exit "$status"
