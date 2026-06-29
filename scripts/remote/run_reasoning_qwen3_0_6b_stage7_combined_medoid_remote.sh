#!/usr/bin/env bash
# Run Stage 7 combined-medoid validation for completed Qwen3-0.6B reasoning lanes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_REL="${CONFIG_REL:-configs/reasoning_qwen3_0_6b_stage7_combined_medoid.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/reasoning_runs}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/reasoning_qwen3_0_6b_stage7_combined_medoid_$(date +%Y%m%d_%H%M%S)}"
HF_ENV_FILE="${HF_ENV_FILE:-/root/.latent_planning/hf_env}"
ONLY="${ONLY:-}"
DRY_RUN="false"

usage() {
  cat <<'EOF'
Usage: bash scripts/remote/run_reasoning_qwen3_0_6b_stage7_combined_medoid_remote.sh [options]

Options:
  --config PATH       Stage 7 JSON config relative to repo root
  --output-root DIR   Reasoning output root
  --log-dir DIR       Directory for per-lane logs
  --only LIST         Comma-separated lanes: gsm8k,math500
  --dry-run           Print commands without running
  --help              Show this help

Runs only 7_validation/7c_baseline_combined_medoid.py for:
  qwen3_0_6b:gsm8k   on CUDA 0
  qwen3_0_6b:math500 on CUDA 1

The config is fixed to h_c_selection=full and top_B=5.
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
    --only)
      ONLY="$2"
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

if [[ -f "$HF_ENV_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$HF_ENV_FILE"
fi

mkdir -p "$LOG_DIR"

if ! config_exports="$(
  uv run python - "$CONFIG_REL" <<'PY'
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as handle:
    config = json.load(handle)

model = config.get("model", {})
clustering = config.get("clustering", {})
steering = config.get("stage_7c_steering", {})
sweeps = steering.get("sweeps", [])
if len(sweeps) != 1:
    raise SystemExit("Stage 7 combined-medoid config must contain exactly one sweep")
sweep = sweeps[0]
if sweep.get("h_c_selections") != ["full"]:
    raise SystemExit("Stage 7 combined-medoid config must use h_c_selections=[full]")
if sweep.get("top_B") != [5]:
    raise SystemExit("Stage 7 combined-medoid config must use top_B=[5]")

def emit(name: str, value: object) -> None:
    print(f"{name}={shlex.quote(str(value))}")

def bool_text(value: object) -> str:
    return "true" if bool(value) else "false"

emit("MODEL_NAME", model.get("base_model", "Qwen/Qwen3-0.6B"))
emit("TRANSCODER", model.get("transcoder", "mwhanna/qwen3-0.6b-transcoders-lowl0"))
emit("POOLING", clustering.get("pooling", "mean"))
emit("K_CLAMP", steering.get("K_clamp", clustering.get("K_clamp", 10)))
emit("MAX_CLUSTER_SAMPLES", steering.get("max_cluster_samples", 10))
emit("MAX_BATCH_SIZE", steering.get("max_batch_size", 16))
emit("PREFIX_BATCH_SIZE", steering.get("prefix_batch_size", 1))
emit("CROSS_PREFIX_BATCHING", bool_text(steering.get("cross_prefix_batching", False)))
emit("SKIP_EXISTING", bool_text(clustering.get("skip_existing", True)))
emit("STEERING_METHOD", sweep.get("steering_method", "sign"))
emit("HC_SELECTIONS", ",".join(sweep.get("h_c_selections", [])))
emit("TOP_B", ",".join(str(x) for x in sweep.get("top_B", [])))
PY
)"; then
  echo "Error: failed to validate Stage 7 config: $CONFIG_REL" >&2
  exit 1
fi
eval "$config_exports"

declare -A LANE_TO_DIR=(
  [gsm8k]="qwen3_0_6b_gsm8k"
  [math500]="qwen3_0_6b_math500"
)
declare -A LANE_TO_GPU=(
  [gsm8k]="0"
  [math500]="1"
)

lanes=(gsm8k math500)
if [[ -n "$ONLY" ]]; then
  lanes=()
  remaining="$ONLY"
  while true; do
    if [[ "$remaining" == *,* ]]; then
      lanes+=("${remaining%%,*}")
      remaining="${remaining#*,}"
    else
      lanes+=("$remaining")
      break
    fi
  done
fi

run_lane() {
  local lane="$1"
  local lane_dir="${LANE_TO_DIR[$lane]:-}"
  local gpu="${LANE_TO_GPU[$lane]:-}"
  if [[ -z "$lane_dir" || -z "$gpu" ]]; then
    echo "Error: unknown lane: $lane" >&2
    return 1
  fi

  local root="$OUTPUT_ROOT/$lane_dir"
  local log_file="$LOG_DIR/${lane_dir}.log"
  local cmd=(
    uv run python 7_validation/7c_baseline_combined_medoid.py
    --samples-dir "$root/2_reasoning_pair_samples"
    --attribution-graphs-dir "$root/3_attribution_graphs"
    --clustering-dir "$root/5_gaussian_clustering"
    --embeddings-dir "$root/4_feature_extraction/embeddings"
    --output-dir "$root/7_validation/7c_combined_medoid"
    --config "$CONFIG_REL"
    --model "$MODEL_NAME"
    --transcoder "$TRANSCODER"
    --pooling "$POOLING"
    --K-clamp "$K_CLAMP"
    --max-cluster-samples "$MAX_CLUSTER_SAMPLES"
    --max-batch-size "$MAX_BATCH_SIZE"
    --beta-values 0.75
    --gamma-values 0.7
  )
  if [[ "$CROSS_PREFIX_BATCHING" == "true" ]]; then
    cmd+=(--cross-prefix-batching --prefix-batch-size "$PREFIX_BATCH_SIZE")
  fi
  if [[ "$SKIP_EXISTING" == "true" ]]; then
    cmd+=(--skip-existing)
  fi

  {
    echo "=== $lane_dir Stage 7 combined-medoid on CUDA_VISIBLE_DEVICES=$gpu ==="
    echo "Config: $CONFIG_REL"
    echo "Model: $MODEL_NAME"
    echo "Transcoder: $TRANSCODER"
    echo "Pooling: $POOLING"
    echo "K clamp: $K_CLAMP"
    echo "Stage 5 filter: beta=0.75 gamma=0.7"
    echo "Sweep: method=$STEERING_METHOD h_c=$HC_SELECTIONS top_B=$TOP_B"
    printf 'Command: CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    printf '%q ' "${cmd[@]}"
    printf '\n'

    if [[ "$DRY_RUN" != "true" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}"
    fi
  } >"$log_file" 2>&1
}

job_pids=()
job_names=()
for lane in "${lanes[@]}"; do
  run_lane "$lane" &
  job_pids+=("$!")
  job_names+=("$lane")
done

status=0
for index in "${!job_pids[@]}"; do
  if ! wait "${job_pids[$index]}"; then
    echo "Error: Stage 7 combined-medoid ${job_names[$index]} failed" >&2
    status=1
  fi
done

echo "Logs: $LOG_DIR"
exit "$status"
