#!/usr/bin/env bash
# Run the Qwen3 small reasoning pipeline sweep across supported model/dataset pairs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$SCRIPT_DIR/run_reasoning_qwen_pipeline.sh"

CONFIG="${REASONING_CONFIG:-configs/reasoning_qwen3_small.yaml}"
OUTPUT_ROOT="${REASONING_OUTPUT_ROOT:-experiments/reasoning_runs}"
POOLING_OVERRIDE="${REASONING_POOLING:-}"
ONLY=""
DRY_RUN="false"

usage() {
  cat <<EOF
Usage: bash scripts/pipeline/run_reasoning_qwen3_small_sweep.sh [options]

Options:
  --config PATH       Reasoning YAML config (default: REASONING_CONFIG or configs/reasoning_qwen3_small.yaml)
  --output-root DIR   Output root for sweep runs (default: REASONING_OUTPUT_ROOT or experiments/reasoning_runs)
  --pooling NAME      Override config clustering.pooling for every run: mean, sum, or max
  --only LIST         Comma-separated model_key:dataset list to run
  --dry-run           Pass --dry-run to the underlying runner
  --help, -h          Show this help

Supported combinations:
  qwen3_0_6b:gsm8k
  qwen3_0_6b:math500
  qwen3_1_7b:gsm8k
  qwen3_1_7b:math500
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

require_value() {
  local option="$1"
  if [[ $# -lt 2 || -z "${2:-}" || "${2:-}" == --* ]]; then
    die "$option requires a value"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      require_value "$1" "${2:-}"
      CONFIG="$2"
      shift 2
      ;;
    --output-root)
      require_value "$1" "${2:-}"
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --pooling)
      require_value "$1" "${2:-}"
      case "$2" in
        mean|sum|max) POOLING_OVERRIDE="$2" ;;
        *) die "--pooling must be one of: mean, sum, max" ;;
      esac
      shift 2
      ;;
    --only)
      require_value "$1" "${2:-}"
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

declare -A MODEL_NAMES=(
  [qwen3_0_6b]="Qwen/Qwen3-0.6B"
  [qwen3_1_7b]="Qwen/Qwen3-1.7B"
)
MODEL_KEYS=(qwen3_0_6b qwen3_1_7b)

declare -A TRANSCODERS=(
  [qwen3_0_6b]="mwhanna/qwen3-0.6b-transcoders-lowl0"
  [qwen3_1_7b]="mwhanna/qwen3-1.7b-transcoders-lowl0"
)

declare -A DATASETS=(
  [gsm8k]="1"
  [math500]="1"
)
DATASET_KEYS=(gsm8k math500)

declare -A DATASET_GPUS=(
  [gsm8k]="0"
  [math500]="1"
)

COMBINATIONS=(
  "qwen3_0_6b:gsm8k"
  "qwen3_0_6b:math500"
  "qwen3_1_7b:gsm8k"
  "qwen3_1_7b:math500"
)

validate_combination() {
  local combo="$1"
  local model_key="${combo%%:*}"
  local dataset=""

  if [[ -z "$combo" ]]; then
    die "--only contains an empty entry"
  fi
  if [[ -z "$model_key" ]]; then
    die "--only entry has an empty model key: $combo"
  fi
  if [[ "$combo" == *:* ]]; then
    dataset="${combo#*:}"
  fi
  if [[ -z "$dataset" ]]; then
    die "--only entry has an empty dataset: $combo"
  fi

  if [[ -z "${MODEL_NAMES[$model_key]+set}" ]]; then
    die "unknown model key in --only: $model_key"
  fi
  if [[ -z "${DATASETS[$dataset]+set}" ]]; then
    die "unknown dataset in --only: $dataset"
  fi
}

if [[ -n "$ONLY" ]]; then
  requested_combinations=()
  remaining_only="$ONLY"
  while true; do
    if [[ "$remaining_only" == *,* ]]; then
      requested_combinations+=("${remaining_only%%,*}")
      remaining_only="${remaining_only#*,}"
    else
      requested_combinations+=("$remaining_only")
      break
    fi
  done

  if [[ "${#requested_combinations[@]}" -eq 0 ]]; then
    die "--only requires at least one combination"
  fi
  for combo in "${requested_combinations[@]}"; do
    validate_combination "$combo"
  done
  COMBINATIONS=("${requested_combinations[@]}")
fi

cd "$ROOT_DIR"

declare -A SELECTED_COMBINATIONS=()
for combo in "${COMBINATIONS[@]}"; do
  SELECTED_COMBINATIONS[$combo]="1"
done

for model_key in "${MODEL_KEYS[@]}"; do
  job_pids=()
  job_names=()

  for dataset in "${DATASET_KEYS[@]}"; do
    combo="${model_key}:${dataset}"
    if [[ -z "${SELECTED_COMBINATIONS[$combo]+set}" ]]; then
      continue
    fi

    output_dir="$OUTPUT_ROOT/${model_key}_${dataset}"
    gpu_id="${DATASET_GPUS[$dataset]}"

    runner_cmd=(
      bash "$RUNNER"
      --config "$CONFIG"
      --model "${MODEL_NAMES[$model_key]}"
      --dataset "$dataset"
      --transcoder "${TRANSCODERS[$model_key]}"
      --output-root "$output_dir"
    )
    if [[ -n "$POOLING_OVERRIDE" ]]; then
      runner_cmd+=(--pooling "$POOLING_OVERRIDE")
    fi
    if [[ "$DRY_RUN" == "true" ]]; then
      runner_cmd+=(--dry-run)
    fi

    echo "=== reasoning ${combo} on CUDA_VISIBLE_DEVICES=${gpu_id} ==="
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu_id"
    printf '%q ' "${runner_cmd[@]}"
    printf '\n'
    CUDA_VISIBLE_DEVICES="$gpu_id" "${runner_cmd[@]}" &
    job_pids+=("$!")
    job_names+=("$combo")
  done

  model_status=0
  for index in "${!job_pids[@]}"; do
    if ! wait "${job_pids[$index]}"; then
      echo "Error: reasoning ${job_names[$index]} failed" >&2
      model_status=1
    fi
  done

  if [[ "$model_status" -ne 0 ]]; then
    exit "$model_status"
  fi
done
