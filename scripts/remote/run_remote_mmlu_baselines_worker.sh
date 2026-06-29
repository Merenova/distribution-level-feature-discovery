#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/misc/mmlu_remote_baseline_lib.sh"

MODEL_TAG=""
CONFIG_PATH=""
GPU_SINGLE="0"
GPU_COMBINED="1"
SKIP_EXISTING="true"
QUIET="false"
PROGRESS_SUBDIR="progress_remote_baselines"
DISABLE_PROGRESS="${MMLU_DISABLE_PROGRESS:-false}"

usage() {
  cat <<EOF
Usage: bash scripts/remote/run_remote_mmlu_baselines_worker.sh --model-tag TAG --config PATH [options]

Options:
  --model-tag TAG       Output directory name, e.g. MMLU_Qwen3-8B
  --config PATH         Config path relative to repo root
  --gpu-single ID       GPU for single baseline (default: 0)
  --gpu-combined ID     GPU for combined medoid baseline (default: 1)
  --skip-existing       Reuse existing outputs when present (default)
  --no-skip-existing    Recompute outputs even if they already exist
  --quiet               Pass --quiet to the baseline scripts
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-tag)
      MODEL_TAG="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --gpu-single)
      GPU_SINGLE="$2"
      shift 2
      ;;
    --gpu-combined)
      GPU_COMBINED="$2"
      shift 2
      ;;
    --skip-existing)
      SKIP_EXISTING="true"
      shift
      ;;
    --no-skip-existing)
      SKIP_EXISTING="false"
      shift
      ;;
    --quiet)
      QUIET="true"
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

if [[ -z "$MODEL_TAG" || -z "$CONFIG_PATH" ]]; then
  usage >&2
  exit 1
fi

OUTPUT_DIR="$ROOT_DIR/$MODEL_TAG"
RESULTS_DIR="$OUTPUT_DIR/results"
LOGS_DIR="$OUTPUT_DIR/logs/remote_baselines"
PROGRESS_DIR="$OUTPUT_DIR/$PROGRESS_SUBDIR"
CONFIG_ABS="$ROOT_DIR/$CONFIG_PATH"
MODEL_STARTED_AT="$(date +%s)"
GPU_LABEL="${GPU_SINGLE},${GPU_COMBINED}"

for required_dir in \
  "$RESULTS_DIR/2_branch_sampling" \
  "$RESULTS_DIR/3_attribution_graphs" \
  "$RESULTS_DIR/5_clustering" \
  "$RESULTS_DIR/4_feature_extraction/embeddings"
do
  if [[ ! -d "$required_dir" ]]; then
    echo "Error: required input directory missing: $required_dir" >&2
    exit 1
  fi
done

if [[ ! -f "$CONFIG_ABS" ]]; then
  echo "Error: config file not found: $CONFIG_ABS" >&2
  exit 1
fi

mkdir -p "$LOGS_DIR"
rm -rf "$PROGRESS_DIR"
mkdir -p "$PROGRESS_DIR/stages"

write_model_state "$PROGRESS_DIR" "$MODEL_TAG" "$GPU_LABEL" "$OUTPUT_DIR" "$LOGS_DIR" "starting" "$MODEL_STARTED_AT" "" "" "" "" ""
write_remote_baseline_meta "$PROGRESS_DIR" "$MODEL_TAG" "$GPU_LABEL" "$OUTPUT_DIR" "$CONFIG_ABS" "$RESULTS_DIR"
load_runtime_vars "$CONFIG_ABS"

baseline_common_args=(
  --samples-dir "$RESULTS_DIR/2_branch_sampling"
  --attribution-graphs-dir "$RESULTS_DIR/3_attribution_graphs"
  --clustering-dir "$RESULTS_DIR/5_clustering"
  --config "$CONFIG_ABS"
  --max-batch-size "$STEERING_MAX_BATCH_SIZE"
  --prefix-batch-size "$STEERING_PREFIX_BATCH_SIZE"
  --log-dir "$LOGS_DIR"
)
if [[ "$STEERING_CROSS_PREFIX_BATCHING" == "true" ]]; then
  baseline_common_args+=(--cross-prefix-batching)
fi
if [[ "$SKIP_EXISTING" == "true" ]]; then
  baseline_common_args+=(--skip-existing)
fi
if [[ "$QUIET" == "true" ]]; then
  baseline_common_args+=(--quiet)
fi

run_stage_job() {
  local stage_key="$1"
  local stage_name="$2"
  local stage_index="$3"
  local gpu_id="$4"
  local log_file="$5"
  shift 5

  local stage_start
  stage_start="$(date +%s)"
  write_stage_state "$PROGRESS_DIR" "$stage_key" "$stage_name" "$stage_index" "running" "$stage_start" "" "$log_file"

  (
    if run_with_env "$gpu_id" "$@" >"$log_file" 2>&1; then
      write_stage_state "$PROGRESS_DIR" "$stage_key" "$stage_name" "$stage_index" "completed" "$stage_start" "$(date +%s)" "$log_file"
      exit 0
    fi

    local exit_code=$?
    write_stage_state "$PROGRESS_DIR" "$stage_key" "$stage_name" "$stage_index" "failed" "$stage_start" "$(date +%s)" "$log_file"
    exit "$exit_code"
  ) &

  JOB_PIDS+=("$!")
  JOB_STAGE_KEYS+=("$stage_key")
}

JOB_PIDS=()
JOB_STAGE_KEYS=()

write_model_state "$PROGRESS_DIR" "$MODEL_TAG" "$GPU_LABEL" "$OUTPUT_DIR" "$LOGS_DIR" "running" "$MODEL_STARTED_AT" "" "" "" "" ""

single_log="$LOGS_DIR/remote_stage_single.log"
if [[ "$SKIP_EXISTING" == "true" ]] && has_named_files "$RESULTS_DIR/7_validation/7c_baseline_single" "*_sweep_results.json"; then
  printf "Skipped by remote worker: single baseline already exists.\n" >"$single_log"
  mark_stage_skipped "$PROGRESS_DIR" "stage7c2_single" "Single Base" "1" "$single_log"
else
  run_stage_job \
    "stage7c2_single" "Single Base" "1" "$GPU_SINGLE" "$single_log" \
    uv run python 7_validation/7c_baseline_single.py \
    --output-dir "$RESULTS_DIR/7_validation/7c_baseline_single" \
    "${baseline_common_args[@]}"
fi

combined_log="$LOGS_DIR/remote_stage_combined.log"
if [[ "$SKIP_EXISTING" == "true" ]] && has_named_files "$RESULTS_DIR/7_validation/7c_baseline_combined_medoid" "*_sweep_results.json"; then
  printf "Skipped by remote worker: combined medoid baseline already exists.\n" >"$combined_log"
  mark_stage_skipped "$PROGRESS_DIR" "stage7c2_combined" "Combined Medoid" "2" "$combined_log"
else
  run_stage_job \
    "stage7c2_combined" "Combined Medoid" "2" "$GPU_COMBINED" "$combined_log" \
    uv run python 7_validation/7c_baseline_combined_medoid.py \
    --embeddings-dir "$RESULTS_DIR/4_feature_extraction/embeddings" \
    --output-dir "$RESULTS_DIR/7_validation/7c_baseline_combined_medoid" \
    "${baseline_common_args[@]}"
fi

if [[ ${#JOB_PIDS[@]} -eq 0 ]]; then
  write_model_state "$PROGRESS_DIR" "$MODEL_TAG" "$GPU_LABEL" "$OUTPUT_DIR" "$LOGS_DIR" "completed" "$MODEL_STARTED_AT" "$(date +%s)" "" "" "" ""
fi

if [[ "$DISABLE_PROGRESS" != "true" ]]; then
  render_progress_dashboard "$PROGRESS_SUBDIR" "$OUTPUT_DIR"
fi

overall_exit_code=0
for idx in "${!JOB_PIDS[@]}"; do
  if ! wait "${JOB_PIDS[$idx]}"; then
    overall_exit_code=1
  fi
done

if [[ $overall_exit_code -ne 0 ]]; then
  write_model_state "$PROGRESS_DIR" "$MODEL_TAG" "$GPU_LABEL" "$OUTPUT_DIR" "$LOGS_DIR" "failed" "$MODEL_STARTED_AT" "$(date +%s)" "" "" "" "One or more remote baseline jobs failed"
  exit "$overall_exit_code"
fi

write_model_state "$PROGRESS_DIR" "$MODEL_TAG" "$GPU_LABEL" "$OUTPUT_DIR" "$LOGS_DIR" "completed" "$MODEL_STARTED_AT" "$(date +%s)" "" "" "" ""
