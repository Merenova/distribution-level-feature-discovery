#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

source "$ROOT_DIR/scripts/misc/mmlu_remote_baseline_lib.sh"

MODEL_TAG=""
CONFIG_PATH=""
GPU_ID="0"
SKIP_EXISTING="true"
QUIET="false"
PROGRESS_SUBDIR="progress_remote_rd"
DISABLE_PROGRESS="${MMLU_DISABLE_PROGRESS:-false}"

usage() {
  cat <<EOF
Usage: bash scripts/remote/run_remote_mmlu_rd_worker.sh --model-tag TAG --config PATH [options]

Options:
  --model-tag TAG       Output directory name, e.g. MMLU_small4_Qwen3-8B
  --config PATH         Config path relative to repo root
  --gpu ID              GPU id for stage 7c_combined_medoid (default: 0)
  --skip-existing       Reuse existing outputs when present (default)
  --no-skip-existing    Recompute outputs even if they already exist
  --quiet               Pass --quiet to run_pipeline.sh
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
    --gpu)
      GPU_ID="$2"
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
LOGS_DIR="$OUTPUT_DIR/logs"
PROGRESS_DIR="$OUTPUT_DIR/$PROGRESS_SUBDIR"
CONFIG_ABS="$ROOT_DIR/$CONFIG_PATH"
MODEL_STARTED_AT="$(date +%s)"
STAGE_LOG="$LOGS_DIR/remote_stage7c_combined_medoid.log"

for required_dir in \
  "$RESULTS_DIR/2_branch_sampling" \
  "$RESULTS_DIR/3_attribution_graphs" \
  "$RESULTS_DIR/5_clustering"
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

write_model_state "$PROGRESS_DIR" "$MODEL_TAG" "$GPU_ID" "$OUTPUT_DIR" "$LOGS_DIR" "starting" "$MODEL_STARTED_AT" "" "" "" "" ""

uv run python - <<PY
import json
from pathlib import Path

progress_dir = Path(${PROGRESS_DIR@Q})
meta_path = progress_dir / "meta.json"
tmp_path = progress_dir / "meta.tmp"
results_dir = Path(${RESULTS_DIR@Q})
config_path = Path(${CONFIG_ABS@Q})
output_dir = Path(${OUTPUT_DIR@Q})

if (results_dir / "test_clozes.json").exists():
    with open(results_dir / "test_clozes.json") as f:
        test_clozes = json.load(f)
    total_samples = len(test_clozes.get("clozes", []))
else:
    total_samples = sum(1 for _ in (results_dir / "2_branch_sampling").glob("*_branches.json"))

with open(config_path) as f:
    config = json.load(f)

steering = config.get("stage_7c_steering", {})
hypotheses = steering.get("hypotheses") or ["H4a"]
hypothesis_count = max(1, len(hypotheses))

payload = {
    "model_tag": ${MODEL_TAG@Q},
    "gpu_id": ${GPU_ID@Q},
    "output_dir": str(output_dir),
    "total_samples": total_samples,
    "hypothesis_count": hypothesis_count,
    "stages": [
        {
            "key": "stage7c_combined_medoid",
            "name": "Combined Medoid Steering",
            "total_units": total_samples * hypothesis_count,
            "count": {
                "mode": "glob",
                "root": "results/7_validation/7c_combined_medoid",
                "pattern": "*_sweep_results.json",
                "recursive": True,
            },
        }
    ],
}

with open(tmp_path, "w") as f:
    json.dump(payload, f, indent=2)
tmp_path.replace(meta_path)
PY

load_runtime_vars "$CONFIG_ABS"

stage_start="$(date +%s)"
write_model_state "$PROGRESS_DIR" "$MODEL_TAG" "$GPU_ID" "$OUTPUT_DIR" "$LOGS_DIR" "running" "$MODEL_STARTED_AT" "" "" "" "" ""
write_stage_state "$PROGRESS_DIR" "stage7c_combined_medoid" "Combined Medoid Steering" "1" "running" "$stage_start" "" "$STAGE_LOG"

PIPELINE_CMD=(
  env CONFIG_FILE="$CONFIG_ABS"
  bash scripts/run_pipeline.sh
  --output_dir "$MODEL_TAG"
  --only 7c_combined_medoid
)
if [[ "$SKIP_EXISTING" == "true" ]]; then
  PIPELINE_CMD+=(--skip-existing)
fi
if [[ "$QUIET" == "true" ]]; then
  PIPELINE_CMD+=(--quiet)
fi

(
  if run_with_env "$GPU_ID" "${PIPELINE_CMD[@]}" >"$STAGE_LOG" 2>&1; then
    write_stage_state "$PROGRESS_DIR" "stage7c_combined_medoid" "Combined Medoid Steering" "1" "completed" "$stage_start" "$(date +%s)" "$STAGE_LOG"
    exit 0
  fi
  exit_code=$?
  write_stage_state "$PROGRESS_DIR" "stage7c_combined_medoid" "Combined Medoid Steering" "1" "failed" "$stage_start" "$(date +%s)" "$STAGE_LOG"
  exit "$exit_code"
) &

JOB_PIDS=("$!")

if [[ "$DISABLE_PROGRESS" != "true" ]]; then
  render_progress_dashboard "$PROGRESS_SUBDIR" "$OUTPUT_DIR"
fi

overall_exit_code=0
if ! wait "${JOB_PIDS[0]}"; then
  overall_exit_code=$?
fi

if [[ $overall_exit_code -ne 0 ]]; then
  write_model_state "$PROGRESS_DIR" "$MODEL_TAG" "$GPU_ID" "$OUTPUT_DIR" "$LOGS_DIR" "failed" "$MODEL_STARTED_AT" "$(date +%s)" "" "" "" "Remote RD steering failed"
  exit "$overall_exit_code"
fi

write_model_state "$PROGRESS_DIR" "$MODEL_TAG" "$GPU_ID" "$OUTPUT_DIR" "$LOGS_DIR" "completed" "$MODEL_STARTED_AT" "$(date +%s)" "" "" "" ""
