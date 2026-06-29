#!/usr/bin/env bash
# Run the latent_planning pipeline on selected cais/mmlu subjects for Qwen3 models.
#
# This wrapper:
# 1. Pulls and snapshots the selected MMLU subset
# 2. Builds model-specific question-only prompts
# 3. Reuses the existing pipeline from Stage 2 onward
# 4. Runs the extra H4a baselines requested for Stage 7

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"
source "$ROOT_DIR/scripts/misc/mmlu_remote_baseline_lib.sh"

DEFAULT_SUBJECTS=(
  "logical_fallacies"
  "moral_disputes"
  "professional_psychology"
  "sociology"
  "philosophy"
  "jurisprudence"
  "international_law"
  "business_ethics"
)
DEFAULT_SUBJECTS_CSV="$(IFS=,; echo "${DEFAULT_SUBJECTS[*]}")"

MODELS="all"
SPLIT="validation"
OUTPUT_ROOT="$ROOT_DIR"
RAW_SAVE_DIR=""
DATASET_ID="cais/mmlu"
SUBJECTS_CSV="$DEFAULT_SUBJECTS_CSV"
RUN_TAG=""
DISTRIBUTED_STAGE7="false"
QUIET="false"
SKIP_EXISTING="false"
GPU_8B=""
GPU_4B=""
PARALLEL_MODE="false"

usage() {
  cat <<EOF
Usage: bash scripts/pipeline/run_mmlu_qwen_pipeline.sh [options]

Options:
  --models MODELS        One of: all, 8b, 4b (default: all)
  --split SPLIT          MMLU split to use (default: validation)
  --subjects CSV         Comma-separated MMLU subjects (default: current 8-subject MMLU subset)
  --run-tag TAG          Optional tag added to output dirs, e.g. small4 -> MMLU_small4_Qwen3-8B
  --distributed-stage7   Run K-means locally and Single/Combined on the designated remote server
  --output-root DIR      Parent directory for model outputs (default: repo root)
  --raw-save-dir DIR     Local merged MMLU snapshot directory
  --dataset-id ID        Hugging Face dataset id (default: cais/mmlu)
  --gpu-8b ID            CUDA_VISIBLE_DEVICES value for Qwen3-8B runs
  --gpu-4b ID            CUDA_VISIBLE_DEVICES value for Qwen3-4B runs
  --quiet                Reduce script output and pass --quiet to python entrypoints
  --skip-existing        Reuse matching prep outputs and existing stage results where possible
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      MODELS="$2"
      shift 2
      ;;
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --subjects)
      SUBJECTS_CSV="$2"
      shift 2
      ;;
    --run-tag)
      RUN_TAG="$2"
      shift 2
      ;;
    --distributed-stage7)
      DISTRIBUTED_STAGE7="true"
      shift
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --raw-save-dir)
      RAW_SAVE_DIR="$2"
      shift 2
      ;;
    --dataset-id)
      DATASET_ID="$2"
      shift 2
      ;;
    --gpu-8b)
      GPU_8B="$2"
      shift 2
      ;;
    --gpu-4b)
      GPU_4B="$2"
      shift 2
      ;;
    --quiet)
      QUIET="true"
      shift
      ;;
    --skip-existing)
      SKIP_EXISTING="true"
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

case "$MODELS" in
  all|8b|4b)
    ;;
  *)
    echo "Error: --models must be one of all, 8b, 4b" >&2
    exit 1
    ;;
esac

if [[ -z "$RAW_SAVE_DIR" ]]; then
  if [[ -n "$RUN_TAG" ]]; then
    RAW_SAVE_DIR="$ROOT_DIR/data/mmlu_cais_${SPLIT}_${RUN_TAG}"
  else
    RAW_SAVE_DIR="$ROOT_DIR/data/mmlu_cais_${SPLIT}_selected"
  fi
fi

dir_has_files() {
  local dir_path="$1"
  [[ -d "$dir_path" ]] && find "$dir_path" -mindepth 1 -print -quit 2>/dev/null | grep -q .
}

has_named_files() {
  local dir_path="$1"
  local pattern="$2"
  [[ -d "$dir_path" ]] && find "$dir_path" -type f -name "$pattern" -print -quit 2>/dev/null | grep -q .
}

run_with_env() {
  local gpu_id="$1"
  shift
  if [[ -n "$gpu_id" ]]; then
    env CUDA_VISIBLE_DEVICES="$gpu_id" "$@"
  else
    "$@"
  fi
}

write_model_state() {
  local progress_dir="$1"
  local model_tag="$2"
  local gpu_id="$3"
  local output_dir="$4"
  local logs_dir="$5"
  local state="$6"
  local started_at="$7"
  local ended_at="$8"
  local current_stage_key="$9"
  local current_stage_name="${10}"
  local current_stage_index="${11}"
  local error_message="${12:-}"

  uv run python - <<PY
import json
from pathlib import Path

progress_dir = Path(${progress_dir@Q})
state_path = progress_dir / "state.json"
tmp_path = progress_dir / "state.tmp"
progress_dir.mkdir(parents=True, exist_ok=True)

def maybe_int(value):
    if value in ("", None):
        return None
    return int(value)

def maybe_value(value):
    return None if value in ("", None) else value

payload = {
    "model_tag": ${model_tag@Q},
    "gpu_id": maybe_value(${gpu_id@Q}),
    "output_dir": ${output_dir@Q},
    "logs_dir": ${logs_dir@Q},
    "state": ${state@Q},
    "started_at": maybe_int(${started_at@Q}),
    "ended_at": maybe_int(${ended_at@Q}),
    "current_stage_key": maybe_value(${current_stage_key@Q}),
    "current_stage_name": maybe_value(${current_stage_name@Q}),
    "current_stage_index": maybe_int(${current_stage_index@Q}),
    "error": maybe_value(${error_message@Q}),
}

with open(tmp_path, "w") as f:
    json.dump(payload, f, indent=2)
tmp_path.replace(state_path)
PY
}

write_stage_state() {
  local progress_dir="$1"
  local stage_key="$2"
  local stage_name="$3"
  local stage_index="$4"
  local status="$5"
  local start_epoch="$6"
  local end_epoch="$7"
  local log_file="$8"

  uv run python - <<PY
import json
from pathlib import Path

progress_dir = Path(${progress_dir@Q})
stage_dir = progress_dir / "stages"
stage_dir.mkdir(parents=True, exist_ok=True)
stage_key = ${stage_key@Q}
stage_path = stage_dir / f"{stage_key}.json"
tmp_path = stage_dir / f"{stage_key}.tmp"

def maybe_int(value):
    if value in ("", None):
        return None
    return int(value)

payload = {
    "key": ${stage_key@Q},
    "name": ${stage_name@Q},
    "index": int(${stage_index@Q}),
    "status": ${status@Q},
    "start_epoch": maybe_int(${start_epoch@Q}),
    "end_epoch": maybe_int(${end_epoch@Q}),
    "log_file": ${log_file@Q},
}

with open(tmp_path, "w") as f:
    json.dump(payload, f, indent=2)
tmp_path.replace(stage_path)
PY
}

write_progress_meta() {
  local progress_dir="$1"
  local model_tag="$2"
  local gpu_id="$3"
  local output_dir="$4"
  local config_path="$5"
  local test_clozes_path="$6"

  uv run python - <<PY
import json
from pathlib import Path

progress_dir = Path(${progress_dir@Q})
meta_path = progress_dir / "meta.json"
tmp_path = progress_dir / "meta.tmp"
progress_dir.mkdir(parents=True, exist_ok=True)

with open(${config_path@Q}) as f:
    config = json.load(f)
with open(${test_clozes_path@Q}) as f:
    test_clozes = json.load(f)

total_samples = len(test_clozes.get("clozes", []))
clustering = config.get("clustering", {})
sweeps = clustering.get("sweeps", {})
beta_values = sweeps.get("beta_values") or [clustering.get("beta", 1.0)]
gamma_values = sweeps.get("gamma_values") or [clustering.get("gamma", 0.0)]
clustering_config_count = max(1, len(beta_values) * len(gamma_values))

steering = config.get("stage_7c_steering", {})
hypotheses = steering.get("hypotheses") or ["H4a"]
hypothesis_count = max(1, len(hypotheses))

stages = [
    {
        "key": "prep",
        "name": "Prepare MMLU",
        "total_units": 1,
        "count": {"mode": "file", "path": "results/test_clozes.json"},
    },
    {
        "key": "stage2",
        "name": "Branch Sampling",
        "total_units": total_samples,
        "count": {"mode": "glob", "root": "results/2_branch_sampling", "pattern": "*_branches.json", "recursive": False},
    },
    {
        "key": "stage3",
        "name": "Attribution",
        "total_units": total_samples,
        "count": {"mode": "glob", "root": "results/3_attribution_graphs", "pattern": "*_prefix_context.pt", "recursive": False},
    },
    {
        "key": "stage4a",
        "name": "Embeddings",
        "total_units": total_samples,
        "count": {"mode": "glob", "root": "results/4_feature_extraction/embeddings", "pattern": "*_embeddings.npy", "recursive": False},
    },
    {
        "key": "stage5",
        "name": "RD Clustering",
        "total_units": total_samples,
        "count": {"mode": "glob", "root": "results/5_clustering", "pattern": "*_sweep_results.json", "recursive": False},
    },
    {
        "key": "stage6",
        "name": "Semantic Graphs",
        "total_units": total_samples * clustering_config_count,
        "count": {"mode": "glob", "root": "results/6_semantic_graphs", "pattern": "*_semantic_graphs.json", "recursive": False},
    },
    {
        "key": "stage7a",
        "name": "Graph Validation",
        "total_units": 1,
        "count": {"mode": "file", "path": "results/7_validation/7a_graph_validation/graph_validation.json"},
    },
    {
        "key": "stage7c2_medoid",
        "name": "K-means Medoid",
        "total_units": total_samples * hypothesis_count,
        "count": {"mode": "glob", "root": "results/7_validation/7c_baseline_kmeans_medoid", "pattern": "*_sweep_results.json", "recursive": True},
    },
    {
        "key": "stage7c2_single",
        "name": "Single Base",
        "total_units": total_samples * hypothesis_count,
        "count": {"mode": "glob", "root": "results/7_validation/7c_baseline_single", "pattern": "*_sweep_results.json", "recursive": True},
    },
    {
        "key": "stage7c2_combined",
        "name": "Combined Medoid",
        "total_units": total_samples * hypothesis_count,
        "count": {"mode": "glob", "root": "results/7_validation/7c_baseline_combined_medoid", "pattern": "*_sweep_results.json", "recursive": True},
    },
]

payload = {
    "model_tag": ${model_tag@Q},
    "gpu_id": None if ${gpu_id@Q} in ("", None) else ${gpu_id@Q},
    "output_dir": ${output_dir@Q},
    "total_samples": total_samples,
    "clustering_config_count": clustering_config_count,
    "hypothesis_count": hypothesis_count,
    "stages": stages,
}

with open(tmp_path, "w") as f:
    json.dump(payload, f, indent=2)
tmp_path.replace(meta_path)
PY
}

initialize_pending_progress() {
  local model_tag="$1"
  local gpu_id="$2"
  local output_dir="$OUTPUT_ROOT/$model_tag"
  local logs_dir="$output_dir/logs"
  local progress_dir="$output_dir/progress"

  mkdir -p "$logs_dir"
  rm -rf "$progress_dir"
  mkdir -p "$progress_dir/stages"

  write_model_state "$progress_dir" "$model_tag" "$gpu_id" "$output_dir" "$logs_dir" "pending" "" "" "" "" "" ""
}

mark_stage_skipped() {
  local progress_dir="$1"
  local stage_key="$2"
  local stage_name="$3"
  local stage_index="$4"
  local log_file="$5"
  local now_epoch
  now_epoch="$(date +%s)"
  write_stage_state "$progress_dir" "$stage_key" "$stage_name" "$stage_index" "skipped" "$now_epoch" "$now_epoch" "$log_file"
}

run_stage_command() {
  local progress_dir="$1"
  local model_tag="$2"
  local gpu_id="$3"
  local output_dir="$4"
  local logs_dir="$5"
  local model_started_at="$6"
  local stage_key="$7"
  local stage_name="$8"
  local stage_index="$9"
  local log_file="${10}"
  shift 10

  local stage_start
  stage_start="$(date +%s)"

  mkdir -p "$(dirname "$log_file")"
  write_model_state "$progress_dir" "$model_tag" "$gpu_id" "$output_dir" "$logs_dir" "running" "$model_started_at" "" "$stage_key" "$stage_name" "$stage_index" ""
  write_stage_state "$progress_dir" "$stage_key" "$stage_name" "$stage_index" "running" "$stage_start" "" "$log_file"

  "$@" >"$log_file" 2>&1 &
  local cmd_pid=$!
  local exit_code
  if wait "$cmd_pid"; then
    exit_code=0
  else
    exit_code=$?
  fi
  local stage_end
  stage_end="$(date +%s)"

  if [[ $exit_code -eq 0 ]]; then
    write_stage_state "$progress_dir" "$stage_key" "$stage_name" "$stage_index" "completed" "$stage_start" "$stage_end" "$log_file"
    return 0
  fi

  write_stage_state "$progress_dir" "$stage_key" "$stage_name" "$stage_index" "failed" "$stage_start" "$stage_end" "$log_file"
  write_model_state "$progress_dir" "$model_tag" "$gpu_id" "$output_dir" "$logs_dir" "failed" "$model_started_at" "$stage_end" "$stage_key" "$stage_name" "$stage_index" "Stage failed: $stage_name"
  return "$exit_code"
}

build_runtime_config() {
  local base_config="$1"
  local config_path="$2"
  local experiment_name="$3"
  local raw_save_dir="$4"

  uv run python - <<PY
import json
from pathlib import Path

base_config = Path(${base_config@Q})
config_path = Path(${config_path@Q})
with open(base_config) as f:
    config = json.load(f)

config["experiment_name"] = ${experiment_name@Q}
data_cfg = config.setdefault("data", {})
data_cfg["cloze_dir"] = ${raw_save_dir@Q}
data_cfg["split"] = ${SPLIT@Q}
stage1_cfg = config.setdefault("stage_1_data_prep", {})
stage1_cfg["mode"] = "question"
stage1_cfg["n_groups"] = 0
clustering_cfg = config.setdefault("clustering", {})
clustering_cfg["skip_existing"] = ${SKIP_EXISTING@Q} == "true"

config_path.parent.mkdir(parents=True, exist_ok=True)
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
PY
}

load_runtime_vars() {
  local config_path="$1"
  local config_vars
  config_vars="$(uv run python - <<PY
import json
with open(${config_path@Q}) as f:
    config = json.load(f)
model = config.get("model", {})
steering = config.get("stage_7c_steering", {})
global_cfg = config.get("global", {})
max_batch_size = steering.get("max_batch_size", global_cfg.get("batch_size", 512))
if max_batch_size is None or int(max_batch_size) <= 0:
    max_batch_size = global_cfg.get("batch_size", 512)
print(f'CFG_MODEL={model.get("base_model", "")}')
print(f'CFG_TRANSCODER={model.get("transcoder", "")}')
print(f'STEERING_MAX_BATCH_SIZE={int(max_batch_size)}')
print(f'STEERING_PREFIX_BATCH_SIZE={int(steering.get("prefix_batch_size", 16))}')
print(f'STEERING_CROSS_PREFIX_BATCHING={str(steering.get("cross_prefix_batching", False)).lower()}')
PY
)"
  eval "$config_vars"
}

build_model_tag() {
  local model_suffix="$1"
  if [[ -n "$RUN_TAG" ]]; then
    printf 'MMLU_%s_%s\n' "$RUN_TAG" "$model_suffix"
  else
    printf 'MMLU_%s\n' "$model_suffix"
  fi
}

validate_existing_test_clozes() {
  local test_clozes_path="$1"
  local model_name="$2"

  if [[ ! -f "$test_clozes_path" ]]; then
    return 0
  fi

  uv run python - <<PY
import json
import sys
from pathlib import Path

path = Path(${test_clozes_path@Q})
subjects = [part.strip() for part in ${SUBJECTS_CSV@Q}.split(",") if part.strip()]
expected = {
    "dataset_id": ${DATASET_ID@Q},
    "split": ${SPLIT@Q},
    "subjects": subjects,
    "model": ${model_name@Q},
    "prompt_style": "question_only",
}

try:
    payload = json.loads(path.read_text())
except Exception as exc:
    print(f"Error: unable to read existing test_clozes file: {path}: {exc}", file=sys.stderr)
    raise SystemExit(1)

metadata = payload.get("metadata", {})
mismatches = []
for key, expected_value in expected.items():
    actual_value = metadata.get(key)
    if actual_value != expected_value:
        mismatches.append((key, actual_value, expected_value))

if mismatches:
    print(f"Error: existing {path} does not match the requested MMLU run.", file=sys.stderr)
    for key, actual_value, expected_value in mismatches:
        print(f"  {key}: found={actual_value!r} expected={expected_value!r}", file=sys.stderr)
    raise SystemExit(1)
PY
}

set_stage_terminal_if_running() {
  local progress_dir="$1"
  local stage_key="$2"
  local stage_name="$3"
  local stage_index="$4"
  local status="$5"
  local log_file="$6"

  uv run python - <<PY
import json
from pathlib import Path

progress_dir = Path(${progress_dir@Q})
stage_key = ${stage_key@Q}
stage_path = progress_dir / "stages" / f"{stage_key}.json"
if not stage_path.exists():
    raise SystemExit(0)

try:
    payload = json.loads(stage_path.read_text())
except Exception:
    raise SystemExit(0)

if payload.get("status") not in {"running", "pending"}:
    raise SystemExit(0)

payload["key"] = stage_key
payload["name"] = ${stage_name@Q}
payload["index"] = int(${stage_index@Q})
payload["status"] = ${status@Q}
payload["end_epoch"] = int(${EPOCHSECONDS@Q})
payload["log_file"] = ${log_file@Q}

tmp_path = stage_path.with_suffix(".tmp")
with open(tmp_path, "w") as f:
    json.dump(payload, f, indent=2)
tmp_path.replace(stage_path)
PY
}

run_distributed_stage7() {
  local model_tag="$1"
  local gpu_id="$2"
  local output_dir="$3"
  local results_dir="$4"
  local logs_dir="$5"
  local progress_dir="$6"
  local model_started_at="$7"
  local config_path="$8"
  shift 8
  local baseline_common_args=("$@")

  local remote_base="$REMOTE_BASE_DEFAULT"
  local remote_stage_start
  local remote_single_log="$logs_dir/remote_baselines/remote_stage_single.log"
  local remote_combined_log="$logs_dir/remote_baselines/remote_stage_combined.log"
  local remote_launcher_log="$logs_dir/remote_baselines/worker_launcher.log"
  local remote_sync_log="$logs_dir/remote_baselines/sync.log"
  local control_path=""
  local remote_worker_pid=""
  local remote_pull_pid=""
  local medoid_exit_code=0
  local remote_exit_code=0

  set_remote_target_for_model_tag "$model_tag"
  require_graph_validation_done "$output_dir"

  mkdir -p "$logs_dir/remote_baselines"
  remote_stage_start="$(date +%s)"

  write_stage_state "$progress_dir" "stage7c2_single" "Single Base" "9" "running" "$remote_stage_start" "" "$remote_single_log"
  write_stage_state "$progress_dir" "stage7c2_combined" "Combined Medoid" "10" "running" "$remote_stage_start" "" "$remote_combined_log"

  control_path="/tmp/ssh-$(printf '%s' "$model_tag" | tr -cd 'A-Za-z0-9_')-$$"
  if ! start_ssh_master "$control_path" "$REMOTE_HOST" "$REMOTE_PORT" "$REMOTE_USER"; then
    write_stage_state "$progress_dir" "stage7c2_single" "Single Base" "9" "failed" "$remote_stage_start" "$(date +%s)" "$remote_single_log"
    write_stage_state "$progress_dir" "stage7c2_combined" "Combined Medoid" "10" "failed" "$remote_stage_start" "$(date +%s)" "$remote_combined_log"
    return 1
  fi

  if ! sync_remote_baseline_workspace \
    "$control_path" "$REMOTE_HOST" "$REMOTE_PORT" "$REMOTE_USER" "$remote_base" \
    "$model_tag" "$(realpath --relative-to="$ROOT_DIR" "$config_path")" "$output_dir" >"$remote_sync_log" 2>&1; then
    write_stage_state "$progress_dir" "stage7c2_single" "Single Base" "9" "failed" "$remote_stage_start" "$(date +%s)" "$remote_single_log"
    write_stage_state "$progress_dir" "stage7c2_combined" "Combined Medoid" "10" "failed" "$remote_stage_start" "$(date +%s)" "$remote_combined_log"
    stop_ssh_master "$control_path" "$REMOTE_HOST" "$REMOTE_PORT" "$REMOTE_USER"
    return 1
  fi

  launch_remote_baseline_worker \
    "$control_path" "$REMOTE_HOST" "$REMOTE_PORT" "$REMOTE_USER" "$remote_base" \
    "$model_tag" "$(realpath --relative-to="$ROOT_DIR" "$config_path")" \
    "$SKIP_EXISTING" "$QUIET" "$remote_launcher_log"
  remote_worker_pid="$REMOTE_WORKER_PID"

  start_remote_pull_loop \
    "$control_path" "$REMOTE_HOST" "$REMOTE_PORT" "$REMOTE_USER" "$remote_base" \
    "$model_tag" "$output_dir" "10"
  remote_pull_pid="$REMOTE_PULL_PID"

  local stage7c2_medoid_log="$logs_dir/wrapper_stage_7c2_medoid.log"
  if [[ "$SKIP_EXISTING" == "true" ]] || ! has_named_files "$results_dir/7_validation/7c_baseline_kmeans_medoid" "*_sweep_results.json"; then
    if run_stage_command \
      "$progress_dir" "$model_tag" "$gpu_id" "$output_dir" "$logs_dir" "$model_started_at" \
      "stage7c2_medoid" "K-means Medoid" "8" "$stage7c2_medoid_log" \
      run_with_env "$gpu_id" \
      uv run python 7_validation/7c_baseline_kmeans.py \
      --embeddings-dir "$results_dir/4_feature_extraction/embeddings" \
      --output-dir "$results_dir/7_validation/7c_baseline_kmeans_medoid" \
      --use-medoid \
      "${baseline_common_args[@]}"; then
      medoid_exit_code=0
    else
      medoid_exit_code=$?
    fi
  else
    printf "Skipped by wrapper: K-means medoid already exists.\n" >"$stage7c2_medoid_log"
    mark_stage_skipped "$progress_dir" "stage7c2_medoid" "K-means Medoid" "8" "$stage7c2_medoid_log"
  fi

  if [[ "$medoid_exit_code" -ne 0 ]]; then
    kill "$remote_pull_pid" 2>/dev/null || true
    kill "$remote_worker_pid" 2>/dev/null || true
    wait "$remote_pull_pid" 2>/dev/null || true
    stop_ssh_master "$control_path" "$REMOTE_HOST" "$REMOTE_PORT" "$REMOTE_USER"
    write_stage_state "$progress_dir" "stage7c2_single" "Single Base" "9" "failed" "$remote_stage_start" "$(date +%s)" "$remote_single_log"
    write_stage_state "$progress_dir" "stage7c2_combined" "Combined Medoid" "10" "failed" "$remote_stage_start" "$(date +%s)" "$remote_combined_log"
    return "$medoid_exit_code"
  fi

  if wait "$remote_worker_pid"; then
    remote_exit_code=0
  else
    remote_exit_code=$?
  fi

  kill "$remote_pull_pid" 2>/dev/null || true
  wait "$remote_pull_pid" 2>/dev/null || true
  pull_remote_baseline_outputs "$control_path" "$REMOTE_HOST" "$REMOTE_PORT" "$REMOTE_USER" "$remote_base" "$model_tag" "$output_dir" || true
  stop_ssh_master "$control_path" "$REMOTE_HOST" "$REMOTE_PORT" "$REMOTE_USER"

  if [[ "$remote_exit_code" -ne 0 ]]; then
    set_stage_terminal_if_running "$progress_dir" "stage7c2_single" "Single Base" "9" "failed" "$remote_single_log"
    set_stage_terminal_if_running "$progress_dir" "stage7c2_combined" "Combined Medoid" "10" "failed" "$remote_combined_log"
    return "$remote_exit_code"
  fi

  set_stage_terminal_if_running "$progress_dir" "stage7c2_single" "Single Base" "9" "completed" "$remote_single_log"
  set_stage_terminal_if_running "$progress_dir" "stage7c2_combined" "Combined Medoid" "10" "completed" "$remote_combined_log"
}

run_model_pipeline() {
  local model_tag="$1"
  local model_name="$2"
  local base_config="$3"
  local gpu_id="$4"

  local output_dir="$OUTPUT_ROOT/$model_tag"
  local results_dir="$output_dir/results"
  local logs_dir="$output_dir/logs"
  local progress_dir="$output_dir/progress"
  local config_path="$output_dir/configs/mmlu_runtime_config.json"
  local experiment_name="${model_tag}_${SPLIT}"
  local model_raw_save_dir="$RAW_SAVE_DIR"
  local model_started_at

  if [[ "$PARALLEL_MODE" == "true" ]]; then
    model_raw_save_dir="$RAW_SAVE_DIR/$model_tag"
  fi

  mkdir -p "$results_dir" "$logs_dir" "$output_dir/configs" "$progress_dir/stages"
  validate_existing_test_clozes "$results_dir/test_clozes.json" "$model_name"
  model_started_at="$(date +%s)"
  write_model_state "$progress_dir" "$model_tag" "$gpu_id" "$output_dir" "$logs_dir" "starting" "$model_started_at" "" "" "" "" ""

  build_runtime_config "$base_config" "$config_path" "$experiment_name" "$model_raw_save_dir"
  load_runtime_vars "$config_path"

  local prep_log="$logs_dir/wrapper_stage_prep.log"
  local prep_cmd=(
    uv run python scripts/prepare_mmlu_questions.py
    --dataset-id "$DATASET_ID"
    --split "$SPLIT"
    --subjects "$SUBJECTS_CSV"
    --model "$model_name"
    --raw-save-dir "$model_raw_save_dir"
    --output "$results_dir/test_clozes.json"
    --log-dir "$logs_dir"
  )
  if [[ "$SKIP_EXISTING" == "true" ]]; then
    prep_cmd+=(--skip-existing)
  fi
  if [[ "$QUIET" == "true" ]]; then
    prep_cmd+=(--quiet)
  fi
  if ! run_stage_command \
    "$progress_dir" "$model_tag" "$gpu_id" "$output_dir" "$logs_dir" "$model_started_at" \
    "prep" "Prepare MMLU" "1" "$prep_log" \
    run_with_env "$gpu_id" "${prep_cmd[@]}"; then
    return 1
  fi

  write_progress_meta "$progress_dir" "$model_tag" "$gpu_id" "$output_dir" "$config_path" "$results_dir/test_clozes.json"

  local pipeline_cmd_base=(env CONFIG_FILE="$config_path")
  if [[ -n "$gpu_id" ]]; then
    pipeline_cmd_base+=(CUDA_VISIBLE_DEVICES="$gpu_id")
  fi
  pipeline_cmd_base+=(bash scripts/run_pipeline.sh --output_dir "$output_dir")
  if [[ "$QUIET" == "true" ]]; then
    pipeline_cmd_base+=(--quiet)
  fi
  if [[ "$SKIP_EXISTING" == "true" ]]; then
    pipeline_cmd_base+=(--skip-existing)
  fi

  local stage2_log="$logs_dir/wrapper_stage_2.log"
  if [[ ! -f "$results_dir/2_branch_sampling/branches_index.json" ]]; then
    if ! run_stage_command \
      "$progress_dir" "$model_tag" "$gpu_id" "$output_dir" "$logs_dir" "$model_started_at" \
      "stage2" "Branch Sampling" "2" "$stage2_log" \
      "${pipeline_cmd_base[@]}" --only 2; then
      return 1
    fi
  else
    printf "Skipped by wrapper: existing branch index found.\n" >"$stage2_log"
    mark_stage_skipped "$progress_dir" "stage2" "Branch Sampling" "2" "$stage2_log"
  fi

  local stage3_log="$logs_dir/wrapper_stage_3.log"
  if ! has_named_files "$results_dir/3_attribution_graphs" "*_prefix_context.pt"; then
    if ! run_stage_command \
      "$progress_dir" "$model_tag" "$gpu_id" "$output_dir" "$logs_dir" "$model_started_at" \
      "stage3" "Attribution" "3" "$stage3_log" \
      "${pipeline_cmd_base[@]}" --only 3; then
      return 1
    fi
  else
    printf "Skipped by wrapper: attribution outputs already exist.\n" >"$stage3_log"
    mark_stage_skipped "$progress_dir" "stage3" "Attribution" "3" "$stage3_log"
  fi

  local stage4a_log="$logs_dir/wrapper_stage_4a.log"
  if ! has_named_files "$results_dir/4_feature_extraction/embeddings" "*_embeddings.npy"; then
    if ! run_stage_command \
      "$progress_dir" "$model_tag" "$gpu_id" "$output_dir" "$logs_dir" "$model_started_at" \
      "stage4a" "Embeddings" "4" "$stage4a_log" \
      "${pipeline_cmd_base[@]}" --only 4a; then
      return 1
    fi
  else
    printf "Skipped by wrapper: embeddings already exist.\n" >"$stage4a_log"
    mark_stage_skipped "$progress_dir" "stage4a" "Embeddings" "4" "$stage4a_log"
  fi

  local stage5_log="$logs_dir/wrapper_stage_5.log"
  if [[ "$SKIP_EXISTING" == "true" ]]; then
    if ! run_stage_command \
      "$progress_dir" "$model_tag" "$gpu_id" "$output_dir" "$logs_dir" "$model_started_at" \
      "stage5" "RD Clustering" "5" "$stage5_log" \
      "${pipeline_cmd_base[@]}" --only 5; then
      return 1
    fi
  elif ! has_named_files "$results_dir/5_clustering" "*_sweep_results.json"; then
    if ! run_stage_command \
      "$progress_dir" "$model_tag" "$gpu_id" "$output_dir" "$logs_dir" "$model_started_at" \
      "stage5" "RD Clustering" "5" "$stage5_log" \
      "${pipeline_cmd_base[@]}" --only 5; then
      return 1
    fi
  else
    printf "Skipped by wrapper: clustering outputs already exist.\n" >"$stage5_log"
    mark_stage_skipped "$progress_dir" "stage5" "RD Clustering" "5" "$stage5_log"
  fi

  local stage6_log="$logs_dir/wrapper_stage_6.log"
  if ! dir_has_files "$results_dir/6_semantic_graphs"; then
    if ! run_stage_command \
      "$progress_dir" "$model_tag" "$gpu_id" "$output_dir" "$logs_dir" "$model_started_at" \
      "stage6" "Semantic Graphs" "6" "$stage6_log" \
      "${pipeline_cmd_base[@]}" --only 6; then
      return 1
    fi
  else
    printf "Skipped by wrapper: semantic graph outputs already exist.\n" >"$stage6_log"
    mark_stage_skipped "$progress_dir" "stage6" "Semantic Graphs" "6" "$stage6_log"
  fi

  local stage7a_log="$logs_dir/wrapper_stage_7a.log"
  if [[ ! -f "$results_dir/7_validation/7a_graph_validation/graph_validation.json" ]]; then
    if ! run_stage_command \
      "$progress_dir" "$model_tag" "$gpu_id" "$output_dir" "$logs_dir" "$model_started_at" \
      "stage7a" "Graph Validation" "7" "$stage7a_log" \
      "${pipeline_cmd_base[@]}" --only 7a; then
      return 1
    fi
  else
    printf "Skipped by wrapper: graph validation already exists.\n" >"$stage7a_log"
    mark_stage_skipped "$progress_dir" "stage7a" "Graph Validation" "7" "$stage7a_log"
  fi

  local baseline_common_args=(
    --samples-dir "$results_dir/2_branch_sampling"
    --attribution-graphs-dir "$results_dir/3_attribution_graphs"
    --clustering-dir "$results_dir/5_clustering"
    --config "$config_path"
    --max-batch-size "$STEERING_MAX_BATCH_SIZE"
    --prefix-batch-size "$STEERING_PREFIX_BATCH_SIZE"
    --log-dir "$logs_dir"
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

  if [[ "$DISTRIBUTED_STAGE7" == "true" ]]; then
    if ! run_distributed_stage7 \
      "$model_tag" "$gpu_id" "$output_dir" "$results_dir" "$logs_dir" "$progress_dir" "$model_started_at" "$config_path" \
      "${baseline_common_args[@]}"; then
      write_model_state "$progress_dir" "$model_tag" "$gpu_id" "$output_dir" "$logs_dir" "failed" "$model_started_at" "$(date +%s)" "" "" "" "Distributed stage 7 failed"
      return 1
    fi
  else
    local stage7c2_medoid_log="$logs_dir/wrapper_stage_7c2_medoid.log"
    if [[ "$SKIP_EXISTING" == "true" ]] || ! has_named_files "$results_dir/7_validation/7c_baseline_kmeans_medoid" "*_sweep_results.json"; then
      if ! run_stage_command \
        "$progress_dir" "$model_tag" "$gpu_id" "$output_dir" "$logs_dir" "$model_started_at" \
        "stage7c2_medoid" "K-means Medoid" "8" "$stage7c2_medoid_log" \
        run_with_env "$gpu_id" \
        uv run python 7_validation/7c_baseline_kmeans.py \
        --embeddings-dir "$results_dir/4_feature_extraction/embeddings" \
        --output-dir "$results_dir/7_validation/7c_baseline_kmeans_medoid" \
        --use-medoid \
        "${baseline_common_args[@]}"; then
        return 1
      fi
    else
      printf "Skipped by wrapper: K-means medoid already exists.\n" >"$stage7c2_medoid_log"
      mark_stage_skipped "$progress_dir" "stage7c2_medoid" "K-means Medoid" "8" "$stage7c2_medoid_log"
    fi

    local stage7c2_single_log="$logs_dir/wrapper_stage_7c2_single.log"
    if [[ "$SKIP_EXISTING" == "true" ]] || ! has_named_files "$results_dir/7_validation/7c_baseline_single" "*_sweep_results.json"; then
      if ! run_stage_command \
        "$progress_dir" "$model_tag" "$gpu_id" "$output_dir" "$logs_dir" "$model_started_at" \
        "stage7c2_single" "Single Base" "9" "$stage7c2_single_log" \
        run_with_env "$gpu_id" \
        uv run python 7_validation/7c_baseline_single.py \
        --output-dir "$results_dir/7_validation/7c_baseline_single" \
        "${baseline_common_args[@]}"; then
        return 1
      fi
    else
      printf "Skipped by wrapper: single baseline already exists.\n" >"$stage7c2_single_log"
      mark_stage_skipped "$progress_dir" "stage7c2_single" "Single Base" "9" "$stage7c2_single_log"
    fi

    local stage7c2_combined_log="$logs_dir/wrapper_stage_7c2_combined.log"
    if [[ "$SKIP_EXISTING" == "true" ]] || ! has_named_files "$results_dir/7_validation/7c_baseline_combined_medoid" "*_sweep_results.json"; then
      if ! run_stage_command \
        "$progress_dir" "$model_tag" "$gpu_id" "$output_dir" "$logs_dir" "$model_started_at" \
        "stage7c2_combined" "Combined Medoid" "10" "$stage7c2_combined_log" \
        run_with_env "$gpu_id" \
        uv run python 7_validation/7c_baseline_combined_medoid.py \
        --embeddings-dir "$results_dir/4_feature_extraction/embeddings" \
        --output-dir "$results_dir/7_validation/7c_baseline_combined_medoid" \
        "${baseline_common_args[@]}"; then
        return 1
      fi
    else
      printf "Skipped by wrapper: combined medoid already exists.\n" >"$stage7c2_combined_log"
      mark_stage_skipped "$progress_dir" "stage7c2_combined" "Combined Medoid" "10" "$stage7c2_combined_log"
    fi
  fi

  write_model_state "$progress_dir" "$model_tag" "$gpu_id" "$output_dir" "$logs_dir" "completed" "$model_started_at" "$(date +%s)" "" "" "" ""
}

if [[ "$MODELS" == "all" && -n "$GPU_8B" && -n "$GPU_4B" && "$GPU_8B" != "$GPU_4B" ]]; then
  PARALLEL_MODE="true"
fi

render_progress_dashboard() {
  local output_dirs=("$@")
  local last_frame=""
  local line_count=0
  local first_frame="true"

  if [[ -t 1 ]]; then
    printf '\033[?25l'
  fi

  while true; do
    local frame
    frame="$(uv run python scripts/remote/render_mmlu_pipeline_progress.py --output-dirs "${output_dirs[@]}")"

    if [[ "$frame" != "$last_frame" ]]; then
      if [[ -t 1 ]]; then
        if [[ "$first_frame" != "true" && "$line_count" -gt 0 ]]; then
          printf '\033[%sA' "$line_count"
          printf '\r\033[J'
        fi
        printf '%s' "$frame"
        [[ "$frame" == *$'\n' ]] || printf '\n'
      else
        printf '%s\n\n' "$frame"
      fi
      line_count="$(printf '%s\n' "$frame" | wc -l | tr -d ' ')"
      last_frame="$frame"
      first_frame="false"
    fi

    local any_alive="false"
    for pid in "${JOB_PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        any_alive="true"
        break
      fi
    done

    if [[ "$any_alive" != "true" ]]; then
      break
    fi
    sleep 1
  done

  if [[ -t 1 ]]; then
    printf '\033[?25h'
  fi
}

launch_model_job() {
  local model_tag="$1"
  local model_name="$2"
  local config_path="$3"
  local gpu_id="$4"
  local log_path="$OUTPUT_ROOT/$model_tag/logs/wrapper_model.log"

  (
    run_model_pipeline "$model_tag" "$model_name" "$config_path" "$gpu_id"
  ) >>"$log_path" 2>&1 &
  JOB_PIDS+=("$!")
  JOB_LABELS+=("$model_tag")
}

JOB_PIDS=()
JOB_LABELS=()
SELECTED_OUTPUT_DIRS=()
MODEL_TAG_8B="$(build_model_tag "Qwen3-8B")"
MODEL_TAG_4B="$(build_model_tag "Qwen3-4B")"
CONFIG_8B="$ROOT_DIR/configs/mmlu_qwen3_8b_config.json"
CONFIG_4B="$ROOT_DIR/configs/mmlu_qwen3_4b_config.json"

if [[ "$MODELS" == "all" || "$MODELS" == "8b" ]]; then
  initialize_pending_progress "$MODEL_TAG_8B" "$GPU_8B"
  SELECTED_OUTPUT_DIRS+=("$OUTPUT_ROOT/$MODEL_TAG_8B")
fi

if [[ "$MODELS" == "all" || "$MODELS" == "4b" ]]; then
  initialize_pending_progress "$MODEL_TAG_4B" "$GPU_4B"
  SELECTED_OUTPUT_DIRS+=("$OUTPUT_ROOT/$MODEL_TAG_4B")
fi

if [[ "$PARALLEL_MODE" == "true" ]]; then
  launch_model_job \
    "$MODEL_TAG_8B" \
    "Qwen/Qwen3-8B" \
    "$CONFIG_8B" \
    "$GPU_8B"
  launch_model_job \
    "$MODEL_TAG_4B" \
    "Qwen/Qwen3-4B" \
    "$CONFIG_4B" \
    "$GPU_4B"
else
  if [[ "$MODELS" == "all" ]]; then
    (
      run_model_pipeline \
        "$MODEL_TAG_8B" \
        "Qwen/Qwen3-8B" \
        "$CONFIG_8B" \
        "$GPU_8B" >>"$OUTPUT_ROOT/$MODEL_TAG_8B/logs/wrapper_model.log" 2>&1
      run_model_pipeline \
        "$MODEL_TAG_4B" \
        "Qwen/Qwen3-4B" \
        "$CONFIG_4B" \
        "$GPU_4B" >>"$OUTPUT_ROOT/$MODEL_TAG_4B/logs/wrapper_model.log" 2>&1
    ) &
    JOB_PIDS+=("$!")
    JOB_LABELS+=("MMLU sequential controller")
  elif [[ "$MODELS" == "8b" ]]; then
    launch_model_job \
      "$MODEL_TAG_8B" \
      "Qwen/Qwen3-8B" \
      "$CONFIG_8B" \
      "$GPU_8B"
  elif [[ "$MODELS" == "4b" ]]; then
    launch_model_job \
      "$MODEL_TAG_4B" \
      "Qwen/Qwen3-4B" \
      "$CONFIG_4B" \
      "$GPU_4B"
  fi
fi

render_progress_dashboard "${SELECTED_OUTPUT_DIRS[@]}"

overall_exit_code=0
for idx in "${!JOB_PIDS[@]}"; do
  if ! wait "${JOB_PIDS[$idx]}"; then
    overall_exit_code=1
  fi
done

if [[ $overall_exit_code -ne 0 ]]; then
  exit "$overall_exit_code"
fi

echo "========================================"
echo "MMLU PIPELINE COMPLETE"
echo "========================================"
