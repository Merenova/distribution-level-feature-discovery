#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$ROOT_DIR/scripts/misc/mmlu_remote_baseline_lib.sh"

HOST="${MMLU_RD_REMOTE_HOST:-}"
PORT="${MMLU_RD_REMOTE_PORT:-40394}"
USER_NAME="${MMLU_RD_REMOTE_USER:-${REMOTE_USER:-${USER:-}}}"
REMOTE_BASE="$REMOTE_BASE_DEFAULT"
MODEL_TAG="MMLU_small4_Qwen3-8B"
CONFIG_REL=""
DRY_RUN="false"
SYNC_ONLY="false"
RUN_ONLY="false"
QUIET="false"
GPU_ID="0"

usage() {
  cat <<EOF
Usage: bash scripts/remote/sync_and_run_mmlu_rd_remote.sh [options]

Options:
  --host HOST            Remote host (default: $HOST)
  --port PORT            Remote SSH port (default: $PORT)
  --user USER            Remote SSH user (default: $USER_NAME)
  --remote-base DIR      Remote repo root (default: $REMOTE_BASE)
  --model-tag TAG        Local/remote model output dir (default: $MODEL_TAG)
  --config-rel PATH      Config path relative to repo root
  --dry-run              Print sync/run plan without mutating remote state
  --sync-only            Sync inputs/code only, do not run stage 7c1
  --run-only             Skip sync, only launch remote stage 7c1 and pull outputs
  --quiet                Pass --quiet to remote stage 7c1
  --gpu ID               Remote CUDA_VISIBLE_DEVICES value for stage 7c1 (default: $GPU_ID)
  --help                 Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --user)
      USER_NAME="$2"
      shift 2
      ;;
    --remote-base)
      REMOTE_BASE="$2"
      shift 2
      ;;
    --model-tag)
      MODEL_TAG="$2"
      shift 2
      ;;
    --config-rel)
      CONFIG_REL="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --sync-only)
      SYNC_ONLY="true"
      shift
      ;;
    --run-only)
      RUN_ONLY="true"
      shift
      ;;
    --quiet)
      QUIET="true"
      shift
      ;;
    --gpu)
      GPU_ID="$2"
      shift 2
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

if [[ -z "$HOST" ]]; then
  echo "Error: set MMLU_RD_REMOTE_HOST or pass --host before running this script." >&2
  exit 1
fi

if [[ "$SYNC_ONLY" == "true" && "$RUN_ONLY" == "true" ]]; then
  echo "Error: --sync-only and --run-only cannot be used together" >&2
  exit 1
fi

if [[ -z "$CONFIG_REL" ]]; then
  CONFIG_REL="$MODEL_TAG/configs/mmlu_runtime_config.json"
fi

OUTPUT_DIR="$ROOT_DIR/$MODEL_TAG"
RESULTS_DIR="$OUTPUT_DIR/results"
LOCAL_CONFIG="$ROOT_DIR/$CONFIG_REL"
REMOTE_MODEL_DIR="$REMOTE_BASE/$MODEL_TAG"
REMOTE_RESULTS_DIR="$REMOTE_MODEL_DIR/results"
REMOTE_CONFIG="$REMOTE_BASE/$CONFIG_REL"
REMOTE_LOG_FILE="$REMOTE_MODEL_DIR/logs/remote_stage7c1.log"
LOCAL_REMOTE_LOG="$OUTPUT_DIR/logs/remote_stage7c1.log"

require_graph_validation_done "$OUTPUT_DIR"

if [[ ! -d "$RESULTS_DIR" ]]; then
  echo "Error: results directory not found: $RESULTS_DIR" >&2
  exit 1
fi

if [[ ! -f "$LOCAL_CONFIG" ]]; then
  echo "Error: config file not found: $LOCAL_CONFIG" >&2
  exit 1
fi

SSH_CONTROL_PATH="/tmp/ssh-mmlu-rd-remote-$$"
SSH_CMD="ssh -p $PORT -o ControlMaster=auto -o ControlPath=$SSH_CONTROL_PATH -o ControlPersist=300"
RSYNC_OPTS="-avz --human-readable --partial --progress"
CODE_EXCLUDES=(
  --exclude="__pycache__"
  --exclude="*.pyc"
  --exclude=".pytest_cache"
  --exclude=".mypy_cache"
  --exclude="*.log"
  --exclude=".DS_Store"
)
if [[ "$DRY_RUN" == "true" ]]; then
  RSYNC_OPTS="$RSYNC_OPTS --dry-run --itemize-changes"
fi

cleanup() {
  stop_ssh_master "$SSH_CONTROL_PATH" "$HOST" "$PORT" "$USER_NAME"
}
trap cleanup EXIT

sync_remote_to_local() {
  local source_path="$1"
  local dest_path="$2"
  shift 2
  rsync $RSYNC_OPTS "$@" -e "$SSH_CMD" "$source_path" "$dest_path"
}

start_ssh_master "$SSH_CONTROL_PATH" "$HOST" "$PORT" "$USER_NAME"

echo "========================================"
echo "SYNC & RUN REMOTE MMLU RD STEERING"
echo "========================================"
echo "Remote: $USER_NAME@$HOST:$PORT"
echo "Remote base: $REMOTE_BASE"
echo "Model tag: $MODEL_TAG"
echo "Local results: $RESULTS_DIR"
echo "Remote results: $REMOTE_RESULTS_DIR"
echo "Config: $CONFIG_REL"
echo ""

if [[ "$RUN_ONLY" != "true" ]]; then
  echo "Step 1/3: Syncing code and inputs..."

  ssh -p "$PORT" -o ControlPath="$SSH_CONTROL_PATH" "$USER_NAME@$HOST" \
    "mkdir -p '$REMOTE_BASE/scripts' '$REMOTE_BASE/7_validation' '$REMOTE_BASE/utils' '$REMOTE_BASE/circuit-tracer' '$REMOTE_MODEL_DIR/results' '$REMOTE_MODEL_DIR/configs' '$REMOTE_MODEL_DIR/logs' '$REMOTE_RESULTS_DIR/7_validation/7a_graph_validation' '$REMOTE_RESULTS_DIR/configs'"

  for code_dir in \
    7_validation \
    utils \
    circuit-tracer
  do
    echo "  Syncing $code_dir/"
    sync_path_to_remote "$SSH_CMD" "$RSYNC_OPTS" \
      "$ROOT_DIR/$code_dir" \
      "$USER_NAME@$HOST:$REMOTE_BASE/" \
      "${CODE_EXCLUDES[@]}"
  done

  for code_file in \
    scripts/run_pipeline.sh \
    scripts/remote/render_mmlu_pipeline_progress.py \
    scripts/misc/mmlu_remote_baseline_lib.sh \
    scripts/remote/run_remote_mmlu_rd_worker.sh \
    pyproject.toml \
    uv.lock \
    "$CONFIG_REL"
  do
    echo "  Syncing $code_file"
    sync_path_to_remote "$SSH_CMD" "$RSYNC_OPTS" \
      "$ROOT_DIR/$code_file" \
      "$USER_NAME@$HOST:$REMOTE_BASE/$(dirname "$code_file")/"
  done

  for subdir in \
    2_branch_sampling \
    3_attribution_graphs
  do
    echo "  Syncing results/$subdir/"
    sync_path_to_remote "$SSH_CMD" "$RSYNC_OPTS" \
      "$RESULTS_DIR/$subdir" \
      "$USER_NAME@$HOST:$REMOTE_RESULTS_DIR/" \
      --exclude="*.log" --exclude="__pycache__"
  done

  echo "  Syncing results/5_clustering/"
  sync_path_to_remote "$SSH_CMD" "$RSYNC_OPTS" \
    "$RESULTS_DIR/5_clustering" \
    "$USER_NAME@$HOST:$REMOTE_RESULTS_DIR/" \
    --exclude="intermediate" --exclude="intermediate/" \
    --exclude="logs" --exclude="logs/" \
    --exclude="*.log" --exclude="__pycache__"

  if [[ -f "$RESULTS_DIR/test_clozes.json" ]]; then
    echo "  Syncing results/test_clozes.json"
    sync_path_to_remote "$SSH_CMD" "$RSYNC_OPTS" \
      "$RESULTS_DIR/test_clozes.json" \
      "$USER_NAME@$HOST:$REMOTE_RESULTS_DIR/"
  fi

  if [[ -f "$RESULTS_DIR/7_validation/7a_graph_validation/graph_validation.json" ]]; then
    echo "  Syncing graph validation gate"
    sync_path_to_remote "$SSH_CMD" "$RSYNC_OPTS" \
      "$RESULTS_DIR/7_validation/7a_graph_validation/graph_validation.json" \
      "$USER_NAME@$HOST:$REMOTE_RESULTS_DIR/7_validation/7a_graph_validation/"
  fi

  echo ""
fi

if [[ "$SYNC_ONLY" == "true" ]]; then
  echo "Sync complete. Skipping remote run due to --sync-only."
  exit 0
fi

REMOTE_CMD="cd '$REMOTE_BASE'; bash scripts/remote/run_remote_mmlu_rd_worker.sh --model-tag '$MODEL_TAG' --config '$CONFIG_REL' --gpu '$GPU_ID' --skip-existing"
if [[ "$QUIET" == "true" ]]; then
  REMOTE_CMD="$REMOTE_CMD --quiet"
fi
printf -v REMOTE_BASH_CMD 'bash -lc %q' "$REMOTE_CMD"

echo "Step 2/3: Running remote stage 7c1..."
if [[ "$DRY_RUN" == "true" ]]; then
  echo "[DRY RUN] Would execute:"
  echo "  ssh -tt -p $PORT $USER_NAME@$HOST \"$REMOTE_BASH_CMD\""
else
  remote_exit_code=0
  if ssh -tt -p "$PORT" -o ControlPath="$SSH_CONTROL_PATH" "$USER_NAME@$HOST" "$REMOTE_BASH_CMD"; then
    remote_exit_code=0
  else
    remote_exit_code=$?
  fi
fi

echo ""
echo "Step 3/3: Pulling remote outputs..."
if [[ "$DRY_RUN" == "true" ]]; then
  echo "[DRY RUN] Would pull:"
  echo "  $USER_NAME@$HOST:$REMOTE_RESULTS_DIR/7_validation/7c_steering -> $RESULTS_DIR/7_validation/"
  echo "  $USER_NAME@$HOST:$REMOTE_RESULTS_DIR/configs/stage_7_config.json -> $RESULTS_DIR/configs/"
  echo "  $USER_NAME@$HOST:$REMOTE_LOG_FILE -> $LOCAL_REMOTE_LOG"
else
  mkdir -p "$RESULTS_DIR/7_validation" "$RESULTS_DIR/configs" "$(dirname "$LOCAL_REMOTE_LOG")"
  if remote_path_exists "$SSH_CONTROL_PATH" "$HOST" "$PORT" "$USER_NAME" "$REMOTE_RESULTS_DIR/7_validation/7c_steering"; then
    sync_remote_to_local \
      "$USER_NAME@$HOST:$REMOTE_RESULTS_DIR/7_validation/7c_steering" \
      "$RESULTS_DIR/7_validation/"
  fi
  if remote_path_exists "$SSH_CONTROL_PATH" "$HOST" "$PORT" "$USER_NAME" "$REMOTE_RESULTS_DIR/configs/stage_7_config.json"; then
    sync_remote_to_local \
      "$USER_NAME@$HOST:$REMOTE_RESULTS_DIR/configs/stage_7_config.json" \
      "$RESULTS_DIR/configs/"
  fi
  if remote_path_exists "$SSH_CONTROL_PATH" "$HOST" "$PORT" "$USER_NAME" "$REMOTE_LOG_FILE"; then
    sync_remote_to_local \
      "$USER_NAME@$HOST:$REMOTE_LOG_FILE" \
      "$LOCAL_REMOTE_LOG"
  fi
  if [[ "${remote_exit_code:-0}" -ne 0 ]]; then
    echo "Remote stage 7c1 failed. Pulled available logs/results." >&2
    exit "$remote_exit_code"
  fi
fi

echo ""
echo "Done."
