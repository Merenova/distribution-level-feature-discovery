#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$ROOT_DIR/scripts/misc/mmlu_remote_baseline_lib.sh"

HOST="${MMLU_QWEN3_4B_HOST:-}"
PORT="${MMLU_QWEN3_4B_PORT:-40394}"
USER_NAME="${MMLU_QWEN3_4B_USER:-${REMOTE_USER:-${USER:-}}}"
REMOTE_BASE="$REMOTE_BASE_DEFAULT"
MODEL_TAG="MMLU_Qwen3-4B"
CONFIG_REL="configs/mmlu_qwen3_4b_config.json"
DRY_RUN="false"

usage() {
  cat <<EOF
Usage: bash scripts/remote/send_mmlu_server_b.sh [--dry-run]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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

OUTPUT_DIR="$ROOT_DIR/$MODEL_TAG"
RESULTS_DIR="$OUTPUT_DIR/results"
if [[ -z "$HOST" ]]; then
  echo "Error: set MMLU_QWEN3_4B_HOST or pass a host-specific wrapper before running this script." >&2
  exit 1
fi
require_graph_validation_done "$OUTPUT_DIR"

SSH_CONTROL_PATH="/tmp/ssh-mmlu-server-b-$$"
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

start_ssh_master "$SSH_CONTROL_PATH" "$HOST" "$PORT" "$USER_NAME"

echo "Syncing $MODEL_TAG to $USER_NAME@$HOST:$PORT"
echo "Remote base: $REMOTE_BASE"
echo "Local results: $RESULTS_DIR"

ssh -p "$PORT" -o ControlPath="$SSH_CONTROL_PATH" "$USER_NAME@$HOST" "mkdir -p '$REMOTE_BASE/$MODEL_TAG/results' '$REMOTE_BASE/scripts' '$REMOTE_BASE/configs'"

sync_path_to_remote "$SSH_CMD" "$RSYNC_OPTS" \
  "$ROOT_DIR/7_validation" \
  "$USER_NAME@$HOST:$REMOTE_BASE/" \
  "${CODE_EXCLUDES[@]}"
sync_path_to_remote "$SSH_CMD" "$RSYNC_OPTS" \
  "$ROOT_DIR/utils" \
  "$USER_NAME@$HOST:$REMOTE_BASE/" \
  "${CODE_EXCLUDES[@]}"
sync_path_to_remote "$SSH_CMD" "$RSYNC_OPTS" \
  "$ROOT_DIR/circuit-tracer" \
  "$USER_NAME@$HOST:$REMOTE_BASE/" \
  "${CODE_EXCLUDES[@]}"

for script_file in \
  scripts/remote/render_mmlu_pipeline_progress.py \
  scripts/misc/mmlu_remote_baseline_lib.sh \
  scripts/remote/run_remote_mmlu_baselines_worker.sh
do
  sync_path_to_remote "$SSH_CMD" "$RSYNC_OPTS" \
    "$ROOT_DIR/$script_file" \
    "$USER_NAME@$HOST:$REMOTE_BASE/$(dirname "$script_file")/"
done

for code_file in pyproject.toml uv.lock "$CONFIG_REL"; do
  sync_path_to_remote "$SSH_CMD" "$RSYNC_OPTS" \
    "$ROOT_DIR/$code_file" \
    "$USER_NAME@$HOST:$REMOTE_BASE/$(dirname "$code_file")/"
done

sync_path_to_remote "$SSH_CMD" "$RSYNC_OPTS" \
  "$RESULTS_DIR/2_branch_sampling" \
  "$USER_NAME@$HOST:$REMOTE_BASE/$MODEL_TAG/results/"
sync_path_to_remote "$SSH_CMD" "$RSYNC_OPTS" \
  "$RESULTS_DIR/3_attribution_graphs" \
  "$USER_NAME@$HOST:$REMOTE_BASE/$MODEL_TAG/results/"

ssh -p "$PORT" -o ControlPath="$SSH_CONTROL_PATH" "$USER_NAME@$HOST" "mkdir -p '$REMOTE_BASE/$MODEL_TAG/results/4_feature_extraction'"
sync_path_to_remote "$SSH_CMD" "$RSYNC_OPTS" \
  "$RESULTS_DIR/4_feature_extraction/embeddings" \
  "$USER_NAME@$HOST:$REMOTE_BASE/$MODEL_TAG/results/4_feature_extraction/"

sync_path_to_remote "$SSH_CMD" "$RSYNC_OPTS" \
  "$RESULTS_DIR/5_clustering" \
  "$USER_NAME@$HOST:$REMOTE_BASE/$MODEL_TAG/results/" \
  --exclude="intermediate" --exclude="intermediate/" \
  --exclude="*.log" --exclude="__pycache__"

if [[ -f "$RESULTS_DIR/test_clozes.json" ]]; then
  sync_path_to_remote "$SSH_CMD" "$RSYNC_OPTS" \
    "$RESULTS_DIR/test_clozes.json" \
    "$USER_NAME@$HOST:$REMOTE_BASE/$MODEL_TAG/results/"
fi

echo "Sync complete for $MODEL_TAG."
