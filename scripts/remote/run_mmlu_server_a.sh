#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$ROOT_DIR/scripts/misc/mmlu_remote_baseline_lib.sh"

HOST="${MMLU_QWEN3_8B_HOST:-}"
PORT="${MMLU_QWEN3_8B_PORT:-40206}"
USER_NAME="${MMLU_QWEN3_8B_USER:-${REMOTE_USER:-${USER:-}}}"
REMOTE_BASE="$REMOTE_BASE_DEFAULT"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash scripts/remote/run_mmlu_server_a.sh"
  echo "Runs the MMLU Qwen3-8B baseline worker on the configured remote host."
  exit 0
fi

if [[ -z "$HOST" ]]; then
  echo "Error: set MMLU_QWEN3_8B_HOST before running this script." >&2
  exit 1
fi

exec ssh -tt -p "$PORT" "$USER_NAME@$HOST" \
  "cd '$REMOTE_BASE' && bash scripts/remote/run_remote_mmlu_baselines_worker.sh --model-tag MMLU_Qwen3-8B --config configs/mmlu_qwen3_8b_config.json --gpu-single 0 --gpu-combined 1 --skip-existing"
