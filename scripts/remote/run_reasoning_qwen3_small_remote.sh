#!/usr/bin/env bash
# Run the Qwen3 small reasoning sweep on a remote host.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_REL="${CONFIG_REL:-configs/reasoning_qwen3_small.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/reasoning_runs}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/reasoning_qwen3_small_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/sweep.log}"
POOLING_OVERRIDE="${POOLING_OVERRIDE:-${REASONING_POOLING:-}}"
ONLY="${ONLY:-}"
DRY_RUN="false"

usage() {
  cat <<'EOF'
Usage: bash scripts/remote/run_reasoning_qwen3_small_remote.sh [options]

Options:
  --config PATH       Config path relative to repo root
  --output-root DIR   Sweep output root
  --pooling NAME      Override config clustering.pooling for every run: mean, sum, or max
  --only LIST         Forwarded model_key:dataset list, e.g. qwen3_0_6b:math500
  --log-dir DIR       Directory for remote run logs
  --dry-run           Print commands without launching real stages
  --help              Show this help

This script is intended to be run after SSHing into the remote server.
The underlying sweep runs GSM8K on GPU 0 and MATH-500 on GPU 1.
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
    --pooling)
      case "${2:-}" in
        mean|sum|max) POOLING_OVERRIDE="$2" ;;
        ""|--*) echo "--pooling requires a value" >&2; usage >&2; exit 1 ;;
        *) echo "--pooling must be one of: mean, sum, max" >&2; usage >&2; exit 1 ;;
      esac
      shift 2
      ;;
    --only)
      ONLY="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      LOG_FILE="$LOG_DIR/sweep.log"
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

runner_cmd=(
  bash scripts/pipeline/run_reasoning_qwen3_small_sweep.sh
  --config "$CONFIG_REL"
  --output-root "$OUTPUT_ROOT"
)
if [[ -n "$POOLING_OVERRIDE" ]]; then
  runner_cmd+=(--pooling "$POOLING_OVERRIDE")
fi
if [[ -n "$ONLY" ]]; then
  runner_cmd+=(--only "$ONLY")
fi
if [[ "$DRY_RUN" == "true" ]]; then
  runner_cmd+=(--dry-run)
fi

if ! config_summary="$(
  uv run python - "$CONFIG_REL" "$POOLING_OVERRIDE" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
pooling_override = sys.argv[2]
with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

attribution = config.get("attribution", {})
clustering = config.get("clustering", {})
store_all = bool(attribution.get("store_all", False))
config_pooling = clustering.get("pooling", "mean")
pooling = pooling_override or config_pooling or "mean"
if pooling not in {"mean", "sum", "max"}:
    raise SystemExit(f"invalid pooling: {pooling}")
if pooling == "max" and not store_all:
    raise SystemExit(
        "pooling=max requires attribution.store_all=true; "
        "compact store_all=false supports mean and sum"
    )

print(f"Store all:         {str(store_all).lower()}")
print(f"Config pooling:    {config_pooling}")
print(f"Effective pooling: {pooling}")
if not store_all and pooling == "mean":
    print("Compact behavior: compact summed attributions are mean-pooled by span length")
elif not store_all and pooling == "sum":
    print("Compact behavior: compact summed attributions are used as sums")
else:
    print("Compact behavior: token-level attributions honor requested pooling")
PY
)"; then
  echo "Error: failed to validate reasoning config/pooling: $CONFIG_REL" >&2
  exit 1
fi

echo "Remote reasoning sweep"
echo "Repo:        $ROOT_DIR"
echo "Config:      $CONFIG_REL"
echo "Output root: $OUTPUT_ROOT"
echo "Log file:    $LOG_FILE"
echo "$config_summary"
if [[ -n "$ONLY" ]]; then
  echo "Only:        $ONLY"
fi
echo ""
printf 'Command: '
printf '%q ' "${runner_cmd[@]}"
printf '\n'

mkdir -p "$LOG_DIR"
if [[ "$DRY_RUN" == "true" ]]; then
  "${runner_cmd[@]}"
else
  "${runner_cmd[@]}" >"$LOG_FILE" 2>&1
fi
