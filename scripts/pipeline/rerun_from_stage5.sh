#!/usr/bin/env bash
# Rerun pipeline from Stage 5 onward (5 -> 6 -> 7c) in an existing output directory.
#
# Usage:
#   bash scripts/rerun_from_stage5.sh --output_dir DIR [--config CONFIG] [--no-backup] [--cross-prefix-batching] [--quiet]
#
# Default behavior:
#   - backs up existing results for stages 5/6/7c into DIR/results/_backup_from_stage5_<timestamp>/
#   - re-runs: 5,6,7c_combined_medoid,7c_single,7c_kmeans

set -euo pipefail

OUTPUT_DIR=""
CONFIG_FILE=""
DO_BACKUP="true"
CROSS_PREFIX="false"
QUIET="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir|--output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --config|--config-file)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --no-backup)
      DO_BACKUP="false"
      shift
      ;;
    --backup-old)
      DO_BACKUP="true"
      shift
      ;;
    --cross-prefix-batching)
      CROSS_PREFIX="true"
      shift
      ;;
    --quiet)
      QUIET="true"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "Error: --output_dir DIR is required"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

RESULTS_DIR="$OUTPUT_DIR/results"

if [[ "$DO_BACKUP" == "true" ]]; then
  TS="$(date -Iseconds | tr ':' '-')"
  BACKUP_DIR="$RESULTS_DIR/_backup_from_stage5_$TS"
  mkdir -p "$BACKUP_DIR"

  echo "Backing up existing stage outputs to: $BACKUP_DIR"
  for p in \
    "$RESULTS_DIR/5_clustering" \
    "$RESULTS_DIR/6_semantic_graphs" \
    "$RESULTS_DIR/7_validation/7c_combined_medoid" \
    "$RESULTS_DIR/7_validation/7c_single" \
    "$RESULTS_DIR/7_validation/7c_kmeans" \
    "$RESULTS_DIR/configs/stage_5_config.json" \
    "$RESULTS_DIR/configs/stage_6_config.json" \
    "$RESULTS_DIR/configs/stage_7_config.json"
  do
    if [[ -e "$p" ]]; then
      mkdir -p "$BACKUP_DIR/$(dirname "${p#$RESULTS_DIR/}")"
      mv "$p" "$BACKUP_DIR/${p#$RESULTS_DIR/}"
    fi
  done
else
  echo "WARNING: --no-backup set; existing results may be overwritten."
fi

EXTRA_ARGS=()
if [[ "$CROSS_PREFIX" == "true" ]]; then
  EXTRA_ARGS+=(--cross-prefix-batching)
fi
if [[ "$QUIET" == "true" ]]; then
  EXTRA_ARGS+=(--quiet)
fi

echo "Re-running stages: 5,6,7c"
if [[ -n "$CONFIG_FILE" ]]; then
  CONFIG_FILE="$CONFIG_FILE" bash run_pipeline.sh --output_dir "$OUTPUT_DIR" --stages 5,6,7c "${EXTRA_ARGS[@]}"
else
  bash run_pipeline.sh --output_dir "$OUTPUT_DIR" --stages 5,6,7c "${EXTRA_ARGS[@]}"
fi

