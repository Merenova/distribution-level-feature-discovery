#!/usr/bin/env bash
# Resume Qwen3-0.6B reasoning runs from existing Stage 3 outputs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/reasoning_runs}"
CONFIG_REL="${CONFIG_REL:-configs/reasoning_qwen3_small.yaml}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/reasoning_qwen3_0_6b_stage4_5_$(date +%Y%m%d_%H%M%S)}"
HF_ENV_FILE="${HF_ENV_FILE:-/root/.latent_planning/hf_env}"
DRY_RUN="false"

usage() {
  cat <<'EOF'
Usage: bash scripts/remote/run_reasoning_qwen3_0_6b_stage4_5_remote.sh [options]

Options:
  --config PATH       Config path relative to repo root
  --output-root DIR   Sweep output root
  --log-dir DIR       Directory for per-lane logs
  --dry-run           Print commands without running
  --help              Show this help

This resumes only:
  qwen3_0_6b:gsm8k
  qwen3_0_6b:math500

It expects existing:
  2_reasoning_pair_samples/
  3_attribution_graphs/
  runtime_clustering_config.json
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

import shlex
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

embedding = config.get("embedding", {})
clustering = config.get("clustering", {})

def emit(name: str, value: object) -> None:
    print(f"{name}={shlex.quote(str(value))}")

def bool_text(value: object) -> str:
    return "true" if bool(value) else "false"

emit("EMBEDDING_MODEL", embedding.get("model_name", "google/embeddinggemma-300m"))
emit("EMBEDDING_BATCH_SIZE", embedding.get("batch_size", 32))
emit("POOLING", clustering.get("pooling", "mean"))
emit("ATTRIBUTION_METRIC", clustering.get("attribution_metric", "l1"))
emit("NORMALIZE_DIMS", bool_text(clustering.get("normalize_dims", False)))
emit("SAVE_INTERMEDIATE", bool_text(clustering.get("save_intermediate", False)))
emit("SKIP_EXISTING", bool_text(clustering.get("skip_existing", False)))
PY
)"; then
  echo "Error: failed to read config: $CONFIG_REL" >&2
  exit 1
fi
eval "$config_exports"

run_lane() {
  local lane="$1"
  local gpu="$2"
  local root="$OUTPUT_ROOT/$lane"
  local log_file="$LOG_DIR/${lane}.log"

  local embedding_cmd=(
    uv run python 4_feature_extraction/compute_embeddings.py
    --samples-dir "$root/2_reasoning_pair_samples"
    --embedding-model "$EMBEDDING_MODEL"
    --batch-size "$EMBEDDING_BATCH_SIZE"
    --output-dir "$root/4_feature_extraction/embeddings"
  )
  local clustering_cmd=(
    uv run python 5_gaussian_clustering/cluster.py
    --samples-dir "$root/2_reasoning_pair_samples"
    --attribution-graphs-dir "$root/3_attribution_graphs"
    --embeddings-dir "$root/4_feature_extraction/embeddings"
    --output-dir "$root/5_gaussian_clustering"
    --config "$root/runtime_clustering_config.json"
    --pooling "$POOLING"
    --attribution-metric "$ATTRIBUTION_METRIC"
  )

  if [[ "$SAVE_INTERMEDIATE" == "true" ]]; then
    clustering_cmd+=(--save-intermediate)
  fi
  if [[ "$NORMALIZE_DIMS" == "true" ]]; then
    clustering_cmd+=(--normalize-dims)
  fi
  if [[ "$SKIP_EXISTING" == "true" ]]; then
    clustering_cmd+=(--skip-existing)
  fi

  {
    echo "=== $lane on CUDA_VISIBLE_DEVICES=$gpu ==="
    echo "Embedding model: $EMBEDDING_MODEL"
    printf 'Command: CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    printf '%q ' "${embedding_cmd[@]}"
    printf '\n'
    printf 'Command: CUDA_VISIBLE_DEVICES=%q ' "$gpu"
    printf '%q ' "${clustering_cmd[@]}"
    printf '\n'

    if [[ "$DRY_RUN" != "true" ]]; then
      CUDA_VISIBLE_DEVICES="$gpu" "${embedding_cmd[@]}"
      CUDA_VISIBLE_DEVICES="$gpu" "${clustering_cmd[@]}"
    fi
  } >"$log_file" 2>&1
}

run_lane qwen3_0_6b_gsm8k 0 &
pid_gsm8k="$!"
run_lane qwen3_0_6b_math500 1 &
pid_math500="$!"

status=0
wait "$pid_gsm8k" || status=1
wait "$pid_math500" || status=1

echo "Logs: $LOG_DIR"
exit "$status"
