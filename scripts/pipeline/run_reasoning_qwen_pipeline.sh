#!/usr/bin/env bash
# Run the reasoning-specific Qwen pipeline from step rollouts through clustering.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG=""
MODEL=""
DATASET=""
TRANSCODER="${REASONING_TRANSCODER:-}"
OUTPUT_ROOT=""
POOLING_OVERRIDE="${REASONING_POOLING:-}"
DRY_RUN="false"

usage() {
  cat <<EOF
Usage: bash scripts/pipeline/run_reasoning_qwen_pipeline.sh \\
  --config configs/reasoning_qwen3_small.yaml \\
  --model Qwen/Qwen3-0.6B \\
  --dataset gsm8k \\
  --transcoder "\$REASONING_TRANSCODER" \\
  --output-root experiments/reasoning_runs/qwen3_0_6b_gsm8k [--dry-run]

Options:
  --config PATH       Reasoning YAML config
  --model NAME        Hugging Face model name
  --dataset NAME      Dataset key, e.g. gsm8k or math500
  --transcoder NAME   Transcoder id/path; falls back to REASONING_TRANSCODER
  --output-root DIR   Output root for all reasoning pipeline stages
  --pooling NAME      Override config clustering.pooling: mean, sum, or max
  --dry-run           Validate, write runtime clustering config, print commands, and exit
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
    --model)
      require_value "$1" "${2:-}"
      MODEL="$2"
      shift 2
      ;;
    --dataset)
      require_value "$1" "${2:-}"
      DATASET="$2"
      shift 2
      ;;
    --transcoder)
      require_value "$1" "${2:-}"
      TRANSCODER="$2"
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

[[ -n "$CONFIG" ]] || die "--config is required"
[[ -f "$CONFIG" ]] || die "--config does not exist: $CONFIG"
[[ -n "$MODEL" ]] || die "--model must be non-empty"
[[ -n "$DATASET" ]] || die "--dataset must be non-empty"
[[ -n "$TRANSCODER" ]] || die "--transcoder must be non-empty or REASONING_TRANSCODER must be set"
[[ -n "$OUTPUT_ROOT" ]] || die "--output-root is required"
command -v uv >/dev/null 2>&1 || die "uv is required but was not found on PATH"

mkdir -p "$OUTPUT_ROOT"

BRANCHES_DIR="$OUTPUT_ROOT/2_branch_sampling"
PAIR_SAMPLES_DIR="$OUTPUT_ROOT/2_reasoning_pair_samples"
PAIR_ATTRIBUTION_DIR="$OUTPUT_ROOT/3_attribution_graphs"
EMBEDDINGS_DIR="$OUTPUT_ROOT/4_feature_extraction/embeddings"
CLUSTERING_DIR="$OUTPUT_ROOT/5_gaussian_clustering"
RUNTIME_CLUSTERING_CONFIG="$OUTPUT_ROOT/runtime_clustering_config.json"

if ! config_exports="$(
  uv run python - "$CONFIG" "$RUNTIME_CLUSTERING_CONFIG" "$POOLING_OVERRIDE" <<'PY'
from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import yaml


def emit(name: str, value: object) -> None:
    print(f"{name}={shlex.quote(str(value))}")


def bool_text(value: object) -> str:
    return "true" if bool(value) else "false"


config_path = Path(sys.argv[1])
runtime_config_path = Path(sys.argv[2])
pooling_override = sys.argv[3]
with config_path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle) or {}

reasoning = config.get("reasoning", {})
models = config.get("models", {})
attribution = config.get("attribution", {})
embedding = config.get("embedding", {})
clustering = dict(config.get("clustering", {}))

beta_values = clustering.pop("beta_values", None)
gamma_values = clustering.pop("gamma_values", None)
sweeps = dict(clustering.get("sweeps", {}))
if beta_values is not None:
    sweeps["beta_values"] = beta_values
if gamma_values is not None:
    sweeps["gamma_values"] = gamma_values
if sweeps:
    clustering["sweeps"] = sweeps
if pooling_override:
    if pooling_override not in {"mean", "sum", "max"}:
        raise SystemExit(f"invalid pooling override: {pooling_override}")
    clustering["pooling"] = pooling_override

runtime_config_path.parent.mkdir(parents=True, exist_ok=True)
with runtime_config_path.open("w", encoding="utf-8") as handle:
    json.dump({"clustering": clustering}, handle, indent=2)
    handle.write("\n")

emit("SAMPLES_PER_STEP", reasoning.get("samples_per_step", ""))
emit("MAX_STEPS", reasoning.get("max_steps", ""))
emit("DTYPE", models.get("dtype", "bfloat16"))
emit("ATTRIBUTION_BACKEND", attribution.get("backend", "auto"))
emit("ATTRIBUTION_BATCH_SIZE", attribution.get("batch_size", 512))
emit("MAX_FEATURE_NODES", attribution.get("max_feature_nodes", 8192))
emit("STORE_ALL", bool_text(attribution.get("store_all", False)))
emit("EMBEDDING_MODEL", embedding.get("model_name", "google/embeddinggemma-300m"))
emit("EMBEDDING_BATCH_SIZE", embedding.get("batch_size", 32))
emit("POOLING", clustering.get("pooling", "mean"))
emit("ATTRIBUTION_METRIC", clustering.get("attribution_metric", "l1"))
emit("NORMALIZE_DIMS", bool_text(clustering.get("normalize_dims", False)))
emit("SAVE_INTERMEDIATE", bool_text(clustering.get("save_intermediate", False)))
emit("SKIP_EXISTING", bool_text(clustering.get("skip_existing", False)))
PY
)"; then
  die "failed to read reasoning config: $CONFIG"
fi
eval "$config_exports"

if [[ "$STORE_ALL" != "true" && "$POOLING" == "max" ]]; then
  die "pooling=max requires attribution.store_all=true; compact store_all=false supports mean and sum"
fi

sample_cmd=(
  uv run python 2_branch_sampling/sample_reasoning_steps.py
  --config "$CONFIG"
  --model "$MODEL"
  --dataset "$DATASET"
  --output-dir "$BRANCHES_DIR"
)
if [[ -n "$SAMPLES_PER_STEP" ]]; then
  sample_cmd+=(--samples-per-step "$SAMPLES_PER_STEP")
fi
if [[ -n "$MAX_STEPS" ]]; then
  sample_cmd+=(--max-steps "$MAX_STEPS")
fi

attribution_cmd=(
  uv run python 3_attribution_graphs/compute_reasoning_step_pair_attribution.py
  --branches-dir "$BRANCHES_DIR"
  --pair-samples-dir "$PAIR_SAMPLES_DIR"
  --model "$MODEL"
  --transcoder "$TRANSCODER"
  --dtype "$DTYPE"
  --batch-size "$ATTRIBUTION_BATCH_SIZE"
  --max-feature-nodes "$MAX_FEATURE_NODES"
  --backend "$ATTRIBUTION_BACKEND"
  --output-dir "$PAIR_ATTRIBUTION_DIR"
)
if [[ "$STORE_ALL" == "true" ]]; then
  attribution_cmd+=(--store-all)
fi

embedding_cmd=(
  uv run python 4_feature_extraction/compute_embeddings.py
  --samples-dir "$PAIR_SAMPLES_DIR"
  --embedding-model "$EMBEDDING_MODEL"
  --batch-size "$EMBEDDING_BATCH_SIZE"
  --output-dir "$EMBEDDINGS_DIR"
)

clustering_cmd=(
  uv run python 5_gaussian_clustering/cluster.py
  --samples-dir "$PAIR_SAMPLES_DIR"
  --attribution-graphs-dir "$PAIR_ATTRIBUTION_DIR"
  --embeddings-dir "$EMBEDDINGS_DIR"
  --output-dir "$CLUSTERING_DIR"
  --config "$RUNTIME_CLUSTERING_CONFIG"
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

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

if [[ "$DRY_RUN" == "true" ]]; then
  print_command "${sample_cmd[@]}"
  print_command "${attribution_cmd[@]}"
  print_command "${embedding_cmd[@]}"
  print_command "${clustering_cmd[@]}"
  exit 0
fi

"${sample_cmd[@]}"
"${attribution_cmd[@]}"
"${embedding_cmd[@]}"
"${clustering_cmd[@]}"
