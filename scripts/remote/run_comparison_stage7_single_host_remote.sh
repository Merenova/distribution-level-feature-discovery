#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

SOURCE_RESULTS_ARG="AmbigQA_Qwen3-8B/results"
COMPARISON_OUTPUT_ARG=""
CONFIG_ARG="configs/beta_gamma_scaled_config.json"
METHODS="combined_medoid,coreg,concat"
HYPOTHESES="H4A"
GPUS="0,1,2,3,4,5,6,7"
K_CLAMP="10"
SKIP_EXISTING="true"
FAST_PLAN="true"
RUN_ID=""
DRY_RUN="false"
QUIET="false"
UV_SYNC="true"
UV_SYNC_ARGS="${UV_SYNC_ARGS:---locked}"

usage() {
  cat <<EOF
Usage: bash scripts/remote/run_comparison_stage7_single_host_remote.sh [options]

Options:
  --source-results-dir PATH    Source results dir (default: $SOURCE_RESULTS_ARG)
  --comparison-output-dir PATH Comparison output dir (default: <source-results-dir>/comparison)
  --config PATH                Config path (default: $CONFIG_ARG)
  --methods CSV                Methods to run (default: $METHODS)
  --hypotheses CSV             Hypotheses (default: $HYPOTHESES)
  --gpus CSV                   GPU ids for slots (default: $GPUS)
  --k-clamp N                  K clamp (default: $K_CLAMP)
  --run-id ID                  Optional run id
  --skip-existing              Reuse existing outputs (default)
  --no-skip-existing           Recompute even if outputs exist
  --fast-plan                  Use prefix-level fast planning (default)
  --no-fast-plan               Use exact manifest planning
  --uv-sync                    Run 'uv sync' before launching (default)
  --no-uv-sync                 Skip the uv sync bootstrap step
  --quiet                      Pass --quiet to slot workers
  --dry-run                    Print the generated inventory and coordinator command
  --help                       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-results-dir)
      SOURCE_RESULTS_ARG="$2"
      shift 2
      ;;
    --comparison-output-dir)
      COMPARISON_OUTPUT_ARG="$2"
      shift 2
      ;;
    --config)
      CONFIG_ARG="$2"
      shift 2
      ;;
    --methods)
      METHODS="$2"
      shift 2
      ;;
    --hypotheses)
      HYPOTHESES="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --k-clamp)
      K_CLAMP="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
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
    --fast-plan)
      FAST_PLAN="true"
      shift
      ;;
    --no-fast-plan)
      FAST_PLAN="false"
      shift
      ;;
    --quiet)
      QUIET="true"
      shift
      ;;
    --uv-sync)
      UV_SYNC="true"
      shift
      ;;
    --no-uv-sync)
      UV_SYNC="false"
      shift
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

resolve_path() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$ROOT_DIR/$value"
  fi
}

stage5_source_method() {
  case "$1" in
    combined_medoid)
      printf 'rd\n'
      ;;
    rd|coreg|concat)
      printf '%s\n' "$1"
      ;;
    *)
      echo "Error: unsupported method: $1" >&2
      exit 1
      ;;
  esac
}

SOURCE_RESULTS_ABS="$(resolve_path "$SOURCE_RESULTS_ARG")"
if [[ -z "$COMPARISON_OUTPUT_ARG" ]]; then
  COMPARISON_OUTPUT_ABS="$SOURCE_RESULTS_ABS/comparison"
else
  COMPARISON_OUTPUT_ABS="$(resolve_path "$COMPARISON_OUTPUT_ARG")"
fi
CONFIG_ABS="$(resolve_path "$CONFIG_ARG")"

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"

if [[ ${#GPU_ARRAY[@]} -eq 0 ]]; then
  echo "Error: GPU list is empty" >&2
  exit 1
fi

NORMALIZED_METHODS=()
IFS=',' read -r -a METHOD_ARRAY <<< "$METHODS"
for method in "${METHOD_ARRAY[@]}"; do
  method="${method//[[:space:]]/}"
  [[ -n "$method" ]] || continue
  NORMALIZED_METHODS+=("$method")
done
if [[ ${#NORMALIZED_METHODS[@]} -eq 0 ]]; then
  echo "Error: methods list is empty" >&2
  exit 1
fi
METHODS="$(IFS=,; echo "${NORMALIZED_METHODS[*]}")"

NORMALIZED_HYPOTHESES=()
IFS=',' read -r -a HYPOTHESIS_ARRAY <<< "$HYPOTHESES"
for hypothesis in "${HYPOTHESIS_ARRAY[@]}"; do
  hypothesis="${hypothesis//[[:space:]]/}"
  [[ -n "$hypothesis" ]] || continue
  NORMALIZED_HYPOTHESES+=("$hypothesis")
done
if [[ ${#NORMALIZED_HYPOTHESES[@]} -eq 0 ]]; then
  echo "Error: hypotheses list is empty" >&2
  exit 1
fi
HYPOTHESES="$(IFS=,; echo "${NORMALIZED_HYPOTHESES[*]}")"

declare -A SEEN_GPUS=()
NORMALIZED_GPUS=()
for gpu_id in "${GPU_ARRAY[@]}"; do
  gpu_id="${gpu_id//[[:space:]]/}"
  [[ -n "$gpu_id" ]] || continue
  if [[ "$gpu_id" =~ [^0-9] ]]; then
    echo "Error: GPU ids must be integers: $gpu_id" >&2
    exit 1
  fi
  if [[ -n "${SEEN_GPUS[$gpu_id]:-}" ]]; then
    echo "Error: duplicate GPU id: $gpu_id" >&2
    exit 1
  fi
  SEEN_GPUS["$gpu_id"]=1
  NORMALIZED_GPUS+=("$gpu_id")
done
GPU_ARRAY=("${NORMALIZED_GPUS[@]}")

if [[ ! -d "$SOURCE_RESULTS_ABS/2_branch_sampling" ]]; then
  echo "Error: missing $SOURCE_RESULTS_ABS/2_branch_sampling" >&2
  exit 1
fi
if [[ ! -d "$SOURCE_RESULTS_ABS/3_attribution_graphs" ]]; then
  echo "Error: missing $SOURCE_RESULTS_ABS/3_attribution_graphs" >&2
  exit 1
fi
if [[ ! -f "$CONFIG_ABS" ]]; then
  echo "Error: config file not found: $CONFIG_ABS" >&2
  exit 1
fi

for method in "${NORMALIZED_METHODS[@]}"; do
  if [[ "$method" == "combined_medoid" ]]; then
    if [[ ! -d "$SOURCE_RESULTS_ABS/4_feature_extraction/embeddings" ]]; then
      echo "Error: combined_medoid requires $SOURCE_RESULTS_ABS/4_feature_extraction/embeddings" >&2
      exit 1
    fi
    for hypothesis in "${NORMALIZED_HYPOTHESES[@]}"; do
      if [[ "${hypothesis^^}" != "H4A" ]]; then
        echo "Error: combined_medoid only supports H4A" >&2
        exit 1
      fi
    done
  fi
  source_method="$(stage5_source_method "$method")"
  if [[ ! -d "$COMPARISON_OUTPUT_ABS/$source_method/5_clustering" ]]; then
    echo "Error: missing $COMPARISON_OUTPUT_ABS/$source_method/5_clustering for method $method" >&2
    exit 1
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t AVAILABLE_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
  declare -A AVAILABLE_GPU_SET=()
  for gpu_id in "${AVAILABLE_GPUS[@]}"; do
    AVAILABLE_GPU_SET["$gpu_id"]=1
  done
  for gpu_id in "${GPU_ARRAY[@]}"; do
    if [[ -z "${AVAILABLE_GPU_SET[$gpu_id]:-}" ]]; then
      echo "Error: requested GPU $gpu_id is not visible to nvidia-smi" >&2
      exit 1
    fi
  done
fi

if [[ -z "$RUN_ID" ]]; then
  RUN_ID="$(date -u +%Y%m%d_%H%M%S)"
fi

INVENTORY_PATH="$(mktemp "${TMPDIR:-/tmp}/comparison-stage7-single-host.XXXXXX.json")"
cleanup() {
  rm -f "$INVENTORY_PATH"
}
trap cleanup EXIT

{
  echo '{'
  echo '  "version": 1,'
  echo '  "slots": ['
  for index in "${!GPU_ARRAY[@]}"; do
    gpu_id="${GPU_ARRAY[$index]}"
    comma=","
    if [[ "$index" -eq "$((${#GPU_ARRAY[@]} - 1))" ]]; then
      comma=""
    fi
    cat <<EOF
    {
      "slot": "slot$index",
      "runner": "local",
      "label": "single-host-gpu$gpu_id",
      "host": "local",
      "repo_root": "$ROOT_DIR",
      "cuda_device": "$gpu_id"
    }$comma
EOF
  done
  echo '  ]'
  echo '}'
} > "$INVENTORY_PATH"

COMMAND=(
  uv run python multiview_comparison/run_stage7_distributed.py
  --source-results-dir "$SOURCE_RESULTS_ABS"
  --comparison-output-dir "$COMPARISON_OUTPUT_ABS"
  --config "$CONFIG_ABS"
  --inventory "$INVENTORY_PATH"
  --methods "$METHODS"
  --hypotheses
)
for hypothesis in "${NORMALIZED_HYPOTHESES[@]}"; do
  COMMAND+=("$hypothesis")
done
COMMAND+=(--k-clamp "$K_CLAMP" --run-id "$RUN_ID")

if [[ "$SKIP_EXISTING" == "true" ]]; then
  COMMAND+=(--skip-existing)
fi
if [[ "$FAST_PLAN" == "true" ]]; then
  COMMAND+=(--fast-plan)
fi
if [[ "$QUIET" == "true" ]]; then
  COMMAND+=(--quiet)
fi
if [[ "$DRY_RUN" == "true" ]]; then
  COMMAND+=(--dry-run)
fi

echo "========================================"
echo "RUN COMPARISON STAGE 7 ON SINGLE HOST"
echo "========================================"
echo "Repo root: $ROOT_DIR"
echo "Source results: $SOURCE_RESULTS_ABS"
echo "Comparison output: $COMPARISON_OUTPUT_ABS"
echo "Config: $CONFIG_ABS"
echo "Methods: $METHODS"
echo "Hypotheses: $HYPOTHESES"
echo "GPUs: $(IFS=,; echo "${GPU_ARRAY[*]}")"
echo "Run ID: $RUN_ID"
echo "Inventory: $INVENTORY_PATH"
echo "uv sync: $UV_SYNC"
if [[ "$UV_SYNC" == "true" && "$DRY_RUN" == "true" ]]; then
  echo "uv sync note: skipped during dry run"
fi
if [[ "$UV_SYNC" == "true" ]]; then
  echo "uv sync args: $UV_SYNC_ARGS"
fi
echo ""
cat "$INVENTORY_PATH"
echo ""
if [[ "$UV_SYNC" == "true" ]]; then
  printf 'Bootstrap:'
  printf ' %q' uv sync
  if [[ -n "$UV_SYNC_ARGS" ]]; then
    # shellcheck disable=SC2206
    UV_SYNC_ARGS_ARRAY=( $UV_SYNC_ARGS )
    for token in "${UV_SYNC_ARGS_ARRAY[@]}"; do
      printf ' %q' "$token"
    done
  fi
  printf '\n'
fi
printf 'Command:'
for token in "${COMMAND[@]}"; do
  printf ' %q' "$token"
done
printf '\n'
echo ""

if [[ "$UV_SYNC" == "true" && "$DRY_RUN" != "true" ]]; then
  UV_SYNC_COMMAND=(uv sync)
  if [[ -n "$UV_SYNC_ARGS" ]]; then
    # shellcheck disable=SC2206
    UV_SYNC_ARGS_ARRAY=( $UV_SYNC_ARGS )
    UV_SYNC_COMMAND+=("${UV_SYNC_ARGS_ARRAY[@]}")
  fi
  "${UV_SYNC_COMMAND[@]}"
fi

"${COMMAND[@]}"
