#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

CONFIG_FILE="configs/default.json"
PREFIXES_FILE=""
DATASET_DIR=""
DATASET_SPLIT=""
N_GROUPS=""
PREPARE_ONLY="false"
OUTPUT_DIR="$ROOT_DIR"
DRY_RUN="false"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_paper_pipeline.sh [--config CONFIG] --dataset-dir DATASET --output-dir DIR [--split SPLIT] [--n-groups N] [--prepare-only] [--dry-run]
  bash scripts/run_paper_pipeline.sh [--config CONFIG] --prefixes-file PREFIXES --output-dir DIR [--dry-run]

Defaults:
  --config defaults to configs/default.json, which resolves to AmbigQA + Qwen3-8B.

Modes:
  --dataset-dir    AmbigQA workflow. Runs Stage 0 and Stage 1 to create prefixes, then continues with Stage 2 to 7.
  --prefixes-file  Fallback workflow for direct prefix runs.
  --dry-run        Print resolved commands without executing model or stage code.
EOF
}

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'

  if [[ "$DRY_RUN" != "true" ]]; then
    "$@"
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --dataset-dir)
      DATASET_DIR="$2"
      shift 2
      ;;
    --split)
      DATASET_SPLIT="$2"
      shift 2
      ;;
    --n-groups)
      N_GROUPS="$2"
      shift 2
      ;;
    --prefixes-file)
      PREFIXES_FILE="$2"
      shift 2
      ;;
    --prepare-only)
      PREPARE_ONLY="true"
      shift
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "$DATASET_DIR" && -n "$PREFIXES_FILE" ]]; then
  echo "Use either --dataset-dir or --prefixes-file, not both." >&2
  exit 1
fi

if [[ -z "$DATASET_DIR" && -z "$PREFIXES_FILE" ]]; then
  echo "One of --dataset-dir or --prefixes-file is required." >&2
  exit 1
fi

if [[ "$PREPARE_ONLY" == "true" && -z "$DATASET_DIR" ]]; then
  echo "--prepare-only is only valid with --dataset-dir." >&2
  exit 1
fi

RESULTS_DIR="$OUTPUT_DIR/results"
LOGS_DIR="$OUTPUT_DIR/logs"
RESOLVED_CONFIG="$RESULTS_DIR/resolved_config.json"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

uv run python -m utils.config "$CONFIG_FILE" --write "$RESOLVED_CONFIG" >/dev/null
eval "$(uv run python -m utils.config "$CONFIG_FILE" --shell-env)"

echo "Config: $CONFIG_FILE"
echo "Resolved config: $RESOLVED_CONFIG"
echo "Experiment: $EXPERIMENT_NAME"
echo "Dataset: $DATASET_NAME"
echo "Model: $MODEL_NAME"

if [[ -n "$DATASET_DIR" && "$DATA_ADAPTER" != "ambigqa" ]]; then
  echo "--dataset-dir currently supports only DATA_ADAPTER=ambigqa, got: $DATA_ADAPTER" >&2
  exit 1
fi

if [[ -n "$DATASET_DIR" ]]; then
  if [[ -z "$DATASET_SPLIT" ]]; then
    DATASET_SPLIT="$DATA_SPLIT_DEFAULT"
  fi
  if [[ -z "$N_GROUPS" ]]; then
    N_GROUPS="$STAGE1_N_GROUPS"
  fi

  PREPROCESS_DIR="$RESULTS_DIR/0_preprocess"
  DATA_PREP_DIR="$RESULTS_DIR/1_data_preparation"
  GROUPS_FILE="$PREPROCESS_DIR/ambigqa_question_groups.json"
  PREFIXES_FILE="$DATA_PREP_DIR/prefixes.json"

  echo "Stage 0: AmbigQA question preparation"
  run_cmd uv run python 0_preprocess/prepare_ambigqa_questions.py \
    --dataset-dir "$DATASET_DIR" \
    --split "$DATASET_SPLIT" \
    --output "$GROUPS_FILE" \
    --log-dir "$LOGS_DIR"

  echo "Stage 1: AmbigQA question formatting"
  run_cmd uv run python 1_data_preparation/format_ambigqa_questions.py \
    --grouped-questions "$GROUPS_FILE" \
    --model "$MODEL_NAME" \
    --n-groups "$N_GROUPS" \
    --seed "$RANDOM_SEED" \
    --output-dir "$DATA_PREP_DIR" \
    --log-dir "$LOGS_DIR"

  if [[ "$PREPARE_ONLY" == "true" ]]; then
    echo "Preparation complete. Prefixes written to $PREFIXES_FILE"
    exit 0
  fi
fi

SAMPLES_DIR="$RESULTS_DIR/2_branch_sampling"
ATTR_DIR="$RESULTS_DIR/3_attribution_graphs"
EMBED_DIR="$RESULTS_DIR/4_feature_extraction"
CLUSTER_DIR="$RESULTS_DIR/5_clustering"
GRAPH_DIR="$RESULTS_DIR/6_semantic_graphs"
VALIDATION_DIR="$RESULTS_DIR/7_validation"

echo "Stage 2: branch sampling"
run_cmd uv run python 2_branch_sampling/sample_branches.py \
  --prefixes-file "$PREFIXES_FILE" \
  --model "$MODEL_NAME" \
  --max-total-continuations "$MAX_TOTAL_CONTINUATIONS" \
  --nucleus-p "$NUCLEUS_P" \
  --temperature "$TEMPERATURE" \
  --max-tokens "$SAMPLING_MAX_TOKENS" \
  --batch-size "$SAMPLING_BATCH_SIZE" \
  --max-batches "$SAMPLING_MAX_BATCHES" \
  --output-dir "$SAMPLES_DIR" \
  --gpu-memory-utilization "$GPU_MEMORY" \
  --tensor-parallel-size "$TENSOR_PARALLEL" \
  --max-model-len "$MAX_MODEL_LEN"

echo "Stage 3: continuation attribution"
run_cmd uv run python 3_attribution_graphs/compute_continuation_attribution.py \
  --branches-dir "$SAMPLES_DIR" \
  --model "$MODEL_NAME" \
  --transcoder "$TRANSCODER_NAME" \
  --dtype "$MODEL_DTYPE" \
  --max-feature-nodes "$ATTR_MAX_FEATURES" \
  --batch-size "$ATTR_BATCH_SIZE" \
  --output-dir "$ATTR_DIR"

echo "Stage 4: contextual continuation embeddings"
run_cmd uv run python 4_feature_extraction/compute_embeddings.py \
  --samples-dir "$SAMPLES_DIR" \
  --embedding-model "$EMBEDDING_MODEL" \
  --batch-size "$EMBEDDING_BATCH_SIZE" \
  --output-dir "$EMBED_DIR"

echo "Stage 5: rate-distortion clustering"
run_cmd uv run python 5_gaussian_clustering/cluster.py \
  --embeddings-dir "$EMBED_DIR" \
  --attribution-graphs-dir "$ATTR_DIR" \
  --samples-dir "$SAMPLES_DIR" \
  --output-dir "$CLUSTER_DIR" \
  --config "$RESOLVED_CONFIG"

echo "Stage 6: semantic graph extraction"
run_cmd uv run python 6_semantic_graphs/extract_graphs.py \
  --clustering-dir "$CLUSTER_DIR" \
  --samples-dir "$SAMPLES_DIR" \
  --attribution-graphs-dir "$ATTR_DIR" \
  --output-dir "$GRAPH_DIR"

echo "Stage 7: RD steering"
run_cmd uv run python 7_validation/rd_medoid.py \
  --samples-dir "$SAMPLES_DIR" \
  --attribution-graphs-dir "$ATTR_DIR" \
  --clustering-dir "$CLUSTER_DIR" \
  --embeddings-dir "$EMBED_DIR" \
  --output-dir "$VALIDATION_DIR" \
  --config "$RESOLVED_CONFIG"

echo "Stage 7: KM-Sem steering"
run_cmd uv run python 7_validation/km_sem.py \
  --samples-dir "$SAMPLES_DIR" \
  --attribution-graphs-dir "$ATTR_DIR" \
  --clustering-dir "$CLUSTER_DIR" \
  --embeddings-dir "$EMBED_DIR" \
  --output-dir "$VALIDATION_DIR" \
  --config "$RESOLVED_CONFIG"

echo "Stage 7: Single steering"
run_cmd uv run python 7_validation/single.py \
  --samples-dir "$SAMPLES_DIR" \
  --attribution-graphs-dir "$ATTR_DIR" \
  --clustering-dir "$CLUSTER_DIR" \
  --output-dir "$VALIDATION_DIR" \
  --config "$RESOLVED_CONFIG"

echo "Stage 7: aggregate summary CSV"
run_cmd uv run python 7_validation/analyze_steering_methods.py \
  "$VALIDATION_DIR/rd" \
  "$VALIDATION_DIR/km_sem" \
  "$VALIDATION_DIR/single" \
  --output-csv "$VALIDATION_DIR/steering_summary.csv"

echo "Complete. Outputs written under $RESULTS_DIR"
