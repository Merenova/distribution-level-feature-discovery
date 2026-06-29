#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the standalone token-attribution extension experiments and visualizations.

This script:
  1) Runs token_attr_extension_experiments.py
  2) Runs visualize_token_attr_extension_experiments.py

Usage:
  ./scripts/run_token_attr_extension_experiments.sh --pt-glob "..." --out-dir /tmp/token_attr_ext [options]

Required:
  --pt-glob GLOB        Glob for *_prefix_context.pt (quote it!)
  --out-dir DIR         Output directory for JSONs and viz/

Options:
  --experiments "LIST"  ("all") Quote it. Example: "consistency span-steer"
  --max-files N         (all)
  --n-samples N         (200)
  --seed N              (0)
  --model NAME          (Qwen/Qwen3-8B)
  --transcoder NAME     (mwhanna/qwen3-8b-transcoders)
  --dtype DTYPE         (bfloat16)
  --device DEV          (auto)
  --epsilons "LIST"     ("-1 -0.5 -0.1 0 0.1 0.5 1") Quote it
  --top-b N             (10)
  --hc-selection MODE   (full)
  --steering-method M   (sign)
  --word-reduction MODE (mean)
  --batch-size N        (16)
  --max-seq-len N       (none)
  --random-baseline-seed N (123)
  --consistency-top-k N (100)
  --span-ratio-max-feature-nodes N (Stage 3 feature cap)
  --span-ratio-max-target-positions N (all target positions)

Example:
  ./scripts/run_token_attr_extension_experiments.sh \
    --pt-glob "$REPO_ROOT/AmbigQA_Qwen3-8B/results/3_attribution_graphs/*_prefix_context.pt" \
    --out-dir "${TMPDIR:-/tmp}/token_attr_ext"
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/repo_paths.sh
source "$SCRIPT_DIR/../lib/repo_paths.sh"
REPO_ROOT="${REPO_ROOT:-$(repo_root_from_script "${BASH_SOURCE[0]}")}"

PT_GLOB=""
OUT_DIR=""
EXPERIMENTS="all"
MAX_FILES=""
N_SAMPLES="200"
SEED="0"
MODEL="Qwen/Qwen3-8B"
TRANSCODER="mwhanna/qwen3-8b-transcoders"
DTYPE="bfloat16"
DEVICE="auto"
EPSILONS="-1 -0.5 -0.1 0 0.1 0.5 1"
TOP_B="10"
HC_SELECTION="full"
STEERING_METHOD="sign"
WORD_REDUCTION="mean"
BATCH_SIZE="16"
MAX_SEQ_LEN=""
RANDOM_BASELINE_SEED="123"
CONSISTENCY_TOP_K="100"
SPAN_RATIO_MAX_FEATURE_NODES=""
SPAN_RATIO_MAX_TARGET_POSITIONS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pt-glob) PT_GLOB="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --experiments) EXPERIMENTS="$2"; shift 2 ;;
    --max-files) MAX_FILES="$2"; shift 2 ;;
    --n-samples) N_SAMPLES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --transcoder) TRANSCODER="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --epsilons) EPSILONS="$2"; shift 2 ;;
    --top-b) TOP_B="$2"; shift 2 ;;
    --hc-selection) HC_SELECTION="$2"; shift 2 ;;
    --steering-method) STEERING_METHOD="$2"; shift 2 ;;
    --word-reduction) WORD_REDUCTION="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
    --random-baseline-seed) RANDOM_BASELINE_SEED="$2"; shift 2 ;;
    --consistency-top-k) CONSISTENCY_TOP_K="$2"; shift 2 ;;
    --span-ratio-max-feature-nodes) SPAN_RATIO_MAX_FEATURE_NODES="$2"; shift 2 ;;
    --span-ratio-max-target-positions) SPAN_RATIO_MAX_TARGET_POSITIONS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${PT_GLOB}" || -z "${OUT_DIR}" ]]; then
  echo "Error: --pt-glob and --out-dir are required" >&2
  usage
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv not found on PATH." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

echo "[1/2] Run extension experiments -> ${OUT_DIR}"
(cd "${REPO_ROOT}" && uv run python scripts/token_attr/token_attr_extension_experiments.py \
  --pt-glob "${PT_GLOB}" \
  --out-dir "${OUT_DIR}" \
  --experiments ${EXPERIMENTS} \
  --n-samples "${N_SAMPLES}" \
  --seed "${SEED}" \
  --consistency-top-k "${CONSISTENCY_TOP_K}" \
  --model "${MODEL}" \
  --transcoder "${TRANSCODER}" \
  --dtype "${DTYPE}" \
  --device "${DEVICE}" \
  --epsilons ${EPSILONS} \
  --top-B "${TOP_B}" \
  --hc-selection "${HC_SELECTION}" \
  --steering-method "${STEERING_METHOD}" \
  --word-reduction "${WORD_REDUCTION}" \
  --batch-size "${BATCH_SIZE}" \
  --random-baseline-seed "${RANDOM_BASELINE_SEED}" \
  $( [[ -n "${MAX_FILES}" ]] && printf -- "--max-files %q " "${MAX_FILES}" ) \
  $( [[ -n "${MAX_SEQ_LEN}" ]] && printf -- "--max-seq-len %q " "${MAX_SEQ_LEN}" ) \
  $( [[ -n "${SPAN_RATIO_MAX_FEATURE_NODES}" ]] && printf -- "--span-ratio-max-feature-nodes %q " "${SPAN_RATIO_MAX_FEATURE_NODES}" ) \
  $( [[ -n "${SPAN_RATIO_MAX_TARGET_POSITIONS}" ]] && printf -- "--span-ratio-max-target-positions %q " "${SPAN_RATIO_MAX_TARGET_POSITIONS}" ))

echo "[2/2] Visualize extension outputs -> ${OUT_DIR}/viz"
(cd "${REPO_ROOT}" && uv run python scripts/token_attr/visualize_token_attr_extension_experiments.py \
  --input-dir "${OUT_DIR}" \
  --output-dir "${OUT_DIR}/viz")

echo "Done. Open: ${OUT_DIR}/viz/index.html"
