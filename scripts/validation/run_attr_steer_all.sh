#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run (1) sample, (2) teacher-forced steering sweep, (3) position stats via the unified script.

Usage:
  ./scripts/run_attr_steer_all.sh [options]

Options (defaults shown):
  --repo-root PATH         (auto) Path to repo root (latent_planning)
  --pt-glob GLOB           (required) Glob for *_prefix_context.pt (quote it!)
  --out-dir DIR            (/tmp/attr_steer_runs) Output directory
  --model NAME             (Qwen/Qwen3-8B) Base model for ReplacementModel
  --transcoder NAME        (mwhanna/qwen3-8b-transcoders) Transcoder for ReplacementModel
  --tokenizer-model NAME   (Qwen/Qwen3-8B) Tokenizer model id for word grouping in sample mode
  --dtype DTYPE            (bfloat16) One of: float32|float16|bfloat16
  --n-samples N            (20) Number of sampled continuations (sample mode)
  --n-steer-samples N      (10) Number of sampled continuations (steer mode)
  --seed N                 (0) RNG seed
  --top-b N                (10) Top-B features for steering
  --hc-selection MODE      (full) One of: full|positive|negative
  --steering-method NAME   (multiplicative) One of: additive|multiplicative|absolute|sign|scaling
  --random-baseline        (off) Also run random-feature baseline steering (same count, shuffled H_c values)
  --random-baseline-seed N (0)   Seed base for random-feature baseline
  --epsilons "LIST"        ("-1 -0.5 -0.1 0 0.1 0.5 1") Space-separated list (quote it!)
  --word-reduction MODE    (mean) One of: sum|mean (word aggregation for steering deltas)

Example:
  ./scripts/run_attr_steer_all.sh \
    --pt-glob "$REPO_ROOT/AmbigQA_Qwen3-8B/results/3_attribution_graphs/*_prefix_context.pt" \
    --out-dir "${TMPDIR:-/tmp}/attr_steer_ambigqa"
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/repo_paths.sh
source "$SCRIPT_DIR/../lib/repo_paths.sh"
REPO_ROOT="${REPO_ROOT:-$(repo_root_from_script "${BASH_SOURCE[0]}")}"

# Defaults
PT_GLOB=""
OUT_DIR="${TMPDIR:-/tmp}/attr_steer_runs"
MODEL="Qwen/Qwen3-8B"
TRANSCODER="mwhanna/qwen3-8b-transcoders"
TOKENIZER_MODEL="Qwen/Qwen3-8B"
DTYPE="bfloat16"
N_SAMPLES="200"
N_STEER_SAMPLES="200"
SEED="0"
TOP_B="10"
HC_SELECTION="full"
STEERING_METHOD="sign"
EPSILONS="-1 -0.5 -0.1 0 0.1 0.5 1"
WORD_REDUCTION="mean"
RANDOM_BASELINE="0"
RANDOM_BASELINE_SEED="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-root) REPO_ROOT="$2"; shift 2 ;;
    --pt-glob) PT_GLOB="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --transcoder) TRANSCODER="$2"; shift 2 ;;
    --tokenizer-model) TOKENIZER_MODEL="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --n-samples) N_SAMPLES="$2"; shift 2 ;;
    --n-steer-samples) N_STEER_SAMPLES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --top-b) TOP_B="$2"; shift 2 ;;
    --hc-selection) HC_SELECTION="$2"; shift 2 ;;
    --steering-method) STEERING_METHOD="$2"; shift 2 ;;
    --random-baseline) RANDOM_BASELINE="1"; shift 1 ;;
    --random-baseline-seed) RANDOM_BASELINE_SEED="$2"; shift 2 ;;
    --epsilons) EPSILONS="$2"; shift 2 ;;
    --word-reduction) WORD_REDUCTION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${PT_GLOB}" ]]; then
  echo "Error: --pt-glob is required" >&2
  usage
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv not found on PATH. Install uv or run with the repo's python env." >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

POS_OUT="${OUT_DIR}/position_stats.json"
SAMPLE_OUT="${OUT_DIR}/samples.json"
STEER_OUT="${OUT_DIR}/steer.json"

echo "[1/3] position-stats -> ${POS_OUT}"
(cd "${REPO_ROOT}" && uv run python scripts/token_attr/token_attribution_magnitude_by_position.py position-stats \
  --pt-glob "${PT_GLOB}" \
  --out "${POS_OUT}")

echo "[2/3] sample -> ${SAMPLE_OUT}"
(cd "${REPO_ROOT}" && uv run python scripts/token_attr/token_attribution_magnitude_by_position.py sample \
  --pt-glob "${PT_GLOB}" \
  --n-samples "${N_SAMPLES}" \
  --seed "${SEED}" \
  --tokenizer-model "${TOKENIZER_MODEL}" \
  --out "${SAMPLE_OUT}")

echo "[3/3] steer -> ${STEER_OUT}"
(cd "${REPO_ROOT}" && uv run python scripts/token_attr/token_attribution_magnitude_by_position.py steer \
  --pt-glob "${PT_GLOB}" \
  --n-samples "${N_STEER_SAMPLES}" \
  --seed "${SEED}" \
  --model "${MODEL}" \
  --transcoder "${TRANSCODER}" \
  --dtype "${DTYPE}" \
  --epsilons ${EPSILONS} \
  --top-B "${TOP_B}" \
  --hc-selection "${HC_SELECTION}" \
  --steering-method "${STEERING_METHOD}" \
  --word-reduction "${WORD_REDUCTION}" \
  $( [[ "${RANDOM_BASELINE}" == "1" ]] && echo "--random-baseline --random-baseline-seed ${RANDOM_BASELINE_SEED}" ) \
  --out "${STEER_OUT}")

echo "Done."
echo "  ${POS_OUT}"
echo "  ${SAMPLE_OUT}"
echo "  ${STEER_OUT}"

