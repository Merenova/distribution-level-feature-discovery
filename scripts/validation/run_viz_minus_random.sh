#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Generate "minus random" visualizations.

This script:
  1) Regenerates steer.json WITH random baseline enabled
  2) Runs visualize_attr_steer_results.py to produce *_minus_random.png plots

Usage:
  ./scripts/run_viz_minus_random.sh --pt-glob "..." --out-dir /tmp/attr_steer_ambigqa [options]

Required:
  --pt-glob GLOB        Glob for *_prefix_context.pt (quote it!)
  --out-dir DIR         Output directory (will write steer.json and viz/)

Options:
  --n-samples N         (200)
  --seed N              (0)
  --model NAME          (Qwen/Qwen3-8B)
  --transcoder NAME     (mwhanna/qwen3-8b-transcoders)
  --dtype DTYPE         (bfloat16)
  --epsilons "LIST"     ("-1 -0.5 -0.1 0 0.1 0.5 1")  (quote it!)
  --top-b N             (10)
  --hc-selection MODE   (full)
  --steering-method M   (sign)
  --word-reduction MODE (mean)
  --random-baseline-seed N (123)
  --viz-max-samples N   (10)
  --viz-max-curves N    (200)
  --viz-heatmap-vmax X  (1.0)

Example:
  ./scripts/run_viz_minus_random.sh \
    --pt-glob "$REPO_ROOT/AmbigQA_Qwen3-8B/results/3_attribution_graphs/*_prefix_context.pt" \
    --out-dir "${TMPDIR:-/tmp}/attr_steer_ambigqa"
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/repo_paths.sh
source "$SCRIPT_DIR/../lib/repo_paths.sh"
REPO_ROOT="${REPO_ROOT:-$(repo_root_from_script "${BASH_SOURCE[0]}")}"

PT_GLOB=""
OUT_DIR=""

N_SAMPLES="200"
SEED="0"
MODEL="Qwen/Qwen3-8B"
TRANSCODER="mwhanna/qwen3-8b-transcoders"
DTYPE="bfloat16"
EPSILONS="-1 -0.5 -0.1 0 0.1 0.5 1"
TOP_B="10"
HC_SELECTION="full"
STEERING_METHOD="sign"
WORD_REDUCTION="mean"
RANDOM_BASELINE_SEED="123"

VIZ_MAX_SAMPLES="10"
VIZ_MAX_CURVES="200"
VIZ_HEATMAP_VMAX="1.0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pt-glob) PT_GLOB="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --n-samples) N_SAMPLES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --transcoder) TRANSCODER="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --epsilons) EPSILONS="$2"; shift 2 ;;
    --top-b) TOP_B="$2"; shift 2 ;;
    --hc-selection) HC_SELECTION="$2"; shift 2 ;;
    --steering-method) STEERING_METHOD="$2"; shift 2 ;;
    --word-reduction) WORD_REDUCTION="$2"; shift 2 ;;
    --random-baseline-seed) RANDOM_BASELINE_SEED="$2"; shift 2 ;;
    --viz-max-samples) VIZ_MAX_SAMPLES="$2"; shift 2 ;;
    --viz-max-curves) VIZ_MAX_CURVES="$2"; shift 2 ;;
    --viz-heatmap-vmax) VIZ_HEATMAP_VMAX="$2"; shift 2 ;;
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

STEER_JSON="${OUT_DIR}/steer.json"
VIZ_DIR="${OUT_DIR}/viz"

echo "[1/2] Regenerate steer.json WITH random baseline -> ${STEER_JSON}"
(cd "${REPO_ROOT}" && uv run python scripts/token_attr/token_attribution_magnitude_by_position.py steer \
  --pt-glob "${PT_GLOB}" \
  --n-samples "${N_SAMPLES}" \
  --seed "${SEED}" \
  --model "${MODEL}" \
  --transcoder "${TRANSCODER}" \
  --dtype "${DTYPE}" \
  --epsilons ${EPSILONS} \
  --top-B "${TOP_B}" \
  --hc-selection "${HC_SELECTION}" \
  --steering-method "${STEERING_METHOD}" \
  --word-reduction "${WORD_REDUCTION}" \
  --random-baseline --random-baseline-seed "${RANDOM_BASELINE_SEED}" \
  --out "${STEER_JSON}")

echo "[2/2] Visualize (includes *_minus_random.png) -> ${VIZ_DIR}"
(cd "${REPO_ROOT}" && uv run python scripts/token_attr/visualize_attr_steer_results.py \
  --input-dir "${OUT_DIR}" \
  --output-dir "${VIZ_DIR}" \
  --max-samples "${VIZ_MAX_SAMPLES}" \
  --max-steer-curves "${VIZ_MAX_CURVES}" \
  --heatmap-vmax "${VIZ_HEATMAP_VMAX}")

echo "Done. Open: ${VIZ_DIR}/index.html"

