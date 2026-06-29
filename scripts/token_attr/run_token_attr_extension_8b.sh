#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run all token-attribution extension experiments on the AmbigQA Qwen3-8B Stage 3 results.

This is a thin wrapper around:
  ./scripts/run_token_attr_extension_experiments.sh

Fixed inputs:
  pt-glob     = $REPO_ROOT/AmbigQA_Qwen3-8B/results/3_attribution_graphs/*_prefix_context.pt
  model       = Qwen/Qwen3-8B
  transcoder  = mwhanna/qwen3-8b-transcoders

Usage:
  ./scripts/run_token_attr_extension_8b.sh [options]

Options:
  --out-dir DIR         Output directory
                        (default: $REPO_ROOT/tmp/attr_extension_ambigqa_qwen3_8b)
  --device DEV          Passed through to the generic runner
  --n-samples N         (200)
  --seed N              (0)
  --experiments "LIST"  ("all")
  --max-files N         (all)
  --epsilons "LIST"     ("-1 -0.5 -0.1 0 0.1 0.5 1")
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

Notes:
  - `--pt-glob`, `--model`, and `--transcoder` are fixed by this script.
  - All other supported options are forwarded to the generic runner.

Example:
  ./scripts/run_token_attr_extension_8b.sh \
    --out-dir "$REPO_ROOT/tmp/attr_extension_ambigqa_qwen3_8b" \
    --device cuda:0
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/repo_paths.sh
source "$SCRIPT_DIR/../lib/repo_paths.sh"
REPO_ROOT="${REPO_ROOT:-$(repo_root_from_script "${BASH_SOURCE[0]}")}"

PT_GLOB="${REPO_ROOT}/AmbigQA_Qwen3-8B/results/3_attribution_graphs/*_prefix_context.pt"
MODEL="Qwen/Qwen3-8B"
TRANSCODER="mwhanna/qwen3-8b-transcoders"
OUT_DIR="${REPO_ROOT}/tmp/attr_extension_ambigqa_qwen3_8b"

EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --pt-glob|--model|--transcoder)
      echo "Error: $1 is fixed by this script." >&2
      usage
      exit 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --device|--n-samples|--seed|--experiments|--max-files|--epsilons|--top-b|--hc-selection|--steering-method|--word-reduction|--batch-size|--max-seq-len|--random-baseline-seed|--consistency-top-k|--span-ratio-max-feature-nodes|--span-ratio-max-target-positions)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires a value." >&2
        usage
        exit 2
      fi
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

exec "${SCRIPT_DIR}/run_token_attr_extension_experiments.sh" \
  "${EXTRA_ARGS[@]}" \
  --out-dir "${OUT_DIR}" \
  --pt-glob "${PT_GLOB}" \
  --model "${MODEL}" \
  --transcoder "${TRANSCODER}"
