#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/repo_paths.sh
source "$SCRIPT_DIR/../lib/repo_paths.sh"
ROOT_DIR="${ROOT_DIR:-$(repo_root_from_script "${BASH_SOURCE[0]}")}"

HOST="${COMPARISON_REMOTE_HOST:-${HOST:-}}"
PORT="${COMPARISON_REMOTE_PORT:-${PORT:-5008}}"
USER_NAME="${COMPARISON_REMOTE_USER:-${USER_NAME:-${REMOTE_USER:-${USER:-}}}}"
REMOTE_BASE="${REMOTE_BASE:-/path/to/latent_planning}"
SOURCE_RESULTS_ARG="${SOURCE_RESULTS_DIR:-AmbigQA_Qwen3-8B/results}"
COMPARISON_OUTPUT_ARG="${COMPARISON_OUTPUT_DIR:-}"
CONFIG_ARG="${CONFIG_REL:-configs/beta_gamma_scaled_config.json}"
METHODS="${METHODS:-combined_medoid,coreg,concat}"
DRY_RUN="false"

usage() {
  cat <<EOF
Usage: bash scripts/remote/send_comparison_stage7_single_host.sh [options]

Options:
  --host HOST                  Remote host (default: $HOST)
  --port PORT                  SSH port (default: $PORT)
  --user USER                  SSH user (default: $USER_NAME)
  --remote-base DIR            Remote repo root (default: $REMOTE_BASE)
  --source-results-dir PATH    Local source results dir (default: $SOURCE_RESULTS_ARG)
  --comparison-output-dir PATH Local comparison dir (default: <source-results-dir>/comparison)
  --config PATH                Config path relative to repo root (default: $CONFIG_ARG)
  --methods CSV                Stage-7 methods (default: $METHODS)
  --dry-run                    Show rsync plan without writing
  --help                       Show this help

Set COMPARISON_REMOTE_HOST or pass --host before running.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --user)
      USER_NAME="$2"
      shift 2
      ;;
    --remote-base)
      REMOTE_BASE="$2"
      shift 2
      ;;
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

if [[ -z "$HOST" ]]; then
  echo "Error: set COMPARISON_REMOTE_HOST or pass --host before running this script." >&2
  exit 1
fi

resolve_path() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "$ROOT_DIR/$value"
  fi
}

repo_relative_path() {
  local absolute="$1"
  absolute="$(cd "$(dirname "$absolute")" && pwd)/$(basename "$absolute")"
  if [[ "$absolute" == "$ROOT_DIR" ]]; then
    printf '.\n'
    return
  fi
  if [[ "$absolute" != "$ROOT_DIR/"* ]]; then
    echo "Error: path must be inside repo root: $absolute" >&2
    exit 1
  fi
  printf '%s\n' "${absolute#"$ROOT_DIR/"}"
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

SOURCE_RESULTS_REL="$(repo_relative_path "$SOURCE_RESULTS_ABS")"
COMPARISON_OUTPUT_REL="$(repo_relative_path "$COMPARISON_OUTPUT_ABS")"
CONFIG_RELATIVE="$(repo_relative_path "$CONFIG_ABS")"

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

declare -A STAGE5_METHODS=()
NORMALIZED_METHODS=()
IFS=',' read -r -a METHOD_ARRAY <<< "$METHODS"
for method in "${METHOD_ARRAY[@]}"; do
  method="${method//[[:space:]]/}"
  [[ -n "$method" ]] || continue
  NORMALIZED_METHODS+=("$method")
  if [[ "$method" == "combined_medoid" ]] && [[ ! -d "$SOURCE_RESULTS_ABS/4_feature_extraction/embeddings" ]]; then
    echo "Error: combined_medoid requires $SOURCE_RESULTS_ABS/4_feature_extraction/embeddings" >&2
    exit 1
  fi
  source_method="$(stage5_source_method "$method")"
  STAGE5_METHODS["$source_method"]=1
  if [[ ! -d "$COMPARISON_OUTPUT_ABS/$source_method/5_clustering" ]]; then
    echo "Error: missing $COMPARISON_OUTPUT_ABS/$source_method/5_clustering for method $method" >&2
    exit 1
  fi
done

if [[ ${#NORMALIZED_METHODS[@]} -eq 0 ]]; then
  echo "Error: methods list is empty" >&2
  exit 1
fi
METHODS="$(IFS=,; echo "${NORMALIZED_METHODS[*]}")"

SSH_CONTROL_PATH="/tmp/ssh-comparison-stage7-single-$$"
SSH_CMD="ssh -p $PORT -o ControlMaster=auto -o ControlPath=$SSH_CONTROL_PATH -o ControlPersist=300"
RSYNC_OPTS=(-az --human-readable --partial --progress)
if [[ "$DRY_RUN" == "true" ]]; then
  RSYNC_OPTS+=(--dry-run --itemize-changes)
fi

cleanup() {
  ssh -p "$PORT" -o ControlPath="$SSH_CONTROL_PATH" -O exit "$USER_NAME@$HOST" >/dev/null 2>&1 || true
}
trap cleanup EXIT

sync_path() {
  local src="$1"
  local dst="$2"
  shift 2
  rsync "${RSYNC_OPTS[@]}" \
    --exclude=__pycache__ \
    --exclude='*.pyc' \
    --exclude=.pytest_cache \
    --exclude=.mypy_cache \
    --exclude='*.log' \
    --exclude=.DS_Store \
    --exclude=.git \
    --exclude=.venv \
    --exclude=venv \
    --exclude=dist \
    --exclude=build \
    "$@" \
    -e "$SSH_CMD" \
    "$src" "$dst"
}

echo "Establishing SSH connection to $USER_NAME@$HOST:$PORT"
ssh -p "$PORT" -o ControlMaster=yes -o ControlPath="$SSH_CONTROL_PATH" -o ControlPersist=300 -fN "$USER_NAME@$HOST"

REMOTE_SOURCE_RESULTS="$REMOTE_BASE/$SOURCE_RESULTS_REL"
REMOTE_COMPARISON_OUTPUT="$REMOTE_BASE/$COMPARISON_OUTPUT_REL"

echo "========================================"
echo "SEND COMPARISON STAGE 7 INPUTS"
echo "========================================"
echo "Remote: $USER_NAME@$HOST:$PORT"
echo "Remote base: $REMOTE_BASE"
echo "Source results: $SOURCE_RESULTS_ABS"
echo "Comparison output: $COMPARISON_OUTPUT_ABS"
echo "Methods: $METHODS"
echo ""
echo "Size breakdown:"
du -sh \
  "$SOURCE_RESULTS_ABS/2_branch_sampling" \
  "$SOURCE_RESULTS_ABS/3_attribution_graphs" \
  "$SOURCE_RESULTS_ABS/4_feature_extraction/embeddings" \
  2>/dev/null || true
for source_method in "${!STAGE5_METHODS[@]}"; do
  du -sh "$COMPARISON_OUTPUT_ABS/$source_method/5_clustering" 2>/dev/null || true
done
echo ""

mkdir_targets=(
  "$REMOTE_BASE"
  "$REMOTE_BASE/scripts"
  "$REMOTE_SOURCE_RESULTS"
  "$REMOTE_SOURCE_RESULTS/4_feature_extraction"
  "$REMOTE_COMPARISON_OUTPUT"
)
for source_method in "${!STAGE5_METHODS[@]}"; do
  mkdir_targets+=(
    "$REMOTE_COMPARISON_OUTPUT/$source_method"
    "$REMOTE_COMPARISON_OUTPUT/$source_method/5_clustering"
  )
done
for method in "${NORMALIZED_METHODS[@]}"; do
  mkdir_targets+=(
    "$REMOTE_COMPARISON_OUTPUT/$method"
    "$REMOTE_COMPARISON_OUTPUT/$method/7_validation"
  )
done

echo "Step 1/4: Creating remote directory structure..."
ssh -p "$PORT" -o ControlPath="$SSH_CONTROL_PATH" "$USER_NAME@$HOST" \
  "mkdir -p $(printf "'%s' " "${mkdir_targets[@]}")"

echo ""
echo "Step 2/4: Syncing code and runtime files..."
for dir in 5_gaussian_clustering multiview_comparison 7_validation utils circuit-tracer; do
  echo "  Syncing $dir/"
  sync_path "$ROOT_DIR/$dir" "$USER_NAME@$HOST:$REMOTE_BASE/"
done
for file in pyproject.toml uv.lock .python-version; do
  if [[ -f "$ROOT_DIR/$file" ]]; then
    echo "  Syncing $file"
    sync_path "$ROOT_DIR/$file" "$USER_NAME@$HOST:$REMOTE_BASE/"
  fi
done
for file in \
  "scripts/remote/render_mmlu_pipeline_progress.py" \
  "scripts/remote/run_comparison_stage7_single_host_remote.sh"
do
  echo "  Syncing $file"
  sync_path "$ROOT_DIR/$file" "$USER_NAME@$HOST:$REMOTE_BASE/$(dirname "$file")/"
done
echo "  Syncing $CONFIG_RELATIVE"
sync_path "$CONFIG_ABS" "$USER_NAME@$HOST:$REMOTE_BASE/$(dirname "$CONFIG_RELATIVE")/"

echo ""
echo "Step 3/4: Syncing source results..."
for subdir in 2_branch_sampling 3_attribution_graphs; do
  echo "  Syncing $SOURCE_RESULTS_REL/$subdir/"
  sync_path "$SOURCE_RESULTS_ABS/$subdir" "$USER_NAME@$HOST:$REMOTE_SOURCE_RESULTS/"
done
if [[ -d "$SOURCE_RESULTS_ABS/4_feature_extraction/embeddings" ]]; then
  echo "  Syncing $SOURCE_RESULTS_REL/4_feature_extraction/embeddings/"
  sync_path "$SOURCE_RESULTS_ABS/4_feature_extraction/embeddings" \
    "$USER_NAME@$HOST:$REMOTE_SOURCE_RESULTS/4_feature_extraction/"
fi
if [[ -f "$SOURCE_RESULTS_ABS/test_clozes.json" ]]; then
  echo "  Syncing $SOURCE_RESULTS_REL/test_clozes.json"
  sync_path "$SOURCE_RESULTS_ABS/test_clozes.json" "$USER_NAME@$HOST:$REMOTE_SOURCE_RESULTS/"
fi

echo ""
echo "Step 4/4: Syncing comparison stage-5 inputs and existing stage-7 outputs..."
for source_method in "${!STAGE5_METHODS[@]}"; do
  echo "  Syncing $COMPARISON_OUTPUT_REL/$source_method/5_clustering/"
  sync_path "$COMPARISON_OUTPUT_ABS/$source_method/5_clustering" \
    "$USER_NAME@$HOST:$REMOTE_COMPARISON_OUTPUT/$source_method/"
done
for method in "${NORMALIZED_METHODS[@]}"; do
  if [[ -d "$COMPARISON_OUTPUT_ABS/$method/7_validation/7c_steering" ]]; then
    echo "  Syncing existing $COMPARISON_OUTPUT_REL/$method/7_validation/7c_steering/"
    sync_path "$COMPARISON_OUTPUT_ABS/$method/7_validation/7c_steering" \
      "$USER_NAME@$HOST:$REMOTE_COMPARISON_OUTPUT/$method/7_validation/"
  fi
done

echo ""
echo "========================================"
echo "SYNC COMPLETE"
echo "========================================"
echo "SSH:"
echo "  ssh -p $PORT $USER_NAME@$HOST"
echo ""
echo "Remote run:"
echo "  cd $REMOTE_BASE"
echo "  # runs 'uv sync --locked' automatically before stage 7"
echo "  bash scripts/remote/run_comparison_stage7_single_host_remote.sh \\"
echo "    --source-results-dir $SOURCE_RESULTS_REL \\"
echo "    --comparison-output-dir $COMPARISON_OUTPUT_REL \\"
echo "    --config $CONFIG_RELATIVE \\"
echo "    --methods $METHODS"
