#!/bin/bash
# =============================================================================
# Sync the multi-view comparison pipeline and required AmbigQA/Qwen3-8B inputs
# to a remote server.
#
# This script only syncs what multiview_comparison/run_comparison_pipeline.py
# needs to run stages 5/6/7:
#   - code: multiview comparison + reused stage 5/6/7 modules
#   - source results: 2/3/4(embeddings)/5 + small metadata files
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/repo_paths.sh
source "$SCRIPT_DIR/../lib/repo_paths.sh"
REPO_ROOT="${REPO_ROOT:-$(repo_root_from_script "${BASH_SOURCE[0]}")}"

HOST="${MULTIVIEW_REMOTE_HOST:-${HOST:-}}"
PORT="${MULTIVIEW_REMOTE_PORT:-${PORT:-40089}}"
USER_NAME="${MULTIVIEW_REMOTE_USER:-${USER_NAME:-${REMOTE_USER:-${USER:-}}}}"
REMOTE_BASE="${REMOTE_BASE:-/path/to/latent_planning}"

MODEL_TAG="${MODEL_TAG:-AmbigQA_Qwen3-8B}"
RESULTS_REL_DEFAULT="$MODEL_TAG/results"
RESULTS_DIR="${RESULTS_DIR:-$RESULTS_REL_DEFAULT}"
REMOTE_RESULTS_DIR="${REMOTE_RESULTS_DIR:-$RESULTS_REL_DEFAULT}"
CONFIG_REL="${CONFIG_REL:-configs/beta_gamma_scaled_config.json}"

DRY_RUN="${DRY_RUN:-0}"
SYNC_ONLY="${SYNC_ONLY:-1}"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --host HOST                 Remote host (default: $HOST)"
    echo "  --port PORT                 SSH port (default: $PORT)"
    echo "  --user USER                 SSH user (default: $USER_NAME)"
    echo "  --remote-base DIR           Remote repo root (default: $REMOTE_BASE)"
    echo "  --model-tag TAG             Model tag prefix (default: $MODEL_TAG)"
    echo "  --results-dir DIR           Local results dir relative to repo root or absolute path"
    echo "                              (default: $RESULTS_DIR)"
    echo "  --remote-results-dir DIR    Remote results dir relative to remote base"
    echo "                              (default: $REMOTE_RESULTS_DIR)"
    echo "  --config RELPATH            Config path relative to repo root for remote run hint"
    echo "                              (default: $CONFIG_REL)"
    echo "  --dry-run                   Show rsync plan without writing"
    echo "  --print-run                 Print remote run command only after sync"
    echo "  --help                      Show this help"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 --dry-run"
    echo "  $0 --remote-base /path/to/latent_planning"
    echo "  $0 --results-dir AmbigQA_Qwen3-8B/results --remote-results-dir AmbigQA_Qwen3-8B/results"
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
        --model-tag)
            MODEL_TAG="$2"
            shift 2
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --remote-results-dir)
            REMOTE_RESULTS_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_REL="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --print-run)
            SYNC_ONLY=0
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$HOST" ]]; then
    echo "Error: set MULTIVIEW_REMOTE_HOST or pass --host before running this script." >&2
    exit 1
fi

if [[ "$RESULTS_DIR" != /* ]]; then
    LOCAL_RESULTS_DIR="$REPO_ROOT/$RESULTS_DIR"
else
    LOCAL_RESULTS_DIR="$RESULTS_DIR"
fi

if [[ ! -d "$LOCAL_RESULTS_DIR" ]]; then
    echo "Error: local results directory not found: $LOCAL_RESULTS_DIR" >&2
    exit 1
fi

SSH_CONTROL_PATH="/tmp/ssh-multiview-remote-$$"
SSH_CMD="ssh -p $PORT -o ControlMaster=auto -o ControlPath=$SSH_CONTROL_PATH -o ControlPersist=300"
RSYNC_OPTS="-az --human-readable --partial --progress"
if [[ "$DRY_RUN" = "1" ]]; then
    RSYNC_OPTS="$RSYNC_OPTS --dry-run --itemize-changes"
fi

cleanup() {
    ssh -p "$PORT" -o ControlPath="$SSH_CONTROL_PATH" -O exit "$USER_NAME@$HOST" 2>/dev/null || true
}
trap cleanup EXIT

echo "Establishing SSH connection..."
ssh -p "$PORT" -o ControlMaster=yes -o ControlPath="$SSH_CONTROL_PATH" -o ControlPersist=300 -fN "$USER_NAME@$HOST"

REMOTE_RESULTS_PATH="$REMOTE_BASE/$REMOTE_RESULTS_DIR"

CODE_DIRS=(
    "5_gaussian_clustering"
    "6_semantic_graphs"
    "7_validation"
    "multiview_comparison"
    "utils"
    "configs"
    "circuit-tracer"
)

TOP_LEVEL_FILES=(
    "pyproject.toml"
    "uv.lock"
    ".python-version"
)

RESULTS_SUBDIRS=(
    "2_branch_sampling"
    "3_attribution_graphs"
    "4_feature_extraction/embeddings"
    "5_clustering"
)

RESULTS_FILES=(
    "pipeline_config.json"
    "test_clozes.json"
    "manifest_stage2.json"
    "manifest_stage3.json"
    "manifest_stage4a.json"
    "manifest_stage5.json"
    "manifest_stage6.json"
)

COMMON_EXCLUDES=(
    "__pycache__"
    "*.pyc"
    ".git"
    "*.log"
    "*.tmp"
    ".DS_Store"
    "*.swp"
    ".ipynb_checkpoints"
    ".venv/"
    "venv/"
    ".mypy_cache/"
    ".pytest_cache/"
    "archive/"
    "figures/"
    "output/"
    "logs/"
    "*.egg-info/"
)

EXCLUDE_ARGS=""
for pattern in "${COMMON_EXCLUDES[@]}"; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude=$pattern"
done

sync_path() {
    local src="$1"
    local dst="$2"
    shift 2
    rsync $RSYNC_OPTS $EXCLUDE_ARGS "$@" -e "$SSH_CMD" "$src" "$dst"
}

echo "========================================"
echo "SEND MULTIVIEW COMPARISON TO REMOTE"
echo "========================================"
echo "Remote: $USER_NAME@$HOST:$PORT"
echo "Remote base: $REMOTE_BASE"
echo "Local results: $LOCAL_RESULTS_DIR"
echo "Remote results: $REMOTE_RESULTS_PATH"
echo ""

echo "Size breakdown (local):"
for subdir in "${RESULTS_SUBDIRS[@]}"; do
    if [[ -e "$LOCAL_RESULTS_DIR/$subdir" ]]; then
        echo "  $subdir: $(du -sh "$LOCAL_RESULTS_DIR/$subdir" | cut -f1)"
    fi
done
echo ""
echo "Skipping:"
echo "  $LOCAL_RESULTS_DIR/5_clustering/intermediate/"
echo "  $LOCAL_RESULTS_DIR/6_semantic_graphs/"
echo "  $LOCAL_RESULTS_DIR/7_validation/"
echo ""

echo "Step 1/4: Creating remote directory structure..."
ssh -p "$PORT" -o ControlPath="$SSH_CONTROL_PATH" "$USER_NAME@$HOST" \
    "mkdir -p '$REMOTE_BASE' '$REMOTE_BASE/scripts/remote' '$REMOTE_RESULTS_PATH' '$REMOTE_RESULTS_PATH/4_feature_extraction'"

echo ""
echo "Step 2/4: Syncing code..."
for dir in "${CODE_DIRS[@]}"; do
    if [[ -d "$REPO_ROOT/$dir" ]]; then
        echo "  Syncing $dir/"
        sync_path "$REPO_ROOT/$dir" "$USER_NAME@$HOST:$REMOTE_BASE/"
    fi
done

for file in "${TOP_LEVEL_FILES[@]}"; do
    if [[ -f "$REPO_ROOT/$file" ]]; then
        echo "  Syncing $file"
        sync_path "$REPO_ROOT/$file" "$USER_NAME@$HOST:$REMOTE_BASE/"
    fi
done

echo ""
echo "Step 3/4: Syncing result inputs..."
for subdir in "${RESULTS_SUBDIRS[@]}"; do
    src="$LOCAL_RESULTS_DIR/$subdir"
    if [[ ! -e "$src" ]]; then
        continue
    fi
    echo "  Syncing $subdir/"
    extra_args=()
    if [[ "$subdir" == "5_clustering" ]]; then
        extra_args+=(--exclude="intermediate" --exclude="intermediate/")
    fi
    parent_dir="$(dirname "$REMOTE_RESULTS_PATH/$subdir")"
    ssh -p "$PORT" -o ControlPath="$SSH_CONTROL_PATH" "$USER_NAME@$HOST" "mkdir -p '$parent_dir'"
    sync_path "$src" "$USER_NAME@$HOST:$parent_dir/" "${extra_args[@]}"
done

if [[ -d "$LOCAL_RESULTS_DIR/configs" ]]; then
    echo "  Syncing results/configs/"
    sync_path "$LOCAL_RESULTS_DIR/configs" "$USER_NAME@$HOST:$REMOTE_RESULTS_PATH/"
fi

for file in "${RESULTS_FILES[@]}"; do
    src="$LOCAL_RESULTS_DIR/$file"
    if [[ -f "$src" ]]; then
        echo "  Syncing $file"
        sync_path "$src" "$USER_NAME@$HOST:$REMOTE_RESULTS_PATH/"
    fi
done

echo ""
echo "Step 4/4: Syncing this helper script..."
sync_path "$REPO_ROOT/scripts/remote/send_multiview_comparison_remote.sh" "$USER_NAME@$HOST:$REMOTE_BASE/scripts/remote/"

echo ""
echo "========================================"
echo "SYNC COMPLETE"
echo "========================================"
echo ""
echo "SSH command:"
echo "  ssh -p $PORT $USER_NAME@$HOST -L 8080:localhost:8080"
echo ""
echo "Remote run example:"
echo "  cd $REMOTE_BASE"
echo "  uv run python multiview_comparison/run_comparison_pipeline.py \\"
echo "    --source-results-dir $REMOTE_RESULTS_DIR \\"
echo "    --config $CONFIG_REL \\"
echo "    --methods rd,coreg,concat \\"
echo "    --stages 5,6,7"
echo ""
echo "Suggested first smoke run:"
echo "  cd $REMOTE_BASE"
echo "  uv run python multiview_comparison/run_comparison_pipeline.py \\"
echo "    --source-results-dir $REMOTE_RESULTS_DIR \\"
echo "    --config $CONFIG_REL \\"
echo "    --methods rd,coreg,concat \\"
echo "    --stages 5,6 \\"
echo "    --max-prefixes 1 \\"
echo "    --k-values 2 \\"
echo "    --lambda-values 0.0,0.1 \\"
echo "    --alpha-values 0.3,0.7 \\"
echo "    --spectral-max-iter 3"
echo ""
if [[ "$SYNC_ONLY" = "0" ]]; then
    echo "Only printing run commands after sync. No remote job was started automatically."
fi
