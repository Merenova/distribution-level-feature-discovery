#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/repo_paths.sh
source "$SCRIPT_DIR/../lib/repo_paths.sh"
REPO_ROOT="${REPO_ROOT:-$(repo_root_from_script "${BASH_SOURCE[0]}")}"

# Remote connection details
HOST="${REMOTE_SYNC_HOST:-${HOST:-}}"
PORT="${REMOTE_SYNC_PORT:-${PORT:-40394}}"
USER="${REMOTE_SYNC_USER:-${REMOTE_USER:-${USER:-}}}"
REMOTE_BASE="${REMOTE_BASE:-/path/to/latent_planning}"

# Source directory base
LOCAL_BASE="${LOCAL_BASE:-$REPO_ROOT}"

# Optional safety toggles:
# - DRY_RUN=1 : show what would change without writing
# - DELETE=1  : delete extraneous files on the destination (dangerous if misconfigured)
DRY_RUN="${DRY_RUN:-0}"
DELETE="${DELETE:-0}"

# Common patterns to exclude
COMMON_EXCLUDES=(
    "__pycache__"
    "*.pyc"
    ".git"
    "*.log"
    "*.tmp"
    ".DS_Store"
    "*.swp"
    ".ipynb_checkpoints"
    "*output*/"
    "*results*/"
    "*log*"
    ".venv/"
    "venv/"
    ".mypy_cache/"
    ".pytest_cache/"
)

# Patterns to include only when pushing (local -> remote)
PUSH_INCLUDES=(
    "0_preprocess"
    "1_data_preparation"
    "2_branch_sampling"
    "3_attribution_graphs"
    "4_feature_extraction"
    "5_gaussian_clustering"
    "6_semantic_graphs"
    "7_validation"
    "8_visualization"
    "configs"
    "scripts"
    "experiments"
    "circuit-tracer"
    "utils"
    "pyproject.toml"
)

# Patterns to include only when pulling (remote -> local)
PULL_INCLUDES=(
    "downstream"
)

MODE=${1:-push} # Default to push if no argument provided

if [ -z "$HOST" ]; then
    echo "Error: set REMOTE_SYNC_HOST or HOST before running this script." >&2
    exit 1
fi

# Build common exclude arguments
EXCLUDE_ARGS=""
for pattern in "${COMMON_EXCLUDES[@]}"; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude=$pattern"
done

# Build include args that work for either files or directories.
# NOTE: rsync include/Exclude rules are evaluated top-to-bottom, so we must
# finish with a catch-all `--exclude='*'` to make "include-only" actually work.
build_include_args() {
    local out=""
    for pattern in "$@"; do
        # Anchor patterns at repo root (relative to the rsync source root).
        out="$out --include=/$pattern --include=/$pattern/ --include=/$pattern/***"
    done
    echo "$out"
}

RSYNC_OPTS="-avz --human-readable --partial"
if [ "$DRY_RUN" = "1" ]; then
    RSYNC_OPTS="$RSYNC_OPTS --dry-run --itemize-changes"
fi
if [ "$DELETE" = "1" ]; then
    RSYNC_OPTS="$RSYNC_OPTS --delete"
fi

if [ "$MODE" == "pull" ]; then
    # Add pull-specific includes (processed before excludes)
    INCLUDE_ARGS="$(build_include_args "${PULL_INCLUDES[@]}")"
    
    echo "Pulling files from $HOST (Remote -> Local)..."

    # Sync from remote base -> local base, but restrict to include-only paths.
    rsync $RSYNC_OPTS --prune-empty-dirs $INCLUDE_ARGS $EXCLUDE_ARGS --exclude='*' \
        -e "ssh -p $PORT" \
        "$USER@$HOST:$REMOTE_BASE/" \
        "$LOCAL_BASE/"
    
elif [ "$MODE" == "push" ]; then
    # Add push-specific includes (processed before excludes)
    INCLUDE_ARGS="$(build_include_args "${PUSH_INCLUDES[@]}")"
    
    echo "Pushing files to $HOST (Local -> Remote)..."

    # Sync from local base -> remote base, but restrict to include-only paths.
    rsync $RSYNC_OPTS --prune-empty-dirs $INCLUDE_ARGS $EXCLUDE_ARGS --exclude='*' \
        -e "ssh -p $PORT" \
        "$LOCAL_BASE/" \
        "$USER@$HOST:$REMOTE_BASE/"
    
else
    echo "Usage: $0 [push|pull]"
    echo "  push: Sync local -> remote (default)"
    echo "  pull: Sync remote -> local"
    exit 1
fi
