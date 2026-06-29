#!/bin/bash
# =============================================================================
# Sync files to remote server and run K-means baseline (7c_kmeans) only
# This script is designed to run in parallel with local execution of 7c
# Local runs other 7c baselines, remote only runs K-means (7c_kmeans)
# =============================================================================

set -euo pipefail

# Remote connection details
HOST="${BASELINE_REMOTE_HOST:-${HOST:-}}"
PORT="${BASELINE_REMOTE_PORT:-${PORT:-21393}}"
USER="${BASELINE_REMOTE_USER:-${REMOTE_USER:-${USER:-}}}"
REMOTE_BASE="${REMOTE_BASE:-/path/to/latent_planning}"

# Local paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/repo_paths.sh
source "$SCRIPT_DIR/../lib/repo_paths.sh"
LOCAL_BASE="${LOCAL_BASE:-$(repo_root_from_script "${BASH_SOURCE[0]}")}"
CONFIG_FILE="configs/beta_gamma_scaled_config.json"

# Results directory - can be specified via --results-dir or environment variable
# Supports both:
#   - Direct results dir (e.g., "results" -> uses $LOCAL_BASE/results/)
#   - Results inside output dir (e.g., "beta_gamma_scaled_results_121/gpu0" -> uses that path)
RESULTS_DIR="${RESULTS_DIR:-results}"
REMOTE_OUTPUT_DIR=""  # Optional: specify a different output dir on remote

# Parse arguments
DRY_RUN="${DRY_RUN:-0}"
SKIP_SYNC="${SKIP_SYNC:-0}"
RUN_ONLY="${RUN_ONLY:-0}"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --results-dir DIR    Local results directory to sync from"
    echo "                       Examples:"
    echo "                         --results-dir results"
    echo "                         --results-dir beta_gamma_scaled_results_121/gpu0"
    echo "  --remote-output DIR  Remote output directory (default: same as results-dir)"
    echo "  --dry-run            Show what would be synced without actually syncing"
    echo "  --skip-sync          Skip syncing, only run remote command"
    echo "  --run-only           Alias for --skip-sync"
    echo "  --help               Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  RESULTS_DIR=...   Same as --results-dir"
    echo "  DRY_RUN=1         Same as --dry-run"
    echo "  SKIP_SYNC=1       Same as --skip-sync"
    echo ""
    echo "Examples:"
    echo "  # Sync from default 'results' directory"
    echo "  $0"
    echo ""
    echo "  # Sync from specific results directory"
    echo "  $0 --results-dir beta_gamma_scaled_results_121/gpu0"
    echo ""
    echo "  # Dry run to see what would be synced"
    echo "  $0 --results-dir results --dry-run"
    echo ""
    echo "This script:"
    echo "  1. Syncs necessary code and results to remote server"
    echo "  2. Runs only stage 7c_kmeans (K-means baseline) on remote"
    echo ""
    echo "The local machine can run other Stage 7c baselines while remote runs 7c_kmeans in parallel."
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --remote-output)
            REMOTE_OUTPUT_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --skip-sync|--run-only)
            SKIP_SYNC=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -z "$HOST" ]; then
    echo "Error: set BASELINE_REMOTE_HOST before running this script." >&2
    exit 1
fi

# Set remote output dir to same as results dir if not specified
if [ -z "$REMOTE_OUTPUT_DIR" ]; then
    REMOTE_OUTPUT_DIR="$RESULTS_DIR"
fi

# Determine the actual results path
# Handle both "results" and "output_dir/results" patterns
if [[ "$RESULTS_DIR" == *"/results" ]]; then
    LOCAL_RESULTS_PATH="$LOCAL_BASE/$RESULTS_DIR"
elif [ -d "$LOCAL_BASE/$RESULTS_DIR/results" ]; then
    LOCAL_RESULTS_PATH="$LOCAL_BASE/$RESULTS_DIR/results"
else
    LOCAL_RESULTS_PATH="$LOCAL_BASE/$RESULTS_DIR"
fi

# Verify the results path exists
if [ ! -d "$LOCAL_RESULTS_PATH" ]; then
    echo "Error: Results directory not found: $LOCAL_RESULTS_PATH"
    echo ""
    echo "Available results directories:"
    ls -d "$LOCAL_BASE"/*results* 2>/dev/null || echo "  (none found)"
    ls -d "$LOCAL_BASE"/results 2>/dev/null || true
    echo ""
    echo "Use --results-dir to specify the correct path"
    exit 1
fi

# SSH connection multiplexing to avoid re-authentication issues
SSH_CONTROL_PATH="/tmp/ssh-baseline-remote-$$"
SSH_OPTS="ssh -p $PORT -o ControlMaster=auto -o ControlPath=$SSH_CONTROL_PATH -o ControlPersist=300"

# Common rsync options
RSYNC_OPTS="-avz --human-readable --partial --progress"
if [ "$DRY_RUN" = "1" ]; then
    RSYNC_OPTS="$RSYNC_OPTS --dry-run --itemize-changes"
fi

# Cleanup function
cleanup() {
    # Close the SSH control connection
    ssh -p $PORT -o ControlPath=$SSH_CONTROL_PATH -O exit $USER@$HOST 2>/dev/null || true
}
trap cleanup EXIT

# Establish initial SSH connection
echo "Establishing SSH connection..."
ssh -p $PORT -o ControlMaster=yes -o ControlPath=$SSH_CONTROL_PATH -o ControlPersist=300 -fN $USER@$HOST

echo "========================================"
echo "SYNC & RUN K-MEANS BASELINE ON REMOTE"
echo "========================================"
echo "Remote: $USER@$HOST:$PORT"
echo "Remote path: $REMOTE_BASE"
echo "Local results: $LOCAL_RESULTS_PATH"
echo "Remote output: $REMOTE_BASE/$REMOTE_OUTPUT_DIR"
echo ""

if [ "$SKIP_SYNC" != "1" ]; then
    # Step 1: Sync code files
    echo "Step 1/3: Syncing code files..."
    
    CODE_EXCLUDES=(
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
        "paper_results/"
        "figures/"
        "output/"
        "logs/"
        "*.egg-info/"
    )
    
    CODE_EXCLUDE_ARGS=""
    for pattern in "${CODE_EXCLUDES[@]}"; do
        CODE_EXCLUDE_ARGS="$CODE_EXCLUDE_ARGS --exclude=$pattern"
    done
    
    # Sync code directories (no trailing slash = preserves directory structure)
    CODE_DIRS=(
        "7_validation"
        "utils"
        "circuit-tracer"
        "configs"
    )
    
    for dir in "${CODE_DIRS[@]}"; do
        if [ -d "$LOCAL_BASE/$dir" ]; then
            echo "  Syncing $dir/..."
            rsync $RSYNC_OPTS $CODE_EXCLUDE_ARGS \
                -e "$SSH_OPTS" \
                "$LOCAL_BASE/$dir" \
                "$USER@$HOST:$REMOTE_BASE/"
        fi
    done
    
    # Sync individual code files
    CODE_FILES=(
        "run_pipeline.sh"
        "pyproject.toml"
        "uv.lock"
    )
    
    for file in "${CODE_FILES[@]}"; do
        if [ -f "$LOCAL_BASE/$file" ]; then
            echo "  Syncing $file..."
            rsync $RSYNC_OPTS \
                -e "$SSH_OPTS" \
                "$LOCAL_BASE/$file" \
                "$USER@$HOST:$REMOTE_BASE/"
        fi
    done
    
    # Step 2: Sync results data needed for baseline
    echo ""
    echo "Step 2/3: Syncing results data..."
    
    # These are required for stage 7c_kmeans
    RESULTS_SUBDIRS=(
        "2_branch_sampling"
        "3_attribution_graphs"
        "4_feature_extraction/embeddings"
        "5_clustering"
    )
    RESULTS_FILES=(
        "test_clozes.json"
        "pipeline_config.json"
    )
    
    # Determine remote results path
    if [[ "$REMOTE_OUTPUT_DIR" == *"/results" ]]; then
        REMOTE_RESULTS_PATH="$REMOTE_BASE/$REMOTE_OUTPUT_DIR"
    else
        REMOTE_RESULTS_PATH="$REMOTE_BASE/$REMOTE_OUTPUT_DIR/results"
    fi
    
    # Create results directory on remote
    echo "  Creating remote directory: $REMOTE_RESULTS_PATH"
    ssh -o ControlPath=$SSH_CONTROL_PATH $USER@$HOST "mkdir -p $REMOTE_RESULTS_PATH"
    
    # Sync subdirectories
    for subdir in "${RESULTS_SUBDIRS[@]}"; do
        src="$LOCAL_RESULTS_PATH/$subdir"
        if [ -e "$src" ]; then
            echo "  Syncing $subdir/..."
            # Create parent directory on remote if needed
            parent_dir=$(dirname "$REMOTE_RESULTS_PATH/$subdir")
            ssh -o ControlPath=$SSH_CONTROL_PATH $USER@$HOST "mkdir -p $parent_dir"
            
            # Exclude large intermediate files for 5_clustering
            EXTRA_EXCLUDES=""
            if [[ "$subdir" == "5_clustering" ]]; then
                EXTRA_EXCLUDES="--exclude=intermediate --exclude=intermediate/"
            fi
            
            rsync $RSYNC_OPTS \
                --exclude="*.log" \
                --exclude="__pycache__" \
                $EXTRA_EXCLUDES \
                -e "$SSH_OPTS" \
                "$src" \
                "$USER@$HOST:$parent_dir/"
        else
            echo "  Warning: $src not found, skipping..."
        fi
    done
    
    # Sync individual files
    for file in "${RESULTS_FILES[@]}"; do
        src="$LOCAL_RESULTS_PATH/$file"
        if [ -e "$src" ]; then
            echo "  Syncing $file..."
            rsync $RSYNC_OPTS \
                -e "$SSH_OPTS" \
                "$src" \
                "$USER@$HOST:$REMOTE_RESULTS_PATH/"
        else
            echo "  Warning: $src not found, skipping..."
        fi
    done
    
    echo ""
    echo "Sync complete!"
fi

# Step 3: Run baseline on remote
echo ""
echo "Step 3/3: Running K-means baseline on remote..."
echo ""

# Determine output_dir argument for pipeline
# If REMOTE_OUTPUT_DIR ends with /results, strip it for --output_dir
if [[ "$REMOTE_OUTPUT_DIR" == *"/results" ]]; then
    PIPELINE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR%/results}"
elif [[ "$REMOTE_OUTPUT_DIR" == "results" ]]; then
    PIPELINE_OUTPUT_DIR=""
else
    PIPELINE_OUTPUT_DIR="$REMOTE_OUTPUT_DIR"
fi

# Build the pipeline command
if [ -n "$PIPELINE_OUTPUT_DIR" ]; then
    PIPELINE_CMD="bash run_pipeline.sh --output_dir '$PIPELINE_OUTPUT_DIR' --only 7c_kmeans --skip-existing"
else
    PIPELINE_CMD="bash run_pipeline.sh --only 7c_kmeans --skip-existing"
fi

LOG_FILE="${REMOTE_OUTPUT_DIR//\//_}_7c_kmeans_remote.log"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would execute on remote:"
    echo "  cd $REMOTE_BASE && \\"
    echo "  CONFIG_FILE='$CONFIG_FILE' $PIPELINE_CMD"
else
    echo "Starting remote execution..."
    echo "----------------------------------------"
    
    # Run only stage 7c_kmeans (K-means baseline) with --skip-existing to avoid redoing work
    ssh -o ControlPath=$SSH_CONTROL_PATH $USER@$HOST "
        cd $REMOTE_BASE && \
        export CONFIG_FILE='$CONFIG_FILE' && \
        echo 'Running K-means baseline (stage 7c_kmeans) on remote...' && \
        echo 'Command: $PIPELINE_CMD' && \
        nohup $PIPELINE_CMD > $LOG_FILE 2>&1 &
        echo 'Background job started. PID: '\$!
        echo 'Log file: $REMOTE_BASE/$LOG_FILE'
    "
    
    echo "----------------------------------------"
    echo ""
    echo "Remote baseline job started in background!"
    echo ""
    echo "To monitor:"
    echo "  ssh -p $PORT $USER@$HOST \"tail -f $REMOTE_BASE/$LOG_FILE\""
    echo ""
    echo "To check process:"
    echo "  ssh -p $PORT $USER@$HOST \"ps aux | grep 7c_baseline\""
    echo ""
    echo "To pull results back after completion:"
    echo "  rsync -avz --progress -e 'ssh -p $PORT' \\"
    echo "    $USER@$HOST:$REMOTE_RESULTS_PATH/7_validation/7c_baseline_kmeans/ \\"
    echo "    $LOCAL_RESULTS_PATH/7_validation/7c_baseline_kmeans_remote/"
fi

echo ""
echo "========================================"
echo "DONE"
echo "========================================"
