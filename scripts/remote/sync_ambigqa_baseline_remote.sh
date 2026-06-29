#!/bin/bash
# =============================================================================
# Sync AmbigQA results to remote server for K-means baseline (7c_kmeans) only
# MINIMAL sync - excludes large unnecessary files like 5_clustering/intermediate/
# =============================================================================

set -euo pipefail

# Remote connection details
HOST="${AMBIGQA_BASELINE_REMOTE_HOST:-${HOST:-}}"
PORT="${AMBIGQA_BASELINE_REMOTE_PORT:-${PORT:-21393}}"
USER="${AMBIGQA_BASELINE_REMOTE_USER:-${REMOTE_USER:-${USER:-}}}"
REMOTE_BASE="${REMOTE_BASE:-/path/to/latent_planning}"

# Local paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/repo_paths.sh
source "$SCRIPT_DIR/../lib/repo_paths.sh"
LOCAL_BASE="${LOCAL_BASE:-$(repo_root_from_script "${BASH_SOURCE[0]}")}"

# Parse arguments
DRY_RUN="${DRY_RUN:-0}"
MODEL="${MODEL:-8B}"  # 8B or 4B
SYNC_ONLY="${SYNC_ONLY:-0}"

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --model 8B|4B   Which model to sync (default: 8B)"
    echo "  --sync-only     Only sync files, don't run baseline on remote"
    echo "  --dry-run       Show what would be synced without actually syncing"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --model 8B"
    echo "  $0 --model 8B --sync-only"
    echo "  $0 --model 4B --dry-run"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --sync-only)
            SYNC_ONLY=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
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
    echo "Error: set AMBIGQA_BASELINE_REMOTE_HOST before running this script." >&2
    exit 1
fi

# Set paths based on model
if [[ "$MODEL" == "8B" ]]; then
    LOCAL_DIR="$LOCAL_BASE/AmbigQA_Qwen3-8B"
    CONFIG_FILE="configs/beta_gamma_scaled_config.json"
elif [[ "$MODEL" == "4B" ]]; then
    LOCAL_DIR="$LOCAL_BASE/AmbigQA_Qwen3-4B"
    CONFIG_FILE="configs/beta_gamma_scaled_qwen4.json"
else
    echo "Error: MODEL must be 8B or 4B"
    exit 1
fi

REMOTE_DIR="$REMOTE_BASE/AmbigQA_Qwen3-$MODEL"
LOCAL_RESULTS="$LOCAL_DIR/results"

# Verify local directory exists
if [ ! -d "$LOCAL_RESULTS" ]; then
    echo "Error: Results directory not found: $LOCAL_RESULTS"
    exit 1
fi

# SSH connection multiplexing to avoid re-authentication issues
SSH_CONTROL_PATH="/tmp/ssh-ambigqa-remote-$$"
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
echo "SYNC AMBIGQA Qwen3-$MODEL FOR BASELINE"
echo "========================================"
echo "Local: $LOCAL_RESULTS"
echo "Remote: $REMOTE_DIR/results"
echo ""

# Calculate sizes
echo "Size breakdown (local):"
echo "  2_branch_sampling:  $(du -sh "$LOCAL_RESULTS/2_branch_sampling" 2>/dev/null | cut -f1)"
echo "  3_attribution_graphs: $(du -sh "$LOCAL_RESULTS/3_attribution_graphs" 2>/dev/null | cut -f1)"
echo "  4_feature_extraction: $(du -sh "$LOCAL_RESULTS/4_feature_extraction" 2>/dev/null | cut -f1)"
echo "  5_clustering (excl. intermediate): ~$(ls -la "$LOCAL_RESULTS/5_clustering/"*.json 2>/dev/null | wc -l) JSON files"
echo ""
echo "SKIPPING:"
echo "  5_clustering/intermediate/ (1+ TB)"
echo "  6_semantic_graphs/"
echo "  7_validation/"
echo ""

# Step 1: Sync code files
echo "Step 1/5: Syncing code files..."

CODE_EXCLUDES="--exclude=__pycache__ --exclude=*.pyc --exclude=.git --exclude=*.log --exclude=archive/ --exclude=paper_results/ --exclude=figures/"

# Sync directories (no trailing slash = copy directory itself)
CODE_DIRS=(
    "7_validation"
    "utils"
    "circuit-tracer"
    "configs"
)

for dir in "${CODE_DIRS[@]}"; do
    if [ -d "$LOCAL_BASE/$dir" ]; then
        echo "  Syncing $dir/..."
        rsync $RSYNC_OPTS $CODE_EXCLUDES \
            -e "$SSH_OPTS" \
            "$LOCAL_BASE/$dir" \
            "$USER@$HOST:$REMOTE_BASE/"
    fi
done

# Sync individual files
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

# Create remote results directory structure
echo ""
echo "Creating remote directory structure..."
ssh -o ControlPath=$SSH_CONTROL_PATH $USER@$HOST "mkdir -p $REMOTE_DIR/results"

# Step 2: Sync 2_branch_sampling
echo ""
echo "Step 2/5: Syncing 2_branch_sampling..."
rsync $RSYNC_OPTS \
    -e "$SSH_OPTS" \
    "$LOCAL_RESULTS/2_branch_sampling" \
    "$USER@$HOST:$REMOTE_DIR/results/"

# Step 3: Sync 3_attribution_graphs
echo ""
echo "Step 3/5: Syncing 3_attribution_graphs..."
rsync $RSYNC_OPTS \
    -e "$SSH_OPTS" \
    "$LOCAL_RESULTS/3_attribution_graphs" \
    "$USER@$HOST:$REMOTE_DIR/results/"

# Step 4: Sync 4_feature_extraction (only embeddings subfolder)
echo ""
echo "Step 4/5: Syncing 4_feature_extraction/embeddings..."
ssh -o ControlPath=$SSH_CONTROL_PATH $USER@$HOST "mkdir -p $REMOTE_DIR/results/4_feature_extraction"
rsync $RSYNC_OPTS \
    -e "$SSH_OPTS" \
    "$LOCAL_RESULTS/4_feature_extraction/embeddings" \
    "$USER@$HOST:$REMOTE_DIR/results/4_feature_extraction/"

# Step 5: Sync 5_clustering (ONLY JSON files, NO intermediate/)
echo ""
echo "Step 5/5: Syncing 5_clustering/*.json (excluding intermediate/)..."
rsync $RSYNC_OPTS \
    --exclude="intermediate" \
    --exclude="intermediate/" \
    -e "$SSH_OPTS" \
    "$LOCAL_RESULTS/5_clustering" \
    "$USER@$HOST:$REMOTE_DIR/results/"

# Step 6: Sync metadata files
echo ""
echo "Syncing metadata files..."
for file in test_clozes.json pipeline_config.json; do
    if [ -f "$LOCAL_RESULTS/$file" ]; then
        rsync $RSYNC_OPTS \
            -e "$SSH_OPTS" \
            "$LOCAL_RESULTS/$file" \
            "$USER@$HOST:$REMOTE_DIR/results/"
    fi
done

# Sync configs directory
if [ -d "$LOCAL_RESULTS/configs" ]; then
    rsync $RSYNC_OPTS \
        -e "$SSH_OPTS" \
        "$LOCAL_RESULTS/configs" \
        "$USER@$HOST:$REMOTE_DIR/results/"
fi

echo ""
echo "========================================"
echo "SYNC COMPLETE!"
echo "========================================"
echo ""

if [ "$DRY_RUN" != "1" ] && [ "$SYNC_ONLY" != "1" ]; then
    echo "To run baseline on remote:"
    echo "  ssh -p $PORT $USER@$HOST"
    echo "  cd $REMOTE_BASE"
    echo "  CONFIG_FILE='$CONFIG_FILE' bash run_pipeline.sh --output_dir AmbigQA_Qwen3-$MODEL --only 7c_kmeans --skip-existing"
    echo ""
    echo "Or run in background:"
    echo "  ssh -p $PORT $USER@$HOST \"cd $REMOTE_BASE && CONFIG_FILE='$CONFIG_FILE' nohup bash run_pipeline.sh --output_dir AmbigQA_Qwen3-$MODEL --only 7c_kmeans --skip-existing > AmbigQA_${MODEL}_7c_kmeans.log 2>&1 &\""
    echo ""
    echo "To pull results back:"
    echo "  rsync -avz --progress -e 'ssh -p $PORT' \\"
    echo "    $USER@$HOST:$REMOTE_DIR/results/7_validation/7c_kmeans/ \\"
    echo "    $LOCAL_RESULTS/7_validation/7c_kmeans_remote/"
fi
