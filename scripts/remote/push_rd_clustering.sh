#!/usr/bin/env bash
set -euo pipefail

# Push selected Python scripts + README to the RD_clustering repo.
# Usage:
#   ./push_rd_clustering.sh
#   GIT_TOKEN=<token> ./push_rd_clustering.sh
#   FORCE_PUSH=1 ./push_rd_clustering.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/repo_paths.sh
source "$SCRIPT_DIR/../lib/repo_paths.sh"
SRC_ROOT="${SRC_ROOT:-$(repo_root_from_script "${BASH_SOURCE[0]}")}"
DEST_REPO="${DEST_REPO:?Set DEST_REPO, e.g. https://github.com/<owner>/<repo>.git}"
BRANCH="main"
DEFAULT_COMMIT_MSG="Add pipeline Python scripts and README"
GIT_USER_NAME="${GIT_USER_NAME:-}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-}"
if [[ -n "$GIT_USER_NAME" ]]; then
  git config --global user.name "$GIT_USER_NAME"
fi
if [[ -n "$GIT_USER_EMAIL" ]]; then
  git config --global user.email "$GIT_USER_EMAIL"
fi
WORKDIR="$(mktemp -d)"
cleanup() {
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

echo "Using temp workdir: $WORKDIR"

mkdir -p "$WORKDIR/repo"

# Configure remote with optional token (for clone and push)
if [[ -n "${GIT_TOKEN:-}" ]]; then
  AUTH_REMOTE="${DEST_REPO/https:\/\//https:\/\/${GIT_TOKEN}@}"
else
  AUTH_REMOTE="$DEST_REPO"
fi

# Clone remote so status shows true diffs vs remote
git clone -q --depth 1 "$AUTH_REMOTE" "$WORKDIR/repo"
cd "$WORKDIR/repo"
git checkout -q "$BRANCH"

# Copy README
if [[ -f "$SRC_ROOT/README.md" ]]; then
  cp "$SRC_ROOT/README.md" .
else
  echo "ERROR: README.md not found at $SRC_ROOT/README.md" >&2
  exit 1
fi

# Copy only .py files (and keep folder structure) from the specified stages.
STAGES=(
  "0_preprocess"
  "1_data_preparation"
  "2_branch_sampling"
  "3_attribution_graphs"
  "4_feature_extraction"
  "5_gaussian_clustering"
  "6_semantic_graphs"
  "7_validation"
  "8_visualization"
  "downstream"
  "circuit-tracer"
)

for stage in "${STAGES[@]}"; do
  src_dir="$SRC_ROOT/$stage"
  if [[ ! -d "$src_dir" ]]; then
    echo "ERROR: Missing directory: $src_dir" >&2
    exit 1
  fi
  mkdir -p "$stage"
  rsync -a \
    --include='*/' \
    --include='*.py' \
    --include='*.sh' \
    --exclude='*' \
    "$src_dir"/ "$stage"/
done

# Optional: remove any accidental __pycache__ copies (safety)
find . -type d -name "__pycache__" -prune -exec rm -rf {} +

git add -A
echo "Updated files to be committed:"
git status -s

if [[ -n "${1:-}" ]]; then
  COMMIT_MSG="$1"
else
  read -r -p "Commit message (default: $DEFAULT_COMMIT_MSG): " COMMIT_MSG
  COMMIT_MSG="${COMMIT_MSG:-$DEFAULT_COMMIT_MSG}"
fi
git commit -qm "$COMMIT_MSG"

echo "Pushing to $DEST_REPO ($BRANCH)..."
if [[ -n "${FORCE_PUSH:-}" ]]; then
  git push -u --force origin "$BRANCH"
else
  git push -u origin "$BRANCH"
fi
echo "Done."
