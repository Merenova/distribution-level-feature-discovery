#!/usr/bin/env bash
set -euo pipefail

# Mirror a filtered copy of latent_planning to a Git repo.
# Default behavior:
# - mirrors the project root more broadly than push_rd_clustering.sh
# - excludes bulky/generated/local-only paths
# - skips files larger than MAX_FILE_SIZE
#
# Usage:
#   ./push_latent_planning_mirror.sh
#   ./push_latent_planning_mirror.sh "Commit message"
#   DEST_REPO=https://github.com/owner/repo.git ./push_latent_planning_mirror.sh
#   DRY_RUN=1 ./push_latent_planning_mirror.sh
#   MAX_FILE_SIZE=50m FORCE_PUSH=1 ./push_latent_planning_mirror.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/repo_paths.sh
source "$SCRIPT_DIR/../lib/repo_paths.sh"
SRC_ROOT="${SRC_ROOT:-$(repo_root_from_script "${BASH_SOURCE[0]}")}"
DEFAULT_DEST_REPO="$(git -C "$SRC_ROOT" remote get-url origin 2>/dev/null || true)"
DEST_REPO="${DEST_REPO:-$DEFAULT_DEST_REPO}"
BRANCH="${BRANCH:-main}"
MAX_FILE_SIZE="${MAX_FILE_SIZE:-20m}"
DEFAULT_COMMIT_MSG="Mirror filtered latent_planning project"
DRY_RUN="${DRY_RUN:-0}"
FORCE_PUSH="${FORCE_PUSH:-0}"
GIT_USER_NAME="${GIT_USER_NAME:-}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-}"

EXCLUDES=(
  ".git/"
  ".venv/"
  "venv/"
  ".vscode/"
  "__pycache__/"
  "*.pyc"
  ".mypy_cache/"
  ".pytest_cache/"
  ".ruff_cache/"
  ".cache/"
  ".coverage"
  ".ipynb_checkpoints/"
  "build/"
  "dist/"
  "wheels/"
  "*.egg-info/"
  "output/"
  "outputs/"
  "results/"
  "logs/"
  "figures/"
  "tmp/"
  "archive/"
  "AmbigQA_Qwen3-4B/"
  "AmbigQA_Qwen3-8B/"
  "*.log"
  "*.out"
)

if [[ ! -d "$SRC_ROOT" ]]; then
  echo "ERROR: Source directory not found: $SRC_ROOT" >&2
  exit 1
fi

if [[ -z "$DEST_REPO" ]]; then
  echo "ERROR: DEST_REPO is not set and no git origin was found for $SRC_ROOT" >&2
  exit 1
fi

if [[ -n "${EXTRA_EXCLUDES:-}" ]]; then
  # EXTRA_EXCLUDES is split on shell whitespace by design.
  for pattern in ${EXTRA_EXCLUDES}; do
    EXCLUDES+=("$pattern")
  done
fi

WORKDIR="$(mktemp -d)"
cleanup() {
  rm -rf "$WORKDIR"
}
trap cleanup EXIT

AUTH_REMOTE="$DEST_REPO"
if [[ -n "${GIT_TOKEN:-}" && "$DEST_REPO" == https://* ]]; then
  AUTH_REMOTE="${DEST_REPO/https:\/\//https:\/\/${GIT_TOKEN}@}"
fi

echo "Using temp workdir: $WORKDIR"
echo "Mirroring filtered contents from $SRC_ROOT"
echo "Destination: $DEST_REPO ($BRANCH)"
echo "Max file size: $MAX_FILE_SIZE"
echo "Excluded patterns:"
printf '  - %s\n' "${EXCLUDES[@]}"

git clone -q "$AUTH_REMOTE" "$WORKDIR/repo"
cd "$WORKDIR/repo"
if [[ -n "$GIT_USER_NAME" ]]; then
  git config user.name "$GIT_USER_NAME"
fi
if [[ -n "$GIT_USER_EMAIL" ]]; then
  git config user.email "$GIT_USER_EMAIL"
fi

if git show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
  git checkout -q -B "$BRANCH" "origin/$BRANCH"
elif git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git checkout -q "$BRANCH"
else
  git checkout --orphan "$BRANCH"
fi

# Rebuild the worktree from scratch so the destination is a mirror of the
# allowed subset, not an additive copy.
find "$WORKDIR/repo" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +

RSYNC_ARGS=(
  -a
  --prune-empty-dirs
  "--max-size=$MAX_FILE_SIZE"
)
for pattern in "${EXCLUDES[@]}"; do
  RSYNC_ARGS+=("--exclude=$pattern")
done

rsync "${RSYNC_ARGS[@]}" "$SRC_ROOT"/ "$WORKDIR/repo"/

git add -A -f .
echo "Updated files to be committed:"
git status -s

if [[ -z "$(git status --porcelain)" ]]; then
  echo "No changes to commit."
  exit 0
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN=1 set; stopping before commit/push."
  exit 0
fi

if [[ -n "${1:-}" ]]; then
  COMMIT_MSG="$1"
else
  read -r -p "Commit message (default: $DEFAULT_COMMIT_MSG): " COMMIT_MSG
  COMMIT_MSG="${COMMIT_MSG:-$DEFAULT_COMMIT_MSG}"
fi

git commit -qm "$COMMIT_MSG"

echo "Pushing to $DEST_REPO ($BRANCH)..."
if [[ "$FORCE_PUSH" == "1" ]]; then
  git push -u --force origin "$BRANCH"
else
  git push -u origin "$BRANCH"
fi
echo "Done."
