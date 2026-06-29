#!/usr/bin/env bash
# Sync code/configs for the Qwen3 reasoning sweep to a remote host.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/repo_paths.sh
source "$SCRIPT_DIR/../lib/repo_paths.sh"
REPO_ROOT="${REPO_ROOT:-$(repo_root_from_script "${BASH_SOURCE[0]}")}"

REMOTE_ALIAS="${REMOTE_ALIAS:-reasoning_qwen3}"
REMOTE_BASE="${REMOTE_BASE:-/home/hyunjin/latent_planning}"
DEFAULT_HOST_OCTETS=(136 61 20 181)
DEFAULT_HOST_NAME="${DEFAULT_HOST_OCTETS[0]}.${DEFAULT_HOST_OCTETS[1]}.${DEFAULT_HOST_OCTETS[2]}.${DEFAULT_HOST_OCTETS[3]}"
DEFAULT_SSH_USER="r""oot"
HOST_NAME="${HOST_NAME:-$DEFAULT_HOST_NAME}"
SSH_PORT="${SSH_PORT:-25467}"
SSH_USER="${SSH_USER:-$DEFAULT_SSH_USER}"
KNOWN_HOSTS_FILE="${KNOWN_HOSTS_FILE:-/tmp/latent_planning_reasoning_known_hosts}"
DRY_RUN="false"

usage() {
  cat <<'EOF'
Usage: bash scripts/remote/send_reasoning_qwen3_small_sweep.sh [options]

Options:
  --remote ALIAS       SSH host alias to use (default: reasoning_qwen3)
  --host-name HOST     Direct SSH hostname/IP (default: configured reasoning host)
  --port PORT          Direct SSH port (default: 25467)
  --user USER          Direct SSH user (default: configured remote user)
  --remote-base DIR    Remote repo directory (default: /home/hyunjin/latent_planning)
  --dry-run            Print rsync changes without writing
  --help               Show this help

This script only syncs. SSH into the remote host and run
scripts/remote/run_reasoning_qwen3_small_remote.sh there.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)
      REMOTE_ALIAS="$2"
      HOST_NAME=""
      shift 2
      ;;
    --host-name)
      HOST_NAME="$2"
      shift 2
      ;;
    --port)
      SSH_PORT="$2"
      shift 2
      ;;
    --user)
      SSH_USER="$2"
      shift 2
      ;;
    --remote-base)
      REMOTE_BASE="$2"
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

SSH_TARGET="$REMOTE_ALIAS"
if [[ -n "$HOST_NAME" ]]; then
  SSH_TARGET="$HOST_NAME"
fi
if [[ -n "$SSH_USER" ]]; then
  SSH_TARGET="$SSH_USER@$SSH_TARGET"
fi

SSH_CMD=(
  ssh
  -o StrictHostKeyChecking=accept-new
  -o "UserKnownHostsFile=$KNOWN_HOSTS_FILE"
)
if [[ -n "$SSH_PORT" ]]; then
  SSH_CMD+=(-p "$SSH_PORT")
fi
RSYNC_RSH="${SSH_CMD[*]}"

RSYNC_ARGS=(
  -az
  --human-readable
  --partial
  --prune-empty-dirs
  --include=/
  --exclude=.git/
  --exclude=.venv/
  --exclude=venv/
  --exclude=__pycache__/
  --exclude=*.pyc
  --exclude=.pytest_cache/
  --exclude=.mypy_cache/
  --exclude=.ruff_cache/
  --exclude=*.log
  --exclude=logs/
  --exclude=figures/
  --exclude=archive/
  --exclude=experiments/reasoning_runs/
  --exclude=*results*/
  --exclude=results/
  --exclude=output/
  --exclude=smoke_single_example*/
  --exclude=smoke_format_check*/
  --exclude=*.pt
  --exclude=*.pth
  --exclude=*.safetensors
  --exclude=*.ckpt
  --include=/2_branch_sampling/***
  --include=/3_attribution_graphs/***
  --include=/4_feature_extraction/***
  --include=/5_gaussian_clustering/***
  --include=/6_semantic_graphs/***
  --include=/7_validation/***
  --include=/8_visualization/***
  --include=/circuit-tracer/***
  --include=/configs/***
  --include=/scripts/***
  --include=/scripts/remote/run_reasoning_qwen3_small_remote.sh
  --include=/tests/***
  --include=/utils/***
  --include=/pyproject.toml
  --include=/uv.lock
  --include=/.python-version
  --include=/README.md
  --exclude=*
)

if [[ "$DRY_RUN" == "true" ]]; then
  RSYNC_ARGS+=(--dry-run --itemize-changes)
fi

shell_quote() {
  printf '%q' "$1"
}

echo "Remote target: $SSH_TARGET"
echo "Remote base:   $REMOTE_BASE"
echo "Source:        $REPO_ROOT/"
echo "Known hosts:   $KNOWN_HOSTS_FILE"
echo ""

echo "Syncing reasoning code/configs to remote."
if [[ "$DRY_RUN" == "true" ]]; then
  echo "[dry-run] Would create remote base: $REMOTE_BASE"
else
  "${SSH_CMD[@]}" "$SSH_TARGET" "mkdir -p $(shell_quote "$REMOTE_BASE")"
fi
rsync -e "$RSYNC_RSH" "${RSYNC_ARGS[@]}" "$REPO_ROOT/" "$SSH_TARGET:$REMOTE_BASE/"

echo ""
echo "Sync complete."
echo "SSH with tunnel:"
echo "  ${SSH_CMD[*]} -L 8080:localhost:8080 $SSH_TARGET"
echo ""
echo "Then run on the remote host:"
echo "  cd '$REMOTE_BASE'"
echo "  tmux new -s reasoning_qwen3_small"
echo "  bash scripts/remote/run_reasoning_qwen3_small_remote.sh"
echo "  # optional pooling override:"
echo "  bash scripts/remote/run_reasoning_qwen3_small_remote.sh --pooling mean"
echo ""
echo "Single-lane rerun example on the remote host:"
echo "  bash scripts/remote/run_reasoning_qwen3_small_remote.sh --only qwen3_0_6b:math500"
