#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-/home/hyunjin/latent_planning}"
REMOTE_HOST="${REMOTE_HOST:-136.61.20.181}"
REMOTE_PORT="${REMOTE_PORT:-25463}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_ROOT="${REMOTE_ROOT:-~/latent_planning}"

SSH_TARGET="$REMOTE_USER@$REMOTE_HOST"
SSH_CMD="ssh -p $REMOTE_PORT -o ClearAllForwardings=yes"

if [[ ! -d "$LOCAL_ROOT" ]]; then
    echo "Missing local root directory: $LOCAL_ROOT" >&2
    exit 1
fi

$SSH_CMD "$SSH_TARGET" "mkdir -p $REMOTE_ROOT"

rsync -avh --partial --progress \
    --exclude='.*/' \
    --include='/AmbigQA_Qwen3-8B/results/***' \
    --exclude='results/' \
    -e "$SSH_CMD" \
    "$LOCAL_ROOT/" \
    "$SSH_TARGET:$REMOTE_ROOT/"
