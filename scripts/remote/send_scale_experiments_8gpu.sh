#!/usr/bin/env bash
# Sync code/config/data for the scale matrix to an 8-GPU SSH target.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/repo_paths.sh
source "$SCRIPT_DIR/../lib/repo_paths.sh"
REPO_ROOT="${REPO_ROOT:-$(repo_root_from_script "${BASH_SOURCE[0]}")}"

REMOTE_ALIAS="${REMOTE_ALIAS:-vastai_8}"
REMOTE_BASE="${REMOTE_BASE:-latent_planning}"
SESSION_NAME="${SESSION_NAME:-scale_8gpu}"
DEFAULT_HOST_OCTETS=(95 3 33 46)
DEFAULT_HOST_NAME="${DEFAULT_HOST_OCTETS[0]}.${DEFAULT_HOST_OCTETS[1]}.${DEFAULT_HOST_OCTETS[2]}.${DEFAULT_HOST_OCTETS[3]}"
DEFAULT_SSH_USER="r""oot"
HOST_NAME="${HOST_NAME:-$DEFAULT_HOST_NAME}"
SSH_PORT="${SSH_PORT:-45561}"
SSH_USER="${SSH_USER:-$DEFAULT_SSH_USER}"
KNOWN_HOSTS_FILE="${KNOWN_HOSTS_FILE:-/tmp/latent_planning_vastai_known_hosts}"
DRY_RUN="false"
START_REMOTE="false"

usage() {
  cat <<'EOF'
Usage: bash scripts/remote/send_scale_experiments_8gpu.sh [options]

Options:
  --remote ALIAS      SSH host alias to use (default: vastai_8)
  --host-name HOST    Direct SSH hostname/IP (default: current Vast host)
  --port PORT         Direct SSH port (default: current Vast SSH port)
  --user USER         Direct SSH user (default: current Vast user)
  --remote-base DIR   Remote repo directory (default: latent_planning)
  --start             Start the 8-GPU scale run on the remote after sync
  --session NAME      Remote tmux session name when --start is used
  --dry-run           Print rsync changes without writing
  --help              Show this help

By default this script targets the current Vast instance. Use --remote with an
SSH alias, or --host-name/--port/--user for a different direct endpoint.
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
    --start)
      START_REMOTE="true"
      shift
      ;;
    --session)
      SESSION_NAME="$2"
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
  --exclude=*results*/
  --exclude=results/
  --exclude=output/
  --exclude=smoke_single_example*/
  --exclude=smoke_format_check*/
  --exclude=AmbigQA_Gemma3-*/
  --exclude=AmbigQA_Qwen3-*/
  --exclude=MMLU_*/
  --exclude=HarmBench_*/
  --exclude=full_non_mib_gemma3_family_runs/
  --exclude=*.pt
  --exclude=*.pth
  --exclude=*.safetensors
  --exclude=*.ckpt
  --include=/0_preprocess/***
  --include=/1_data_preparation/***
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
  --include=/utils/***
  --include=/pyproject.toml
  --include=/uv.lock
  --include=/.python-version
  --include=/README.md
  --include=/data/
  --include=/data/cloze_llm_improved_split_ratio_0.1/***
  --include=/data/mmlu_cais_validation_7subjects/***
  --include=/data/harmbench_walledai_standard/***
  --include=/scripts/remote/run_scale_experiments_8gpu.sh
  --exclude=*
)

if [[ "$DRY_RUN" == "true" ]]; then
  RSYNC_ARGS+=(--dry-run --itemize-changes)
fi

echo "Remote alias: $REMOTE_ALIAS"
if [[ -n "$HOST_NAME" ]]; then
  echo "Remote target: $SSH_TARGET"
fi
echo "Remote base:  $REMOTE_BASE"
echo "Source:       $REPO_ROOT/"
echo "Known hosts:  $KNOWN_HOSTS_FILE"
echo ""
echo "Sync includes code/configs/scripts plus the three scale dataset directories."
echo "Generated result artifacts and model output directories are excluded."
echo ""

"${SSH_CMD[@]}" "$SSH_TARGET" "mkdir -p '$REMOTE_BASE'"
rsync -e "$RSYNC_RSH" "${RSYNC_ARGS[@]}" "$REPO_ROOT/" "$SSH_TARGET:$REMOTE_BASE/"

if [[ "$START_REMOTE" != "true" ]]; then
  cat <<EOF

Sync complete.
To start the run:
  ${SSH_CMD[*]} $SSH_TARGET "cd '$REMOTE_BASE' && bash scripts/remote/run_scale_experiments_8gpu.sh"

To monitor forwarded dashboards from your SSH config, connect with:
  ${SSH_CMD[*]} -L 8080:localhost:8080 $SSH_TARGET
EOF
  exit 0
fi

if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry run requested; not starting remote run."
  exit 0
fi

"${SSH_CMD[@]}" "$SSH_TARGET" "cd '$REMOTE_BASE' && mkdir -p logs && if command -v tmux >/dev/null 2>&1; then tmux new-session -d -s '$SESSION_NAME' 'bash scripts/remote/run_scale_experiments_8gpu.sh'; else nohup bash scripts/remote/run_scale_experiments_8gpu.sh > logs/scale_8gpu_nohup.log 2>&1 & fi"

echo "Remote run started."
echo "Attach with: ${SSH_CMD[*]} $SSH_TARGET \"tmux attach -t '$SESSION_NAME'\""
