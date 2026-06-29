#!/usr/bin/env bash
# Sync code, scale datasets, and full intermediate results to a resume host.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/repo_paths.sh
source "$SCRIPT_DIR/../lib/repo_paths.sh"
REPO_ROOT="${REPO_ROOT:-$(repo_root_from_script "${BASH_SOURCE[0]}")}"

REMOTE_ALIAS="${REMOTE_ALIAS:-vastai_resume}"
REMOTE_BASE="${REMOTE_BASE:-latent_planning}"
SESSION_NAME="${SESSION_NAME:-scale_8gpu_resume}"
DEFAULT_HOST_OCTETS=(70 78 140 110)
DEFAULT_HOST_NAME="${DEFAULT_HOST_OCTETS[0]}.${DEFAULT_HOST_OCTETS[1]}.${DEFAULT_HOST_OCTETS[2]}.${DEFAULT_HOST_OCTETS[3]}"
DEFAULT_SSH_USER="r""oot"
HOST_NAME="${HOST_NAME:-$DEFAULT_HOST_NAME}"
SSH_PORT="${SSH_PORT:-46474}"
SSH_USER="${SSH_USER:-$DEFAULT_SSH_USER}"
KNOWN_HOSTS_FILE="${KNOWN_HOSTS_FILE:-/tmp/latent_planning_vastai_resume_known_hosts}"
DRY_RUN="false"
START_REMOTE="false"
SETUP_REMOTE="false"
PARALLEL="${PARALLEL:-6}"
RESUME_STAGE="${RESUME_STAGE:-5}"
RESUME_ONLY="${RESUME_ONLY:-}"

TAGS=(
  "AmbigQA_Gemma3-1B-it"
  "AmbigQA_Gemma3-4B-it"
  "MMLU_Qwen3-8B"
  "MMLU_Qwen3-4B"
  "MMLU_Gemma3-4B-it"
  "HarmBench_Qwen3-8B"
  "HarmBench_Qwen3-4B"
  "HarmBench_Gemma3-4B-it"
)

usage() {
  cat <<'EOF'
Usage: bash scripts/remote/send_scale_resume_full_results.sh [options]

Options:
  --remote ALIAS        SSH host alias to use (default: vastai_resume)
  --host-name HOST      Direct SSH hostname/IP (default: current resume host)
  --port PORT           Direct SSH port (default: current resume SSH port)
  --user USER           Direct SSH user (default: current resume SSH user)
  --remote-base DIR     Remote repo directory (default: latent_planning)
  --parallel N          Number of result-directory rsync lanes (default: 6)
  --setup               Run uv sync on the remote after copying files
  --start               Start resumed 8-GPU run in tmux after sync
  --resume-stage STAGE  Stage passed to run_scale_experiments_8gpu.sh (default: 5)
  --resume-only TAGS    Comma-separated tags to resume (default: copied tags)
  --session NAME        Remote tmux session name when --start is used
  --dry-run             Print rsync changes without writing
  --help                Show this help

This script is intentionally broader than send_scale_experiments_8gpu.sh: it
copies full scale result directories, including Stage 3 attribution graphs,
Stage 4 feature files, Stage 5 intermediates, and .pt/.pth tensors. It still
skips repository environments, caches, and model checkpoint files.
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
    --parallel)
      PARALLEL="$2"
      shift 2
      ;;
    --setup)
      SETUP_REMOTE="true"
      shift
      ;;
    --start)
      START_REMOTE="true"
      shift
      ;;
    --resume-stage)
      RESUME_STAGE="$2"
      shift 2
      ;;
    --resume-only)
      RESUME_ONLY="$2"
      shift 2
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

if [[ -z "$RESUME_ONLY" ]]; then
  RESUME_ONLY="$(IFS=,; printf '%s' "${TAGS[*]}")"
fi

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

RSYNC_CODE_COMMON=(
  -az
  --compress-choice=zstd
  --compress-level=1
  --human-readable
  --partial
  --info=progress2,stats2
  --exclude=*.safetensors
  --exclude=*.ckpt
)

if [[ "$DRY_RUN" == "true" ]]; then
  RSYNC_CODE_COMMON+=(--dry-run --itemize-changes)
fi

CODE_ARGS=(
  "${RSYNC_CODE_COMMON[@]}"
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
  --exclude=archive/
  --exclude=figures/
  --exclude=smoke_single_example*/
  --exclude=smoke_format_check*/
  --exclude=AmbigQA_Gemma3-*/
  --exclude=AmbigQA_Qwen3-*/
  --exclude=MMLU_*/
  --exclude=HarmBench_*/
  --exclude=full_non_mib_gemma3_family_runs/
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
  --include=/tests/***
  --include=/utils/***
  --include=/logs/***
  --include=/pyproject.toml
  --include=/uv.lock
  --include=/.python-version
  --include=/README.md
  --include=/data/
  --include=/data/cloze_llm_improved_split_ratio_0.1/***
  --include=/data/mmlu_cais_validation_7subjects/***
  --include=/data/harmbench_walledai_standard/***
  --exclude=*
)

run_rsync_tag() {
  local tag="$1"
  local rsync_args=(
    -az
    --compress-choice=zstd
    --compress-level=1
    --human-readable
    --partial
    --append-verify
    --info=progress2,stats2
    --exclude=*.safetensors
    --exclude=*.ckpt
  )
  if [[ "${DRY_RUN:-false}" == "true" ]]; then
    rsync_args+=(--dry-run --itemize-changes)
  fi
  if [[ ! -d "$REPO_ROOT/$tag" ]]; then
    echo "Skipping missing local result directory: $tag" >&2
    return 0
  fi
  echo "[$(date '+%F %T')] Syncing full result dir: $tag"
  rsync -e "$RSYNC_RSH" "${rsync_args[@]}" "$REPO_ROOT/$tag/" "$SSH_TARGET:$REMOTE_BASE/$tag/"
}

export REPO_ROOT REMOTE_BASE SSH_TARGET RSYNC_RSH DRY_RUN
export -f run_rsync_tag

echo "Remote alias: $REMOTE_ALIAS"
if [[ -n "$HOST_NAME" ]]; then
  echo "Remote target: $SSH_TARGET"
fi
echo "Remote base:  $REMOTE_BASE"
echo "Source:       $REPO_ROOT/"
echo "Known hosts:  $KNOWN_HOSTS_FILE"
echo "Parallel result lanes: $PARALLEL"
echo ""
echo "Syncing code/configs/scripts/data first, then full scale result directories."
echo ""

"${SSH_CMD[@]}" "$SSH_TARGET" "mkdir -p '$REMOTE_BASE'"
rsync -e "$RSYNC_RSH" "${CODE_ARGS[@]}" "$REPO_ROOT/" "$SSH_TARGET:$REMOTE_BASE/"

printf '%s\n' "${TAGS[@]}" | xargs -n1 -P "$PARALLEL" bash -c 'run_rsync_tag "$1"' _

if [[ "$SETUP_REMOTE" == "true" && "$DRY_RUN" != "true" ]]; then
  "${SSH_CMD[@]}" "$SSH_TARGET" "cd '$REMOTE_BASE' && uv sync"
fi

if [[ "$START_REMOTE" != "true" ]]; then
  cat <<EOF

Resume migration sync complete.
To resume the scale run from Stage $RESUME_STAGE:
  ${SSH_CMD[*]} $SSH_TARGET "cd '$REMOTE_BASE' && bash scripts/remote/run_scale_experiments_8gpu.sh --only '$RESUME_ONLY' --resume '$RESUME_STAGE' --skip-existing"

To monitor forwarded dashboards:
  ${SSH_CMD[*]} -L 8080:localhost:8080 $SSH_TARGET
EOF
  exit 0
fi

if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry run requested; not starting remote run."
  exit 0
fi

"${SSH_CMD[@]}" "$SSH_TARGET" "cd '$REMOTE_BASE' && mkdir -p logs && if command -v tmux >/dev/null 2>&1; then tmux new-session -d -s '$SESSION_NAME' 'bash scripts/remote/run_scale_experiments_8gpu.sh --only \"$RESUME_ONLY\" --resume \"$RESUME_STAGE\" --skip-existing'; else nohup bash scripts/remote/run_scale_experiments_8gpu.sh --only '$RESUME_ONLY' --resume '$RESUME_STAGE' --skip-existing > logs/scale_8gpu_resume_nohup.log 2>&1 & fi"

echo "Remote resumed run started."
echo "Attach with: ${SSH_CMD[*]} $SSH_TARGET \"tmux attach -t '$SESSION_NAME'\""
