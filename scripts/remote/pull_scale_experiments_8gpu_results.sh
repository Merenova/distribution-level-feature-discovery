#!/usr/bin/env bash
# Pull generated outputs from the 8-GPU scale experiment host.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/repo_paths.sh
source "$SCRIPT_DIR/../lib/repo_paths.sh"
REPO_ROOT="${REPO_ROOT:-$(repo_root_from_script "${BASH_SOURCE[0]}")}"

REMOTE_ALIAS="${REMOTE_ALIAS:-vastai_8}"
REMOTE_BASE="${REMOTE_BASE:-latent_planning}"
LOCAL_BASE="${LOCAL_BASE:-$REPO_ROOT}"
DEFAULT_HOST_OCTETS=(95 3 33 46)
DEFAULT_HOST_NAME="${DEFAULT_HOST_OCTETS[0]}.${DEFAULT_HOST_OCTETS[1]}.${DEFAULT_HOST_OCTETS[2]}.${DEFAULT_HOST_OCTETS[3]}"
DEFAULT_SSH_USER="r""oot"
HOST_NAME="${HOST_NAME:-$DEFAULT_HOST_NAME}"
SSH_PORT="${SSH_PORT:-45561}"
SSH_USER="${SSH_USER:-$DEFAULT_SSH_USER}"
KNOWN_HOSTS_FILE="${KNOWN_HOSTS_FILE:-/tmp/latent_planning_vastai_known_hosts}"
DRY_RUN="false"
INCLUDE_INTERMEDIATE="false"
FULL_RESULTS="false"
INCLUDE_TENSORS="false"
ONLY=""

TAGS=(
  "AmbigQA_Gemma3-1B-it"
  "AmbigQA_Gemma3-4B-it"
  "MMLU_Qwen3-8B"
  "MMLU_Qwen3-4B"
  "MMLU_Gemma3-1B-it"
  "MMLU_Gemma3-4B-it"
  "HarmBench_Qwen3-8B"
  "HarmBench_Qwen3-4B"
  "HarmBench_Gemma3-1B-it"
  "HarmBench_Gemma3-4B-it"
)

usage() {
  cat <<'EOF'
Usage: bash scripts/remote/pull_scale_experiments_8gpu_results.sh [options]

Options:
  --remote ALIAS              SSH host alias to use (default: vastai_8)
  --host-name HOST            Direct SSH hostname/IP (default: current Vast host)
  --port PORT                 Direct SSH port (default: current Vast SSH port)
  --user USER                 Direct SSH user (default: current Vast user)
  --remote-base DIR           Remote repo directory (default: latent_planning)
  --local-base DIR            Local destination repo directory (default: repo root)
  --only TAG1,TAG2            Pull only selected scale tags
  --full                      Pull all results/ and logs/ content except excluded artifacts
  --include-intermediate      Also pull results/5_clustering/intermediate/
  --include-tensors           Also pull .pt/.pth result tensor files
  --dry-run                   Print rsync changes without writing
  --help                      Show this help

By default this pulls analysis outputs: logs, manifests/configs, branch samples,
Stage 5 sweep JSONs, semantic graphs, validation, and visualizations. It skips
large Stage 3 attribution graphs, Stage 4 embeddings, Stage 5 intermediate
snapshots, tensors, and model/checkpoint files.
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
    --local-base)
      LOCAL_BASE="$2"
      shift 2
      ;;
    --only)
      ONLY="$2"
      shift 2
      ;;
    --full)
      FULL_RESULTS="true"
      shift
      ;;
    --include-intermediate)
      INCLUDE_INTERMEDIATE="true"
      shift
      ;;
    --include-tensors)
      INCLUDE_TENSORS="true"
      shift
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
  --info=progress2,stats2
  --include=/
)

if [[ "$INCLUDE_INTERMEDIATE" != "true" ]]; then
  RSYNC_ARGS+=(--exclude=/results/5_clustering/intermediate/***)
fi
if [[ "$INCLUDE_TENSORS" != "true" ]]; then
  RSYNC_ARGS+=(--exclude=*.pt --exclude=*.pth)
fi
RSYNC_ARGS+=(--exclude=*.safetensors --exclude=*.ckpt)

if [[ "$FULL_RESULTS" == "true" ]]; then
  RSYNC_ARGS+=(--include=/results/*** --include=/logs/***)
else
  RSYNC_ARGS+=(
    --include=/logs/***
    --include=/results/
    --include=/results/*.json
    --include=/results/configs/***
    --include=/results/2_branch_sampling/***
    --include=/results/5_clustering/
    --include=/results/5_clustering/*.json
    --include=/results/5_clustering/logs/***
    --include=/results/6_semantic_graphs/***
    --include=/results/7_validation/***
    --include=/results/8_visualization/***
    --exclude=/results/3_attribution_graphs/***
    --exclude=/results/4_feature_extraction/***
  )
fi
RSYNC_ARGS+=(--exclude=*)

if [[ "$DRY_RUN" == "true" ]]; then
  RSYNC_ARGS+=(--dry-run --itemize-changes)
fi

declare -A WANT=()
if [[ -n "$ONLY" ]]; then
  IFS=',' read -ra requested_tags <<<"$ONLY"
  for tag in "${requested_tags[@]}"; do
    WANT["$tag"]=1
  done
else
  for tag in "${TAGS[@]}"; do
    WANT["$tag"]=1
  done
fi

for tag in "${!WANT[@]}"; do
  known="false"
  for allowed in "${TAGS[@]}"; do
    if [[ "$tag" == "$allowed" ]]; then
      known="true"
      break
    fi
  done
  if [[ "$known" != "true" ]]; then
    echo "Unknown scale tag: $tag" >&2
    echo "Use --help to see supported tags." >&2
    exit 1
  fi
done

echo "Remote alias: $REMOTE_ALIAS"
if [[ -n "$HOST_NAME" ]]; then
  echo "Remote target: $SSH_TARGET"
fi
echo "Remote base:  $REMOTE_BASE"
echo "Local base:   $LOCAL_BASE"
echo "Known hosts:  $KNOWN_HOSTS_FILE"
echo ""
if [[ "$FULL_RESULTS" == "true" ]]; then
  echo "Pulling full results/ and logs/ for selected scale tags."
else
  echo "Pulling analysis outputs and logs for selected scale tags."
  echo "Skipping results/3_attribution_graphs/ and results/4_feature_extraction/."
fi
if [[ "$INCLUDE_INTERMEDIATE" != "true" ]]; then
  echo "Skipping results/5_clustering/intermediate/."
fi
if [[ "$INCLUDE_TENSORS" != "true" ]]; then
  echo "Skipping .pt and .pth tensor files."
fi
echo ""

mkdir -p "$LOCAL_BASE"

for tag in "${TAGS[@]}"; do
  [[ -n "${WANT[$tag]:-}" ]] || continue

  remote_tag_dir="$REMOTE_BASE/$tag"
  local_tag_dir="$LOCAL_BASE/$tag"

  if ! "${SSH_CMD[@]}" "$SSH_TARGET" "test -d '$remote_tag_dir'"; then
    echo "Warning: missing remote directory, skipping: $remote_tag_dir" >&2
    continue
  fi

  echo "==> Pulling $tag"
  mkdir -p "$local_tag_dir"
  rsync -e "$RSYNC_RSH" \
    "${RSYNC_ARGS[@]}" \
    "$SSH_TARGET:$remote_tag_dir/" \
    "$local_tag_dir/"
done

echo ""
echo "Pull complete."
