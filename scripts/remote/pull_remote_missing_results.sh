#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/repo_paths.sh
source "$SCRIPT_DIR/../lib/repo_paths.sh"
REPO_ROOT="${REPO_ROOT:-$(repo_root_from_script "${BASH_SOURCE[0]}")}"
LOCAL_RESULTS_ROOT="${LOCAL_RESULTS_ROOT:-$REPO_ROOT}"
REMOTE_AMBIGQA_ROOT="${REMOTE_AMBIGQA_ROOT:-/path/to/remote}"
REMOTE_RESULTS_ROOT="${REMOTE_RESULTS_ROOT:-/path/to/latent_planning}"

DRY_RUN="false"
VERIFY_ONLY="false"

usage() {
  cat <<EOF
Usage: bash scripts/remote/pull_remote_missing_results.sh [--dry-run] [--verify-only]

Pull remote-only or locally incomplete results from the reachable experiment hosts.

Sources:
  - AMBIGQA_COMPARE_HOST: AmbigQA_Qwen3-8B comparison outputs
  - GEMMA4B_REMOTE_HOST: Gemma 3 4B family runs
  - GEMMA1B_REMOTE_HOST: Gemma 3 1B family runs

Options:
  --dry-run      Show planned rsync changes without writing files
  --verify-only  Skip rsync and only print local counts for the managed targets
  --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --verify-only)
      VERIFY_ONLY="true"
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

SSH_CONTROL_BASE="/tmp/ssh-pull-missing-results-$$"
RSYNC_OPTS="-av --human-readable --partial --progress --ignore-existing"
if [[ "$DRY_RUN" == "true" ]]; then
  RSYNC_OPTS="$RSYNC_OPTS --dry-run --itemize-changes"
fi

AMBIGQA_COMPARE_HOST="${AMBIGQA_COMPARE_HOST:-}"
AMBIGQA_COMPARE_PORT="${AMBIGQA_COMPARE_PORT:-40089}"
AMBIGQA_COMPARE_USER="${AMBIGQA_COMPARE_USER:-${REMOTE_USER:-${USER:-}}}"
GEMMA4B_REMOTE_HOST="${GEMMA4B_REMOTE_HOST:-}"
GEMMA4B_REMOTE_PORT="${GEMMA4B_REMOTE_PORT:-40394}"
GEMMA4B_REMOTE_USER="${GEMMA4B_REMOTE_USER:-${REMOTE_USER:-${USER:-}}}"
GEMMA1B_REMOTE_HOST="${GEMMA1B_REMOTE_HOST:-}"
GEMMA1B_REMOTE_PORT="${GEMMA1B_REMOTE_PORT:-40206}"
GEMMA1B_REMOTE_USER="${GEMMA1B_REMOTE_USER:-${REMOTE_USER:-${USER:-}}}"

declare -a SSH_SPECS=(
  "ambigqa_compare|$AMBIGQA_COMPARE_HOST|$AMBIGQA_COMPARE_PORT|$AMBIGQA_COMPARE_USER"
  "gemma4b|$GEMMA4B_REMOTE_HOST|$GEMMA4B_REMOTE_PORT|$GEMMA4B_REMOTE_USER"
  "gemma1b|$GEMMA1B_REMOTE_HOST|$GEMMA1B_REMOTE_PORT|$GEMMA1B_REMOTE_USER"
)

cleanup() {
  local spec label host port user control_path
  for spec in "${SSH_SPECS[@]}"; do
    IFS="|" read -r label host port user <<<"$spec"
    control_path="${SSH_CONTROL_BASE}-${label}"
    ssh -p "$port" -o ControlPath="$control_path" -O exit "$user@$host" 2>/dev/null || true
  done
}
trap cleanup EXIT

start_master() {
  local label="$1"
  local host="$2"
  local port="$3"
  local user="$4"
  local control_path="${SSH_CONTROL_BASE}-${label}"
  ssh -p "$port" -o ControlMaster=yes -o ControlPath="$control_path" -o ControlPersist=300 -fN "$user@$host"
}

sync_from_remote() {
  local label="$1"
  local host="$2"
  local port="$3"
  local user="$4"
  local remote_path="$5"
  local local_path="$6"
  local control_path="${SSH_CONTROL_BASE}-${label}"

  if [[ "$remote_path" == */ ]]; then
    local local_dir="${local_path%/}"
    mkdir -p "$(dirname "$local_dir")"
    rsync $RSYNC_OPTS \
      --mkpath \
      -e "ssh -p $port -o ControlMaster=auto -o ControlPath=$control_path -o ControlPersist=300" \
      "$user@$host:${remote_path%/}" \
      "$(dirname "$local_dir")/"
  else
    mkdir -p "$(dirname "$local_path")"
    rsync $RSYNC_OPTS \
      --mkpath \
      -e "ssh -p $port -o ControlMaster=auto -o ControlPath=$control_path -o ControlPersist=300" \
      "$user@$host:$remote_path" \
      "$local_path"
  fi
}

verify_counts() {
  export LOCAL_RESULTS_ROOT
  python3 - <<'PY'
import glob
import os

local_root = os.environ["LOCAL_RESULTS_ROOT"]
targets = [
    ("AmbigQA_Qwen3-8B/results/5_clustering", 345),
    ("AmbigQA_Qwen3-8B/results/comparison/coreg/5_clustering", 343),
    ("AmbigQA_Qwen3-8B/results/comparison/concat/5_clustering", 343),
    ("full_non_mib_gemma3_family_runs/ambigqa_gemma-3-4b-it/results/5_clustering", 399),
    ("full_non_mib_gemma3_family_runs/ambigqa_gemma-3-4b-it/results/7_validation/7c_steering/H4a", 350),
    ("full_non_mib_gemma3_family_runs/ambigqa_gemma-3-1b-it/results/5_clustering", 403),
    ("full_non_mib_gemma3_family_runs/ambigqa_gemma-3-1b-it/results/7_validation/7c_steering/H4a", 279),
    ("full_non_mib_gemma3_family_runs/harmbench_gemma-3-1b-it/results/5_clustering", 36),
    ("full_non_mib_gemma3_family_runs/harmbench_gemma-3-1b-it/results/7_validation/7c_steering/H4a", 4),
]

for rel_path, expected in targets:
    path = os.path.join(local_root, rel_path)
    count = len(glob.glob(os.path.join(path, "*_sweep_results.json"))) if os.path.isdir(path) else 0
    status = "OK" if count == expected else "MISMATCH"
    print(f"{status:8} {count:4d} / {expected:4d}  {path}")
PY
}

if [[ "$VERIFY_ONLY" == "true" ]]; then
  verify_counts
  exit 0
fi

for spec in "${SSH_SPECS[@]}"; do
  IFS="|" read -r label host port user <<<"$spec"
  if [[ -z "$host" ]]; then
    echo "Error: set host environment variable for $label before pulling remote results." >&2
    exit 1
  fi
done

echo "Establishing SSH control connections..."
for spec in "${SSH_SPECS[@]}"; do
  IFS="|" read -r label host port user <<<"$spec"
  start_master "$label" "$host" "$port" "$user"
done

echo "Pulling AmbigQA comparison results..."
sync_from_remote \
  "ambigqa_compare" "$AMBIGQA_COMPARE_HOST" "$AMBIGQA_COMPARE_PORT" "$AMBIGQA_COMPARE_USER" \
  "$REMOTE_AMBIGQA_ROOT/AmbigQA_Qwen3-8B/results/manifest_stage5.json" \
  "$LOCAL_RESULTS_ROOT/AmbigQA_Qwen3-8B/results/"
sync_from_remote \
  "ambigqa_compare" "$AMBIGQA_COMPARE_HOST" "$AMBIGQA_COMPARE_PORT" "$AMBIGQA_COMPARE_USER" \
  "$REMOTE_AMBIGQA_ROOT/AmbigQA_Qwen3-8B/results/5_clustering/" \
  "$LOCAL_RESULTS_ROOT/AmbigQA_Qwen3-8B/results/5_clustering/"
sync_from_remote \
  "ambigqa_compare" "$AMBIGQA_COMPARE_HOST" "$AMBIGQA_COMPARE_PORT" "$AMBIGQA_COMPARE_USER" \
  "$REMOTE_AMBIGQA_ROOT/AmbigQA_Qwen3-8B/results/comparison/coreg/5_clustering/" \
  "$LOCAL_RESULTS_ROOT/AmbigQA_Qwen3-8B/results/comparison/coreg/5_clustering/"
sync_from_remote \
  "ambigqa_compare" "$AMBIGQA_COMPARE_HOST" "$AMBIGQA_COMPARE_PORT" "$AMBIGQA_COMPARE_USER" \
  "$REMOTE_AMBIGQA_ROOT/AmbigQA_Qwen3-8B/results/comparison/concat/5_clustering/" \
  "$LOCAL_RESULTS_ROOT/AmbigQA_Qwen3-8B/results/comparison/concat/5_clustering/"

echo "Pulling Gemma 3 4B results..."
sync_from_remote \
  "gemma4b" "$GEMMA4B_REMOTE_HOST" "$GEMMA4B_REMOTE_PORT" "$GEMMA4B_REMOTE_USER" \
  "$REMOTE_RESULTS_ROOT/full_non_mib_gemma3_family_runs/ambigqa_gemma-3-4b-it/results/5_clustering/" \
  "$LOCAL_RESULTS_ROOT/full_non_mib_gemma3_family_runs/ambigqa_gemma-3-4b-it/results/5_clustering/"
sync_from_remote \
  "gemma4b" "$GEMMA4B_REMOTE_HOST" "$GEMMA4B_REMOTE_PORT" "$GEMMA4B_REMOTE_USER" \
  "$REMOTE_RESULTS_ROOT/full_non_mib_gemma3_family_runs/ambigqa_gemma-3-4b-it/results/7_validation/7c_steering/H4a/" \
  "$LOCAL_RESULTS_ROOT/full_non_mib_gemma3_family_runs/ambigqa_gemma-3-4b-it/results/7_validation/7c_steering/H4a/"

echo "Pulling Gemma 3 1B results..."
sync_from_remote \
  "gemma1b" "$GEMMA1B_REMOTE_HOST" "$GEMMA1B_REMOTE_PORT" "$GEMMA1B_REMOTE_USER" \
  "$REMOTE_RESULTS_ROOT/full_non_mib_gemma3_family_runs/ambigqa_gemma-3-1b-it/results/5_clustering/" \
  "$LOCAL_RESULTS_ROOT/full_non_mib_gemma3_family_runs/ambigqa_gemma-3-1b-it/results/5_clustering/"
sync_from_remote \
  "gemma1b" "$GEMMA1B_REMOTE_HOST" "$GEMMA1B_REMOTE_PORT" "$GEMMA1B_REMOTE_USER" \
  "$REMOTE_RESULTS_ROOT/full_non_mib_gemma3_family_runs/ambigqa_gemma-3-1b-it/results/7_validation/7c_steering/H4a/" \
  "$LOCAL_RESULTS_ROOT/full_non_mib_gemma3_family_runs/ambigqa_gemma-3-1b-it/results/7_validation/7c_steering/H4a/"
sync_from_remote \
  "gemma1b" "$GEMMA1B_REMOTE_HOST" "$GEMMA1B_REMOTE_PORT" "$GEMMA1B_REMOTE_USER" \
  "$REMOTE_RESULTS_ROOT/full_non_mib_gemma3_family_runs/harmbench_gemma-3-1b-it/results/5_clustering/" \
  "$LOCAL_RESULTS_ROOT/full_non_mib_gemma3_family_runs/harmbench_gemma-3-1b-it/results/5_clustering/"
sync_from_remote \
  "gemma1b" "$GEMMA1B_REMOTE_HOST" "$GEMMA1B_REMOTE_PORT" "$GEMMA1B_REMOTE_USER" \
  "$REMOTE_RESULTS_ROOT/full_non_mib_gemma3_family_runs/harmbench_gemma-3-1b-it/results/7_validation/7c_steering/H4a/" \
  "$LOCAL_RESULTS_ROOT/full_non_mib_gemma3_family_runs/harmbench_gemma-3-1b-it/results/7_validation/7c_steering/H4a/"

echo ""
echo "Verification:"
verify_counts
