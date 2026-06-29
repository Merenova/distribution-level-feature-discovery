#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/repo_paths.sh
source "$SCRIPT_DIR/../lib/repo_paths.sh"
ROOT_DIR="${ROOT_DIR:-$(repo_root_from_script "${BASH_SOURCE[0]}")}"
REMOTE_BASE_DEFAULT="${REMOTE_BASE_DEFAULT:-/path/to/latent_planning}"

dir_has_files() {
  local dir_path="$1"
  [[ -d "$dir_path" ]] && find "$dir_path" -mindepth 1 -print -quit 2>/dev/null | grep -q .
}

has_named_files() {
  local dir_path="$1"
  local pattern="$2"
  [[ -d "$dir_path" ]] && find "$dir_path" -type f -name "$pattern" -print -quit 2>/dev/null | grep -q .
}

run_with_env() {
  local gpu_id="$1"
  shift
  if [[ -n "$gpu_id" ]]; then
    env CUDA_VISIBLE_DEVICES="$gpu_id" "$@"
  else
    "$@"
  fi
}

write_model_state() {
  local progress_dir="$1"
  local model_tag="$2"
  local gpu_id="$3"
  local output_dir="$4"
  local logs_dir="$5"
  local state="$6"
  local started_at="$7"
  local ended_at="$8"
  local current_stage_key="$9"
  local current_stage_name="${10}"
  local current_stage_index="${11}"
  local error_message="${12:-}"

  uv run python - <<PY
import json
from pathlib import Path

progress_dir = Path(${progress_dir@Q})
state_path = progress_dir / "state.json"
tmp_path = progress_dir / "state.tmp"
progress_dir.mkdir(parents=True, exist_ok=True)

def maybe_int(value):
    if value in ("", None):
        return None
    return int(value)

def maybe_value(value):
    return None if value in ("", None) else value

payload = {
    "model_tag": ${model_tag@Q},
    "gpu_id": maybe_value(${gpu_id@Q}),
    "output_dir": ${output_dir@Q},
    "logs_dir": ${logs_dir@Q},
    "state": ${state@Q},
    "started_at": maybe_int(${started_at@Q}),
    "ended_at": maybe_int(${ended_at@Q}),
    "current_stage_key": maybe_value(${current_stage_key@Q}),
    "current_stage_name": maybe_value(${current_stage_name@Q}),
    "current_stage_index": maybe_int(${current_stage_index@Q}),
    "error": maybe_value(${error_message@Q}),
}

with open(tmp_path, "w") as f:
    json.dump(payload, f, indent=2)
tmp_path.replace(state_path)
PY
}

write_stage_state() {
  local progress_dir="$1"
  local stage_key="$2"
  local stage_name="$3"
  local stage_index="$4"
  local status="$5"
  local start_epoch="$6"
  local end_epoch="$7"
  local log_file="$8"

  uv run python - <<PY
import json
from pathlib import Path

progress_dir = Path(${progress_dir@Q})
stage_dir = progress_dir / "stages"
stage_dir.mkdir(parents=True, exist_ok=True)
stage_key = ${stage_key@Q}
stage_path = stage_dir / f"{stage_key}.json"
tmp_path = stage_dir / f"{stage_key}.tmp"

def maybe_int(value):
    if value in ("", None):
        return None
    return int(value)

payload = {
    "key": ${stage_key@Q},
    "name": ${stage_name@Q},
    "index": int(${stage_index@Q}),
    "status": ${status@Q},
    "start_epoch": maybe_int(${start_epoch@Q}),
    "end_epoch": maybe_int(${end_epoch@Q}),
    "log_file": ${log_file@Q},
}

with open(tmp_path, "w") as f:
    json.dump(payload, f, indent=2)
tmp_path.replace(stage_path)
PY
}

write_remote_baseline_meta() {
  local progress_dir="$1"
  local model_tag="$2"
  local gpu_label="$3"
  local output_dir="$4"
  local config_path="$5"
  local results_dir="$6"

  uv run python - <<PY
import json
from pathlib import Path

progress_dir = Path(${progress_dir@Q})
meta_path = progress_dir / "meta.json"
tmp_path = progress_dir / "meta.tmp"
progress_dir.mkdir(parents=True, exist_ok=True)

results_dir = Path(${results_dir@Q})
config_path = Path(${config_path@Q})

if (results_dir / "test_clozes.json").exists():
    with open(results_dir / "test_clozes.json") as f:
        test_clozes = json.load(f)
    total_samples = len(test_clozes.get("clozes", []))
else:
    total_samples = sum(1 for _ in (results_dir / "2_branch_sampling").glob("*_branches.json"))

with open(config_path) as f:
    config = json.load(f)

steering = config.get("stage_7c_steering", {})
hypotheses = steering.get("hypotheses") or ["H4a"]
hypothesis_count = max(1, len(hypotheses))

stages = [
    {
        "key": "stage7c2_single",
        "name": "Single Base",
        "total_units": total_samples * hypothesis_count,
        "count": {"mode": "glob", "root": "results/7_validation/7c_baseline_single", "pattern": "*_sweep_results.json", "recursive": True},
    },
    {
        "key": "stage7c2_combined",
        "name": "Combined Medoid",
        "total_units": total_samples * hypothesis_count,
        "count": {"mode": "glob", "root": "results/7_validation/7c_baseline_combined_medoid", "pattern": "*_sweep_results.json", "recursive": True},
    },
]

payload = {
    "model_tag": ${model_tag@Q},
    "gpu_id": None if ${gpu_label@Q} in ("", None) else ${gpu_label@Q},
    "output_dir": ${output_dir@Q},
    "total_samples": total_samples,
    "hypothesis_count": hypothesis_count,
    "stages": stages,
}

with open(tmp_path, "w") as f:
    json.dump(payload, f, indent=2)
tmp_path.replace(meta_path)
PY
}

mark_stage_skipped() {
  local progress_dir="$1"
  local stage_key="$2"
  local stage_name="$3"
  local stage_index="$4"
  local log_file="$5"
  local now_epoch
  now_epoch="$(date +%s)"
  write_stage_state "$progress_dir" "$stage_key" "$stage_name" "$stage_index" "skipped" "$now_epoch" "$now_epoch" "$log_file"
}

load_runtime_vars() {
  local config_path="$1"
  local config_vars
  config_vars="$(uv run python - <<PY
import json
with open(${config_path@Q}) as f:
    config = json.load(f)
steering = config.get("stage_7c_steering", {})
global_cfg = config.get("global", {})
max_batch_size = steering.get("max_batch_size", global_cfg.get("batch_size", 512))
if max_batch_size is None or int(max_batch_size) <= 0:
    max_batch_size = global_cfg.get("batch_size", 512)
print(f'STEERING_MAX_BATCH_SIZE={int(max_batch_size)}')
print(f'STEERING_PREFIX_BATCH_SIZE={int(steering.get("prefix_batch_size", 16))}')
print(f'STEERING_CROSS_PREFIX_BATCHING={str(steering.get("cross_prefix_batching", False)).lower()}')
PY
)"
  eval "$config_vars"
}

render_progress_dashboard() {
  local progress_subdir="$1"
  shift
  local output_dirs=("$@")
  local last_frame=""
  local line_count=0
  local first_frame="true"

  if [[ -t 1 ]]; then
    printf '\033[?25l'
  fi

  while true; do
    local frame
    frame="$(uv run python scripts/remote/render_mmlu_pipeline_progress.py --progress-subdir "$progress_subdir" --output-dirs "${output_dirs[@]}")"

    if [[ "$frame" != "$last_frame" ]]; then
      if [[ -t 1 ]]; then
        if [[ "$first_frame" != "true" && "$line_count" -gt 0 ]]; then
          printf '\033[%sA' "$line_count"
          printf '\r\033[J'
        fi
        printf '%s' "$frame"
        [[ "$frame" == *$'\n' ]] || printf '\n'
      else
        printf '%s\n\n' "$frame"
      fi
      line_count="$(printf '%s\n' "$frame" | wc -l | tr -d ' ')"
      last_frame="$frame"
      first_frame="false"
    fi

    local any_alive="false"
    for pid in "${JOB_PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        any_alive="true"
        break
      fi
    done

    if [[ "$any_alive" != "true" ]]; then
      break
    fi
    sleep 1
  done

  if [[ -t 1 ]]; then
    printf '\033[?25h'
  fi
}

start_ssh_master() {
  local control_path="$1"
  local host="$2"
  local port="$3"
  local user="$4"
  ssh -p "$port" -o ControlMaster=yes -o ControlPath="$control_path" -o ControlPersist=300 -fN "$user@$host"
}

stop_ssh_master() {
  local control_path="$1"
  local host="$2"
  local port="$3"
  local user="$4"
  ssh -p "$port" -o ControlPath="$control_path" -O exit "$user@$host" 2>/dev/null || true
}

require_graph_validation_done() {
  local output_dir="$1"
  local gate_file="$output_dir/results/7_validation/7a_graph_validation/graph_validation.json"
  if [[ ! -f "$gate_file" ]]; then
    echo "Error: graph validation gate not satisfied: $gate_file" >&2
    echo "Run through stage 7/10 Graph Validation before syncing remote baselines." >&2
    return 1
  fi
}

sync_path_to_remote() {
  local ssh_cmd="$1"
  local rsync_opts="$2"
  local source_path="$3"
  local dest_path="$4"
  shift 4
  rsync $rsync_opts "$@" -e "$ssh_cmd" "$source_path" "$dest_path"
}

set_remote_target_for_model_tag() {
  local model_tag="$1"
  case "$model_tag" in
    *Qwen3-8B)
      REMOTE_HOST="${MMLU_QWEN3_8B_HOST:-}"
      REMOTE_PORT="${MMLU_QWEN3_8B_PORT:-40206}"
      REMOTE_USER="${MMLU_QWEN3_8B_USER:-${REMOTE_USER:-${USER:-}}}"
      ;;
    *Qwen3-4B)
      REMOTE_HOST="${MMLU_QWEN3_4B_HOST:-}"
      REMOTE_PORT="${MMLU_QWEN3_4B_PORT:-40394}"
      REMOTE_USER="${MMLU_QWEN3_4B_USER:-${REMOTE_USER:-${USER:-}}}"
      ;;
    *)
      echo "Error: unsupported remote target for model tag: $model_tag" >&2
      return 1
      ;;
  esac
  if [[ -z "$REMOTE_HOST" ]]; then
    echo "Error: set the remote host for $model_tag via MMLU_QWEN3_8B_HOST or MMLU_QWEN3_4B_HOST." >&2
    return 1
  fi
}

remote_path_exists() {
  local control_path="$1"
  local host="$2"
  local port="$3"
  local user="$4"
  local remote_path="$5"
  ssh -p "$port" -o ControlPath="$control_path" "$user@$host" "test -e '$remote_path'"
}

mirror_remote_progress_to_local() {
  local remote_progress_dir="$1"
  local local_progress_dir="$2"
  local local_logs_dir="$3"

  uv run python - <<PY
import json
from pathlib import Path

remote_progress_dir = Path(${remote_progress_dir@Q})
local_progress_dir = Path(${local_progress_dir@Q})
local_logs_dir = Path(${local_logs_dir@Q})

stage_map = {
    "stage7c2_single": {"name": "Single Base", "index": 9, "log_file": local_logs_dir / "remote_stage_single.log"},
    "stage7c2_combined": {"name": "Combined Medoid", "index": 10, "log_file": local_logs_dir / "remote_stage_combined.log"},
}

remote_stage_dir = remote_progress_dir / "stages"
if not remote_stage_dir.exists():
    raise SystemExit(0)

local_stage_dir = local_progress_dir / "stages"
local_stage_dir.mkdir(parents=True, exist_ok=True)

for stage_key, overrides in stage_map.items():
    src = remote_stage_dir / f"{stage_key}.json"
    if not src.exists():
        continue
    try:
        payload = json.loads(src.read_text())
    except Exception:
        continue

    payload["key"] = stage_key
    payload["name"] = overrides["name"]
    payload["index"] = overrides["index"]
    payload["log_file"] = str(overrides["log_file"])

    dest = local_stage_dir / f"{stage_key}.json"
    tmp = local_stage_dir / f"{stage_key}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(dest)
PY
}

sync_remote_baseline_workspace() {
  local control_path="$1"
  local host="$2"
  local port="$3"
  local user="$4"
  local remote_base="$5"
  local model_tag="$6"
  local config_rel="$7"
  local output_dir="$8"

  local results_dir="$output_dir/results"
  local ssh_cmd="ssh -p $port -o ControlMaster=auto -o ControlPath=$control_path -o ControlPersist=300"
  local rsync_opts="-az --human-readable --partial"
  local code_excludes=(
    --exclude="__pycache__"
    --exclude="*.pyc"
    --exclude=".pytest_cache"
    --exclude=".mypy_cache"
    --exclude="*.log"
    --exclude=".DS_Store"
  )

  ssh -p "$port" -o ControlPath="$control_path" "$user@$host" \
    "mkdir -p \
      '$remote_base/$model_tag/results/4_feature_extraction' \
      '$remote_base/$model_tag/results/7_validation' \
      '$remote_base/$model_tag/logs' \
      '$remote_base/scripts' \
      '$remote_base/configs'"

  sync_path_to_remote "$ssh_cmd" "$rsync_opts" \
    "$ROOT_DIR/7_validation" \
    "$user@$host:$remote_base/" \
    "${code_excludes[@]}"
  sync_path_to_remote "$ssh_cmd" "$rsync_opts" \
    "$ROOT_DIR/utils" \
    "$user@$host:$remote_base/" \
    "${code_excludes[@]}"
  sync_path_to_remote "$ssh_cmd" "$rsync_opts" \
    "$ROOT_DIR/circuit-tracer" \
    "$user@$host:$remote_base/" \
    "${code_excludes[@]}"

  for script_file in \
    scripts/remote/render_mmlu_pipeline_progress.py \
    scripts/misc/mmlu_remote_baseline_lib.sh \
    scripts/remote/run_remote_mmlu_baselines_worker.sh
  do
    sync_path_to_remote "$ssh_cmd" "$rsync_opts" \
      "$ROOT_DIR/$script_file" \
      "$user@$host:$remote_base/$(dirname "$script_file")/"
  done

  for code_file in pyproject.toml uv.lock "$config_rel"; do
    sync_path_to_remote "$ssh_cmd" "$rsync_opts" \
      "$ROOT_DIR/$code_file" \
      "$user@$host:$remote_base/$(dirname "$code_file")/"
  done

  sync_path_to_remote "$ssh_cmd" "$rsync_opts" \
    "$results_dir/2_branch_sampling" \
    "$user@$host:$remote_base/$model_tag/results/"
  sync_path_to_remote "$ssh_cmd" "$rsync_opts" \
    "$results_dir/3_attribution_graphs" \
    "$user@$host:$remote_base/$model_tag/results/"
  sync_path_to_remote "$ssh_cmd" "$rsync_opts" \
    "$results_dir/4_feature_extraction/embeddings" \
    "$user@$host:$remote_base/$model_tag/results/4_feature_extraction/"
  sync_path_to_remote "$ssh_cmd" "$rsync_opts" \
    "$results_dir/5_clustering" \
    "$user@$host:$remote_base/$model_tag/results/" \
    --exclude="intermediate" --exclude="intermediate/" \
    --exclude="*.log" --exclude="__pycache__"

  if [[ -f "$results_dir/test_clozes.json" ]]; then
    sync_path_to_remote "$ssh_cmd" "$rsync_opts" \
      "$results_dir/test_clozes.json" \
      "$user@$host:$remote_base/$model_tag/results/"
  fi

  if [[ -d "$results_dir/7_validation/7c_baseline_single" ]]; then
    sync_path_to_remote "$ssh_cmd" "$rsync_opts" \
      "$results_dir/7_validation/7c_baseline_single" \
      "$user@$host:$remote_base/$model_tag/results/7_validation/"
  fi

  if [[ -d "$results_dir/7_validation/7c_baseline_combined_medoid" ]]; then
    sync_path_to_remote "$ssh_cmd" "$rsync_opts" \
      "$results_dir/7_validation/7c_baseline_combined_medoid" \
      "$user@$host:$remote_base/$model_tag/results/7_validation/"
  fi
}

launch_remote_baseline_worker() {
  local control_path="$1"
  local host="$2"
  local port="$3"
  local user="$4"
  local remote_base="$5"
  local model_tag="$6"
  local config_rel="$7"
  local skip_existing="$8"
  local quiet="$9"
  local launcher_log="${10}"
  local remote_cmd

  remote_cmd="cd '$remote_base' && MMLU_DISABLE_PROGRESS=1 bash scripts/remote/run_remote_mmlu_baselines_worker.sh --model-tag '$model_tag' --config '$config_rel' --gpu-single 0 --gpu-combined 1"
  if [[ "$skip_existing" == "true" ]]; then
    remote_cmd+=" --skip-existing"
  else
    remote_cmd+=" --no-skip-existing"
  fi
  if [[ "$quiet" == "true" ]]; then
    remote_cmd+=" --quiet"
  fi

  mkdir -p "$(dirname "$launcher_log")"
  ssh -p "$port" -o ControlPath="$control_path" "$user@$host" "$remote_cmd" >"$launcher_log" 2>&1 &
  REMOTE_WORKER_PID="$!"
}

pull_remote_baseline_outputs() {
  local control_path="$1"
  local host="$2"
  local port="$3"
  local user="$4"
  local remote_base="$5"
  local model_tag="$6"
  local output_dir="$7"

  local ssh_cmd="ssh -p $port -o ControlMaster=auto -o ControlPath=$control_path -o ControlPersist=300"
  local rsync_opts="-az --human-readable --partial"
  local remote_root="$remote_base/$model_tag"
  local local_results_dir="$output_dir/results/7_validation"
  local local_logs_dir="$output_dir/logs/remote_baselines"
  local local_remote_progress_root="$output_dir/progress_remote_snapshot"
  local local_remote_progress_dir="$local_remote_progress_root/progress_remote_baselines"
  local local_progress_dir="$output_dir/progress"

  mkdir -p "$local_results_dir" "$local_logs_dir" "$local_remote_progress_root"

  if remote_path_exists "$control_path" "$host" "$port" "$user" "$remote_root/results/7_validation/7c_baseline_single"; then
    sync_path_to_remote "$ssh_cmd" "$rsync_opts" \
      "$user@$host:$remote_root/results/7_validation/7c_baseline_single" \
      "$local_results_dir/"
  fi

  if remote_path_exists "$control_path" "$host" "$port" "$user" "$remote_root/results/7_validation/7c_baseline_combined_medoid"; then
    sync_path_to_remote "$ssh_cmd" "$rsync_opts" \
      "$user@$host:$remote_root/results/7_validation/7c_baseline_combined_medoid" \
      "$local_results_dir/"
  fi

  if remote_path_exists "$control_path" "$host" "$port" "$user" "$remote_root/logs/remote_baselines"; then
    sync_path_to_remote "$ssh_cmd" "$rsync_opts" \
      "$user@$host:$remote_root/logs/remote_baselines" \
      "$output_dir/logs/"
  fi

  if remote_path_exists "$control_path" "$host" "$port" "$user" "$remote_root/progress_remote_baselines"; then
    sync_path_to_remote "$ssh_cmd" "$rsync_opts" \
      "$user@$host:$remote_root/progress_remote_baselines" \
      "$local_remote_progress_root/"
    mirror_remote_progress_to_local "$local_remote_progress_dir" "$local_progress_dir" "$local_logs_dir"
  fi
}

start_remote_pull_loop() {
  local control_path="$1"
  local host="$2"
  local port="$3"
  local user="$4"
  local remote_base="$5"
  local model_tag="$6"
  local output_dir="$7"
  local interval_seconds="${8:-10}"

  (
    while true; do
      pull_remote_baseline_outputs "$control_path" "$host" "$port" "$user" "$remote_base" "$model_tag" "$output_dir" || true
      sleep "$interval_seconds"
    done
  ) &
  REMOTE_PULL_PID="$!"
}
