#!/usr/bin/env bash
# Run the publication scale matrix on one 8-GPU host.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/scale_8gpu_$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="false"
ONLY=""
STAGE7_ACCELERATED="false"
STAGE7_SHARD_COUNT="${STAGE7_SHARD_COUNT:-8}"
STAGE7_BASELINES="${STAGE7_BASELINES:-combined_medoid}"
STAGE7_CONFIG_OVERLAY="${STAGE7_CONFIG_OVERLAY:-configs/stage7_fast_gemma_overlay.json}"
SCALE_ARGS=()

PARALLEL_COMBOS=(
  "0:AmbigQA_Gemma3-4B-it"
  "1:MMLU_Qwen3-8B"
  "2:MMLU_Qwen3-4B"
  "3:MMLU_Gemma3-4B-it"
  "4:HarmBench_Qwen3-8B"
  "5:HarmBench_Qwen3-4B"
  "6:HarmBench_Gemma3-4B-it"
)

GEMMA3_1B_GPU="${GEMMA3_1B_GPU:-7}"
GEMMA3_1B_COMBOS=(
  "AmbigQA_Gemma3-1B-it"
  "MMLU_Gemma3-1B-it"
  "HarmBench_Gemma3-1B-it"
)

usage() {
  cat <<'EOF'
Usage: bash scripts/remote/run_scale_experiments_8gpu.sh [options]

Options:
  --dry-run   Print the GPU schedule without launching jobs
  --only TAG1,TAG2
              Run only selected combo tags from the 8-GPU schedule
  --resume STAGE
              Pass --resume STAGE through to each run_pipeline.sh invocation
  --skip-existing
              Pass --skip-existing through to each run_pipeline.sh invocation
  --scale-arg ARG
              Pass one extra literal arg through to scripts/run_scale_experiments.sh
  --stage7-accelerated
              Run selected Stage 7c baselines directly with prefix sharding
  --stage7-shard-count N
              Number of prefix shards per selected tag in accelerated mode (default: 8)
  --stage7-baselines LIST
              Comma-separated Stage 7 baselines in accelerated mode: combined_medoid,single,kmeans
  --stage7-config-overlay PATH
              JSON config overlay merged over each tag config in accelerated mode
              (default: configs/stage7_fast_gemma_overlay.json; use "" to disable)
  --help      Show this help

GPU schedule:
  GPUs 0-6 run non-Gemma3-1B combos in parallel.
  GPU 7 runs all Gemma3-1B combos sequentially.
  With --stage7-accelerated, selected Stage 7c baseline shards are scheduled
  across GPUs with separate per-shard logs.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --only)
      ONLY="$2"
      shift 2
      ;;
    --resume)
      SCALE_ARGS+=(--resume "$2")
      shift 2
      ;;
    --skip-existing|--skip_existing)
      SCALE_ARGS+=(--skip-existing)
      shift
      ;;
    --scale-arg)
      SCALE_ARGS+=("$2")
      shift 2
      ;;
    --stage7-accelerated)
      STAGE7_ACCELERATED="true"
      shift
      ;;
    --stage7-shard-count)
      STAGE7_SHARD_COUNT="$2"
      shift 2
      ;;
    --stage7-baselines)
      STAGE7_BASELINES="$2"
      shift 2
      ;;
    --stage7-config-overlay)
      STAGE7_CONFIG_OVERLAY="$2"
      shift 2
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

declare -A WANT=()
if [[ -n "$ONLY" ]]; then
  IFS=',' read -ra ONLY_PARTS <<<"$ONLY"
  for tag in "${ONLY_PARTS[@]}"; do
    [[ -n "$tag" ]] && WANT["$tag"]=1
  done
fi

tag_selected() {
  local tag="$1"
  [[ -z "$ONLY" || -n "${WANT[$tag]:-}" ]]
}

sanitize_tag() {
  printf '%s' "$1" | tr -c 'A-Za-z0-9_.-' '_'
}

config_for_tag() {
  case "$1" in
    AmbigQA_Gemma3-1B-it) echo "configs/ambigqa_gemma3_1b_it_config.json" ;;
    AmbigQA_Gemma3-4B-it) echo "configs/ambigqa_gemma3_4b_it_config.json" ;;
    MMLU_Qwen3-8B) echo "configs/mmlu_qwen3_8b_config.json" ;;
    MMLU_Qwen3-4B) echo "configs/mmlu_qwen3_4b_config.json" ;;
    MMLU_Gemma3-1B-it) echo "configs/mmlu_gemma3_1b_it_config.json" ;;
    MMLU_Gemma3-4B-it) echo "configs/mmlu_gemma3_4b_it_config.json" ;;
    HarmBench_Qwen3-8B) echo "configs/harmbench_qwen3_8b_config.json" ;;
    HarmBench_Qwen3-4B) echo "configs/harmbench_qwen3_4b_config.json" ;;
    HarmBench_Gemma3-1B-it) echo "configs/harmbench_gemma3_1b_it_config.json" ;;
    HarmBench_Gemma3-4B-it) echo "configs/harmbench_gemma3_4b_it_config.json" ;;
    *) echo "Unknown tag: $1" >&2; return 1 ;;
  esac
}

declare -A STAGE7_TAG_CONFIGS=()
declare -A STAGE7_TAG_MANIFESTS=()

stage7_config_for_tag() {
  local tag="$1"
  local base_config
  base_config="$(config_for_tag "$tag")"

  if [[ -z "$STAGE7_CONFIG_OVERLAY" ]]; then
    echo "$base_config"
    return 0
  fi

  local safe_tag out_config
  safe_tag="$(sanitize_tag "$tag")"
  out_config="$LOG_DIR/stage7_accelerated_configs/${safe_tag}.json"

  mkdir -p "$(dirname "$out_config")"
  BASE_CONFIG="$base_config" OVERLAY_CONFIG="$STAGE7_CONFIG_OVERLAY" OUT_CONFIG="$out_config" \
    uv run python - <<'PY'
import json
import os
from pathlib import Path


def deep_merge(base, overlay):
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


base_path = Path(os.environ["BASE_CONFIG"])
overlay_path = Path(os.environ["OVERLAY_CONFIG"])
out_path = Path(os.environ["OUT_CONFIG"])
base = json.loads(base_path.read_text())
overlay = json.loads(overlay_path.read_text())
merged = deep_merge(base, overlay)
out_path.write_text(json.dumps(merged, indent=2) + "\n")
PY
  echo "$out_config"
}

stage7_manifest_for_tag() {
  local tag="$1"
  local config="$2"

  local manifest top_k score_key score_order min_k max_k max_prefixes
  eval "$(
    CONFIG_PATH="$config" uv run python - <<'PY'
import json
import os
import shlex
from pathlib import Path

config_path = Path(os.environ["CONFIG_PATH"])
if config_path.exists():
    config = json.loads(config_path.read_text())
else:
    config = {}
steering = config.get("stage_7c_steering", {})
selection = steering.get("clustering_selection", {}) if isinstance(steering.get("clustering_selection"), dict) else {}

def value(*keys, default=""):
    for key in keys:
        if key in steering and steering[key] is not None:
            return steering[key]
        if key in selection and selection[key] is not None:
            return selection[key]
    return default

for name, raw in {
    "manifest": value("clustering_manifest"),
    "top_k": value("clustering_top_k", "top_k"),
    "score_key": value("clustering_score_key", "score_key", default="harmonic"),
    "score_order": value("clustering_score_order", "score_order", default="desc"),
    "min_k": value("clustering_min_k", "min_k", default=2),
    "max_k": value("clustering_max_k", "max_k"),
    "max_prefixes": value("max_samples"),
}.items():
    print(f"{name}={shlex.quote(str(raw))}")
PY
  )"

  if [[ -n "${manifest:-}" ]]; then
    echo "$manifest"
    return 0
  fi
  if [[ -z "${top_k:-}" || "${top_k:-0}" == "0" ]]; then
    echo ""
    return 0
  fi

  local out_manifest="$ROOT_DIR/$tag/results/manifests/stage7_clustering_topk.json"
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "$out_manifest"
    return 0
  fi

  mkdir -p "$(dirname "$out_manifest")"
  local -a max_k_args=()
  if [[ -n "${max_k:-}" ]]; then
    max_k_args=(--max-k "$max_k")
  fi
  local -a max_prefixes_args=()
  if [[ -n "${max_prefixes:-}" ]]; then
    max_prefixes_args=(--max-prefixes "$max_prefixes")
  fi
  uv run python 7_validation/select_stage7_clustering_manifest.py \
    --stage5-dir "$ROOT_DIR/$tag/results/5_clustering" \
    --output "$out_manifest" \
    --top-k "$top_k" \
    --score-key "${score_key:-harmonic}" \
    --score-order "${score_order:-desc}" \
    --min-k "${min_k:-2}" \
    "${max_k_args[@]}" \
    "${max_prefixes_args[@]}" >/dev/null
  echo "$out_manifest"
}

check_gpus() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "Warning: nvidia-smi not found; skipping GPU visibility check." >&2
    return 0
  fi

  local visible_count
  visible_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
  if [[ "$visible_count" -lt 8 ]]; then
    echo "Error: expected at least 8 visible GPUs, found $visible_count." >&2
    return 1
  fi
}

run_lane() {
  local gpu_id="$1"
  local tag="$2"
  local safe_tag
  safe_tag="$(sanitize_tag "$tag")"
  local log_file="$LOG_DIR/${safe_tag}.gpu${gpu_id}.log"

  echo "[$(date '+%F %T')] GPU $gpu_id -> $tag"
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "  CUDA_VISIBLE_DEVICES=$gpu_id bash scripts/run_scale_experiments.sh --only $tag ${SCALE_ARGS[*]}"
    return 0
  fi

  mkdir -p "$LOG_DIR"
  CUDA_VISIBLE_DEVICES="$gpu_id" \
    bash "$ROOT_DIR/scripts/run_scale_experiments.sh" --only "$tag" "${SCALE_ARGS[@]}" \
    >"$log_file" 2>&1
}

run_gemma3_1b_lane() {
  local tag
  for tag in "${GEMMA3_1B_COMBOS[@]}"; do
    tag_selected "$tag" || continue
    run_lane "$GEMMA3_1B_GPU" "$tag"
  done
}

validate_stage7_accelerated_args() {
  if [[ ! "$STAGE7_SHARD_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --stage7-shard-count must be a positive integer, got: $STAGE7_SHARD_COUNT" >&2
    return 1
  fi

  if [[ -z "$STAGE7_BASELINES" ]]; then
    echo "Error: --stage7-baselines must not be empty" >&2
    return 1
  fi

  local baseline
  IFS=',' read -ra STAGE7_BASELINE_PARTS <<<"$STAGE7_BASELINES"
  for baseline in "${STAGE7_BASELINE_PARTS[@]}"; do
    case "$baseline" in
      combined_medoid|single|kmeans) ;;
      *)
        echo "Error: unknown Stage 7 baseline '$baseline'." >&2
        echo "Valid baselines: combined_medoid,single,kmeans" >&2
        return 1
        ;;
    esac
  done
}

run_stage7_baseline_shard() {
  local gpu_id="$1"
  local tag="$2"
  local baseline="$3"
  local shard_index="$4"
  local shard_count="$5"
  local config manifest
  config="${STAGE7_TAG_CONFIGS[$tag]:-$(config_for_tag "$tag")}"
  manifest="${STAGE7_TAG_MANIFESTS[$tag]:-}"

  local safe_tag safe_baseline shard_log_dir log_file out_dir
  safe_tag="$(sanitize_tag "$tag")"
  safe_baseline="$(sanitize_tag "$baseline")"
  shard_log_dir="$LOG_DIR/${safe_tag}.${safe_baseline}.shard${shard_index}.gpu${gpu_id}"
  log_file="$shard_log_dir/stdout_stderr.log"

  local -a cmd
  case "$baseline" in
    combined_medoid)
      out_dir="$ROOT_DIR/$tag/results/7_validation/7c_combined_medoid"
      cmd=(
        uv run python 7_validation/7c_baseline_combined_medoid.py
        --samples-dir "$ROOT_DIR/$tag/results/2_branch_sampling"
        --embeddings-dir "$ROOT_DIR/$tag/results/4_feature_extraction/embeddings"
        --attribution-graphs-dir "$ROOT_DIR/$tag/results/3_attribution_graphs"
        --clustering-dir "$ROOT_DIR/$tag/results/5_clustering"
        --output-dir "$out_dir"
        --config "$config"
        --skip-existing
        --prefix-shard-index "$shard_index"
        --prefix-shard-count "$shard_count"
        --log-dir "$shard_log_dir"
      )
      ;;
    single)
      out_dir="$ROOT_DIR/$tag/results/7_validation/7c_single"
      cmd=(
        uv run python 7_validation/7c_baseline_single.py
        --samples-dir "$ROOT_DIR/$tag/results/2_branch_sampling"
        --attribution-graphs-dir "$ROOT_DIR/$tag/results/3_attribution_graphs"
        --clustering-dir "$ROOT_DIR/$tag/results/5_clustering"
        --output-dir "$out_dir"
        --config "$config"
        --skip-existing
        --prefix-shard-index "$shard_index"
        --prefix-shard-count "$shard_count"
        --log-dir "$shard_log_dir"
      )
      ;;
    kmeans)
      out_dir="$ROOT_DIR/$tag/results/7_validation/7c_kmeans"
      cmd=(
        uv run python 7_validation/7c_baseline_kmeans.py
        --samples-dir "$ROOT_DIR/$tag/results/2_branch_sampling"
        --embeddings-dir "$ROOT_DIR/$tag/results/4_feature_extraction/embeddings"
        --attribution-graphs-dir "$ROOT_DIR/$tag/results/3_attribution_graphs"
        --clustering-dir "$ROOT_DIR/$tag/results/5_clustering"
        --output-dir "$out_dir"
        --config "$config"
        --skip-existing
        --prefix-shard-index "$shard_index"
        --prefix-shard-count "$shard_count"
        --log-dir "$shard_log_dir"
      )
      ;;
    *)
      echo "Unknown Stage 7 baseline: $baseline" >&2
      return 1
      ;;
  esac

  if [[ -n "$manifest" ]]; then
    cmd+=(--clustering-manifest "$manifest")
  fi

  echo "[$(date '+%F %T')] GPU $gpu_id -> $tag $baseline shard $shard_index/$shard_count"
  if [[ "$DRY_RUN" == "true" ]]; then
    printf '  CUDA_VISIBLE_DEVICES=%s' "$gpu_id"
    printf ' %q' "${cmd[@]}"
    printf ' >%q 2>&1\n' "$log_file"
    return 0
  fi

  mkdir -p "$shard_log_dir"
  CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}" >"$log_file" 2>&1
}

declare -a ACCEL_WAVE_PIDS=()
declare -a ACCEL_WAVE_LABELS=()

wait_stage7_wave() {
  local status=0
  local i
  for i in "${!ACCEL_WAVE_PIDS[@]}"; do
    if ! wait "${ACCEL_WAVE_PIDS[$i]}"; then
      echo "Stage 7 shard failed: ${ACCEL_WAVE_LABELS[$i]}" >&2
      status=1
    fi
  done
  ACCEL_WAVE_PIDS=()
  ACCEL_WAVE_LABELS=()
  return "$status"
}

run_stage7_accelerated() {
  validate_stage7_accelerated_args

  IFS=',' read -ra STAGE7_BASELINE_PARTS <<<"$STAGE7_BASELINES"

  local -a selected_tags=()
  local assignment tag gpu_id baseline shard_index status wave_gpu
  for assignment in "${PARALLEL_COMBOS[@]}"; do
    IFS=":" read -r _gpu_id tag <<<"$assignment"
    tag_selected "$tag" && selected_tags+=("$tag")
  done
  for tag in "${GEMMA3_1B_COMBOS[@]}"; do
    tag_selected "$tag" && selected_tags+=("$tag")
  done

  if [[ "${#selected_tags[@]}" -eq 0 ]]; then
    echo "No selected scale lanes matched: ${ONLY:-<all>}" >&2
    return 1
  fi

  echo "Stage 7 accelerated mode: baselines=$STAGE7_BASELINES shard_count=$STAGE7_SHARD_COUNT selected_tags=${#selected_tags[@]}"
  if [[ -n "$STAGE7_CONFIG_OVERLAY" ]]; then
    echo "Stage 7 accelerated config overlay: $STAGE7_CONFIG_OVERLAY"
  fi

  for tag in "${selected_tags[@]}"; do
    local prepared_config prepared_manifest
    prepared_config="$(stage7_config_for_tag "$tag")"
    prepared_manifest="$(stage7_manifest_for_tag "$tag" "$prepared_config")"
    STAGE7_TAG_CONFIGS["$tag"]="$prepared_config"
    STAGE7_TAG_MANIFESTS["$tag"]="$prepared_manifest"
    if [[ -n "$prepared_manifest" ]]; then
      echo "Prepared $tag config=$prepared_config manifest=$prepared_manifest"
    else
      echo "Prepared $tag config=$prepared_config"
    fi
  done

  status=0
  wave_gpu=0
  for baseline in "${STAGE7_BASELINE_PARTS[@]}"; do
    for tag in "${selected_tags[@]}"; do
      for ((shard_index=0; shard_index<STAGE7_SHARD_COUNT; shard_index++)); do
        gpu_id="$wave_gpu"
        if [[ "$DRY_RUN" == "true" ]]; then
          run_stage7_baseline_shard "$gpu_id" "$tag" "$baseline" "$shard_index" "$STAGE7_SHARD_COUNT"
        else
          run_stage7_baseline_shard "$gpu_id" "$tag" "$baseline" "$shard_index" "$STAGE7_SHARD_COUNT" &
          ACCEL_WAVE_PIDS+=("$!")
          ACCEL_WAVE_LABELS+=("$tag $baseline shard $shard_index/$STAGE7_SHARD_COUNT gpu $gpu_id")
        fi
        wave_gpu=$(( (wave_gpu + 1) % 8 ))

        if [[ "${#ACCEL_WAVE_PIDS[@]}" -eq 8 ]]; then
          if ! wait_stage7_wave; then
            status=1
          fi
          wave_gpu=0
        fi
      done
    done
  done

  if [[ "${#ACCEL_WAVE_PIDS[@]}" -gt 0 ]]; then
    if ! wait_stage7_wave; then
      status=1
    fi
  fi

  if [[ "$status" -ne 0 ]]; then
    echo "One or more accelerated Stage 7 shards failed. See logs in $LOG_DIR" >&2
    return "$status"
  fi

  echo "Accelerated Stage 7 shards finished. Logs: $LOG_DIR"
}

echo "Scale 8-GPU run logs: $LOG_DIR"

if [[ "$DRY_RUN" != "true" ]]; then
  mkdir -p "$LOG_DIR"
  check_gpus
fi

if [[ "$STAGE7_ACCELERATED" == "true" ]]; then
  run_stage7_accelerated
  exit "$?"
fi

declare -a PIDS=()
declare -a LABELS=()

for assignment in "${PARALLEL_COMBOS[@]}"; do
  IFS=":" read -r gpu_id tag <<<"$assignment"
  tag_selected "$tag" || continue
  run_lane "$gpu_id" "$tag" &
  PIDS+=("$!")
  LABELS+=("$tag")
done

if [[ -z "$ONLY" || -n "${WANT[AmbigQA_Gemma3-1B-it]:-}" || -n "${WANT[MMLU_Gemma3-1B-it]:-}" || -n "${WANT[HarmBench_Gemma3-1B-it]:-}" ]]; then
  run_gemma3_1b_lane &
  PIDS+=("$!")
  LABELS+=("Gemma3-1B-sequential")
fi

if [[ "${#PIDS[@]}" -eq 0 ]]; then
  echo "No selected scale lanes matched: ${ONLY:-<all>}" >&2
  exit 1
fi

status=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "Lane failed: ${LABELS[$i]}" >&2
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "One or more scale lanes failed. See logs in $LOG_DIR" >&2
  exit "$status"
fi

echo "All 8-GPU scale lanes finished. Logs: $LOG_DIR"
