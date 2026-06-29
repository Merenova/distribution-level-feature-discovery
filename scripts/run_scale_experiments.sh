#!/usr/bin/env bash
# Run all (model x dataset) combos for the scale-up experiments.
# Each combo invokes scripts/run_pipeline.sh with the right config and an
# output dir of the form {Dataset}_{ModelTag}/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# tag                        config                                              cloze_dir
COMBOS_RAW='
AmbigQA_Gemma3-1B-it       configs/ambigqa_gemma3_1b_it_config.json     data/cloze_llm_improved_split_ratio_0.1
AmbigQA_Gemma3-4B-it       configs/ambigqa_gemma3_4b_it_config.json     data/cloze_llm_improved_split_ratio_0.1
MMLU_Qwen3-8B              configs/mmlu_qwen3_8b_config.json            data/mmlu_cais_validation_7subjects
MMLU_Qwen3-4B              configs/mmlu_qwen3_4b_config.json            data/mmlu_cais_validation_7subjects
MMLU_Gemma3-1B-it          configs/mmlu_gemma3_1b_it_config.json        data/mmlu_cais_validation_7subjects
MMLU_Gemma3-4B-it          configs/mmlu_gemma3_4b_it_config.json        data/mmlu_cais_validation_7subjects
HarmBench_Qwen3-8B         configs/harmbench_qwen3_8b_config.json       data/harmbench_walledai_standard
HarmBench_Qwen3-4B         configs/harmbench_qwen3_4b_config.json       data/harmbench_walledai_standard
HarmBench_Gemma3-1B-it     configs/harmbench_gemma3_1b_it_config.json   data/harmbench_walledai_standard
HarmBench_Gemma3-4B-it     configs/harmbench_gemma3_4b_it_config.json   data/harmbench_walledai_standard
'

MMLU_SUBJECTS_CSV="${MMLU_SUBJECTS_CSV:-international_law,logical_fallacies,moral_disputes,philosophy,professional_psychology,sociology,world_religions}"
MMLU_PREP_MODEL="${MMLU_PREP_MODEL:-Qwen/Qwen3-4B}"
MMLU_N_GROUPS="${MMLU_N_GROUPS:-209}"

dataset_is_prepped() {
  local cloze_dir="$1"
  [[ -f "$cloze_dir/dataset_dict.json" ]]
}

ensure_ambigqa_real_dataset() {
  local cloze_dir="$1"
  if dataset_is_prepped "$cloze_dir" && uv run python - "$cloze_dir" <<'PY'
import sys
from datasets import load_from_disk

path = sys.argv[1]
try:
    ds = load_from_disk(path)
    split = ds["train"] if "train" in ds else ds
    n_rows = len(split)
    rows = [split[i] for i in range(min(n_rows, 10))]
except Exception:
    sys.exit(1)

if n_rows < 200:
    sys.exit(2)

for row in rows:
    values = [
        str(row.get("id", "")),
        str(row.get("source_dataset", "")),
        str(row.get("subject", "")),
        str(row.get("category", "")),
    ]
    if any("ambigqa_smoke" in value for value in values):
        sys.exit(3)
sys.exit(0)
PY
  then
    return 0
  fi

  cat >&2 <<EOF
AmbigQA source at $cloze_dir is missing, too small, or still contains the one-row smoke dataset.
Restore the full AmbigQA DatasetDict used by AmbigQA_Qwen3-4B / AmbigQA_Qwen3-8B before running scale experiments.
EOF
  exit 1
}

ensure_mmlu_prepped() {
  local cloze_dir="$1"
  if dataset_is_prepped "$cloze_dir" && uv run python - "$cloze_dir" "$MMLU_SUBJECTS_CSV" "$MMLU_N_GROUPS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
subjects = [part.strip() for part in sys.argv[2].split(",") if part.strip()]
min_rows = int(sys.argv[3])
try:
    metadata = json.loads((path / "metadata.json").read_text())
except Exception:
    sys.exit(1)

num_rows = int(metadata.get("num_rows", -1))
ok = (
    metadata.get("dataset_id") == "cais/mmlu"
    and metadata.get("split") == "validation"
    and metadata.get("subjects") == subjects
    and num_rows >= min_rows
)
sys.exit(0 if ok else 1)
PY
  then
    return 0
  fi

  echo ">>> Preparing MMLU DatasetDict at $cloze_dir"
  uv run python scripts/prepare_mmlu_questions.py \
    --subjects "$MMLU_SUBJECTS_CSV" \
    --model "$MMLU_PREP_MODEL" \
    --raw-save-dir "$cloze_dir" \
    --output "${TMPDIR:-/tmp}/scale_mmlu_preview.json"
}

ensure_harmbench_prepped() {
  local cloze_dir="$1"
  if dataset_is_prepped "$cloze_dir"; then
    return 0
  fi
  echo ">>> Preparing HarmBench DatasetDict at $cloze_dir"
  uv run python scripts/prepare_harmbench_questions.py \
    --save-dir "$cloze_dir"
}

usage() {
  cat <<'EOF'
Usage: scripts/run_scale_experiments.sh [options]

Options:
  --only TAG1,TAG2,...  Run only selected tags.
  --resume STAGE         Pass --resume STAGE through to run_pipeline.sh.
  --skip-existing        Pass --skip-existing through to run_pipeline.sh.
  --pipeline-arg ARG     Pass one extra literal arg through to run_pipeline.sh.
  --list                List available tags and exit.
  -h, --help            Show this help.

Examples:
  scripts/run_scale_experiments.sh --list
  scripts/run_scale_experiments.sh --only MMLU_Qwen3-4B,HarmBench_Gemma3-4B-it
EOF
}

ONLY=""
LIST=false
PIPELINE_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --resume) PIPELINE_ARGS+=(--resume "$2"); shift 2 ;;
    --skip-existing|--skip_existing) PIPELINE_ARGS+=(--skip-existing); shift ;;
    --pipeline-arg) PIPELINE_ARGS+=("$2"); shift 2 ;;
    --list) LIST=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# Parse combos
declare -a TAGS CONFIGS CLOZE_DIRS
while IFS= read -r line; do
  line="${line## }"
  line="${line%% }"
  [[ -z "$line" ]] && continue
  read -r tag config cloze_dir <<<"$line"
  TAGS+=("$tag")
  CONFIGS+=("$config")
  CLOZE_DIRS+=("$cloze_dir")
done <<<"$COMBOS_RAW"

if $LIST; then
  echo "Available combos:"
  for i in "${!TAGS[@]}"; do
    printf "  %s\n    config: %s\n    cloze_dir: %s\n" "${TAGS[$i]}" "${CONFIGS[$i]}" "${CLOZE_DIRS[$i]}"
  done
  exit 0
fi

# Build the set of tags to run
declare -A WANT
if [[ -n "$ONLY" ]]; then
  IFS=',' read -ra parts <<<"$ONLY"
  for t in "${parts[@]}"; do WANT["$t"]=1; done
else
  for t in "${TAGS[@]}"; do WANT["$t"]=1; done
fi

for i in "${!TAGS[@]}"; do
  tag="${TAGS[$i]}"
  config="${CONFIGS[$i]}"
  cloze_dir="${CLOZE_DIRS[$i]}"
  [[ -z "${WANT[$tag]:-}" ]] && continue

  if [[ ! -f "$config" ]]; then
    echo "Missing config: $config" >&2
    exit 1
  fi

  echo "==> $tag  config=$config  cloze_dir=$cloze_dir"

  case "$cloze_dir" in
    data/cloze_llm_improved_split_ratio_0.1)
      ensure_ambigqa_real_dataset "$cloze_dir"
      ;;
    data/mmlu_cais_validation_7subjects)
      ensure_mmlu_prepped "$cloze_dir"
      ;;
    data/harmbench_walledai_standard)
      ensure_harmbench_prepped "$cloze_dir"
      ;;
    *)
      echo "No dataset preparation rule for $cloze_dir" >&2
      exit 1
      ;;
  esac

  out_dir="$ROOT_DIR/$tag"
  mkdir -p "$out_dir/results" "$out_dir/logs"

  CONFIG_FILE="$config" bash "$SCRIPT_DIR/run_pipeline.sh" --output_dir "$out_dir" "${PIPELINE_ARGS[@]}"
done

echo "All requested combos finished."
