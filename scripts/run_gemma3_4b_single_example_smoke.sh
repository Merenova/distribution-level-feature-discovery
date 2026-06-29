#!/usr/bin/env bash
# Run Gemma-3 4B-it single-example smoke pipelines for the extended datasets.
#
# These configs keep one selected Stage 1 group while sampling many rollouts
# from that single prefix so clustering and steering still exercise K > 1 paths.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# tag                         config                                                        cloze_dir
COMBOS_RAW='
AmbigQA_Gemma3-4B-it_single   configs/smoke_ambigqa_gemma3_4b_it_config.json     data/cloze_llm_improved_split_ratio_0.1
MMLU_Gemma3-4B-it_single      configs/smoke_mmlu_gemma3_4b_it_config.json        data/mmlu_cais_validation_7subjects
HarmBench_Gemma3-4B-it_single configs/smoke_harmbench_gemma3_4b_it_config.json   data/harmbench_walledai_standard
'

MODEL_ID="google/gemma-3-4b-it"
MMLU_SUBJECTS_CSV="${MMLU_SUBJECTS_CSV:-international_law,logical_fallacies,moral_disputes,philosophy,professional_psychology,sociology,world_religions}"

dataset_is_prepped() {
  local cloze_dir="$1"
  [[ -f "$cloze_dir/dataset_dict.json" ]]
}

ensure_ambigqa_smoke_prepped() {
  local cloze_dir="$1"
  if dataset_is_prepped "$cloze_dir" && ! uv run python - "$cloze_dir" <<'PY'
import sys
from datasets import load_from_disk

path = sys.argv[1]
desired_question = "Give one unusual but plausible reason someone might postpone a long-planned trip. Answer in one or two sentences, and choose a different reason each time."
try:
    ds = load_from_disk(path)
    train = ds["train"] if "train" in ds else ds
    row = train[0] if len(train) == 1 else None
except Exception:
    row = None

is_old_smoke = (
    row is not None
    and str(row.get("id", "")) == "ambigqa_smoke_0000"
    and str(row.get("question", "")) != desired_question
)
sys.exit(0 if is_old_smoke else 1)
PY
  then
    return 0
  fi
  echo ">>> Preparing one-row AmbigQA smoke DatasetDict at $cloze_dir"
  uv run python - <<'PY'
from pathlib import Path
import shutil

from datasets import Dataset, DatasetDict

question = "Give one unusual but plausible reason someone might postpone a long-planned trip. Answer in one or two sentences, and choose a different reason each time."
rows = [
    {
        "id": "ambigqa_smoke_0000",
        "question": question,
        "prompt": question,
        "answer_text": "open-ended",
        "target": "open-ended",
        "subject": "ambigqa_smoke",
        "category": "ambiguous_question_answering",
        "source_dataset": "synthetic_ambigqa_smoke",
        "source_split": "train",
    }
]
out = Path("data/cloze_llm_improved_split_ratio_0.1")
if out.exists():
    shutil.rmtree(out)
out.parent.mkdir(parents=True, exist_ok=True)
DatasetDict({"train": Dataset.from_list(rows)}).save_to_disk(str(out))
PY
}

ensure_mmlu_prepped() {
  local cloze_dir="$1"
  if dataset_is_prepped "$cloze_dir" && uv run python - "$cloze_dir" "$MMLU_SUBJECTS_CSV" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
subjects = [part.strip() for part in sys.argv[2].split(",") if part.strip()]
try:
    metadata = json.loads((path / "metadata.json").read_text())
except Exception:
    sys.exit(1)

ok = (
    metadata.get("dataset_id") == "cais/mmlu"
    and metadata.get("split") == "validation"
    and metadata.get("subjects") == subjects
)
sys.exit(0 if ok else 1)
PY
  then
    return 0
  fi
  echo ">>> Preparing MMLU DatasetDict at $cloze_dir"
  uv run python scripts/prepare_mmlu_questions.py \
    --subjects "$MMLU_SUBJECTS_CSV" \
    --model "$MODEL_ID" \
    --raw-save-dir "$cloze_dir" \
    --output "${TMPDIR:-/tmp}/gemma3_4b_single_example_mmlu_preview.json" \
    --skip-existing
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
Usage: scripts/run_gemma3_4b_single_example_smoke.sh [options] [-- PIPELINE_ARGS...]

Options:
  --only TAG1,TAG2,...  Run only selected tags.
  --output-root DIR     Output root (default: smoke_single_example_gemma3_4b).
  --list                List available tags and exit.
  -h, --help            Show this help.

Any args after -- are forwarded to scripts/run_pipeline.sh.
Examples:
  scripts/run_gemma3_4b_single_example_smoke.sh --list
  scripts/run_gemma3_4b_single_example_smoke.sh --only AmbigQA_Gemma3-4B-it_single
  scripts/run_gemma3_4b_single_example_smoke.sh --only MMLU_Gemma3-4B-it_single -- --stages 1,2,3,5,7c
EOF
}

ONLY=""
LIST=false
OUTPUT_ROOT="$ROOT_DIR/smoke_single_example_gemma3_4b"
declare -a PIPELINE_ARGS

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      ONLY="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --list)
      LIST=true
      shift
      ;;
    --)
      shift
      PIPELINE_ARGS=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

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
  echo "Available Gemma-3 4B-it single-example smoke combos:"
  for i in "${!TAGS[@]}"; do
    printf "  %s\n    config: %s\n    cloze_dir: %s\n" "${TAGS[$i]}" "${CONFIGS[$i]}" "${CLOZE_DIRS[$i]}"
  done
  exit 0
fi

declare -A WANT
if [[ -n "$ONLY" ]]; then
  IFS=',' read -ra parts <<<"$ONLY"
  for tag in "${parts[@]}"; do
    WANT["$tag"]=1
  done
else
  for tag in "${TAGS[@]}"; do
    WANT["$tag"]=1
  done
fi

mkdir -p "$OUTPUT_ROOT"

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
      ensure_ambigqa_smoke_prepped "$cloze_dir"
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

  out_dir="$OUTPUT_ROOT/$tag"
  mkdir -p "$out_dir/results" "$out_dir/logs"

  CONFIG_FILE="$config" bash "$SCRIPT_DIR/run_pipeline.sh" \
    --output_dir "$out_dir" \
    "${PIPELINE_ARGS[@]}"
done

echo "All requested Gemma-3 4B-it single-example smoke runs finished."
