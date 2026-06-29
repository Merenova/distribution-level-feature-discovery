#!/usr/bin/env bash
set -euo pipefail

ORIGINAL_ROOT="/home/hyunjin/latent_planning"
CLEAN_ROOT="/home/hyunjin/latent_planning_paper_clean"
DATASET_DIR="/home/hyunjin/latent_planning/data/cloze_llm_improved_split_ratio_0.1"
CONFIG=""
WORKDIR=""
QUIET=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_integrity_compare.sh [options]

Options:
  --original-root PATH   Original repo root. Default: /home/hyunjin/latent_planning
  --clean-root PATH      Clean repo root. Default: /home/hyunjin/latent_planning_paper_clean
  --dataset-dir PATH     Local Hugging Face dataset dir. Default: /home/hyunjin/latent_planning/data/cloze_llm_improved_split_ratio_0.1
  --config PATH          Integrity config JSON. Default: <clean-root>/configs/presets/integrity_qwen3_4b_single.json
  --workdir PATH         Output workdir. Default: /tmp/latent_planning_integrity_<UTC timestamp>
  --quiet                Pass --quiet through to stage scripts where supported.
  --help                 Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --original-root)
      ORIGINAL_ROOT="$2"
      shift 2
      ;;
    --clean-root)
      CLEAN_ROOT="$2"
      shift 2
      ;;
    --dataset-dir)
      DATASET_DIR="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --workdir)
      WORKDIR="$2"
      shift 2
      ;;
    --quiet)
      QUIET=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${CONFIG}" ]]; then
  CONFIG="${CLEAN_ROOT}/configs/presets/integrity_qwen3_4b_single.json"
fi

if [[ -z "${WORKDIR}" ]]; then
  WORKDIR="/tmp/latent_planning_integrity_$(date -u +%Y%m%d_%H%M%S)"
fi

for path in "$ORIGINAL_ROOT" "$CLEAN_ROOT" "$DATASET_DIR" "$CONFIG"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path not found: $path" >&2
    exit 1
  fi
done

run_clean_py() {
  (cd "$CLEAN_ROOT" && uv run python "$@")
}

run_original_py() {
  (cd "$ORIGINAL_ROOT" && uv run python "$@")
}

QUIET_ARGS=()
if [[ "$QUIET" -eq 1 ]]; then
  QUIET_ARGS+=(--quiet)
fi

CLEAN_RESULTS="$WORKDIR/clean/results"
CLEAN_LOGS="$WORKDIR/clean/logs"
ORIGINAL_RESULTS="$WORKDIR/original/results"
ORIGINAL_LOGS="$WORKDIR/original/logs"
FIXTURES_DIR="$WORKDIR/fixtures"
REPORT_DIR="$WORKDIR/report"

mkdir -p \
  "$CLEAN_RESULTS/0_preprocess" \
  "$CLEAN_RESULTS/1_data_preparation" \
  "$CLEAN_RESULTS/2_branch_sampling" \
  "$CLEAN_RESULTS/3_attribution_graphs" \
  "$CLEAN_RESULTS/4_feature_extraction" \
  "$CLEAN_RESULTS/5_clustering" \
  "$CLEAN_RESULTS/6_semantic_graphs" \
  "$CLEAN_RESULTS/7_validation" \
  "$CLEAN_LOGS" \
  "$ORIGINAL_RESULTS/2_branch_sampling" \
  "$ORIGINAL_RESULTS/3_attribution_graphs" \
  "$ORIGINAL_RESULTS/4_feature_extraction" \
  "$ORIGINAL_RESULTS/5_clustering" \
  "$ORIGINAL_RESULTS/6_semantic_graphs" \
  "$ORIGINAL_RESULTS/7_validation" \
  "$ORIGINAL_LOGS" \
  "$FIXTURES_DIR" \
  "$REPORT_DIR"

RESOLVED_CONFIG="$WORKDIR/resolved_integrity_config.json"
run_clean_py -m utils.config "$CONFIG" --write "$RESOLVED_CONFIG" >/dev/null
CONFIG="$RESOLVED_CONFIG"

CONFIG_EXPORTS="$(
run_clean_py - "$CONFIG" <<'PY'
import json
import shlex
import sys

cfg = json.load(open(sys.argv[1]))
model = cfg.get("model", {})
sampling = cfg.get("sampling", {})
embedding = cfg.get("embedding", {})
attribution = cfg.get("attribution", {})
data = cfg.get("data", {})
stage1 = cfg.get("stage_1_data_prep", {})
clustering = cfg.get("clustering", {})
stage7 = cfg.get("stage_7c_steering", {})
sweeps = clustering.get("sweeps", {})
stage7_sweeps = stage7.get("sweeps", [])
first_stage7 = stage7_sweeps[0] if stage7_sweeps else {}
steering_method = first_stage7.get("steering_method", "sign")
hc_selection = (first_stage7.get("h_c_selections") or ["full"])[0]
top_b = (first_stage7.get("top_B") or [5])[0]
values = {
    "SEED": cfg.get("random_seed", 42),
    "MODEL": model.get("base_model", "Qwen/Qwen3-4B"),
    "TRANSCODER": model.get("transcoder", "mwhanna/qwen3-4b-transcoders"),
    "DTYPE": model.get("dtype", "bfloat16"),
    "DEVICE": model.get("device", "cuda"),
    "SPLIT": data.get("split", "train"),
    "N_GROUPS": stage1.get("n_groups", 1),
    "EMBEDDING_MODEL": embedding.get("model_name", "google/embeddinggemma-300m"),
    "EMBEDDING_BATCH_SIZE": embedding.get("batch_size", 16),
    "ATTR_BATCH_SIZE": attribution.get("batch_size", 64),
    "MAX_FEATURE_NODES": attribution.get("max_feature_nodes", 4096),
    "MAX_TOTAL_CONTINUATIONS": sampling.get("max_total_continuations", 64),
    "NUCLEUS_P": sampling.get("nucleus_p", 0.9),
    "TEMPERATURE": sampling.get("temperature", 1.0),
    "MAX_TOKENS": sampling.get("max_tokens", 20),
    "SAMPLING_BATCH_SIZE": sampling.get("batch_size", 16),
    "MAX_BATCHES": sampling.get("max_batches", 32),
    "GPU_MEMORY_UTILIZATION": cfg.get("vllm", {}).get("gpu_memory_utilization", 0.95),
    "TENSOR_PARALLEL_SIZE": cfg.get("vllm", {}).get("tensor_parallel_size", 1),
    "MAX_MODEL_LEN": cfg.get("vllm", {}).get("max_model_len", 96),
    "BETA": (sweeps.get("beta_values") or [0.75])[0],
    "GAMMA": (sweeps.get("gamma_values") or [0.5])[0],
    "K_CLAMP": clustering.get("K_clamp", clustering.get("K_max", 10)),
    "MAX_CLUSTER_SAMPLES": stage7.get("max_cluster_samples", 10),
    "MAX_STEERING_BATCH_SIZE": stage7.get("max_batch_size", 256),
    "STAGE7_RESULT_KEY": f"{steering_method}_{hc_selection}_B{top_b}",
}
for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"
eval "$CONFIG_EXPORTS"

echo "Workdir: $WORKDIR"
echo "Config:  $CONFIG"
echo "Model:   $MODEL"

run_clean_py 0_preprocess/prepare_ambigqa_questions.py \
  --dataset-dir "$DATASET_DIR" \
  --split "$SPLIT" \
  --output "$CLEAN_RESULTS/0_preprocess/ambigqa_question_groups.json" \
  --log-dir "$CLEAN_LOGS" \
  "${QUIET_ARGS[@]}"

run_clean_py 1_data_preparation/format_ambigqa_questions.py \
  --grouped-questions "$CLEAN_RESULTS/0_preprocess/ambigqa_question_groups.json" \
  --model "$MODEL" \
  --n-groups "$N_GROUPS" \
  --seed "$SEED" \
  --output-dir "$CLEAN_RESULTS/1_data_preparation" \
  --log-dir "$CLEAN_LOGS" \
  "${QUIET_ARGS[@]}"

run_original_py 1_data_preparation/select_test_clozes.py \
  --cloze-dir "$DATASET_DIR" \
  --split "$SPLIT" \
  --n-samples "$N_GROUPS" \
  --mode question \
  --model "$MODEL" \
  --output "$ORIGINAL_RESULTS/test_clozes.json" \
  --seed "$SEED" \
  "${QUIET_ARGS[@]}"

FIXTURE_EXPORTS="$(
run_clean_py - \
  "$CLEAN_RESULTS/1_data_preparation/prefixes.json" \
  "$CLEAN_RESULTS/1_data_preparation/prefix_metadata.json" \
  "$ORIGINAL_RESULTS/test_clozes.json" \
  "$FIXTURES_DIR" <<'PY'
import json
import shlex
import sys
from pathlib import Path

clean_prefixes = json.load(open(sys.argv[1]))
clean_metadata = json.load(open(sys.argv[2]))
original = json.load(open(sys.argv[3]))
fixtures_dir = Path(sys.argv[4])
fixtures_dir.mkdir(parents=True, exist_ok=True)

if not clean_prefixes:
    raise SystemExit("No clean Stage 1 prefixes produced")
if not original.get("clozes"):
    raise SystemExit("No original Stage 1 clozes produced")

selected_prefix = clean_prefixes[0]
selected_prefix_id = selected_prefix["prefix_id"]

clean_subset_path = fixtures_dir / "clean_prefixes_first.json"
with clean_subset_path.open("w") as f:
    json.dump([selected_prefix], f, indent=2)
    f.write("\n")

original_subset = {
    "metadata": dict(original.get("metadata", {})),
    "clozes": [original["clozes"][0]],
}
original_subset_path = fixtures_dir / "original_test_clozes_first.json"
with original_subset_path.open("w") as f:
    json.dump(original_subset, f, indent=2)
    f.write("\n")

print(f"PREFIX_ID={shlex.quote(selected_prefix_id)}")
print(f"CLEAN_PREFIX_FIXTURE={shlex.quote(str(clean_subset_path))}")
print(f"ORIGINAL_PREFIX_FIXTURE={shlex.quote(str(original_subset_path))}")
PY
)"
eval "$FIXTURE_EXPORTS"

run_original_py 2_branch_sampling/sample_branches.py \
  --test-clozes "$ORIGINAL_PREFIX_FIXTURE" \
  --model "$MODEL" \
  --max-total-continuations "$MAX_TOTAL_CONTINUATIONS" \
  --nucleus-p "$NUCLEUS_P" \
  --temperature "$TEMPERATURE" \
  --max-tokens "$MAX_TOKENS" \
  --batch-size "$SAMPLING_BATCH_SIZE" \
  --max-batches "$MAX_BATCHES" \
  --output-dir "$ORIGINAL_RESULTS/2_branch_sampling" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-complete 1 \
  "${QUIET_ARGS[@]}"

run_clean_py - "$ORIGINAL_RESULTS" "$PREFIX_ID" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

results_dir = Path(sys.argv[1])
prefix_id = sys.argv[2]
payload = {
    "stage": "stage2",
    "timestamp": datetime.utcnow().isoformat(),
    "completed": [prefix_id],
    "failed": [],
    "skipped": [],
    "n_completed": 1,
    "n_failed": 0,
    "n_skipped": 0,
    "errors": {},
}
path = results_dir / "manifest_stage2.json"
path.write_text(json.dumps(payload, indent=2) + "\n")
PY

run_clean_py 2_branch_sampling/sample_branches.py \
  --prefixes-file "$CLEAN_PREFIX_FIXTURE" \
  --model "$MODEL" \
  --max-total-continuations "$MAX_TOTAL_CONTINUATIONS" \
  --nucleus-p "$NUCLEUS_P" \
  --temperature "$TEMPERATURE" \
  --max-tokens "$MAX_TOKENS" \
  --batch-size "$SAMPLING_BATCH_SIZE" \
  --max-batches "$MAX_BATCHES" \
  --output-dir "$CLEAN_RESULTS/2_branch_sampling" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  "${QUIET_ARGS[@]}"

CANONICAL_BRANCHES_DIR="$ORIGINAL_RESULTS/2_branch_sampling"

run_original_py 3_attribution_graphs/compute_continuation_attribution.py \
  --branches-dir "$CANONICAL_BRANCHES_DIR" \
  --model "$MODEL" \
  --transcoder "$TRANSCODER" \
  --dtype "$DTYPE" \
  --max-feature-nodes "$MAX_FEATURE_NODES" \
  --batch-size "$ATTR_BATCH_SIZE" \
  --output-dir "$ORIGINAL_RESULTS/3_attribution_graphs" \
  "${QUIET_ARGS[@]}"

run_clean_py 3_attribution_graphs/compute_continuation_attribution.py \
  --branches-dir "$CANONICAL_BRANCHES_DIR" \
  --model "$MODEL" \
  --transcoder "$TRANSCODER" \
  --dtype "$DTYPE" \
  --max-feature-nodes "$MAX_FEATURE_NODES" \
  --batch-size "$ATTR_BATCH_SIZE" \
  --output-dir "$CLEAN_RESULTS/3_attribution_graphs" \
  "${QUIET_ARGS[@]}"

run_original_py 4_feature_extraction/compute_embeddings.py \
  --samples-dir "$CANONICAL_BRANCHES_DIR" \
  --embedding-model "$EMBEDDING_MODEL" \
  --batch-size "$EMBEDDING_BATCH_SIZE" \
  --device "$DEVICE" \
  --output-dir "$ORIGINAL_RESULTS/4_feature_extraction" \
  "${QUIET_ARGS[@]}"

run_clean_py 4_feature_extraction/compute_embeddings.py \
  --samples-dir "$CANONICAL_BRANCHES_DIR" \
  --embedding-model "$EMBEDDING_MODEL" \
  --batch-size "$EMBEDDING_BATCH_SIZE" \
  --device "$DEVICE" \
  --output-dir "$CLEAN_RESULTS/4_feature_extraction" \
  "${QUIET_ARGS[@]}"

run_original_py 5_gaussian_clustering/cluster.py \
  --embeddings-dir "$ORIGINAL_RESULTS/4_feature_extraction" \
  --attribution-graphs-dir "$ORIGINAL_RESULTS/3_attribution_graphs" \
  --samples-dir "$CANONICAL_BRANCHES_DIR" \
  --output-dir "$ORIGINAL_RESULTS/5_clustering" \
  --config "$CONFIG" \
  --n-workers 1 \
  --log-dir "$ORIGINAL_LOGS" \
  "${QUIET_ARGS[@]}"

run_clean_py 5_gaussian_clustering/cluster.py \
  --embeddings-dir "$CLEAN_RESULTS/4_feature_extraction" \
  --attribution-graphs-dir "$CLEAN_RESULTS/3_attribution_graphs" \
  --samples-dir "$CANONICAL_BRANCHES_DIR" \
  --output-dir "$CLEAN_RESULTS/5_clustering" \
  --config "$CONFIG" \
  --n-workers 1 \
  --log-dir "$CLEAN_LOGS" \
  "${QUIET_ARGS[@]}"

run_original_py 6_semantic_graphs/extract_graphs.py \
  --clustering-dir "$ORIGINAL_RESULTS/5_clustering" \
  --samples-dir "$CANONICAL_BRANCHES_DIR" \
  --attribution-graphs-dir "$ORIGINAL_RESULTS/3_attribution_graphs" \
  --output-dir "$ORIGINAL_RESULTS/6_semantic_graphs" \
  --log-dir "$ORIGINAL_LOGS" \
  "${QUIET_ARGS[@]}"

run_clean_py 6_semantic_graphs/extract_graphs.py \
  --clustering-dir "$CLEAN_RESULTS/5_clustering" \
  --samples-dir "$CANONICAL_BRANCHES_DIR" \
  --attribution-graphs-dir "$CLEAN_RESULTS/3_attribution_graphs" \
  --output-dir "$CLEAN_RESULTS/6_semantic_graphs" \
  --log-dir "$CLEAN_LOGS" \
  "${QUIET_ARGS[@]}"

run_original_py 7_validation/7c_baseline_combined_medoid.py \
  --samples-dir "$CANONICAL_BRANCHES_DIR" \
  --attribution-graphs-dir "$ORIGINAL_RESULTS/3_attribution_graphs" \
  --clustering-dir "$ORIGINAL_RESULTS/5_clustering" \
  --embeddings-dir "$ORIGINAL_RESULTS/4_feature_extraction" \
  --output-dir "$ORIGINAL_RESULTS/7_validation" \
  --config "$CONFIG" \
  --model "$MODEL" \
  --transcoder "$TRANSCODER" \
  --max-cluster-samples "$MAX_CLUSTER_SAMPLES" \
  --max-batch-size "$MAX_STEERING_BATCH_SIZE" \
  --K-clamp "$K_CLAMP" \
  --beta-values "$BETA" \
  --gamma-values "$GAMMA" \
  --log-dir "$ORIGINAL_LOGS" \
  "${QUIET_ARGS[@]}"

run_clean_py 7_validation/rd_medoid.py \
  --samples-dir "$CANONICAL_BRANCHES_DIR" \
  --attribution-graphs-dir "$CLEAN_RESULTS/3_attribution_graphs" \
  --clustering-dir "$CLEAN_RESULTS/5_clustering" \
  --embeddings-dir "$CLEAN_RESULTS/4_feature_extraction" \
  --output-dir "$CLEAN_RESULTS/7_validation" \
  --config "$CONFIG" \
  --model "$MODEL" \
  --transcoder "$TRANSCODER" \
  --max-cluster-samples "$MAX_CLUSTER_SAMPLES" \
  --max-batch-size "$MAX_STEERING_BATCH_SIZE" \
  --K-clamp "$K_CLAMP" \
  --beta-values "$BETA" \
  --gamma-values "$GAMMA" \
  --log-dir "$CLEAN_LOGS" \
  "${QUIET_ARGS[@]}"

run_clean_py scripts/compare_integrity_artifacts.py \
  --clean-stage1-prefixes "$CLEAN_RESULTS/1_data_preparation/prefixes.json" \
  --clean-stage1-metadata "$CLEAN_RESULTS/1_data_preparation/prefix_metadata.json" \
  --original-stage1 "$ORIGINAL_RESULTS/test_clozes.json" \
  --clean-stage2-dir "$CLEAN_RESULTS/2_branch_sampling" \
  --original-stage2-dir "$ORIGINAL_RESULTS/2_branch_sampling" \
  --clean-stage3-dir "$CLEAN_RESULTS/3_attribution_graphs" \
  --original-stage3-dir "$ORIGINAL_RESULTS/3_attribution_graphs" \
  --clean-stage4-dir "$CLEAN_RESULTS/4_feature_extraction" \
  --original-stage4-dir "$ORIGINAL_RESULTS/4_feature_extraction" \
  --clean-stage5-dir "$CLEAN_RESULTS/5_clustering" \
  --original-stage5-dir "$ORIGINAL_RESULTS/5_clustering" \
  --clean-stage6-file "$CLEAN_RESULTS/6_semantic_graphs/${PREFIX_ID}_beta${BETA}_gamma${GAMMA}_semantic_graphs.pt" \
  --original-stage6-file "$ORIGINAL_RESULTS/6_semantic_graphs/${PREFIX_ID}_beta${BETA}_gamma${GAMMA}_semantic_graphs.pt" \
  --clean-stage7-file "$CLEAN_RESULTS/7_validation/rd/${PREFIX_ID}_sweep_results.json" \
  --original-stage7-file "$ORIGINAL_RESULTS/7_validation/H4a_combined_medoid/${PREFIX_ID}_sweep_results.json" \
  --prefix-id "$PREFIX_ID" \
  --beta "$BETA" \
  --gamma "$GAMMA" \
  --stage7-result-key "$STAGE7_RESULT_KEY" \
  --output-json "$REPORT_DIR/integrity_summary.json" \
  --output-md "$REPORT_DIR/integrity_report.md"

echo "Integrity summary: $REPORT_DIR/integrity_summary.json"
echo "Integrity report:  $REPORT_DIR/integrity_report.md"
