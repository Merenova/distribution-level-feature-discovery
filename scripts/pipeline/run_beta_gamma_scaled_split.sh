#!/usr/bin/env bash
# Scaled beta/gamma run split across two GPUs (0 and 1)

set -euo pipefail

export CONFIG_FILE="configs/beta_gamma_scaled_config.json"

BASE_OUTPUT="beta_gamma_scaled_results"
STAGE1_OUTPUT="$BASE_OUTPUT/stage1"
GPU0_OUTPUT="$BASE_OUTPUT/gpu0"
GPU1_OUTPUT="$BASE_OUTPUT/gpu1"
GPU0_LOG="$BASE_OUTPUT/gpu0_pipeline.log"
GPU1_LOG="$BASE_OUTPUT/gpu1_pipeline.log"

mkdir -p "$BASE_OUTPUT"

echo "Running Stage 1 once to generate test_clozes..."
bash run_pipeline.sh --output_dir "$STAGE1_OUTPUT" --stages 1 --quiet

echo "Splitting test_clozes 50/50 by group_id..."
python - <<'PY'
import json
from pathlib import Path

base = Path("beta_gamma_scaled_results")
stage1 = base / "stage1" / "results" / "test_clozes.json"
out0 = base / "gpu0" / "results"
out1 = base / "gpu1" / "results"

out0.mkdir(parents=True, exist_ok=True)
out1.mkdir(parents=True, exist_ok=True)

data = json.loads(stage1.read_text())
clozes = data["clozes"]

# Preserve group integrity and order
groups = []
group_map = {}
for cloze in clozes:
    gid = cloze.get("group_id")
    if gid not in group_map:
        group_map[gid] = []
        groups.append(gid)
    group_map[gid].append(cloze)

mid = len(groups) // 2
groups0 = groups[:mid]
groups1 = groups[mid:]

def build_payload(groups_subset):
    subset = []
    for gid in groups_subset:
        subset.extend(group_map[gid])
    meta = dict(data.get("metadata", {}))
    meta["selected_groups"] = len(groups_subset)
    meta["total_samples"] = len(subset)
    return {"metadata": meta, "clozes": subset}

(out0 / "test_clozes.json").write_text(json.dumps(build_payload(groups0), indent=2))
(out1 / "test_clozes.json").write_text(json.dumps(build_payload(groups1), indent=2))
print(f"Wrote {len(groups0)} groups to {out0 / 'test_clozes.json'}")
print(f"Wrote {len(groups1)} groups to {out1 / 'test_clozes.json'}")
PY

echo "Launching GPU 0 run..."
(
  export CUDA_VISIBLE_DEVICES=0
  bash run_pipeline.sh --output_dir "$GPU0_OUTPUT" --stages 2,3,4a,5,6,7c --quiet >"$GPU0_LOG" 2>&1
) &

echo "Launching GPU 1 run..."
(
  export CUDA_VISIBLE_DEVICES=1
  bash run_pipeline.sh --output_dir "$GPU1_OUTPUT" --stages 2,3,4a,5,6,7c --quiet >"$GPU1_LOG" 2>&1
) &

wait

echo ""
echo "Done. Results in:"
echo "  $GPU0_OUTPUT/results"
echo "  $GPU1_OUTPUT/results"
echo "Logs:"
echo "  $GPU0_LOG"
echo "  $GPU1_LOG"

