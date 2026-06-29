#!/usr/bin/env bash
# Sweep script for Qwen3-8B experiments
# Sweeps over: pooling (span_mode fixed to full)

set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

BASE_CONFIG="configs/Qwen3_8B_config.json"
BASE_OUTPUT_DIR="Qwen3_8B_results"
LOG_DIR="$BASE_OUTPUT_DIR/logs"

POOLINGS=("sum")

mkdir -p "$BASE_OUTPUT_DIR/configs" "$LOG_DIR"

TOTAL_SWEEPS=$((${#POOLINGS[@]}))

echo "========================================"
echo "QWEN3-8B PARAMETER SWEEP"
echo "========================================"
echo "Grid: ${#POOLINGS[@]} poolings = $TOTAL_SWEEPS configs"
echo "========================================"
echo ""

# ========================================
# SHARED STAGES: 1, 2, 3, 4a
# ========================================
SHARED_CONFIG="${BASE_OUTPUT_DIR}/configs/config_shared.json"

uv run python -c "
import json
with open('$BASE_CONFIG') as f:
    config = json.load(f)
config['attribution']['span_mode'] = 'full'
config['attribution']['store_all'] = True
with open('$SHARED_CONFIG', 'w') as f:
    json.dump(config, f, indent=2)
" 2>/dev/null

export CONFIG_FILE="$SHARED_CONFIG"

if [ ! -d "$BASE_OUTPUT_DIR/results/2_branch_sampling" ]; then
  echo "Stages 1-2 (branch sampling)... "
  bash run_pipeline.sh --output_dir "$BASE_OUTPUT_DIR" --stages 1,2 --quiet
fi

if [ ! -d "$BASE_OUTPUT_DIR/results/3_attribution_graphs" ]; then
  echo "Stage 3 (attribution, full span)... "
  bash run_pipeline.sh --output_dir "$BASE_OUTPUT_DIR" --only 3 --quiet
fi

if [ ! -d "$BASE_OUTPUT_DIR/results/4_feature_extraction/embeddings" ]; then
  echo "Stage 4a (embeddings)... "
  bash run_pipeline.sh --output_dir "$BASE_OUTPUT_DIR" --stages 4a --quiet
fi

echo ""
echo "Shared stages complete. Starting sweep..."
echo ""

# ========================================
# SWEEP: Stages 5, 6, 7
# ========================================
SWEEP_SUMMARY="$BASE_OUTPUT_DIR/sweep_summary.json"
echo '{"sweep_configs": [], "completed": []}' > "$SWEEP_SUMMARY"

CURRENT=0
START_TIME=$(date +%s)

for pooling in "${POOLINGS[@]}"; do
  CURRENT=$((CURRENT + 1))
  CONFIG_NAME="pool_${pooling}"
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/sweep_${CONFIG_NAME}"
  TEMP_CONFIG="${BASE_OUTPUT_DIR}/configs/config_${CONFIG_NAME}.json"

  echo "[$CURRENT/$TOTAL_SWEEPS] $CONFIG_NAME"

  mkdir -p "$OUTPUT_DIR/results"

  # Generate config
  uv run python -c "
import json
with open('$BASE_CONFIG') as f:
    config = json.load(f)
config['attribution']['span_mode'] = 'full'
config['attribution']['store_all'] = True
config['clustering']['pooling'] = '$pooling'
with open('$TEMP_CONFIG', 'w') as f:
    json.dump(config, f, indent=2)
" 2>/dev/null

  # Link shared results
  for src in test_clozes.json 2_branch_sampling 3_attribution_graphs 4_feature_extraction; do
    ln -sf "$(realpath $BASE_OUTPUT_DIR/results/$src)" "$OUTPUT_DIR/results/$src" 2>/dev/null || \
      cp -r "$BASE_OUTPUT_DIR/results/$src" "$OUTPUT_DIR/results/$src"
  done

  # Run stages 5, 6, 7c only if missing
  export CONFIG_FILE="$TEMP_CONFIG"

  if [ ! -d "$OUTPUT_DIR/results/5_clustering" ]; then
    echo "Running stage 5 (clustering)..."
    bash run_pipeline.sh --output_dir "$OUTPUT_DIR" --only 5 --quiet
  else
    echo "Skipping stage 5 (clustering) - already exists."
  fi

  if [ ! -d "$OUTPUT_DIR/results/6_semantic_graphs" ]; then
    echo "Running stage 6 (semantic graphs)..."
    bash run_pipeline.sh --output_dir "$OUTPUT_DIR" --only 6 --quiet
  else
    echo "Skipping stage 6 (semantic graphs) - already exists."
  fi

  if [ ! -d "$OUTPUT_DIR/results/7_validation/7c_steering" ]; then
    echo "Running stage 7c (steering validation)..."
    bash run_pipeline.sh --output_dir "$OUTPUT_DIR" --only 7c --quiet
  else
    echo "Skipping stage 7c (steering validation) - already exists."
  fi

  # Update summary
  uv run python -c "
import json
with open('$SWEEP_SUMMARY') as f:
    summary = json.load(f)
summary['sweep_configs'].append({
    'name': '$CONFIG_NAME',
    'pooling': '$pooling',
    'output_dir': '$OUTPUT_DIR'
})
summary['completed'].append('$CONFIG_NAME')
with open('$SWEEP_SUMMARY', 'w') as f:
    json.dump(summary, f, indent=2)
" 2>/dev/null

done

echo ""
echo ""
echo "========================================"
echo "SWEEP COMPLETE!"
echo "========================================"
echo "Results: $BASE_OUTPUT_DIR/sweep_*/"
echo "Logs: $LOG_DIR/"
echo "Summary: $SWEEP_SUMMARY"
