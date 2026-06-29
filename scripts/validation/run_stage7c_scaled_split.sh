#!/usr/bin/env bash
# Stage 7c (RD steering) + K-means baseline, split outputs across gpu0/gpu1 results
#
# Runs steering on both GPUs in parallel, then baseline on both GPUs in parallel.
# This ensures only one model per GPU at a time.

set -euo pipefail

export CONFIG_FILE="configs/beta_gamma_scaled_config.json"

BASE_OUTPUT="beta_gamma_scaled_results"
GPU0_OUTPUT="$BASE_OUTPUT/gpu0"
GPU1_OUTPUT="$BASE_OUTPUT/gpu1"

# Log files
GPU0_STEERING_LOG="$BASE_OUTPUT/gpu0_stage7c_steering.log"
GPU1_STEERING_LOG="$BASE_OUTPUT/gpu1_stage7c_steering.log"
GPU0_BASELINE_LOG="$BASE_OUTPUT/gpu0_stage7c_baseline.log"
GPU1_BASELINE_LOG="$BASE_OUTPUT/gpu1_stage7c_baseline.log"

# Parse arguments
RUN_STEERING=true
RUN_BASELINE=true
for arg in "$@"; do
  case $arg in
    --steering-only)
      RUN_BASELINE=false
      ;;
    --baseline-only)
      RUN_STEERING=false
      ;;
  esac
done

# ============================================================
# Phase 1: RD Steering (both GPUs in parallel)
# ============================================================
if $RUN_STEERING; then
  echo "=== Phase 1: RD Steering ==="
  
  echo "Launching Stage 7c (RD steering) on GPU 0..."
  (
    export CUDA_VISIBLE_DEVICES=0
    bash run_pipeline.sh --output_dir "$GPU0_OUTPUT" --stages 7c --quiet >"$GPU0_STEERING_LOG" 2>&1
  ) &
  
  echo "Launching Stage 7c (RD steering) on GPU 1..."
  (
    export CUDA_VISIBLE_DEVICES=1
    bash run_pipeline.sh --output_dir "$GPU1_OUTPUT" --stages 7c --quiet >"$GPU1_STEERING_LOG" 2>&1
  ) &
  
  echo "Waiting for steering to complete..."
  wait
  echo "Steering complete."
  echo ""
fi

# ============================================================
# Phase 2: K-means Baseline (both GPUs in parallel)
# ============================================================
if $RUN_BASELINE; then
  echo "=== Phase 2: K-means Baseline ==="
  
  echo "Launching K-means baseline on GPU 0..."
  (
    export CUDA_VISIBLE_DEVICES=0
    python 7_validation/7c_baseline_kmeans.py \
      --samples-dir "$GPU0_OUTPUT/results/2_branch_sampling" \
      --embeddings-dir "$GPU0_OUTPUT/results/4_feature_extraction/embeddings" \
      --attribution-graphs-dir "$GPU0_OUTPUT/results/3_attribution_graphs" \
      --clustering-dir "$GPU0_OUTPUT/results/5_clustering" \
      --output-dir "$GPU0_OUTPUT/results/7_validation/7c_baseline_kmeans" \
      --config "$CONFIG_FILE" \
      --cross-prefix-batching \
      --prefix-batch-size 16 \
      --quiet >"$GPU0_BASELINE_LOG" 2>&1
  ) &
  
  echo "Launching K-means baseline on GPU 1..."
  (
    export CUDA_VISIBLE_DEVICES=1
    python 7_validation/7c_baseline_kmeans.py \
      --samples-dir "$GPU1_OUTPUT/results/2_branch_sampling" \
      --embeddings-dir "$GPU1_OUTPUT/results/4_feature_extraction/embeddings" \
      --attribution-graphs-dir "$GPU1_OUTPUT/results/3_attribution_graphs" \
      --clustering-dir "$GPU1_OUTPUT/results/5_clustering" \
      --output-dir "$GPU1_OUTPUT/results/7_validation/7c_baseline_kmeans" \
      --config "$CONFIG_FILE" \
      --cross-prefix-batching \
      --prefix-batch-size 16 \
      --quiet >"$GPU1_BASELINE_LOG" 2>&1
  ) &
  
  echo "Waiting for baseline to complete..."
  wait
  echo "Baseline complete."
  echo ""
fi

echo ""
echo "=== Stage 7c complete ==="
echo ""
echo "Logs:"
if $RUN_STEERING; then
  echo "  RD Steering (GPU 0): $GPU0_STEERING_LOG"
  echo "  RD Steering (GPU 1): $GPU1_STEERING_LOG"
fi
if $RUN_BASELINE; then
  echo "  K-means Baseline (GPU 0): $GPU0_BASELINE_LOG"
  echo "  K-means Baseline (GPU 1): $GPU1_BASELINE_LOG"
fi

