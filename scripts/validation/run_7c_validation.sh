#!/bin/bash
# Run 7c Steering Validation
# Usage:
#   ./scripts/run_7c_validation.sh [baseline|steering|both] [gpu_id]
#
# Examples:
#   ./scripts/run_7c_validation.sh baseline 0    # Run baseline kmeans on GPU 0
#   ./scripts/run_7c_validation.sh steering 1    # Run steering hypotheses on GPU 1
#   ./scripts/run_7c_validation.sh both 0        # Run both on GPU 0

set -e

# Parse arguments
MODE="${1:-both}"
GPU_ID="${2:-0}"

# Set CUDA device
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# Paths (adjust these for your setup)
RESULTS_BASE="beta_gamma_scaled_results/gpu${GPU_ID}/results"
SAMPLES_DIR="${RESULTS_BASE}/2_branch_sampling"
EMBEDDINGS_DIR="${RESULTS_BASE}/4_feature_extraction/embeddings"
CLUSTERING_DIR="${RESULTS_BASE}/5_clustering"
ATTRIBUTION_DIR="${RESULTS_BASE}/3_attribution_graphs"
CONFIG="configs/beta_gamma_scaled_config.json"

# Output directories
BASELINE_OUTPUT="${RESULTS_BASE}/7_validation/7c_baseline_kmeans"
STEERING_OUTPUT="${RESULTS_BASE}/7_validation/7c_steering"
LOG_DIR="${RESULTS_BASE}/7_validation/logs"

# Create directories
mkdir -p "${BASELINE_OUTPUT}/H4a"
mkdir -p "${STEERING_OUTPUT}/H4a"
mkdir -p "${LOG_DIR}"

# Activate virtual environment
source .venv/bin/activate

echo "========================================"
echo "7c Steering Validation"
echo "========================================"
echo "Mode: ${MODE}"
echo "GPU: ${GPU_ID}"
echo "Results base: ${RESULTS_BASE}"
echo ""

# Function to run baseline kmeans
run_baseline() {
    echo "========================================"
    echo "Running 7c Baseline K-Means Validation"
    echo "========================================"
    
    python 7_validation/7c_baseline_kmeans.py \
        --samples-dir "${SAMPLES_DIR}" \
        --embeddings-dir "${EMBEDDINGS_DIR}" \
        --clustering-dir "${CLUSTERING_DIR}" \
        --attribution-graphs-dir "${ATTRIBUTION_DIR}" \
        --output-dir "${BASELINE_OUTPUT}" \
        --config "${CONFIG}" \
        --cross-prefix-batching \
        --prefix-batch-size 4 \
        --max-batch-size 512 \
        --log-dir "${LOG_DIR}" \
        2>&1 | tee "${LOG_DIR}/7c_baseline_kmeans_$(date +%Y%m%d_%H%M%S).log"
    
    echo "Baseline K-Means validation complete!"
    echo "Results saved to: ${BASELINE_OUTPUT}"
}

# Function to run steering hypotheses
run_steering() {
    echo "========================================"
    echo "Running 7c Steering Hypotheses Validation"
    echo "========================================"
    
    python 7_validation/7c_hypotheses.py \
        --samples-dir "${SAMPLES_DIR}" \
        --clustering-dir "${CLUSTERING_DIR}" \
        --attribution-graphs-dir "${ATTRIBUTION_DIR}" \
        --output-dir "${STEERING_OUTPUT}" \
        --config "${CONFIG}" \
        --cross-prefix-batching \
        --log-dir "${LOG_DIR}" \
        2>&1 | tee "${LOG_DIR}/7c_hypotheses_$(date +%Y%m%d_%H%M%S).log"
    
    echo "Steering hypotheses validation complete!"
    echo "Results saved to: ${STEERING_OUTPUT}"
}

# Run based on mode
case "${MODE}" in
    baseline)
        run_baseline
        ;;
    steering)
        run_steering
        ;;
    both)
        run_baseline
        echo ""
        run_steering
        ;;
    *)
        echo "Unknown mode: ${MODE}"
        echo "Usage: $0 [baseline|steering|both] [gpu_id]"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "All validations complete!"
echo "========================================"
