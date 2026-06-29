#!/usr/bin/env bash
# Install dependencies for latent_planning using uv add
#
# Usage:
#   bash install_deps.sh
#
# This script adds all required dependencies for the latent_planning project.

set -euo pipefail

# Change to project root so relative paths resolve from the repository.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================"
echo "Installing dependencies for latent_planning"
echo "========================================"
echo ""

# Install all dependencies in batches to speed up resolution

echo "Step 1: Core ML/AI libraries..."
uv add \
    "torch>=2.4.0" \
    transformers \
    sentence-transformers \
    datasets

echo ""
echo "Step 2: vllm (pre-built wheel, may take a moment)..."
# vllm >= 0.6.0 provides pre-built wheels for common CUDA versions
uv add "vllm>=0.6.0"

echo ""
echo "Step 3: Scientific computing..."
uv add \
    numpy \
    scipy \
    scikit-learn

echo ""
echo "Step 4: Visualization..."
uv add \
    matplotlib \
    plotly \
    kaleido

echo ""
echo "Step 5: Utilities and HuggingFace ecosystem..."
uv add \
    tqdm \
    einops \
    pydantic \
    huggingface-hub \
    safetensors \
    tokenizers

echo ""
echo "Step 6: TransformerLens (for circuit-tracer)..."
uv add transformer-lens

echo ""
echo "Step 7: Jupyter support..."
uv add \
    ipykernel \
    ipywidgets

echo ""
echo "Step 8: Local circuit-tracer package..."
# Need --prerelease=allow for transformer-lens 3.x which supports numpy 2.x
uv add --editable ./circuit-tracer --prerelease=allow

echo ""
echo "========================================"
echo "All dependencies installed successfully!"
echo "========================================"
echo ""
echo "To verify installation, run:"
echo "  uv run python -c \"import torch; import transformers; import vllm; print('OK')\""
