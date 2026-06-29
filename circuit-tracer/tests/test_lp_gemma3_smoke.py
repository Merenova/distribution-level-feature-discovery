from __future__ import annotations

import os

import pytest
import torch

from circuit_tracer import ReplacementModel
from circuit_tracer.attribution import attribute_prefix_to_continuations

GEMMA3_1B_TRANSCODER_ID = (
    "mwhanna/gemma-scope-2-1b-it/transcoder_all/width_16k_l0_small_affine"
)

pytestmark = pytest.mark.skipif(
    os.getenv("LP_RUN_GEMMA3_SMOKE") != "1",
    reason="Set LP_RUN_GEMMA3_SMOKE=1 to run GPU/HF Gemma-3 smoke tests.",
)


def test_gemma3_nnsight_loads_and_prefix_attribution_runs():
    model = ReplacementModel.from_pretrained(
        "google/gemma-3-1b-it",
        GEMMA3_1B_TRANSCODER_ID,
        backend="nnsight",
        dtype=torch.bfloat16,
    )
    assert model.backend == "nnsight"
    assert model.cfg.n_layers > 0

    prefix = model.tokenizer.encode("The capital of France is", add_special_tokens=False)
    continuation = model.tokenizer.encode(" Paris.", add_special_tokens=False)
    result = attribute_prefix_to_continuations(
        prefix=prefix,
        continuations=[continuation],
        model=model,
        batch_size=64,
        add_bos=True,
        max_feature_nodes=256,
        verbose=True,
    )
    first = result.continuation_attributions[0][0]
    assert first.source_attribution.ndim == 1
    assert first.source_attribution.numel() > 0
    assert torch.isfinite(first.source_attribution).all()
