from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load_nnsight_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "circuit-tracer"
        / "circuit_tracer"
        / "replacement_model"
        / "replacement_model_nnsight.py"
    )
    spec = importlib.util.spec_from_file_location("lp_replacement_model_nnsight_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _TokenizerFailsOnTensor:
    def __call__(self, value):
        if isinstance(value, torch.Tensor):
            raise ValueError("tokenizer should not be called for tensor inputs")
        return SimpleNamespace(input_ids=[1, 2, 3, 4])


def test_infer_input_n_positions_accepts_tensor_without_retokenizing():
    mod = _load_nnsight_module()
    model = SimpleNamespace(tokenizer=_TokenizerFailsOnTensor())

    n_pos = mod._infer_input_n_positions(model, torch.tensor([11, 12, 13]), None, None)

    assert n_pos == 3


def test_infer_input_n_positions_rejects_batched_tensor_inputs():
    mod = _load_nnsight_module()
    model = SimpleNamespace(tokenizer=_TokenizerFailsOnTensor())

    with pytest.raises(ValueError, match="single sequence"):
        mod._infer_input_n_positions(model, torch.tensor([[1, 2], [3, 4]]), None, None)


def test_feature_steering_spec_computes_delta_from_current_value():
    mod = _load_nnsight_module()
    spec = mod.FeatureSteeringSpec(
        h_c_value=0.5,
        h_c_norm=2.0,
        steering_method="multiplicative",
        epsilon=0.4,
    )

    delta = mod._compute_feature_intervention_delta(torch.tensor(10.0), spec)

    assert torch.equal(delta, torch.tensor(1.0))
