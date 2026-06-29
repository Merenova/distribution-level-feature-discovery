from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_nnsight_steering():
    path = Path(__file__).resolve().parents[1] / "7_validation" / "7c_steering_nnsight.py"
    spec = importlib.util.spec_from_file_location("lp_7c_steering_nnsight_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_steered_pass_uses_single_dynamic_intervention_trace(monkeypatch):
    steering = _load_nnsight_steering()
    captured = {}

    def fail_activation_cache(*args, **kwargs):
        raise AssertionError("separate activation cache pass should not run")

    def feature_intervention(tokens, interventions, **kwargs):
        captured["tokens"] = tokens
        captured["interventions"] = interventions
        return torch.zeros(1, tokens.numel(), 8), None

    model = SimpleNamespace(
        cfg=SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32),
        device=torch.device("cpu"),
        dtype=torch.float32,
        feature_intervention=feature_intervention,
    )
    monkeypatch.setattr(steering, "_get_activation_cache", fail_activation_cache)

    logits, mask = steering.run_batched_steered_pass_on_the_fly(
        model,
        [[1, 2, 3]],
        [(0, 1, 7, 0.5)],
        encoder_cache={},
        decoder_cache={},
        steering_method="additive",
        epsilon=0.2,
    )

    assert logits.shape == (1, 3, 8)
    assert mask.tolist() == [[1, 1, 1]]
    assert captured["tokens"].tolist() == [1, 2, 3]
    assert captured["interventions"][0][:3] == (0, 1, 7)
    spec = captured["interventions"][0][3]
    assert spec.h_c_value == 0.5
    assert spec.h_c_norm == 0.5
    assert spec.steering_method == "additive"
    assert spec.epsilon == 0.2
