from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_nnsight_module():
    path = Path(__file__).resolve().parents[1] / "7_validation" / "7c_steering_nnsight.py"
    spec = importlib.util.spec_from_file_location("lp_7c_steering_nnsight_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_absolute_interventions_from_activation_cache():
    mod = _load_nnsight_module()
    model = SimpleNamespace(cfg=SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32))
    activation_cache = torch.zeros(2, 4, 10)
    activation_cache[1, 2, 7] = 3.0
    interventions = mod.build_absolute_interventions(
        model=model,
        activation_cache=activation_cache,
        features=[(1, 2, 7, 0.5)],
        steering_method="additive",
        epsilon=2.0,
    )
    assert interventions == [(1, 2, 7, 5.0)]


def test_pad_logits_to_batch_shape():
    mod = _load_nnsight_module()
    a = torch.ones(1, 2, 5)
    b = torch.ones(1, 4, 5) * 2
    padded, mask = mod.pad_logits([a, b], pad_to=4)
    assert padded.shape == (2, 4, 5)
    assert mask.tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]
    assert torch.all(padded[0, 2:] == 0)
