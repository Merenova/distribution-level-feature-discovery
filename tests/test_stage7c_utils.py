from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_utils_module():
    path = Path(__file__).resolve().parents[1] / "7_validation" / "7c_utils.py"
    spec = importlib.util.spec_from_file_location("lp_7c_utils_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _FakeLayerTranscoder:
    def __init__(self, offset: int):
        self.W_enc = torch.arange(24, dtype=torch.float32).reshape(6, 4) + offset
        self.b_enc = torch.arange(6, dtype=torch.float32) + offset


class _FakeTranscoderSet:
    def __init__(self):
        self.transcoders = [_FakeLayerTranscoder(0), _FakeLayerTranscoder(100)]

    def __getitem__(self, idx: int):
        return self.transcoders[idx]


class _FakeEnvoy:
    def __init__(self, module):
        self._module = module
        self.transcoders = [object()]

    def __getitem__(self, idx: int):
        return self.transcoders[idx]


def test_get_encoder_weights_unwraps_nnsight_envoy_transcoder_set():
    mod = _load_utils_module()
    model = SimpleNamespace(transcoders=_FakeEnvoy(_FakeTranscoderSet()))
    feat_ids = torch.tensor([1, 3], dtype=torch.long)

    W_enc, b_enc = mod.get_encoder_weights(model, layer=1, feat_ids_tensor=feat_ids)

    expected_layer = model.transcoders._module[1]
    assert torch.equal(W_enc, expected_layer.W_enc[feat_ids])
    assert torch.equal(b_enc, expected_layer.b_enc[feat_ids])
