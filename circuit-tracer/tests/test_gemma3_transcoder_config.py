from __future__ import annotations

import pytest

from circuit_tracer.utils.gemma3_transcoder_config import build_gemma3_transcoder_config
from circuit_tracer.utils.hf_utils import HfUri


def test_build_gemma3_config_from_repo_subfolder(monkeypatch):
    files = [
        "transcoder_all/width_16k_l0_small_affine/layer_1.safetensors",
        "transcoder_all/width_16k_l0_small_affine/layer_0.safetensors",
        "transcoder_all/width_16k_l0_big/layer_0.safetensors",
    ]
    monkeypatch.setattr(
        "circuit_tracer.utils.gemma3_transcoder_config.list_repo_files",
        lambda repo_id, revision=None: files,
    )

    uri = HfUri(
        repo_id="mwhanna/gemma-scope-2-1b-it",
        file_path="transcoder_all/width_16k_l0_small_affine",
        revision=None,
    )

    config = build_gemma3_transcoder_config(uri)

    assert config is not None
    assert config["model_kind"] == "transcoder_set"
    assert config["repo_id"] == "mwhanna/gemma-scope-2-1b-it"
    assert config["feature_input_hook"] == "ln2.hook_normalized"
    assert config["feature_output_hook"] == "hook_mlp_out"
    assert config["transcoders"] == [
        "hf://mwhanna/gemma-scope-2-1b-it/transcoder_all/width_16k_l0_small_affine/layer_0.safetensors",
        "hf://mwhanna/gemma-scope-2-1b-it/transcoder_all/width_16k_l0_small_affine/layer_1.safetensors",
    ]


def test_build_gemma3_config_preserves_revision(monkeypatch):
    monkeypatch.setattr(
        "circuit_tracer.utils.gemma3_transcoder_config.list_repo_files",
        lambda repo_id, revision=None: ["sub/layer_0.safetensors"],
    )
    uri = HfUri(
        repo_id="mwhanna/gemma-scope-2-1b-it",
        file_path="sub",
        revision="abc123",
    )

    config = build_gemma3_transcoder_config(uri)

    assert config is not None
    assert config["scan"] == "mwhanna/gemma-scope-2-1b-it/sub@abc123"
    assert config["transcoders"] == [
        "hf://mwhanna/gemma-scope-2-1b-it/sub/layer_0.safetensors?revision=abc123"
    ]


def test_build_gemma3_config_returns_none_for_non_gemma_scope():
    uri = HfUri(repo_id="mwhanna/qwen3-8b-transcoders", file_path=None, revision=None)
    assert build_gemma3_transcoder_config(uri) is None


def test_build_gemma3_config_rejects_non_contiguous_layers(monkeypatch):
    monkeypatch.setattr(
        "circuit_tracer.utils.gemma3_transcoder_config.list_repo_files",
        lambda repo_id, revision=None: ["sub/layer_0.safetensors", "sub/layer_2.safetensors"],
    )
    uri = HfUri(repo_id="mwhanna/gemma-scope-2-1b-it", file_path="sub", revision=None)

    with pytest.raises(ValueError, match="not contiguous"):
        build_gemma3_transcoder_config(uri)
