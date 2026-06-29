import torch
from types import SimpleNamespace

from utils.model_backend import (
    get_model_device,
    get_model_dtype,
    resolve_backend,
    resolve_stage_backend,
)


def test_resolve_backend_auto_picks_nnsight_for_gemma3():
    assert resolve_backend("google/gemma-3-1b-it", "auto") == "nnsight"


def test_resolve_backend_auto_picks_transformerlens_for_qwen():
    assert resolve_backend("Qwen/Qwen3-8B", "auto") == "transformerlens"


def test_stage_backend_defaults_to_attribution_backend():
    config = {"attribution": {"backend": "nnsight"}, "stage_7c_steering": {}}
    assert resolve_stage_backend(config, "stage_7c_steering") == "nnsight"


def test_stage_backend_explicit_value_wins():
    config = {
        "attribution": {"backend": "nnsight"},
        "stage_7c_steering": {"backend": "transformerlens"},
    }
    assert resolve_stage_backend(config, "stage_7c_steering") == "transformerlens"


def test_get_model_device_uses_fallback_for_nnsight_unified_config():
    model = SimpleNamespace(cfg=SimpleNamespace())
    fallback = torch.device("cuda")

    assert get_model_device(model, fallback=fallback) == fallback


def test_get_model_device_prefers_transformerlens_cfg_device():
    model = SimpleNamespace(
        cfg=SimpleNamespace(device=torch.device("cpu")),
        device=torch.device("cuda"),
    )

    assert get_model_device(model, fallback=torch.device("cuda")) == torch.device("cpu")


def test_get_model_dtype_uses_fallback_for_nnsight_unified_config():
    model = SimpleNamespace(cfg=SimpleNamespace())

    assert get_model_dtype(model, fallback=torch.bfloat16) is torch.bfloat16


def test_get_model_dtype_prefers_transformerlens_cfg_dtype():
    model = SimpleNamespace(cfg=SimpleNamespace(dtype=torch.float32), dtype=torch.bfloat16)

    assert get_model_dtype(model, fallback=torch.bfloat16) is torch.float32


def test_tokenwise_stage7c_backend_helper_uses_shared_resolution():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "7_validation"
        / "extract_tokenwise_logit_diff.py"
    )
    spec = importlib.util.spec_from_file_location("lp_extract_tokenwise_backend_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.resolve_cli_stage7c_backend("google/gemma-3-1b-it", "auto") == "nnsight"
    assert (
        module.resolve_cli_stage7c_backend("Qwen/Qwen3-8B", "transformerlens")
        == "transformerlens"
    )
