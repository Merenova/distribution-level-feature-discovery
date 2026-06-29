"""Backend selection shared by attribution and steering stages."""

from __future__ import annotations

import torch

VALID_BACKENDS = {"auto", "transformerlens", "nnsight"}


def resolve_backend(model_name: str, backend: str) -> str:
    """Resolve a concrete circuit-tracer backend for a model name."""
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"Invalid backend {backend!r}; expected one of {sorted(VALID_BACKENDS)}"
        )
    if backend != "auto":
        return backend
    return "nnsight" if "gemma-3" in model_name.lower() else "transformerlens"


def resolve_stage_backend(config: dict, stage_key: str) -> str:
    """Read a stage backend, defaulting to attribution.backend, then auto."""
    stage_cfg = config.get(stage_key, {}) or {}
    attribution_cfg = config.get("attribution", {}) or {}
    backend = stage_cfg.get("backend", attribution_cfg.get("backend", "auto"))
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"Invalid {stage_key}.backend {backend!r}; expected one of {sorted(VALID_BACKENDS)}"
        )
    return backend


def get_model_device(model, fallback: torch.device | str | None = None) -> torch.device:
    """Return a model device across TransformerLens and nnsight backends."""
    cfg = getattr(model, "cfg", None)
    device = getattr(cfg, "device", None)
    if device is None:
        device = getattr(model, "device", None)
    if device is None:
        device = fallback
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def get_model_dtype(model, fallback: torch.dtype = torch.float32) -> torch.dtype:
    """Return a model dtype across TransformerLens and nnsight backends."""
    cfg = getattr(model, "cfg", None)
    dtype = getattr(cfg, "dtype", None)
    if dtype is None:
        dtype = getattr(model, "dtype", None)
    if dtype is None:
        dtype = fallback
    return dtype
