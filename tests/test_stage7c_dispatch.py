from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_dispatcher():
    path = Path(__file__).resolve().parents[1] / "7_validation" / "7c_steering.py"
    spec = importlib.util.spec_from_file_location("lp_7c_steering_dispatch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dispatcher_selects_nnsight(monkeypatch):
    steering = _load_dispatcher()
    monkeypatch.setattr(
        steering.nnsight,
        "run_batched_steered_pass_on_the_fly",
        lambda model, *args, **kwargs: ("nnsight", model.backend),
    )
    result = steering.run_batched_steered_pass_on_the_fly(
        SimpleNamespace(backend="nnsight"),
        [[1, 2]],
        [],
        {},
        {},
        "additive",
        0.0,
    )
    assert result == ("nnsight", "nnsight")


def test_dispatcher_selects_transformerlens(monkeypatch):
    steering = _load_dispatcher()
    monkeypatch.setattr(
        steering.tl,
        "run_batched_steered_pass_on_the_fly",
        lambda model, *args, **kwargs: ("tl", model.backend),
    )
    result = steering.run_batched_steered_pass_on_the_fly(
        SimpleNamespace(backend="transformerlens"),
        [[1, 2]],
        [],
        {},
        {},
        "additive",
        0.0,
    )
    assert result == ("tl", "transformerlens")


def test_dispatcher_exposes_centered_logits_helper():
    steering = _load_dispatcher()

    assert steering.compute_per_token_centered_logits_batched is steering.tl.compute_per_token_centered_logits_batched
