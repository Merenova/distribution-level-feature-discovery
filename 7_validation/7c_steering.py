#!/usr/bin/env python3
"""Backend dispatcher for Stage 7c steering."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _import_module(name: str, file_name: str):
    path = Path(__file__).resolve().parent / file_name
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tl = _import_module("lp_7c_steering_tl", "7c_steering_tl.py")
nnsight = _import_module("lp_7c_steering_nnsight", "7c_steering_nnsight.py")

compute_per_token_centered_logits_batched = tl.compute_per_token_centered_logits_batched


def _impl(model):
    return nnsight if getattr(model, "backend", "transformerlens") == "nnsight" else tl


def run_batched_steered_pass_on_the_fly(model, *args, **kwargs):
    return _impl(model).run_batched_steered_pass_on_the_fly(model, *args, **kwargs)


def run_steered_pass_on_the_fly(model, *args, **kwargs):
    return _impl(model).run_steered_pass_on_the_fly(model, *args, **kwargs)


def prepare_heterogeneous_layer_metadata(model, *args, **kwargs):
    return _impl(model).prepare_heterogeneous_layer_metadata(model, *args, **kwargs)


def run_heterogeneous_steered_pass(model, *args, **kwargs):
    return _impl(model).run_heterogeneous_steered_pass(model, *args, **kwargs)


def compute_branch_log_probs_batch(model, *args, **kwargs):
    return _impl(model).compute_branch_log_probs_batch(model, *args, **kwargs)


def compute_baseline_metadata(*args, **kwargs):
    return tl.compute_baseline_metadata(*args, **kwargs)


def generate_steered_sequences(model, *args, **kwargs):
    return _impl(model).generate_steered_sequences(model, *args, **kwargs)
