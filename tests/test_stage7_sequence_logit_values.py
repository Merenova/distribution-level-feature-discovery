import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_per_token_logit_values_batched_returns_raw_and_demeaned_values():
    metrics = load_module(ROOT / "7_validation" / "7c_metrics.py", "stage7c_metrics_sequence_values")
    logits = torch.tensor(
        [
            [
                [1.0, 2.0, 4.0],
                [0.0, 3.0, 6.0],
                [5.0, 1.0, 0.0],
            ]
        ]
    )

    values = metrics.compute_per_token_logit_values_batched(
        logits,
        [([2, 1], 0)],
    )

    assert values == [
        {
            "raw_target_logits": [4.0, 3.0],
            "target_probs": pytest.approx([0.8437947, 0.0473142]),
            "demeaned_logits": pytest.approx([1.6666666, 0.0]),
            "mean_raw_target_logit": 3.5,
            "mean_target_prob": pytest.approx(0.4455544),
            "mean_demeaned_logit": pytest.approx(0.8333333),
        }
    ]


def test_build_per_sequence_logit_record_contains_tokens_raw_values_and_deltas():
    hypotheses = load_module(
        ROOT / "7_validation" / "7c_hypotheses.py",
        "stage7c_hypotheses_sequence_values",
    )

    record = hypotheses._build_per_sequence_logit_record(
        branch_id=7,
        continuation_token_ids=[11, 12],
        original_values={
            "raw_target_logits": [1.0, 2.0],
            "target_probs": [0.2, 0.4],
            "demeaned_logits": [0.5, -0.5],
            "mean_raw_target_logit": 1.5,
            "mean_target_prob": 0.3,
            "mean_demeaned_logit": 0.0,
        },
        steered_values={
            "raw_target_logits": [1.5, 1.0],
            "target_probs": [0.3, 0.1],
            "demeaned_logits": [0.75, -0.25],
            "mean_raw_target_logit": 1.25,
            "mean_target_prob": 0.2,
            "mean_demeaned_logit": 0.25,
        },
        log_prob_original=-3.0,
        log_prob_steered=-2.0,
    )

    assert record == {
        "branch_id": 7,
        "continuation_token_ids": [11, 12],
        "raw_target_logits_original": [1.0, 2.0],
        "raw_target_logits_steered": [1.5, 1.0],
        "raw_target_logit_delta": [0.5, -1.0],
        "target_probs_original": [0.2, 0.4],
        "target_probs_steered": [0.3, 0.1],
        "target_prob_delta": [0.1, -0.3],
        "demeaned_logits_original": [0.5, -0.5],
        "demeaned_logits_steered": [0.75, -0.25],
        "demeaned_logit_delta": [0.25, 0.25],
        "mean_raw_target_logit_original": 1.5,
        "mean_raw_target_logit_steered": 1.25,
        "mean_raw_target_logit_delta": -0.25,
        "mean_target_prob_original": 0.3,
        "mean_target_prob_steered": 0.2,
        "mean_target_prob_delta": -0.1,
        "mean_demeaned_logit_original": 0.0,
        "mean_demeaned_logit_steered": 0.25,
        "mean_demeaned_logit_delta": 0.25,
        "log_prob_original": -3.0,
        "log_prob_steered": -2.0,
        "log_prob_delta": 1.0,
    }
