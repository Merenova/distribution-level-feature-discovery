from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from import_helpers import load_module_from_path


sampler = load_module_from_path(
    "sample_reasoning_steps",
    "2_branch_sampling/sample_reasoning_steps.py",
)


def candidate(text: str, probability: float, token_ids: list[int] | None = None):
    token_ids = token_ids or [len(text)]
    return sampler.ReasoningStepCandidate(
        text=text,
        token_ids=token_ids,
        full_token_ids=list(token_ids),
        logprob=-1.0,
        probability=probability,
    )


def encode_prefix(prefix_text: str) -> list[int]:
    return [0] + list(range(1, len(prefix_text) + 1))


def test_top_confidence_step_becomes_next_fixed_context():
    prompt = "PROMPT\n"
    selected_step_1 = "high confidence step\n\n"
    seen_prefixes: list[tuple[int, str]] = []

    def sample_step(prefix_text: str, step_index: int):
        seen_prefixes.append((step_index, prefix_text))
        if step_index == 1:
            return [
                candidate("low confidence step\n\n", 0.2),
                candidate(selected_step_1, 0.9, [11, 12]),
            ]
        return [candidate("second step\n\n", 0.8, [21])]

    branches, metadata = sampler.rollout_example_with_sampler(
        dataset_key="gsm8k",
        example={"example_id": "ex1"},
        prompt_text=prompt,
        initial_prefix_text=prompt,
        initial_prefix_tokens_with_bos=encode_prefix(prompt),
        model_name="test-model",
        max_steps=2,
        final_answer_markers=["####"],
        sample_step=sample_step,
        encode_prefix=encode_prefix,
    )

    assert seen_prefixes[1] == (2, prompt + selected_step_1)
    assert branches[1]["prefix"] == prompt + selected_step_1
    assert branches[0]["reasoning_metadata"]["selected_candidate_index"] == 0
    assert metadata["committed_steps"][0]["text"] == selected_step_1
    assert metadata["committed_steps"][1]["text"] == "second step\n\n"


def test_rollout_metadata_includes_committed_steps_and_stop_marker():
    prompt = "Question?\n"

    def sample_step(prefix_text: str, step_index: int):
        assert step_index == 1
        return [
            candidate("work then answer #### 42\n\n", 0.7, [1, 2, 3]),
            candidate("more work\n\n", 0.6, [4]),
        ]

    branches, metadata = sampler.rollout_example_with_sampler(
        dataset_key="gsm8k",
        example={"id": "gsm-row-1"},
        prompt_text=prompt,
        initial_prefix_text=prompt,
        initial_prefix_tokens_with_bos=encode_prefix(prompt),
        model_name="test-model",
        max_steps=5,
        final_answer_markers=["####"],
        sample_step=sample_step,
        encode_prefix=encode_prefix,
    )

    assert len(branches) == 1
    assert metadata["stop_reason"] == "final_answer_marker"
    assert metadata["completed"] is True
    assert metadata["committed_steps"] == [
        {
            "step_index": 1,
            "text": "work then answer #### 42\n\n",
            "probability": 0.7,
            "token_span": {
                "step_index": 1,
                "start": len(encode_prefix(prompt)),
                "end": len(encode_prefix(prompt + "work then answer #### 42\n\n")),
            },
        }
    ]


def test_committed_previous_step_token_span_recorded_for_step_2_branch():
    prompt = "PROMPT\n"
    selected_step_1 = "chosen reasoning\n\n"

    def sample_step(prefix_text: str, step_index: int):
        if step_index == 1:
            return [candidate(selected_step_1, 0.95, [7, 8])]
        return [candidate("target branch\n\n", 0.5, [9])]

    branches, _metadata = sampler.rollout_example_with_sampler(
        dataset_key="math500",
        example={"example_id": "math/1"},
        prompt_text=prompt,
        initial_prefix_text=prompt,
        initial_prefix_tokens_with_bos=encode_prefix(prompt),
        model_name="test-model",
        max_steps=2,
        final_answer_markers=["Answer:"],
        sample_step=sample_step,
        encode_prefix=encode_prefix,
    )

    expected_span = {
        "step_index": 1,
        "start": len(encode_prefix(prompt)),
        "end": len(encode_prefix(prompt + selected_step_1)),
    }
    assert branches[1]["reasoning_metadata"]["committed_previous_steps"] == [
        selected_step_1
    ]
    assert branches[1]["reasoning_metadata"][
        "committed_previous_step_token_spans"
    ] == [expected_span]


def test_rollout_rejects_non_prefix_stable_tokenization_for_spans():
    prompt = "P"

    def sample_step(prefix_text: str, step_index: int):
        return [candidate(" appended\n\n", 0.9, [2, 3])]

    def non_prefix_stable_encode(prefix_text: str) -> list[int]:
        if prefix_text == prompt:
            return [0, 1]
        return [0, 99, 2, 3]

    with pytest.raises(ValueError, match="prefix tokenization changed"):
        sampler.rollout_example_with_sampler(
            dataset_key="gsm8k",
            example={"example_id": "unstable"},
            prompt_text=prompt,
            initial_prefix_text=prompt,
            initial_prefix_tokens_with_bos=[0, 1],
            model_name="test-model",
            max_steps=1,
            final_answer_markers=["####"],
            sample_step=sample_step,
            encode_prefix=non_prefix_stable_encode,
        )


def test_rollout_stops_before_sampling_when_prefix_reaches_context_limit():
    calls: list[tuple[str, int]] = []

    def sample_step(prefix_text: str, step_index: int):
        calls.append((prefix_text, step_index))
        return [candidate("should not sample\n\n", 0.9, [2, 3])]

    branches, metadata = sampler.rollout_example_with_sampler(
        dataset_key="math500",
        example={"example_id": "too-long"},
        prompt_text="PROMPT",
        initial_prefix_text="PROMPT",
        initial_prefix_tokens_with_bos=[0, 1, 2],
        model_name="test-model",
        max_steps=3,
        max_prefix_tokens=3,
        final_answer_markers=["Answer:"],
        sample_step=sample_step,
        encode_prefix=lambda _text: [0, 1, 2],
    )

    assert calls == []
    assert branches == []
    assert metadata["stop_reason"] == "max_context_length"
    assert metadata["completed"] is False
    assert metadata["final_prefix_token_length"] == 3
    assert metadata["max_prefix_tokens"] == 3


def test_candidate_retokenizes_when_delimiter_is_appended():
    completion = SimpleNamespace(
        text="partial step",
        token_ids=[1, 2],
        cumulative_logprob=-2.0,
    )

    class FakeTokenizer:
        def __call__(self, text: str, add_special_tokens: bool):
            assert add_special_tokens is False
            assert text == "partial step\n\n"
            return {"input_ids": [10, 11, 12, 13]}

    candidate_obj = sampler.candidate_from_vllm_completion(
        completion,
        prefix_tokens_with_bos=[0, 5],
        tokenizer=FakeTokenizer(),
    )

    assert candidate_obj.text == "partial step\n\n"
    assert candidate_obj.token_ids == [10, 11, 12, 13]
    assert candidate_obj.full_token_ids == [0, 5, 10, 11, 12, 13]


def test_candidate_probability_uses_raw_sampled_token_count_after_retokenizing():
    completion = SimpleNamespace(
        text="partial step",
        token_ids=[1, 2],
        cumulative_logprob=math.log(0.25) * 2,
    )

    class FakeTokenizer:
        def __call__(self, text: str, add_special_tokens: bool):
            assert text == "partial step\n\n"
            return {"input_ids": [10, 11, 12, 13]}

    candidate_obj = sampler.candidate_from_vllm_completion(
        completion,
        prefix_tokens_with_bos=[0],
        tokenizer=FakeTokenizer(),
    )

    assert candidate_obj.token_ids == [10, 11, 12, 13]
    assert candidate_obj.probability == pytest.approx(0.25)


def test_logprob_fallback_uses_sampled_token_id_from_dict_entries():
    completion = SimpleNamespace(
        cumulative_logprob=None,
        token_ids=[101, 102],
        logprobs=[
            {
                101: SimpleNamespace(logprob=-1.0),
                999: SimpleNamespace(logprob=-0.01),
            },
            {
                102: {"logprob": -2.0},
                888: {"logprob": -0.02},
            },
        ],
    )

    assert sampler.extract_completion_logprob(completion) == -3.0


def test_final_answer_markers_prefer_dataset_config():
    config = {
        "reasoning": {
            "final_answer_markers": {
                "math500": ["\\boxed", "The answer is"],
            },
        },
    }

    assert sampler.final_answer_markers_for_dataset(config, "math500") == [
        "\\boxed",
        "The answer is",
    ]


def test_final_answer_markers_fall_back_when_config_missing_or_empty():
    assert sampler.final_answer_markers_for_dataset({}, "math500") == ["Answer:"]
    assert sampler.final_answer_markers_for_dataset(
        {"reasoning": {"final_answer_markers": {"math500": []}}},
        "math500",
    ) == ["Answer:"]


def test_samples_per_step_prefers_reasoning_config_over_global():
    config = {
        "reasoning": {"samples_per_step": 64},
        "global": {"samples_per_example": 128},
    }

    assert sampler.samples_per_step_from_config(config) == 64


def test_samples_per_step_falls_back_to_global_config_and_validation_uses_helper():
    config = {
        "models": {"names": ["model-a"]},
        "datasets": {"gsm8k": {"enabled": True}},
        "global": {"samples_per_example": 128},
        "reasoning": {"samples_per_step": 64},
    }

    assert sampler.samples_per_step_from_config(
        {"global": {"samples_per_example": 128}}
    ) == 128
    assert sampler.validate_cli_runtime_config(config, max_steps=1) == (
        "model-a",
        ["gsm8k"],
        64,
    )


def test_samples_per_step_cli_override_path_updates_reasoning_value():
    config = {
        "global": {"samples_per_example": 128},
        "reasoning": {"samples_per_step": 64},
    }

    sampler.set_samples_per_step_override(config, 32)

    assert config["reasoning"]["samples_per_step"] == 32
    assert config["global"]["samples_per_example"] == 128
    assert sampler.samples_per_step_from_config(config) == 32


def test_cli_runtime_validation_rejects_multiple_models_and_bad_max_steps():
    config = {
        "models": {"names": ["model-a", "model-b"]},
        "datasets": {"gsm8k": {"enabled": True}},
        "global": {"samples_per_example": 4},
    }

    with pytest.raises(ValueError, match="exactly one model"):
        sampler.validate_cli_runtime_config(config, max_steps=1)

    config["models"]["names"] = ["model-a"]
    with pytest.raises(ValueError, match="max_steps must be > 0"):
        sampler.validate_cli_runtime_config(config, max_steps=0)
