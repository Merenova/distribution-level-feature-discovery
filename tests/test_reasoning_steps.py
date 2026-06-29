import math

import pytest

from utils.reasoning_steps import (
    ReasoningStepCandidate,
    build_reasoning_branch_payload,
    build_reasoning_pair_id,
    build_reasoning_prefix_id,
    deduplicate_candidates,
    normalize_step_text,
    select_top_confidence,
    sequence_probability,
    should_stop_after_step,
)


def candidate(text: str, probability: float) -> ReasoningStepCandidate:
    return ReasoningStepCandidate(
        text=text,
        token_ids=[1, 2],
        full_token_ids=[10, 11, 1, 2],
        logprob=math.log(probability) * 2,
        probability=probability,
    )


def test_sequence_probability_uses_length_normalized_confidence():
    assert sequence_probability(math.log(0.25) * 4, 4) == pytest.approx(0.25)
    assert sequence_probability(-1.0, 0) == 0.0


def test_deduplicate_candidates_keeps_highest_confidence_per_text():
    candidates = deduplicate_candidates(
        [
            candidate("first step", 0.20),
            candidate("first step", 0.35),
            candidate("second step", 0.25),
            candidate("   ", 0.99),
        ]
    )

    assert [item.text for item in candidates] == ["first step", "second step"]
    assert [item.probability for item in candidates] == [0.35, 0.25]


def test_select_top_confidence_returns_original_index():
    candidates = [candidate("a", 0.10), candidate("b", 0.80), candidate("c", 0.30)]

    index, selected = select_top_confidence(candidates)

    assert index == 1
    assert selected.text == "b"


def test_select_top_confidence_rejects_empty_candidate_list():
    with pytest.raises(ValueError, match="Cannot select from an empty candidate list"):
        select_top_confidence([])


def test_branch_payload_uses_fixed_former_context_and_current_step_candidates():
    candidates = [candidate("try path A\n\n", 0.7), candidate("try path B\n\n", 0.2)]

    payload = build_reasoning_branch_payload(
        dataset_key="gsm8k",
        example_id="42",
        prompt_text="Question: What is 2+2?",
        prefix_text="Question: What is 2+2?\n\nStep 1\n\n",
        prefix_tokens_with_bos=[0, 10, 11],
        committed_previous_steps=["Step 1\n\n"],
        committed_previous_step_token_spans=[{"step_index": 1, "start": 2, "end": 3}],
        step_index=2,
        candidates=candidates,
        selected_candidate_index=0,
        model_name="Qwen/Qwen3-0.6B",
        extra_metadata={"split": "train"},
    )

    assert payload["prefix_id"] == "gsm8k_42_step_02"
    assert payload["prefix"] == "Question: What is 2+2?\n\nStep 1\n\n"
    assert payload["prefix_tokens_with_bos"] == [0, 10, 11]
    assert payload["continuations"] == [
        {
            "text": "try path A\n\n",
            "token_ids": [1, 2],
            "full_token_ids": [10, 11, 1, 2],
            "logprob": candidates[0].logprob,
            "probability": 0.7,
        },
        {
            "text": "try path B\n\n",
            "token_ids": [1, 2],
            "full_token_ids": [10, 11, 1, 2],
            "logprob": candidates[1].logprob,
            "probability": 0.2,
        },
    ]
    assert payload["metadata"] == {
        "dataset": "gsm8k",
        "example_id": "42",
        "model": "Qwen/Qwen3-0.6B",
        "task_type": "reasoning_step",
        "split": "train",
    }
    assert payload["reasoning_metadata"] == {
        "prompt_text": "Question: What is 2+2?",
        "example_id": "42",
        "step_index": 2,
        "committed_previous_steps": ["Step 1\n\n"],
        "committed_previous_step_token_spans": [
            {"step_index": 1, "start": 2, "end": 3}
        ],
        "selected_candidate_index": 0,
        "selected_candidate_probability": 0.7,
    }


def test_branch_payload_rejects_missing_or_invalid_selected_candidate():
    with pytest.raises(ValueError, match="Cannot build reasoning branch payload without candidates"):
        build_reasoning_branch_payload(
            dataset_key="gsm8k",
            example_id="42",
            prompt_text="Question",
            prefix_text="Question\n\n",
            prefix_tokens_with_bos=[0, 10],
            committed_previous_steps=[],
            committed_previous_step_token_spans=[],
            step_index=1,
            candidates=[],
            selected_candidate_index=0,
            model_name="model",
        )

    with pytest.raises(ValueError, match="selected_candidate_index 2 is out of range"):
        build_reasoning_branch_payload(
            dataset_key="gsm8k",
            example_id="42",
            prompt_text="Question",
            prefix_text="Question\n\n",
            prefix_tokens_with_bos=[0, 10],
            committed_previous_steps=[],
            committed_previous_step_token_spans=[],
            step_index=1,
            candidates=[candidate("try path A\n\n", 0.7)],
            selected_candidate_index=2,
            model_name="model",
        )


def test_stop_marker_detection():
    assert should_stop_after_step("Therefore #### 12", ["####"])
    assert not should_stop_after_step("intermediate reasoning", ["####", "Final:"])
    assert not should_stop_after_step("empty marker ignored", [""])


def test_prefix_id_and_pair_id_formatting():
    assert normalize_step_text("  reason  ") == "reason\n\n"
    assert normalize_step_text("  ") == ""
    assert (
        build_reasoning_prefix_id("math500", "level/1 with space", 3)
        == "math500_level_1_with_space_step_03"
    )
    assert build_reasoning_pair_id("math500_level_1_step_03", 1, 3) == (
        "math500_level_1_step_03_src_01_tgt_03"
    )
