from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ReasoningStepCandidate:
    text: str
    token_ids: list[int]
    full_token_ids: list[int]
    logprob: float
    probability: float


def sequence_probability(logprob: float, token_count: int) -> float:
    if token_count <= 0:
        return 0.0
    return float(__import__("math").exp(logprob / token_count))


def normalize_step_text(text: str, delimiter: str = "\n\n") -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return stripped + delimiter


def deduplicate_candidates(
    candidates: list[ReasoningStepCandidate],
) -> list[ReasoningStepCandidate]:
    best_by_text: dict[str, ReasoningStepCandidate] = {}
    for candidate in candidates:
        key = candidate.text.strip()
        if not key:
            continue
        current = best_by_text.get(key)
        if current is None or candidate.probability > current.probability:
            best_by_text[key] = candidate
    return sorted(best_by_text.values(), key=lambda item: item.probability, reverse=True)


def select_top_confidence(
    candidates: list[ReasoningStepCandidate],
) -> tuple[int, ReasoningStepCandidate]:
    if not candidates:
        raise ValueError("Cannot select from an empty candidate list")
    best_index, best_candidate = max(
        enumerate(candidates),
        key=lambda pair: pair[1].probability,
    )
    return best_index, best_candidate


def should_stop_after_step(step_text: str, final_answer_markers: list[str]) -> bool:
    return any(marker and marker in step_text for marker in final_answer_markers)


def build_reasoning_prefix_id(dataset_key: str, example_id: str, step_index: int) -> str:
    safe_example_id = str(example_id).replace("/", "_").replace(" ", "_")
    return f"{dataset_key}_{safe_example_id}_step_{step_index:02d}"


def build_reasoning_pair_id(
    target_prefix_id: str,
    source_step_index: int,
    target_step_index: int,
) -> str:
    return f"{target_prefix_id}_src_{source_step_index:02d}_tgt_{target_step_index:02d}"


def build_reasoning_branch_payload(
    *,
    dataset_key: str,
    example_id: str,
    prompt_text: str,
    prefix_text: str,
    prefix_tokens_with_bos: list[int],
    committed_previous_steps: list[str],
    committed_previous_step_token_spans: list[dict[str, int]],
    step_index: int,
    candidates: list[ReasoningStepCandidate],
    selected_candidate_index: int,
    model_name: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("Cannot build reasoning branch payload without candidates")
    if selected_candidate_index < 0 or selected_candidate_index >= len(candidates):
        raise ValueError(
            f"selected_candidate_index {selected_candidate_index} is out of range "
            f"for {len(candidates)} candidates"
        )

    return {
        "prefix_id": build_reasoning_prefix_id(dataset_key, example_id, step_index),
        "prefix": prefix_text,
        "prefix_tokens_with_bos": prefix_tokens_with_bos,
        "continuations": [
            {
                "text": candidate.text,
                "token_ids": candidate.token_ids,
                "full_token_ids": candidate.full_token_ids,
                "logprob": candidate.logprob,
                "probability": candidate.probability,
            }
            for candidate in candidates
        ],
        "metadata": {
            "dataset": dataset_key,
            "example_id": str(example_id),
            "model": model_name,
            "task_type": "reasoning_step",
            **(extra_metadata or {}),
        },
        "reasoning_metadata": {
            "prompt_text": prompt_text,
            "example_id": str(example_id),
            "step_index": step_index,
            "committed_previous_steps": committed_previous_steps,
            "committed_previous_step_token_spans": committed_previous_step_token_spans,
            "selected_candidate_index": selected_candidate_index,
            "selected_candidate_probability": candidates[
                selected_candidate_index
            ].probability,
        },
    }
