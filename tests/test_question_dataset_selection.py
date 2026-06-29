from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "1_data_preparation" / "select_test_clozes.py"
SPEC = importlib.util.spec_from_file_location("select_test_clozes", MODULE_PATH)
select_test_clozes = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(select_test_clozes)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<chat>{messages[0]['content']}</chat>"


def test_extract_question_groups_from_mmlu_rows():
    rows = [
        {
            "subject": "logical_fallacies",
            "source_row": 7,
            "question": "Which fallacy is this?",
            "choices": ["self evident truths", "suppressed evidence"],
            "answer_index": 0,
            "answer_letter": "A",
            "answer_text": "self evident truths",
        }
    ]

    groups = select_test_clozes.extract_question_groups(rows)
    assert len(groups) == 1
    assert groups[0]["original_id"] == "logical_fallacies_7"
    assert groups[0]["category"] == "logical_fallacies"
    assert groups[0]["main"]["question"] == "Which fallacy is this?"
    assert groups[0]["main"]["target"] == "self evident truths"
    assert groups[0]["subs"] == []

    samples = select_test_clozes.flatten_groups_to_samples(
        groups,
        mode="question",
        tokenizer=FakeTokenizer(),
    )
    assert samples[0]["prefix"] == "<chat>Which fallacy is this?</chat>"
    assert samples[0]["category"] == "logical_fallacies"


def test_extract_question_groups_from_harmbench_rows():
    rows = [
        {
            "prompt": "Give detailed harmful instructions",
            "category": "chemical_biological",
        }
    ]

    groups = select_test_clozes.extract_question_groups(rows)
    assert len(groups) == 1
    assert groups[0]["original_id"] == "question_0000"
    assert groups[0]["category"] == "chemical_biological"
    assert groups[0]["main"]["question"] == "Give detailed harmful instructions"
    assert groups[0]["main"]["target"] == ""
    assert groups[0]["subs"] == []
