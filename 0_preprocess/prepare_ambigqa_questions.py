#!/usr/bin/env -S uv run python
"""Prepare AmbigQA-style raw questions into grouped question records."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import Dataset, DatasetDict, load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data_utils import save_json
from utils.logging_utils import setup_logger


def _annotation_record(raw_annotations: Any) -> Dict[str, Any]:
    if isinstance(raw_annotations, dict):
        return raw_annotations
    if isinstance(raw_annotations, list):
        for item in raw_annotations:
            if isinstance(item, dict):
                return item
    return {}


def _normalize_answers(raw_answers: Any) -> List[str]:
    values: List[str] = []

    def visit(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            text = value.strip()
            if text:
                values.append(text)
            return
        if isinstance(value, dict):
            for key in ("text", "answer"):
                maybe_text = value.get(key)
                if isinstance(maybe_text, str):
                    visit(maybe_text)
                    return
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(raw_answers)

    deduped: List[str] = []
    seen = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _answers_from_frequencies(raw_answer_frequencies: Any) -> List[str]:
    parsed = raw_answer_frequencies
    if isinstance(raw_answer_frequencies, str):
        try:
            parsed = json.loads(raw_answer_frequencies)
        except json.JSONDecodeError:
            return []

    if isinstance(parsed, dict):
        answers = []
        for key in parsed.keys():
            if isinstance(key, str) and key.strip():
                answers.append(key.strip())
        return answers
    return []


def _extract_main_answers(row: Dict[str, Any], annotations: Dict[str, Any]) -> List[str]:
    answers = _normalize_answers(annotations.get("answer"))
    if answers:
        return answers

    answers = _answers_from_frequencies(row.get("answer_frequencies"))
    if answers:
        return answers

    for key in ("answer", "answers"):
        answers = _normalize_answers(row.get(key))
        if answers:
            return answers

    return []


def _extract_sub_questions(
    row: Dict[str, Any],
    annotations: Dict[str, Any],
) -> List[Dict[str, Any]]:
    qa_pairs = annotations.get("qaPairs") or []
    if isinstance(qa_pairs, dict):
        qa_pairs = [qa_pairs]

    primary_pair = None
    if isinstance(qa_pairs, list):
        for pair in qa_pairs:
            if isinstance(pair, dict):
                primary_pair = pair
                break

    if primary_pair is None:
        return []

    raw_questions = primary_pair.get("question") or []
    raw_answers = primary_pair.get("answer") or []

    if not isinstance(raw_questions, list):
        raw_questions = [raw_questions]
    if not isinstance(raw_answers, list):
        raw_answers = [raw_answers]

    main_question = str(row.get("question", "")).strip()
    sub_entries: List[Dict[str, Any]] = []
    seen_questions = {main_question} if main_question else set()

    for idx, raw_question in enumerate(raw_questions):
        if not isinstance(raw_question, str):
            continue
        question = raw_question.strip()
        if not question or question in seen_questions:
            continue

        answers = []
        if idx < len(raw_answers):
            answers = _normalize_answers(raw_answers[idx])

        sub_entries.append(
            {
                "question": question,
                "answers": answers,
            }
        )
        seen_questions.add(question)

    return sub_entries


def load_split(dataset_dir: Path, split: str) -> Tuple[Dataset, str]:
    dataset_or_dict = load_from_disk(str(dataset_dir))
    if isinstance(dataset_or_dict, DatasetDict):
        if split not in dataset_or_dict:
            raise ValueError(
                f"Split '{split}' not found in {dataset_dir}. Available splits: {list(dataset_or_dict.keys())}"
            )
        return dataset_or_dict[split], split

    if isinstance(dataset_or_dict, Dataset):
        return dataset_or_dict, "single_dataset"

    raise TypeError(f"Unsupported dataset object loaded from {dataset_dir}: {type(dataset_or_dict)!r}")


def prepare_groups(dataset_dir: Path, split: str, logger) -> Dict[str, Any]:
    dataset, resolved_split = load_split(dataset_dir, split)
    groups: List[Dict[str, Any]] = []
    skipped_empty_question = 0

    for row_idx, row in enumerate(dataset):
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            skipped_empty_question += 1
            continue

        annotations = _annotation_record(row.get("annotations"))
        group = {
            "group_id": f"group_{len(groups):04d}",
            "original_id": str(row.get("id", row_idx)),
            "question_type": str(row.get("question_type", "")),
            "main": {
                "question": question.strip(),
                "answers": _extract_main_answers(row, annotations),
            },
            "subs": _extract_sub_questions(row, annotations),
        }
        groups.append(group)

    logger.info(f"Loaded {len(dataset)} rows from split '{resolved_split}'")
    logger.info(f"Prepared {len(groups)} grouped questions")
    if skipped_empty_question:
        logger.info(f"Skipped {skipped_empty_question} rows with empty questions")

    return {
        "metadata": {
            "dataset_dir": str(dataset_dir),
            "requested_split": split,
            "resolved_split": resolved_split,
            "n_rows": len(dataset),
            "n_groups": len(groups),
            "skipped_empty_question": skipped_empty_question,
            "format": "ambigqa_question_groups_v1",
        },
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare AmbigQA raw questions into grouped question records")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Local Hugging Face dataset directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to load when using a DatasetDict")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results/0_preprocess/ambigqa_question_groups.json",
        help="Output JSON path for grouped question records",
    )
    parser.add_argument("--log-dir", type=Path, default=None, help="Optional log directory")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    log_file = args.log_dir / "prepare_ambigqa_questions.log" if args.log_dir else None
    logger = setup_logger("prepare_ambigqa_questions", log_file=log_file, level=log_level)

    logger.info("=" * 60)
    logger.info("AMBIGQA QUESTION PREPARATION")
    logger.info("=" * 60)
    logger.info(f"Dataset directory: {args.dataset_dir}")
    logger.info(f"Requested split: {args.split}")
    logger.info(f"Output file: {args.output}")

    payload = prepare_groups(args.dataset_dir, args.split, logger)
    save_json(payload, args.output)

    logger.info(f"Saved grouped questions to: {args.output}")


if __name__ == "__main__":
    main()
