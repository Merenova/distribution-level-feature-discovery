#!/usr/bin/env -S uv run python
"""Pull selected MMLU subjects and emit pipeline-ready question prompts.

This script adapts `cais/mmlu` to the latent_planning Stage 2 input contract by:
1. Loading selected subjects from Hugging Face
2. Saving a merged local snapshot for reuse
3. Rendering question-only chat prompts with the target model tokenizer
4. Writing `test_clozes.json` compatible with the existing pipeline
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, List

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data_utils import save_json
from utils.logging_utils import setup_logger


DEFAULT_SUBJECTS = [
    "logical_fallacies",
    "moral_disputes",
    "professional_psychology",
    "sociology",
    "philosophy",
    "jurisprudence",
    "international_law",
    "business_ethics",
]


def parse_subjects(subjects_arg: str) -> List[str]:
    subjects = [part.strip() for part in subjects_arg.split(",") if part.strip()]
    if not subjects:
        raise ValueError("No MMLU subjects provided")
    return subjects


def apply_chat_template(question: str, tokenizer) -> str:
    messages = [{"role": "user", "content": question}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def normalize_answer(example: dict, answer_value) -> tuple[int, str, str]:
    answer_feature = example["answer_feature"]

    if isinstance(answer_value, str):
        answer_letter = answer_value
        if hasattr(answer_feature, "str2int"):
            answer_idx = int(answer_feature.str2int(answer_value))
        else:
            answer_idx = ["A", "B", "C", "D"].index(answer_value)
    else:
        answer_idx = int(answer_value)
        if hasattr(answer_feature, "int2str"):
            answer_letter = str(answer_feature.int2str(answer_idx))
        else:
            answer_letter = ["A", "B", "C", "D"][answer_idx]

    answer_text = str(example["choices"][answer_idx])
    return answer_idx, answer_letter, answer_text


def snapshot_metadata_path(raw_save_dir: Path) -> Path:
    return raw_save_dir / "metadata.json"


def snapshot_matches(
    raw_save_dir: Path,
    dataset_id: str,
    split: str,
    subjects: List[str],
) -> bool:
    metadata_path = snapshot_metadata_path(raw_save_dir)
    if not raw_save_dir.exists() or not metadata_path.exists():
        return False

    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception:
        return False

    return (
        metadata.get("dataset_id") == dataset_id
        and metadata.get("split") == split
        and metadata.get("subjects") == subjects
    )


def load_selected_dataset(
    dataset_id: str,
    split: str,
    subjects: List[str],
    raw_save_dir: Path,
    logger,
):
    if snapshot_matches(raw_save_dir, dataset_id, split, subjects):
        logger.info("Loading cached MMLU snapshot from %s", raw_save_dir)
        loaded = load_from_disk(str(raw_save_dir))
        if isinstance(loaded, DatasetDict):
            return loaded[split]
        return loaded

    logger.info("Pulling %s split=%s for %d subjects", dataset_id, split, len(subjects))

    rows = []
    for subject in subjects:
        subject_ds = load_dataset(dataset_id, subject, split=split)
        answer_feature = subject_ds.features["answer"]
        logger.info("  %s: %d rows", subject, len(subject_ds))

        for row_idx, example in enumerate(subject_ds):
            example = dict(example)
            example["answer_feature"] = answer_feature
            answer_idx, answer_letter, answer_text = normalize_answer(example, example["answer"])
            rows.append(
                {
                    "subject": subject,
                    "source_row": row_idx,
                    "question": str(example["question"]),
                    "choices": [str(choice) for choice in example["choices"]],
                    "answer_index": answer_idx,
                    "answer_letter": answer_letter,
                    "answer_text": answer_text,
                }
            )

    merged = Dataset.from_list(rows)
    merged_dict = DatasetDict({split: merged})

    if raw_save_dir.exists():
        shutil.rmtree(raw_save_dir)
    raw_save_dir.parent.mkdir(parents=True, exist_ok=True)
    merged_dict.save_to_disk(str(raw_save_dir))
    save_json(
        {
            "dataset_id": dataset_id,
            "split": split,
            "subjects": subjects,
            "num_rows": len(merged),
        },
        snapshot_metadata_path(raw_save_dir),
    )
    logger.info("Saved merged MMLU snapshot to %s", raw_save_dir)
    return merged


def output_matches(
    output_path: Path,
    dataset_id: str,
    split: str,
    subjects: List[str],
    model: str,
) -> bool:
    if not output_path.exists():
        return False
    try:
        payload = json.loads(output_path.read_text())
    except Exception:
        return False

    metadata = payload.get("metadata", {})
    return (
        metadata.get("dataset_id") == dataset_id
        and metadata.get("split") == split
        and metadata.get("subjects") == subjects
        and metadata.get("model") == model
        and metadata.get("prompt_style") == "question_only"
    )


def build_samples(dataset, tokenizer, dataset_id: str, split: str) -> list[dict]:
    samples = []
    for idx, example in enumerate(dataset):
        subject = str(example["subject"])
        question = str(example["question"])
        choices = [str(choice) for choice in example["choices"]]
        answer_index = int(example["answer_index"])
        answer_letter = str(example["answer_letter"])
        answer_text = str(example["answer_text"])

        samples.append(
            {
                "cloze_id": f"cloze_{idx:04d}",
                "group_id": f"group_{idx:04d}",
                "prefix": apply_chat_template(question, tokenizer),
                "target": answer_text,
                "category": subject,
                "subject": subject,
                "original_id": f"{subject}:{int(example['source_row']):04d}",
                "question": question,
                "cloze": question,
                "cloze_type": "main",
                "mode": "question",
                "choices": choices,
                "answer_index": answer_index,
                "answer_letter": answer_letter,
                "answer_text": answer_text,
                "source_dataset": dataset_id,
                "source_split": split,
            }
        )
    return samples


def build_output_payload(
    dataset_id: str,
    split: str,
    subjects: List[str],
    raw_save_dir: Path,
    model: str,
    samples: Iterable[dict],
) -> dict:
    sample_list = list(samples)

    return {
        "metadata": {
            "dataset_id": dataset_id,
            "source_dir": str(raw_save_dir),
            "split": split,
            "subjects": subjects,
            "mode": "question",
            "prompt_style": "question_only",
            "total_groups": len(sample_list),
            "selected_groups": len(sample_list),
            "total_samples": len(sample_list),
            "random_seed": None,
            "model": model,
        },
        "clozes": sample_list,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare selected MMLU questions for latent_planning")
    parser.add_argument("--dataset-id", type=str, default="cais/mmlu")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument(
        "--subjects",
        type=str,
        default=",".join(DEFAULT_SUBJECTS),
        help="Comma-separated MMLU subjects",
    )
    parser.add_argument("--model", type=str, required=True, help="Tokenizer model used for prompt rendering")
    parser.add_argument("--raw-save-dir", type=Path, required=True, help="Merged local MMLU snapshot directory")
    parser.add_argument("--output", type=Path, required=True, help="Pipeline-ready test_clozes.json path")
    parser.add_argument("--log-dir", type=Path, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects)

    log_dir = args.log_dir or (Path(__file__).resolve().parents[1] / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    import logging

    logger = setup_logger(
        "prepare_mmlu_questions",
        log_file=log_dir / "prepare_mmlu_questions.log",
        level=logging.WARNING if args.quiet else logging.INFO,
    )

    logger.info("=" * 60)
    logger.info("PREPARE MMLU QUESTIONS")
    logger.info("=" * 60)
    logger.info("Dataset: %s", args.dataset_id)
    logger.info("Split: %s", args.split)
    logger.info("Subjects: %s", ", ".join(subjects))
    logger.info("Model: %s", args.model)
    logger.info("Snapshot dir: %s", args.raw_save_dir)
    logger.info("Output: %s", args.output)

    if args.skip_existing and output_matches(args.output, args.dataset_id, args.split, subjects, args.model):
        logger.info("Existing output matches requested dataset/model; skipping regeneration")
        return

    dataset = load_selected_dataset(
        dataset_id=args.dataset_id,
        split=args.split,
        subjects=subjects,
        raw_save_dir=args.raw_save_dir,
        logger=logger,
    )

    logger.info("Loading tokenizer for prompt rendering...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    logger.info("Tokenizer loaded")

    logger.info("Rendering question-only prompts...")
    samples = build_samples(
        dataset,
        tokenizer,
        dataset_id=args.dataset_id,
        split=args.split,
    )

    payload = build_output_payload(
        dataset_id=args.dataset_id,
        split=args.split,
        subjects=subjects,
        raw_save_dir=args.raw_save_dir,
        model=args.model,
        samples=samples,
    )
    save_json(payload, args.output)

    logger.info("Wrote %d samples to %s", len(samples), args.output)
    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
