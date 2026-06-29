#!/usr/bin/env -S uv run python
"""Format grouped AmbigQA questions into Stage-2 prefix entries."""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data_utils import load_json, save_json
from utils.logging_utils import setup_logger


def load_grouped_questions(grouped_questions_path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = load_json(grouped_questions_path)
    if isinstance(raw, dict) and isinstance(raw.get("groups"), list):
        return raw["groups"], raw.get("metadata", {})
    if isinstance(raw, list):
        return raw, {}
    raise ValueError(f"Unsupported grouped-question format in {grouped_questions_path}")


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


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


def select_groups(groups: List[Dict[str, Any]], n_groups: int, seed: int) -> List[Dict[str, Any]]:
    if n_groups <= 0:
        raise ValueError("--n-groups must be positive")
    if n_groups >= len(groups):
        return groups

    rng = random.Random(seed)
    selected_indices = rng.sample(range(len(groups)), n_groups)
    return [groups[idx] for idx in selected_indices]


def build_prefix_payload(groups: List[Dict[str, Any]], tokenizer, logger) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    prefixes: List[Dict[str, str]] = []
    metadata_entries: List[Dict[str, Any]] = []
    prefix_counter = 0

    for group in groups:
        group_id = str(group.get("group_id", ""))
        original_id = str(group.get("original_id", ""))
        question_type = str(group.get("question_type", ""))

        main = group.get("main", {}) or {}
        main_question = str(main.get("question", "")).strip()
        if main_question:
            prefix_id = f"cloze_{prefix_counter:04d}"
            prefixes.append(
                {
                    "prefix_id": prefix_id,
                    "prefix": apply_chat_template(main_question, tokenizer),
                }
            )
            metadata_entries.append(
                {
                    "prefix_id": prefix_id,
                    "group_id": group_id,
                    "original_id": original_id,
                    "question_role": "main",
                    "question_index": 0,
                    "question_type": question_type,
                    "question": main_question,
                    "main_question": main_question,
                    "answers": main.get("answers", []),
                }
            )
            prefix_counter += 1

        subs = group.get("subs", []) or []
        for sub_idx, sub in enumerate(subs, start=1):
            question = str(sub.get("question", "")).strip()
            if not question:
                continue
            prefix_id = f"cloze_{prefix_counter:04d}"
            prefixes.append(
                {
                    "prefix_id": prefix_id,
                    "prefix": apply_chat_template(question, tokenizer),
                }
            )
            metadata_entries.append(
                {
                    "prefix_id": prefix_id,
                    "group_id": group_id,
                    "original_id": original_id,
                    "question_role": f"sub_{sub_idx}",
                    "question_index": sub_idx,
                    "question_type": question_type,
                    "question": question,
                    "main_question": main_question,
                    "answers": sub.get("answers", []),
                }
            )
            prefix_counter += 1

    logger.info(f"Built {len(prefixes)} prefix entries from {len(groups)} groups")
    return prefixes, {"entries": metadata_entries}


def main() -> None:
    parser = argparse.ArgumentParser(description="Format grouped AmbigQA questions into Stage-2 prefix entries")
    parser.add_argument("--grouped-questions", type=Path, required=True, help="Grouped question JSON from Stage 0")
    parser.add_argument("--model", type=str, required=True, help="Model name used for chat template formatting")
    parser.add_argument("--n-groups", type=int, required=True, help="Number of question groups to select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for group selection")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results/1_data_preparation",
        help="Output directory for prefixes.json and prefix_metadata.json",
    )
    parser.add_argument("--log-dir", type=Path, default=None, help="Optional log directory")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    log_file = args.log_dir / "format_ambigqa_questions.log" if args.log_dir else None
    logger = setup_logger("format_ambigqa_questions", log_file=log_file, level=log_level)

    logger.info("=" * 60)
    logger.info("AMBIGQA QUESTION FORMATTING")
    logger.info("=" * 60)
    logger.info(f"Grouped questions: {args.grouped_questions}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Requested groups: {args.n_groups}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Output directory: {args.output_dir}")

    groups, source_metadata = load_grouped_questions(args.grouped_questions)
    selected_groups = select_groups(groups, args.n_groups, args.seed)
    tokenizer = load_tokenizer(args.model)
    prefixes, metadata_payload = build_prefix_payload(selected_groups, tokenizer, logger)

    prefixes_path = args.output_dir / "prefixes.json"
    metadata_path = args.output_dir / "prefix_metadata.json"

    save_json(prefixes, prefixes_path)
    save_json(
        {
            "metadata": {
                "source_grouped_questions": str(args.grouped_questions),
                "source_metadata": source_metadata,
                "model": args.model,
                "random_seed": args.seed,
                "n_available_groups": len(groups),
                "n_selected_groups": len(selected_groups),
                "n_prefixes": len(prefixes),
            },
            "entries": metadata_payload["entries"],
        },
        metadata_path,
    )

    logger.info(f"Saved prefixes to: {prefixes_path}")
    logger.info(f"Saved prefix metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
