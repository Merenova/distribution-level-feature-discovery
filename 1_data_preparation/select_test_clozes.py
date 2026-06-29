#!/usr/bin/env -S uv run python
"""Select a subset of test clozes/questions from knowledge_attribution dataset.

This script selects a random subset of clozes for testing the Gaussian
optimization algorithm. It keeps main and sub-clozes together as groups.

Supports two modes:
- cloze: Uses prefix completion format (e.g., "The capital of France is ____.")
- question: Uses chat template format with the question

Features:
- Groups main + sub clozes together (never splits them)
- Keeps all sub-clozes for each selected group
- Supports both cloze and question input formats
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data_utils import save_json
from utils.logging_utils import setup_logger
from utils.manifest import save_manifest


def extract_cloze_groups(split_data: List[Dict], quiet: bool = False) -> List[Dict]:
    """
    Extract cloze groups from dataset, keeping main and sub-clozes together.

    Args:
        split_data: Raw dataset items
        quiet: If True, show progress bar

    Returns:
        List of cloze groups, each containing main cloze and sub-clozes
    """
    cloze_groups = []
    
    iterator = split_data
    if quiet:
        from tqdm import tqdm
        iterator = tqdm(split_data, desc="Extracting groups")

    for item in iterator:
        # Extract annotations
        annotations = item.get("annotations", {})
        answers = annotations.get("answer", [[]])
        main_target = answers[0][0] if answers and answers[0] else ""

        # Extract category
        category = item.get("question_type", "unknown")

        # Extract main question
        main_question = item.get("question", "")

        # Extract sub-questions and sub-answers from annotations.qaPairs[0]
        qa_pairs = annotations.get("qaPairs", [{}])
        sub_questions = []
        sub_answers = []
        if qa_pairs and qa_pairs[0]:
            sub_questions = qa_pairs[0].get("question", []) or []
            sub_answers = qa_pairs[0].get("answer", []) or []

        # Extract clozes from final_filter
        final_filter = item.get("final_filter", {})
        main_cloze = final_filter.get("main_cloze", "")
        sub_clozes = final_filter.get("sub_clozes", [])

        # Skip if no main cloze
        if not main_cloze or "____" not in main_cloze:
            continue

        # Build main entry
        main_prefix = main_cloze[:main_cloze.find("____")].strip()

        group = {
            "original_id": item.get("id", ""),
            "category": category,
            "main": {
                "question": main_question,
                "cloze": main_cloze,
                "prefix": main_prefix,
                "target": main_target,
            },
            "subs": []
        }

        # Build sub entries
        for sub_idx in range(len(sub_clozes)):
            sub_cloze = sub_clozes[sub_idx]
            if not sub_cloze or "____" not in sub_cloze:
                continue

            sub_prefix = sub_cloze[:sub_cloze.find("____")].strip()

            # Get sub-question if available
            sub_question = sub_questions[sub_idx] if sub_idx < len(sub_questions) else main_question

            # Get sub-answer if available
            sub_target = main_target
            if sub_idx < len(sub_answers) and sub_answers[sub_idx]:
                sub_target = sub_answers[sub_idx][0] if isinstance(sub_answers[sub_idx], list) else sub_answers[sub_idx]

            group["subs"].append({
                "question": sub_question,
                "cloze": sub_cloze,
                "prefix": sub_prefix,
                "target": sub_target,
            })

        cloze_groups.append(group)

    return cloze_groups


def _first_nonempty(item: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        value = item.get(key)
        if value is not None and value != "":
            return value
    return None


def extract_question_groups(split_data: List[Dict], quiet: bool = False) -> List[Dict]:
    """Extract one-question groups from MMLU/HarmBench-style datasets.

    These datasets do not have AmbigQA's `final_filter.main_cloze` structure.
    In question mode we treat each row as a single main question with no sub
    clozes and let `flatten_groups_to_samples` apply the target model's chat
    template.
    """
    question_groups = []

    iterator = split_data
    if quiet:
        from tqdm import tqdm
        iterator = tqdm(split_data, desc="Extracting question rows")

    for idx, item in enumerate(iterator):
        question = _first_nonempty(item, ["question", "prompt", "behavior", "Behavior"])
        if question is None:
            continue

        category = _first_nonempty(
            item,
            [
                "subject",
                "category",
                "FunctionalCategory",
                "functional_category",
                "SemanticCategory",
                "question_type",
            ],
        ) or "unknown"

        original_id = _first_nonempty(item, ["id", "BehaviorID", "original_id"])
        if original_id is None:
            if "subject" in item and "source_row" in item:
                original_id = f"{item['subject']}_{item['source_row']}"
            else:
                original_id = f"question_{idx:04d}"

        target = _first_nonempty(item, ["answer_text", "target", "answer"]) or ""

        question_groups.append(
            {
                "original_id": str(original_id),
                "category": str(category),
                "main": {
                    "question": str(question),
                    "cloze": str(question),
                    "prefix": str(question),
                    "target": str(target),
                },
                "subs": [],
            }
        )

    return question_groups


def flatten_groups_to_samples(
    groups: List[Dict],
    mode: str = "cloze",
    tokenizer=None,
) -> List[Dict]:
    """
    Flatten cloze groups into individual samples.

    Args:
        groups: List of cloze groups
        mode: "cloze" or "question"
        tokenizer: Tokenizer for chat template (required for question mode)

    Returns:
        List of individual samples ready for the pipeline
    """
    samples = []
    sample_counter = 0

    for group in groups:
        original_id = group["original_id"]
        category = group["category"]
        main = group["main"]
        subs = group["subs"]

        # Create group_id for tracking related samples
        group_id = f"group_{len(samples):04d}"

        # Process main cloze/question
        if mode == "cloze":
            prefix = main["prefix"]
        else:  # question mode
            prefix = apply_chat_template(main["question"], tokenizer)

        samples.append({
            "cloze_id": f"cloze_{sample_counter:04d}",
            "group_id": group_id,
            "prefix": prefix,
            "target": main["target"],
            "category": category,
            "original_id": original_id,
            "question": main["question"],
            "cloze": main["cloze"],
            "cloze_type": "main",
            "mode": mode,
        })
        sample_counter += 1

        # Process sub clozes/questions
        for sub_idx, sub in enumerate(subs):
            if mode == "cloze":
                prefix = sub["prefix"]
            else:  # question mode
                prefix = apply_chat_template(sub["question"], tokenizer)

            samples.append({
                "cloze_id": f"cloze_{sample_counter:04d}",
                "group_id": group_id,
                "prefix": prefix,
                "target": sub["target"],
                "category": category,
                "original_id": original_id,
                "question": sub["question"],
                "main_question": main["question"],
                "cloze": sub["cloze"],
                "cloze_type": f"sub_{sub_idx + 1}",
                "mode": mode,
            })
            sample_counter += 1

    return samples


def apply_chat_template(question: str, tokenizer) -> str:
    """
    Apply chat template to a question.

    Args:
        question: The question text
        tokenizer: HuggingFace tokenizer with chat template

    Returns:
        Formatted prompt string
    """
    messages = [{"role": "user", "content": question}]

    # Try with Qwen-specific enable_thinking param, fall back to standard
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Qwen-specific: disable thinking mode
        )
    except TypeError:
        # Other models don't have enable_thinking parameter
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    return text


def load_tokenizer(model_name: str):
    """Load tokenizer for chat template."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    return tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Select test clozes/questions for Gaussian optimization"
    )
    parser.add_argument(
        "--cloze-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/cloze_llm_improved_split_ratio_0.1",
        help="Path to source cloze dataset directory (HuggingFace format)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (train/test/validation)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of question groups to select (each group has main + sub clozes)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["cloze", "question"],
        default="cloze",
        help="Input mode: 'cloze' (prefix completion) or 'question' (chat template)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model name for tokenizer (used in question mode)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: results/test_clozes.json)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selection"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode (only progress bars)"
    )
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.output is None:
        args.output = results_dir / "test_clozes.json"

    # Setup logger
    import logging
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_level = logging.WARNING if args.quiet else logging.INFO
    logger = setup_logger(
        "data_preparation",
        log_file=logs_dir / "select_clozes.log",
        level=log_level
    )

    logger.info("=" * 60)
    logger.info("TEST CLOZE/QUESTION SELECTION")
    logger.info("=" * 60)
    logger.info(f"Source cloze directory: {args.cloze_dir}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Number of groups: {args.n_samples}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Output file: {args.output}")

    # Check if source directory exists
    if not args.cloze_dir.exists():
        logger.error(f"Cloze directory not found: {args.cloze_dir}")
        sys.exit(1)

    # Load tokenizer if question mode
    tokenizer = None
    if args.mode == "question":
        logger.info(f"Loading tokenizer for model: {args.model}")
        tokenizer = load_tokenizer(args.model)
        logger.info("Tokenizer loaded")

    # Load cloze data
    logger.info("Loading cloze data...")

    try:
        from datasets import load_from_disk
        dataset = load_from_disk(str(args.cloze_dir))

        if args.split not in dataset:
            logger.error(f"Split '{args.split}' not found. Available: {list(dataset.keys())}")
            sys.exit(1)

        split_dataset = dataset[args.split]
        split_data = [dict(item) for item in split_dataset]
        logger.info(f"Loaded {len(split_data)} items from dataset")

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        sys.exit(1)

    # Extract cloze groups (keeping main + subs together)
    logger.info("Extracting cloze groups...")
    cloze_groups = extract_cloze_groups(split_data, quiet=args.quiet)
    if not cloze_groups and args.mode == "question":
        logger.info("No AmbigQA-style cloze groups found; trying question-row extraction")
        cloze_groups = extract_question_groups(split_data, quiet=args.quiet)
    logger.info(f"Found {len(cloze_groups)} cloze/question groups")

    # Count total clozes
    total_clozes = sum(1 + len(g["subs"]) for g in cloze_groups)
    logger.info(f"Total clozes (main + sub): {total_clozes}")

    # Select random subset of groups
    random.seed(args.seed)
    if args.n_samples >= len(cloze_groups):
        selected_groups = cloze_groups
        logger.info(f"Selecting all {len(cloze_groups)} groups (requested {args.n_samples})")
    else:
        selected_groups = random.sample(cloze_groups, args.n_samples)
        logger.info(f"Randomly selected {len(selected_groups)} groups")

    # Flatten groups into individual samples
    logger.info(f"Flattening groups to samples (mode={args.mode})...")
    samples = flatten_groups_to_samples(selected_groups, mode=args.mode, tokenizer=tokenizer)
    logger.info(f"Created {len(samples)} individual samples")

    # Build output
    output_data = {
        "metadata": {
            "source_dir": str(args.cloze_dir),
            "split": args.split,
            "mode": args.mode,
            "total_groups": len(cloze_groups),
            "selected_groups": len(selected_groups),
            "total_samples": len(samples),
            "random_seed": args.seed,
            "model": args.model if args.mode == "question" else None,
        },
        "clozes": samples
    }

    # Save output
    logger.info(f"Saving to {args.output}...")
    save_json(output_data, args.output)

    # Write manifest with all selected cloze IDs
    cloze_ids = [c["cloze_id"] for c in samples]
    save_manifest(results_dir, "stage1", completed=cloze_ids)
    logger.info(f"Wrote manifest_stage1.json with {len(cloze_ids)} cloze IDs")

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Test clozes saved to: {args.output}")
    logger.info(f"Number of groups: {len(selected_groups)}")
    logger.info(f"Number of samples: {len(samples)}")

    # Print first few examples
    logger.info("\nFirst 3 selected samples:")
    for i, sample in enumerate(samples[:3]):
        logger.info(f"\n{i+1}. ID: {sample['cloze_id']}, Type: {sample['cloze_type']}")
        logger.info(f"   Question: {sample['question'][:80]}...")
        logger.info(f"   Prefix: {sample['prefix'][:80]}...")
        logger.info(f"   Target: {sample['target']}")


if __name__ == "__main__":
    main()
