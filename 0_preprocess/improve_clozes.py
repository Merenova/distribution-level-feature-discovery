#!/usr/bin/env -S uv run python
"""
Rule-based cloze improvement script.

This script:
1. Improves existing clozes by adding appropriate prepositions/functional words
2. Generates clozes for sub-questions with minimal changes from original clozes

Adapted from knowledge_attribution/scripts/1_data_generation/improve_clozes.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_from_disk, Dataset, DatasetDict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logging_utils import setup_logger


def add_preposition_to_cloze(cloze: str, question: str, question_type: str, answers: List[str]) -> str:
    """
    Add appropriate prepositions or functional words to make cloze more natural.

    Args:
        cloze: Original cloze sentence
        question: Original question
        question_type: Type of question (who, what, when, where, how, why)
        answers: List of possible answers

    Returns:
        Improved cloze with appropriate prepositions
    """
    # Remove the placeholder to work with the base
    base_cloze = cloze.replace("____.", "").replace("____", "").rstrip()

    # Check if a preposition is already present
    prepositions = ['in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'during', 'since', 'until']
    last_word = base_cloze.split()[-1] if base_cloze.split() else ""

    # Rules for adding prepositions based on question type and context
    if question_type == "when":
        if not any(base_cloze.endswith(f" {prep}") for prep in ['in', 'on', 'at', 'during', 'since', 'until', 'by', 'from']):
            sample_answer = answers[0] if answers else ""

            if re.match(r'^\d{4}$', sample_answer):
                base_cloze += " in"
            elif re.match(r'^(january|february|march|april|may|june|july|august|september|october|november|december)', sample_answer.lower()):
                base_cloze += " on" if ',' in sample_answer else " in"
            elif 'began' in base_cloze or 'started' in base_cloze or 'begin' in base_cloze:
                if 'began and ended' not in base_cloze and 'begin and end' not in base_cloze:
                    base_cloze += " in"
            elif not base_cloze.endswith(('was', 'were', 'is', 'are')):
                base_cloze += " on"

    elif question_type == "where":
        if not any(base_cloze.endswith(f" {prep}") for prep in ['in', 'at', 'on', 'from', 'to', 'near', 'by']):
            if 'heard' in base_cloze or 'held' in base_cloze or 'located' in base_cloze:
                base_cloze += " in"
            elif 'from' not in base_cloze and 'come from' not in question.lower():
                base_cloze += " in"

    elif question_type == "who":
        if base_cloze.endswith(" by"):
            pass
        elif 'by' in question.lower() and not base_cloze.endswith(" by"):
            base_cloze += " by"

    elif question_type == "what":
        if answers and not base_cloze.endswith(('is', 'are', 'was', 'were', 'be')):
            sample_answer = answers[0]
            if not sample_answer.lower().startswith(('a ', 'an ', 'the ')):
                if 'color' in question.lower() or 'colour' in question.lower():
                    pass
                elif base_cloze.endswith('called'):
                    pass

    elif question_type == "how":
        if 'how many' in question.lower() or 'how much' in question.lower():
            pass
        elif 'how long' in question.lower():
            if not base_cloze.endswith('for'):
                base_cloze += " for"

    return base_cloze + " ____."


def find_diff_context(orig_words: List[str], sub_words: List[str]) -> Tuple[List[str], str]:
    """Find the additional context that distinguishes the sub-question from the original."""
    prefix_len = 0
    for i in range(min(len(orig_words), len(sub_words))):
        if orig_words[i] == sub_words[i]:
            prefix_len += 1
        else:
            break

    suffix_len = 0
    for i in range(1, min(len(orig_words) - prefix_len, len(sub_words) - prefix_len) + 1):
        if orig_words[-i] == sub_words[-i]:
            suffix_len += 1
        else:
            break

    orig_middle = orig_words[prefix_len:len(orig_words) - suffix_len if suffix_len > 0 else None]
    sub_middle = sub_words[prefix_len:len(sub_words) - suffix_len if suffix_len > 0 else None]

    if len(sub_middle) > len(orig_middle):
        additional = []
        insertion_type = 'phrase'

        if 'as' in sub_middle and 'as' not in orig_middle:
            as_idx = sub_middle.index('as')
            additional = sub_middle[as_idx:]
            insertion_type = 'phrase'
        elif len(sub_middle) == len(orig_middle) + 1:
            for i in range(len(sub_middle)):
                if i >= len(orig_middle) or sub_middle[i] != orig_middle[i]:
                    additional = [sub_middle[i]]
                    insertion_type = 'modifier'
                    break
        else:
            additional = sub_middle
            insertion_type = 'clause'

        return additional, insertion_type
    elif sub_middle != orig_middle:
        return sub_middle, 'replacement'

    return [], 'none'


def generate_sub_question_cloze(original_question: str, sub_question: str, original_cloze: str,
                                 question_type: str, answers: List[str]) -> str:
    """Generate cloze for sub-question based on the difference from original question."""
    orig_lower = original_question.lower().rstrip('?').strip()
    sub_lower = sub_question.lower().rstrip('?').strip()

    base_cloze = original_cloze.replace("____.", "").replace("____", "").strip()

    prepositions = ['on', 'in', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'during', 'since', 'until']
    cloze_words = base_cloze.split()
    if cloze_words and cloze_words[-1] in prepositions:
        base_cloze = ' '.join(cloze_words[:-1])
        cloze_words = cloze_words[:-1]

    orig_words = orig_lower.split()
    sub_words = sub_lower.split()

    additional_context, insertion_type = find_diff_context(orig_words, sub_words)

    result_cloze = base_cloze

    if insertion_type == 'phrase' and additional_context:
        context_phrase = ' '.join(additional_context)
        result_cloze = result_cloze + ' ' + context_phrase

    elif insertion_type == 'modifier' and additional_context:
        modifier = additional_context[0]
        target_words = ['championship', 'title', 'award', 'medal', 'competition', 'game', 'match']
        inserted = False

        for target in target_words:
            for i, word in enumerate(cloze_words):
                if target in word.lower():
                    cloze_words.insert(i, modifier)
                    result_cloze = ' '.join(cloze_words)
                    inserted = True
                    break
            if inserted:
                break

        if not inserted:
            for i, word in enumerate(sub_words):
                if word == modifier:
                    insert_pos = min(i - 1, len(cloze_words) - 1)
                    if insert_pos > 0:
                        cloze_words.insert(insert_pos, modifier)
                        result_cloze = ' '.join(cloze_words)
                    break

    elif insertion_type == 'replacement' and additional_context:
        for i in range(min(len(orig_words), len(sub_words))):
            if orig_words[i] != sub_words[i]:
                old_word = orig_words[i]
                new_word = sub_words[i]

                pattern = re.compile(r'\b' + re.escape(old_word) + r'\b', re.IGNORECASE)

                def replace_preserve_case(match):
                    original = match.group()
                    if original.isupper():
                        return new_word.upper()
                    elif original[0].isupper():
                        return new_word.capitalize()
                    else:
                        return new_word

                result_cloze = pattern.sub(replace_preserve_case, result_cloze)
                break

    elif insertion_type == 'clause' and additional_context:
        clause = ' '.join(additional_context)
        if 'without' in clause or 'with' in clause:
            if len(cloze_words) > 3:
                cloze_words.insert(3, clause)
                result_cloze = ' '.join(cloze_words)
            else:
                result_cloze = result_cloze + ' ' + clause

    result_cloze = add_preposition_to_cloze(result_cloze + " ____.", sub_question, question_type, answers)
    result_cloze = re.sub(r'\s+', ' ', result_cloze).strip()
    result_cloze = result_cloze.replace(' ,', ',')

    if result_cloze:
        result_cloze = result_cloze[0].upper() + result_cloze[1:]

    if not result_cloze.endswith("____."):
        if result_cloze.endswith("____"):
            result_cloze += "."
        elif "____" not in result_cloze:
            result_cloze += " ____."

    return result_cloze


def process_dataset(input_path: str, output_path: str, logger=None, quiet: bool = False):
    """Process the dataset to improve clozes and add sub-question clozes."""
    if not quiet:
        if logger:
            logger.info(f"Loading dataset from {input_path}...")
        else:
            print(f"Loading dataset from {input_path}...")

    dataset = load_from_disk(input_path)

    def process_sample(sample):
        answer_freq = json.loads(sample['answer_frequencies'])
        answers = list(answer_freq.keys())

        original_cloze = sample['cloze']

        improved_cloze = add_preposition_to_cloze(
            sample['cloze'],
            sample['question'],
            sample['question_type'],
            answers
        )

        sub_clozes = []
        if sample['annotations']['qaPairs'] and sample['annotations']['qaPairs'][0]['question']:
            for i, sub_q in enumerate(sample['annotations']['qaPairs'][0]['question']):
                sub_answers = []
                if (sample['annotations']['qaPairs'][0]['answer'] and
                    i < len(sample['annotations']['qaPairs'][0]['answer'])):
                    sub_answers = sample['annotations']['qaPairs'][0]['answer'][i]

                if not sub_answers:
                    sub_answers = answers

                sub_cloze = generate_sub_question_cloze(
                    sample['question'],
                    sub_q,
                    original_cloze,
                    sample['question_type'],
                    sub_answers
                )
                sub_clozes.append(sub_cloze)

        return {
            **sample,
            'cloze_original': original_cloze,
            'cloze': improved_cloze,
            'sub_question_clozes': sub_clozes
        }

    if not quiet:
        if logger:
            logger.info("Processing samples...")
        else:
            print("Processing samples...")

    processed_dataset = {}

    for split in dataset.keys():
        if not quiet:
            if logger:
                logger.info(f"Processing {split} split...")
        processed_dataset[split] = dataset[split].map(
            process_sample,
            desc=f"Improving clozes in {split}"
        )

    processed_dataset = DatasetDict(processed_dataset)

    if not quiet:
        if logger:
            logger.info(f"Saving processed dataset to {output_path}...")
        else:
            print(f"Saving processed dataset to {output_path}...")

    processed_dataset.save_to_disk(output_path)

    if not quiet:
        if logger:
            logger.info("Done!")
        else:
            print("Done!")

    return processed_dataset


def main():
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="Improve clozes and generate sub-question clozes")
    parser.add_argument("--input", type=Path, required=True, help="Path to input dataset")
    parser.add_argument("--output", type=Path, required=True, help="Path to save processed dataset")
    parser.add_argument("--log-dir", type=Path, default=None, help="Log directory")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (only progress bars)")

    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logger = setup_logger("improve_clozes", 
                          log_file=args.log_dir / "improve_clozes.log" if args.log_dir else None,
                          level=log_level)

    process_dataset(str(args.input), str(args.output), logger, quiet=args.quiet)


if __name__ == "__main__":
    main()
