#!/usr/bin/env -S uv run python
"""
LLM-based cloze improvement script using Qwen model.

This script uses an LLM to improve clozes by ensuring they are natural and grammatically correct
while maintaining distinctness between main and sub-question clozes.

Adapted from knowledge_attribution/scripts/1_data_generation/llm_improve_clozes.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from datasets import load_from_disk, Dataset, DatasetDict
from vllm import LLM, SamplingParams

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logging_utils import setup_logger


class LLMClozeImprover:
    def __init__(self, model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
                 gpu_memory_utilization: float = 0.8):
        self.model_name = model_name
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm = None

    def load_model(self):
        """Load the vLLM model."""
        print(f"Loading vLLM model {self.model_name}...")
        self.llm = LLM(
            model=self.model_name,
            tensor_parallel_size=1,
            max_model_len=8192,
            dtype="auto",
            trust_remote_code=True,
            gpu_memory_utilization=self.gpu_memory_utilization
        )
        print("vLLM model loaded successfully!")

    def create_few_shot_examples(self) -> str:
        """Create few-shot examples from good samples."""
        good_examples = [
            {
                "question": "When did the simpsons first air on television?",
                "main_cloze": "The Simpsons first aired on television on ____.",
                "sub_questions": [
                    "When did the Simpsons first air on television as an animated short on the Tracey Ullman Show?",
                    "When did the Simpsons first air as a half-hour prime time show?"
                ],
                "sub_clozes": [
                    "The Simpsons first aired on television as an animated short on the Tracey Ullman Show on ____.",
                    "The Simpsons first aired as a half-hour prime time show on ____."
                ],
                "answers": ["december 17, 1989", "april 19, 1987"]
            },
            {
                "question": "When was the last time uga won a national championship?",
                "main_cloze": "The last time UGA won a national championship was in ____.",
                "sub_questions": [
                    "When was the last time UGA won a national football championship?",
                    "When was the last time UGA won a national gymnastics championship?"
                ],
                "sub_clozes": [
                    "The last time UGA won a national football championship was in ____.",
                    "The last time UGA won a national gymnastics championship was in ____."
                ],
                "answers": ["2019", "2005"]
            },
            {
                "question": "What color is a negative benedict's test?",
                "main_cloze": "A negative benedict's test is ____.",
                "sub_questions": [],
                "sub_clozes": [],
                "answers": ["blue"]
            }
        ]

        examples = []
        for i, example in enumerate(good_examples):
            examples.append(f"""Example {i+1}:
Question: {example['question']}
Answers: {', '.join(example['answers'])}
Sub-questions: {len(example['sub_questions'])} total
{chr(10).join(f'  {j+1}. {sq}' for j, sq in enumerate(example['sub_questions']))}

CLOZES:
Main: {example['main_cloze']}
{chr(10).join(f'Sub-{j+1}: {sc}' for j, sc in enumerate(example['sub_clozes']))}""")

        return "\n\n".join(examples)

    def create_prompt(self, question: str, answers: List[str], sub_questions: List[str],
                      current_main_cloze: str, current_sub_clozes: List[str]) -> str:
        """Create a prompt for the LLM to improve clozes."""
        few_shot_examples = self.create_few_shot_examples()

        prompt = f"""You are an expert at creating cloze sentences (fill-in-the-blank sentences) from questions. Your task is to improve existing clozes to make them more natural and grammatically correct.

CRITICAL RULES - FOLLOW EXACTLY:
1. The blank (____)  must ALWAYS be at the END of the sentence, followed by a period: "____."
2. NEVER include any part of the answer in the cloze sentence itself
3. Each cloze must be DISTINCT from others - no two clozes should be identical
4. Only use information present in the questions - do not add external context or dates/numbers
5. The sentence before the blank should be incomplete and require the answer to make sense
6. For sub-questions, incorporate the distinguishing context that makes them different from the main question

EXAMPLES OF WHAT NOT TO DO:
❌ "The Manhattan Project began in 1942 and ended ____." (contains answer "1942")
❌ "World War II ended ____." (blank not at end)
❌ "The war that ended on May 8, 1945 was ____." (contains specific date not in question)

EXAMPLES OF CORRECT FORMAT:
✅ "The Manhattan Project began and ended ____."
✅ "World War II ended in ____."
✅ "The Simpsons first aired on television as an animated short on ____."

{few_shot_examples}

Now improve the following clozes:

Question: {question}
Answers: {', '.join(answers)}
Sub-questions: {len(sub_questions)} total
{chr(10).join(f'  {i+1}. {sq}' for i, sq in enumerate(sub_questions))}

CURRENT CLOZES:
Main: {current_main_cloze}
{chr(10).join(f'Sub-{i+1}: {sc}' for i, sc in enumerate(current_sub_clozes))}

IMPROVED CLOZES:
Main: """

        return prompt

    def validate_cloze(self, cloze: str, answers: List[str]) -> Tuple[bool, List[str]]:
        """Validate a cloze sentence against the rules."""
        issues = []

        if not cloze.strip().endswith("____."):
            issues.append("blank_not_at_end")

        cloze_without_blank = cloze.replace("____", "").lower()

        common_words = {
            'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of',
            'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'shall',
            'when', 'where', 'what', 'who', 'why', 'how', 'which', 'that', 'this', 'these', 'those',
            'age', 'time', 'year', 'day', 'began', 'begin', 'end', 'ended', 'first', 'last', 'project',
            'war', 'world', 'states', 'usa', 'america', 'united'
        }

        for answer in answers:
            answer_lower = answer.lower()

            if answer_lower in cloze_without_blank:
                issues.append(f"answer_contained: {answer}")
                break

            answer_parts = answer_lower.replace(',', ' ').replace('-', ' ').split()
            for part in answer_parts:
                if (len(part) > 3 or part.isdigit()) and part not in common_words and part in cloze_without_blank:
                    if re.search(r'\b' + re.escape(part) + r'\b', cloze_without_blank):
                        issues.append(f"answer_part_contained: {part} from {answer}")
                        break
            if issues:
                break

        if not cloze.strip():
            issues.append("empty_cloze")

        if "____" not in cloze:
            issues.append("no_blank")

        return len(issues) == 0, issues

    def parse_llm_output(self, output: str, num_sub_clozes: int, answers: List[str],
                         original_main: str, original_subs: List[str]) -> Tuple[str, List[str]]:
        """Parse the LLM output to extract improved clozes with validation."""
        lines = output.strip().split('\n')

        main_cloze = ""
        sub_clozes = []

        for line in lines:
            line = line.strip()
            if line and not line.startswith('Sub-'):
                main_cloze = line
                break

        for line in lines:
            line = line.strip()
            if line.startswith('Sub-'):
                if ':' in line:
                    sub_cloze = line.split(':', 1)[1].strip()
                    if sub_cloze:
                        sub_clozes.append(sub_cloze)

        is_valid, issues = self.validate_cloze(main_cloze, answers)
        if not is_valid:
            main_cloze = original_main

        validated_sub_clozes = []
        for i, sub_cloze in enumerate(sub_clozes):
            if i < len(original_subs):
                is_valid, issues = self.validate_cloze(sub_cloze, answers)
                if is_valid:
                    validated_sub_clozes.append(sub_cloze)
                else:
                    validated_sub_clozes.append(original_subs[i])
            else:
                break

        while len(validated_sub_clozes) < num_sub_clozes:
            if len(validated_sub_clozes) < len(original_subs):
                validated_sub_clozes.append(original_subs[len(validated_sub_clozes)])
            else:
                validated_sub_clozes.append(main_cloze)

        validated_sub_clozes = validated_sub_clozes[:num_sub_clozes]

        return main_cloze, validated_sub_clozes

    def improve_clozes_batch(self, samples: List[Dict]) -> List[Dict]:
        """Improve clozes for a batch of samples."""
        prompts = []

        for sample in samples:
            answer_freq = json.loads(sample['answer_frequencies'])
            answers = list(answer_freq.keys())[:3]

            sub_questions = []
            if sample['annotations']['qaPairs'] and sample['annotations']['qaPairs'][0]['question']:
                sub_questions = sample['annotations']['qaPairs'][0]['question']

            current_main = sample['cloze']
            current_subs = sample['sub_question_clozes']

            prompt = self.create_prompt(
                sample['question'],
                answers,
                sub_questions,
                current_main,
                current_subs
            )
            prompts.append(prompt)

        sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=500,
            stop=["\n\nQuestion:", "Example "]
        )

        outputs = self.llm.generate(prompts, sampling_params)

        improved_samples = []
        for sample, output in zip(samples, outputs):
            generated_text = output.outputs[0].text.strip()

            answer_freq = json.loads(sample['answer_frequencies'])
            answers = list(answer_freq.keys())[:3]

            num_sub_clozes = len(sample['sub_question_clozes'])
            improved_main, improved_subs = self.parse_llm_output(
                generated_text,
                num_sub_clozes,
                answers,
                sample['cloze'],
                sample['sub_question_clozes']
            )

            improved_sample = {
                **sample,
                'cloze_llm_improved': improved_main,
                'sub_question_clozes_llm_improved': improved_subs,
                'llm_generation': generated_text,
                'final_filter': {
                    'main_cloze': improved_main,
                    'sub_clozes': improved_subs
                }
            }

            improved_samples.append(improved_sample)

        return improved_samples

    def process_dataset(self, input_path: str, output_path: str, max_samples: int = None,
                        batch_size: int = 16, logger=None, quiet: bool = False):
        """Process the entire dataset to improve clozes."""
        log = logger.info if logger else print
        
        if quiet:
            # No-op log for quiet mode
            def noop(*args, **kwargs): pass
            log = noop

        log(f"Loading dataset from {input_path}...")
        dataset = load_from_disk(input_path)

        train_samples = list(dataset['train'])
        if max_samples:
            train_samples = train_samples[:max_samples]

        log(f"Processing {len(train_samples)} samples...")

        improved_train_samples = []

        # Use tqdm for progress bar
        from tqdm import tqdm
        
        train_iter = range(0, len(train_samples), batch_size)
        if quiet:
            train_iter = tqdm(train_iter, desc="Processing Train", total=(len(train_samples) + batch_size - 1)//batch_size)

        for i in train_iter:
            batch = train_samples[i:i+batch_size]
            if not quiet:
                log(f"Processing batch {i//batch_size + 1}/{(len(train_samples) + batch_size - 1)//batch_size}")

            improved_batch = self.improve_clozes_batch(batch)
            improved_train_samples.extend(improved_batch)

        # Process validation split if exists
        improved_val_samples = []
        if 'validation' in dataset:
            val_samples = list(dataset['validation'])
            if max_samples:
                val_samples = val_samples[:min(max_samples//5, len(val_samples))]

            log(f"Processing {len(val_samples)} validation samples...")

            val_iter = range(0, len(val_samples), batch_size)
            if quiet:
                val_iter = tqdm(val_iter, desc="Processing Validation", total=(len(val_samples) + batch_size - 1)//batch_size)

            for i in val_iter:
                batch = val_samples[i:i+batch_size]
                if not quiet:
                    log(f"Processing validation batch {i//batch_size + 1}/{(len(val_samples) + batch_size - 1)//batch_size}")

                improved_batch = self.improve_clozes_batch(batch)
                improved_val_samples.extend(improved_batch)

        # Create new dataset
        dataset_dict = {'train': Dataset.from_list(improved_train_samples)}
        if improved_val_samples:
            dataset_dict['validation'] = Dataset.from_list(improved_val_samples)

        improved_dataset = DatasetDict(dataset_dict)

        log(f"Saving improved dataset to {output_path}...")
        improved_dataset.save_to_disk(output_path)
        log("Done!")

        return improved_dataset


def main():
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="Improve clozes using LLM")
    parser.add_argument("--input", type=Path, required=True, help="Path to input dataset")
    parser.add_argument("--output", type=Path, required=True, help="Path to save improved dataset")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
                        help="Model name for vLLM")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    parser.add_argument("--gpu-memory", type=float, default=0.8, help="GPU memory utilization")
    parser.add_argument("--log-dir", type=Path, default=None, help="Log directory")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    log_level = logging.WARNING if args.quiet else logging.INFO
    logger = setup_logger("llm_improve_clozes",
                          log_file=args.log_dir / "llm_improve_clozes.log" if args.log_dir else None,
                          level=log_level)

    improver = LLMClozeImprover(model_name=args.model, gpu_memory_utilization=args.gpu_memory)
    improver.load_model()
    improver.process_dataset(str(args.input), str(args.output), args.max_samples, args.batch_size, logger, quiet=args.quiet)


if __name__ == "__main__":
    main()
