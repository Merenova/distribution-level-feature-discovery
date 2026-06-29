#!/usr/bin/env -S uv run python
"""Prepare walledai/HarmBench for latent_planning Stage 1.

The pipeline's Stage 1 expects `data.cloze_dir` to be a HuggingFace
DatasetDict loaded by `datasets.load_from_disk`. This script saves raw
HarmBench behavior prompts in that format. It can also emit a legacy
pipeline-shaped JSON sample for quick manual inspection when `--output` is
provided.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, List

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.data_utils import save_json
from utils.logging_utils import setup_logger


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


def _first_nonempty(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def build_dataset_rows(
    rows: List[dict],
    prompt_col: str,
    category_col: str | None,
    id_col: str | None,
    source_split: str,
) -> list[dict]:
    out = []
    for idx, row in enumerate(rows):
        question = str(row[prompt_col])
        category = str(row[category_col]) if category_col and category_col in row else "unknown"
        original_id = str(row[id_col]) if id_col and id_col in row else f"harmbench_{idx:04d}"
        out.append(
            {
                "id": original_id,
                "question": question,
                "prompt": question,
                "category": category,
                "source_dataset": "walledai/HarmBench",
                "source_split": source_split,
            }
        )
    return out


def build_json_samples(rows: list[dict], tokenizer) -> list[dict]:
    samples = []
    for idx, row in enumerate(rows):
        question = row["question"]
        category = row["category"]
        samples.append(
            {
                "cloze_id": f"cloze_{idx:04d}",
                "group_id": f"group_{idx:04d}",
                "prefix": apply_chat_template(question, tokenizer),
                "target": "",
                "category": category,
                "subject": category,
                "original_id": row["id"],
                "question": question,
                "cloze": question,
                "cloze_type": "main",
                "mode": "question",
                "choices": [],
                "answer_index": -1,
                "answer_letter": "",
                "answer_text": "",
                "source_dataset": row["source_dataset"],
                "source_split": row["source_split"],
            }
        )
    return samples


def main():
    p = argparse.ArgumentParser(description="Prepare walledai/HarmBench prompts for latent_planning")
    p.add_argument("--dataset-id", default="walledai/HarmBench")
    p.add_argument("--config-name", default="standard")
    p.add_argument("--split", default="train")
    p.add_argument("--save-dir", type=Path, default=None, help="DatasetDict output directory for Stage 1")
    p.add_argument("--output", type=Path, default=None, help="Optional legacy test_clozes.json output")
    p.add_argument("--model", default=None, help="HF model id for optional JSON chat template")
    p.add_argument("--n-samples", type=int, default=None, help="Subsample to N rows before saving")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt-col", default=None, help="Column holding prompt text")
    p.add_argument("--category-col", default=None, help="Column holding category")
    p.add_argument("--id-col", default=None, help="Column holding behavior id")
    args = p.parse_args()

    if args.save_dir is None and args.output is None:
        raise SystemExit("Provide --save-dir for Stage 1 data, --output for JSON, or both")
    if args.output is not None and args.model is None:
        raise SystemExit("--output requires --model so the JSON prompt uses the right chat template")

    log = setup_logger("prepare_harmbench")
    log.info("Loading %s/%s split=%s", args.dataset_id, args.config_name, args.split)
    ds = load_dataset(args.dataset_id, args.config_name, split=args.split)
    log.info("Loaded %d rows; columns=%s", len(ds), ds.column_names)

    prompt_col = args.prompt_col or next(
        (c for c in ["prompt", "behavior", "Behavior", "question"] if c in ds.column_names),
        None,
    )
    if prompt_col is None:
        raise SystemExit(f"Cannot find prompt column in {ds.column_names}")

    category_col = args.category_col or next(
        (
            c
            for c in [
                "category",
                "FunctionalCategory",
                "functional_category",
                "SemanticCategory",
            ]
            if c in ds.column_names
        ),
        None,
    )
    id_col = args.id_col or next((c for c in ["BehaviorID", "id"] if c in ds.column_names), None)
    log.info("prompt_col=%s category_col=%s id_col=%s", prompt_col, category_col, id_col)

    source_rows = list(ds)
    if args.n_samples is not None and args.n_samples < len(source_rows):
        random.seed(args.seed)
        source_rows = random.sample(source_rows, args.n_samples)
        log.info("Subsampled to %d rows (seed=%d)", len(source_rows), args.seed)

    rows = build_dataset_rows(source_rows, prompt_col, category_col, id_col, args.split)
    metadata = {
        "dataset_id": args.dataset_id,
        "config_name": args.config_name,
        "split": args.split,
        "total_samples": len(rows),
        "random_seed": args.seed,
    }

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        DatasetDict({args.split: Dataset.from_list(rows)}).save_to_disk(str(args.save_dir))
        save_json(metadata, args.save_dir / "metadata.json")
        log.info("Saved %d rows as DatasetDict to %s", len(rows), args.save_dir)

    if args.output is not None:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        samples = build_json_samples(rows, tok)
        payload = {
            "metadata": {
                **metadata,
                "mode": "question",
                "prompt_style": "question_only",
                "model": args.model,
            },
            "clozes": samples,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_json(payload, args.output)
        log.info("Wrote %d JSON samples to %s", len(samples), args.output)


if __name__ == "__main__":
    main()
