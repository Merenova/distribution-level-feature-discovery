#!/usr/bin/env -S uv run python
"""Sample responses with vLLM and score them on GSM8K and MATH-500."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import re
import subprocess
import sys
import traceback
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import yaml
from datasets import load_dataset
from sympy import Eq, simplify, sympify
from sympy.parsing.latex import parse_latex
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Add project root to import path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data_utils import save_json
from utils.logging_utils import setup_logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "reasoning_qwen3_small.yaml"
GSM_INVALID_ANSWER = "[invalid]"
ANSWER_LINE_RE = re.compile(r"(?im)^Answer\s*:\s*(.+?)\s*$")
GSM_NUMERIC_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


@dataclass
class ExampleRecord:
    """Normalized example record for a benchmark item."""

    dataset_key: str
    example_id: str
    index: int
    prompt_question: str
    raw_reference: str
    metadata: Dict[str, Any]


def build_default_config() -> Dict[str, Any]:
    """Build the default config used for YAML/JSON merging."""
    return {
        "experiment_name": "reasoning_vllm_eval",
        "random_seed": 42,
        "global": {
            "samples_per_example": 128,
            "prompt_batch_size": 4,
            "max_examples": None,
            "output_root": "experiments/reasoning_runs",
        },
        "models": {
            "names": [
                "Qwen/Qwen3-0.6B",
                "Qwen/Qwen3-1.7B",
            ],
            "dtype": "bfloat16",
        },
        "datasets": {
            "gsm8k": {
                "enabled": True,
                "name": "openai/gsm8k",
                "config": "main",
                "split": "test",
            },
            "math500": {
                "enabled": True,
                "name": "HuggingFaceH4/MATH-500",
                "config": "default",
                "split": "test",
            },
        },
        "vllm": {
            "gpu_memory_utilization": 0.9,
            "tensor_parallel_size": 1,
            "max_model_len": 4096,
            "trust_remote_code": True,
        },
        "sampling": {
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 512,
            "logprobs": 1,
        },
        "prompting": {
            "gsm8k_final_answer_format": "#### <answer>",
            "math_final_answer_format": "Answer: <answer>",
        },
        "evaluation": {
            "gsm8k": {
                "extractor": "grade_school_math",
            },
            "math500": {
                "mode": "rule_based",
                "latex_parser": "sympy",
            },
        },
        "external": {
            "root_dir": "external/reasoning_eval",
            "auto_clone": True,
            "simple_evals_repo": "https://github.com/openai/simple-evals.git",
            "grade_school_math_repo": "https://github.com/openai/grade-school-math.git",
        },
    }


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge config dictionaries."""
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config_file(config_path: Path) -> Dict[str, Any]:
    """Load YAML or JSON config from disk."""
    suffix = config_path.suffix.lower()
    with open(config_path, "r", encoding="utf-8") as handle:
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(handle)
        elif suffix == ".json":
            data = json.load(handle)
        else:
            raise ValueError(f"Unsupported config type: {config_path}")
    return data or {}


def resolve_project_path(path_str: str) -> Path:
    """Resolve relative repo paths against the project root."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI overrides on top of the merged config."""
    merged = deepcopy(config)
    if args.models:
        merged["models"]["names"] = args.models

    if args.datasets:
        selected = set(args.datasets)
        for key in merged["datasets"]:
            merged["datasets"][key]["enabled"] = key in selected

    if args.samples_per_example is not None:
        merged["global"]["samples_per_example"] = args.samples_per_example
    if args.prompt_batch_size is not None:
        merged["global"]["prompt_batch_size"] = args.prompt_batch_size
    if args.max_examples is not None:
        merged["global"]["max_examples"] = args.max_examples
    if args.output_dir is not None:
        merged["global"]["output_root"] = str(Path(args.output_dir).resolve())
    if args.temperature is not None:
        merged["sampling"]["temperature"] = args.temperature
    if args.top_p is not None:
        merged["sampling"]["top_p"] = args.top_p
    if args.max_tokens is not None:
        merged["sampling"]["max_tokens"] = args.max_tokens
    if args.trust_remote_code is not None:
        merged["vllm"]["trust_remote_code"] = args.trust_remote_code
    return merged


def normalize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize config values and materialize project-relative paths."""
    normalized = deepcopy(config)
    normalized["global"]["output_root"] = str(
        resolve_project_path(normalized["global"]["output_root"]).resolve()
    )
    normalized["external"]["root_dir"] = str(
        resolve_project_path(normalized["external"]["root_dir"]).resolve()
    )

    if not normalized["models"]["names"]:
        raise ValueError("Config must contain at least one model name")

    enabled_datasets = [
        key for key, value in normalized["datasets"].items() if value.get("enabled", False)
    ]
    if not enabled_datasets:
        raise ValueError("Config must enable at least one dataset")

    samples_per_example = int(normalized["global"]["samples_per_example"])
    if samples_per_example <= 0:
        raise ValueError("global.samples_per_example must be > 0")
    prompt_batch_size = int(normalized["global"]["prompt_batch_size"])
    if prompt_batch_size <= 0:
        raise ValueError("global.prompt_batch_size must be > 0")

    return normalized


def dump_yaml(data: Dict[str, Any], path: Path) -> None:
    """Write YAML with stable ordering."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def slugify(value: str) -> str:
    """Build filesystem-safe names for models and datasets."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    slug = slug.strip("._")
    return slug or "item"


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    """Append one JSON record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def load_existing_example_ids(path: Path) -> set[str]:
    """Load processed example ids from an existing examples.jsonl file."""
    if not path.exists():
        return set()

    processed: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                processed.add(json.loads(line)["example_id"])
            except Exception:
                continue
    return processed


def ensure_repo_cloned(dest: Path, repo_url: str, auto_clone: bool, logger) -> Path:
    """Clone a helper repo if it is missing."""
    if dest.exists() and any(dest.iterdir()):
        return dest

    if not auto_clone:
        raise FileNotFoundError(
            f"Required external repo is missing: {dest}. "
            f"Set external.auto_clone=true or clone {repo_url} manually."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Cloning external repo: %s -> %s", repo_url, dest)
    subprocess.run(
        ["git", "clone", "--depth", "1", repo_url, str(dest)],
        check=True,
    )
    return dest


def load_grade_school_math_extractor(repo_root: Path, logger) -> Tuple[Callable[[str], str], str]:
    """Load grade-school-math's extract_answer function, or fall back identically."""
    dataset_path = repo_root / "grade_school_math" / "dataset.py"
    if dataset_path.exists():
        try:
            spec = importlib.util.spec_from_file_location("grade_school_math_dataset", dataset_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module.extract_answer, getattr(module, "INVALID_ANS", GSM_INVALID_ANSWER)
        except Exception as exc:
            logger.warning("Failed to import grade-school-math extractor: %s: %s", type(exc).__name__, exc)

    logger.warning("Falling back to local GSM8K extractor; external module was not importable")

    def fallback_extract_answer(completion: str) -> str:
        match = GSM_NUMERIC_RE.search(completion)
        if not match:
            return GSM_INVALID_ANSWER
        return match.group(1).strip().replace(",", "")

    return fallback_extract_answer, GSM_INVALID_ANSWER


def load_examples(dataset_key: str, dataset_cfg: Dict[str, Any], max_examples: Optional[int], logger) -> List[ExampleRecord]:
    """Load and normalize examples for a configured dataset."""
    dataset_name = dataset_cfg["name"]
    dataset_config = dataset_cfg.get("config")
    dataset_split = dataset_cfg["split"]

    logger.info("Loading dataset %s (%s, split=%s)", dataset_key, dataset_name, dataset_split)
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)
    else:
        dataset = load_dataset(dataset_name, split=dataset_split)

    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    examples: List[ExampleRecord] = []
    for idx, row in enumerate(dataset):
        if dataset_key == "gsm8k":
            example_id = f"{dataset_key}_{dataset_split}_{idx:05d}"
            examples.append(
                ExampleRecord(
                    dataset_key=dataset_key,
                    example_id=example_id,
                    index=idx,
                    prompt_question=row["question"],
                    raw_reference=row["answer"],
                    metadata={},
                )
            )
        elif dataset_key == "math500":
            example_id = row.get("unique_id") or f"{dataset_key}_{dataset_split}_{idx:05d}"
            examples.append(
                ExampleRecord(
                    dataset_key=dataset_key,
                    example_id=example_id,
                    index=idx,
                    prompt_question=row["problem"],
                    raw_reference=row["answer"],
                    metadata={
                        "solution": row.get("solution"),
                        "subject": row.get("subject"),
                        "level": row.get("level"),
                    },
                )
            )
        else:
            raise ValueError(f"Unsupported dataset key: {dataset_key}")

    logger.info("Loaded %d examples for %s", len(examples), dataset_key)
    return examples


def build_user_prompt(dataset_key: str, question: str, config: Dict[str, Any]) -> str:
    """Build the user-facing benchmark prompt."""
    prompting_cfg = config["prompting"]
    if dataset_key == "gsm8k":
        final_format = prompting_cfg["gsm8k_final_answer_format"]
        return (
            "Solve the following grade school math problem step by step. "
            f"The last line of your response should be exactly of the form {final_format}.\n\n"
            f"{question}"
        )

    if dataset_key == "math500":
        final_format = prompting_cfg["math_final_answer_format"]
        return (
            "Solve the following math problem step by step. "
            "The last line of your response should be of the form "
            f"{final_format} where <answer> is the final answer.\n\n"
            f"{question}\n\n"
            'Remember to put your answer on its own line after "Answer:", '
            "and you do not need to use a \\boxed command."
        )

    raise ValueError(f"Unsupported dataset key: {dataset_key}")


def apply_chat_template(tokenizer, prompt: str) -> str:
    """Apply the model's chat template without forcing empty think tags."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_think_blocks(text: str) -> List[str]:
    """Extract reasoning inside think tags when present."""
    return [match.group(1).strip() for match in re.finditer(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL)]


def extract_braced_content(text: str, open_brace_idx: int) -> Tuple[Optional[str], Optional[int]]:
    """Extract balanced content inside braces starting at a given index."""
    if open_brace_idx >= len(text) or text[open_brace_idx] != "{":
        return None, None

    depth = 0
    for idx in range(open_brace_idx, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_idx + 1 : idx], idx
    return None, None


def extract_last_boxed(text: str) -> Tuple[Optional[str], Optional[int]]:
    """Extract the last \\boxed{...} expression from text."""
    last_content = None
    last_start = None
    search_idx = 0
    while True:
        start = text.find("\\boxed", search_idx)
        if start == -1:
            break
        brace_start = text.find("{", start)
        if brace_start == -1:
            break
        content, brace_end = extract_braced_content(text, brace_start)
        if content is not None and brace_end is not None:
            last_content = content.strip()
            last_start = start
            search_idx = brace_end + 1
        else:
            search_idx = brace_start + 1
    return last_content, last_start


def extract_last_nonempty_line(text: str) -> Optional[str]:
    """Return the last non-empty line in a completion."""
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def extract_reasoning_text(raw_text: str, answer_start: Optional[int]) -> Optional[str]:
    """Extract reasoning, preferring think tags when they exist."""
    think_blocks = extract_think_blocks(raw_text)
    if think_blocks:
        joined = "\n\n".join(block for block in think_blocks if block)
        return joined.strip() or None

    if answer_start is not None:
        reasoning = raw_text[:answer_start].strip()
        return reasoning or None

    stripped = raw_text.strip()
    return stripped or None


def parse_gsm_response(raw_text: str, extract_answer: Callable[[str], str], invalid_answer: str) -> Dict[str, Any]:
    """Parse GSM8K reasoning and final answer from a completion."""
    answer = extract_answer(raw_text)
    marker_idx = raw_text.rfind("####")
    answer_text = None if answer == invalid_answer else answer
    return {
        "answer_text": answer_text,
        "answer_source": "grade_school_math" if answer_text is not None else None,
        "reasoning_text": extract_reasoning_text(raw_text, marker_idx if marker_idx >= 0 else None),
    }


def parse_math_response(raw_text: str) -> Dict[str, Any]:
    """Parse MATH reasoning and final answer from a completion."""
    answer_text = None
    answer_source = None
    answer_start = None

    matches = list(ANSWER_LINE_RE.finditer(raw_text))
    if matches:
        match = matches[-1]
        answer_text = match.group(1).strip()
        answer_source = "answer_line"
        answer_start = match.start()
    else:
        boxed_text, boxed_start = extract_last_boxed(raw_text)
        if boxed_text:
            answer_text = boxed_text
            answer_source = "boxed"
            answer_start = boxed_start
        else:
            answer_text = extract_last_nonempty_line(raw_text)
            answer_source = "last_line" if answer_text else None
            answer_start = None

    return {
        "answer_text": answer_text,
        "answer_source": answer_source,
        "reasoning_text": extract_reasoning_text(raw_text, answer_start),
    }


def strip_outer_math_delimiters(text: str) -> str:
    """Remove surrounding math-mode wrappers when they wrap the whole string."""
    stripped = text.strip()
    while True:
        changed = False
        if stripped.startswith("$") and stripped.endswith("$") and len(stripped) >= 2:
            stripped = stripped[1:-1].strip()
            changed = True
        elif stripped.startswith("\\(") and stripped.endswith("\\)"):
            stripped = stripped[2:-2].strip()
            changed = True
        elif stripped.startswith("\\[") and stripped.endswith("\\]"):
            stripped = stripped[2:-2].strip()
            changed = True
        if not changed:
            break
    return stripped


def unwrap_boxed_wrapper(text: str) -> str:
    """Remove a surrounding \\boxed{...} wrapper if present."""
    stripped = text.strip()
    if not stripped.startswith("\\boxed"):
        return stripped
    brace_idx = stripped.find("{")
    if brace_idx == -1:
        return stripped
    content, brace_end = extract_braced_content(stripped, brace_idx)
    if content is None or brace_end is None or brace_end != len(stripped) - 1:
        return stripped
    return content.strip()


def clean_math_text(text: Optional[str]) -> Optional[str]:
    """Normalize common formatting wrappers without destroying LaTeX structure."""
    if text is None:
        return None

    cleaned = text.replace("**", "").replace("`", "").strip()
    cleaned = re.sub(r"(?im)^Answer\s*:\s*", "", cleaned).strip()
    cleaned = strip_outer_math_delimiters(cleaned)

    prev = None
    while prev != cleaned:
        prev = cleaned
        cleaned = unwrap_boxed_wrapper(cleaned)
        cleaned = strip_outer_math_delimiters(cleaned)

    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    cleaned = cleaned.replace("\\!", "").replace("\\,", "")
    cleaned = cleaned.replace("\\;", "").replace("\\:", "")
    cleaned = cleaned.replace("\\$", "$")
    cleaned = cleaned.replace("\u2212", "-")
    cleaned = re.sub(r"\\mathrm\s*{([^{}]*)}", r"\1", cleaned)
    cleaned = re.sub(r"\\text\s*{([^{}]*)}", r"\1", cleaned)
    cleaned = re.sub(r"\\operatorname\s*{([^{}]*)}", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.rstrip(". ")
    return cleaned or None


def math_answer_key(text: Optional[str]) -> Optional[str]:
    """Canonical text key used for logging and majority vote bucketing."""
    cleaned = clean_math_text(text)
    if cleaned is None:
        return None
    compact = cleaned.replace(" ", "")
    compact = compact.replace("{", "").replace("}", "")
    return compact or None


def split_top_level(text: str, delimiter: str) -> List[str]:
    """Split on a delimiter while respecting bracket nesting."""
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)

        if char == delimiter and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue

        current.append(char)

    parts.append("".join(current).strip())
    return [part for part in parts if part]


def split_sequence(text: str) -> Tuple[Optional[str], Optional[List[str]]]:
    """Split a tuple/list style answer into elements."""
    cleaned = clean_math_text(text)
    if cleaned is None or len(cleaned) < 2:
        return None, None

    first = cleaned[0]
    last = cleaned[-1]
    matching = {"(": ")", "[": "]", "{": "}"}
    if first not in matching or matching[first] != last:
        return None, None

    inner = cleaned[1:-1].strip()
    parts = split_top_level(inner, ",")
    if len(parts) <= 1:
        return None, None

    return f"{first}{last}", parts


def split_equation(text: str) -> Optional[Tuple[str, str]]:
    """Split a top-level equation into lhs and rhs."""
    cleaned = clean_math_text(text)
    if cleaned is None:
        return None

    parts = split_top_level(cleaned, "=")
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def maybe_parse_latex(text: str):
    """Parse a math expression using SymPy's LaTeX parser."""
    return parse_latex(text)


def maybe_parse_sympy(text: str):
    """Parse a plain-text expression using sympy."""
    prepared = text.replace("^", "**")
    return sympify(prepared)


def parse_math_expression(text: str):
    """Try multiple parsers for a scalar expression."""
    cleaned = clean_math_text(text)
    if cleaned is None:
        raise ValueError("Cannot parse empty math text")

    errors: List[str] = []
    parsers = []
    if any(token in cleaned for token in ["\\", "{", "}", "^"]):
        parsers = [maybe_parse_latex, maybe_parse_sympy]
    else:
        parsers = [maybe_parse_sympy, maybe_parse_latex]

    for parser in parsers:
        try:
            return parser(cleaned)
        except Exception as exc:
            errors.append(f"{parser.__name__}: {type(exc).__name__}: {exc}")

    raise ValueError("; ".join(errors))


def compare_scalar_math(prediction: str, reference: str) -> Tuple[bool, Dict[str, Any]]:
    """Compare two scalar math answers."""
    pred_clean = clean_math_text(prediction)
    ref_clean = clean_math_text(reference)
    meta: Dict[str, Any] = {
        "normalized_prediction": pred_clean,
        "normalized_reference": ref_clean,
    }

    if pred_clean is None or ref_clean is None:
        meta["match_type"] = "missing"
        return False, meta

    if math_answer_key(pred_clean) == math_answer_key(ref_clean):
        meta["match_type"] = "normalized_exact"
        return True, meta

    pred_equation = split_equation(pred_clean)
    ref_equation = split_equation(ref_clean)
    if pred_equation and ref_equation:
        lhs_ok, lhs_meta = compare_scalar_math(pred_equation[0], ref_equation[0])
        rhs_ok, rhs_meta = compare_scalar_math(pred_equation[1], ref_equation[1])
        if lhs_ok and rhs_ok:
            meta["match_type"] = "equation_pairwise"
            meta["lhs"] = lhs_meta
            meta["rhs"] = rhs_meta
            return True, meta

        lhs_ok, lhs_meta = compare_scalar_math(pred_equation[0], ref_equation[1])
        rhs_ok, rhs_meta = compare_scalar_math(pred_equation[1], ref_equation[0])
        if lhs_ok and rhs_ok:
            meta["match_type"] = "equation_pairwise_swapped"
            meta["lhs"] = lhs_meta
            meta["rhs"] = rhs_meta
            return True, meta

    try:
        pred_expr = parse_math_expression(pred_clean)
        ref_expr = parse_math_expression(ref_clean)
        if isinstance(pred_expr, Eq) or isinstance(ref_expr, Eq):
            if pred_expr == ref_expr:
                meta["match_type"] = "equation_exact"
                return True, meta
            meta["match_type"] = "equation_mismatch"
            return False, meta

        difference = simplify(pred_expr - ref_expr)
        if difference == 0:
            meta["match_type"] = "symbolic_equivalence"
            return True, meta

        equals_result = getattr(pred_expr, "equals", None)
        if callable(equals_result) and pred_expr.equals(ref_expr):
            meta["match_type"] = "symbolic_equals"
            return True, meta

        meta["match_type"] = "symbolic_mismatch"
        meta["difference"] = str(difference)
        return False, meta
    except Exception as exc:
        meta["match_type"] = "parse_error"
        meta["parse_error"] = f"{type(exc).__name__}: {exc}"
        return False, meta


def compare_math_answers(prediction: Optional[str], reference: str) -> Tuple[bool, Dict[str, Any]]:
    """Compare possibly structured MATH answers."""
    if prediction is None:
        return False, {
            "match_type": "missing_prediction",
            "normalized_prediction": None,
            "normalized_reference": clean_math_text(reference),
        }

    pred_container, pred_parts = split_sequence(prediction)
    ref_container, ref_parts = split_sequence(reference)
    if pred_parts is not None or ref_parts is not None:
        if (
            pred_parts is None
            or ref_parts is None
            or pred_container != ref_container
            or len(pred_parts) != len(ref_parts)
        ):
            return False, {
                "match_type": "sequence_shape_mismatch",
                "normalized_prediction": clean_math_text(prediction),
                "normalized_reference": clean_math_text(reference),
            }

        part_metas = []
        for pred_part, ref_part in zip(pred_parts, ref_parts):
            ok, part_meta = compare_math_answers(pred_part, ref_part)
            part_metas.append(part_meta)
            if not ok:
                return False, {
                    "match_type": "sequence_element_mismatch",
                    "normalized_prediction": clean_math_text(prediction),
                    "normalized_reference": clean_math_text(reference),
                    "parts": part_metas,
                }

        return True, {
            "match_type": "sequence_equivalence",
            "normalized_prediction": clean_math_text(prediction),
            "normalized_reference": clean_math_text(reference),
            "parts": part_metas,
        }

    return compare_scalar_math(prediction, reference)


def summarize_example_scores(sample_records: List[Dict[str, Any]], reference_answer: str, dataset_key: str) -> Dict[str, Any]:
    """Aggregate sample-level correctness for one example."""
    total = len(sample_records)
    n_correct = sum(int(sample["is_correct"]) for sample in sample_records)
    valid_answers = []
    first_index_by_key: Dict[str, int] = {}

    for idx, sample in enumerate(sample_records):
        answer_key = sample.get("answer_key")
        if not answer_key:
            continue
        if answer_key not in first_index_by_key:
            first_index_by_key[answer_key] = idx
        valid_answers.append(answer_key)

    majority_answer = None
    majority_count = 0
    majority_correct = False
    if valid_answers:
        counts = Counter(valid_answers)
        majority_answer, majority_count = max(
            counts.items(),
            key=lambda item: (item[1], -first_index_by_key[item[0]]),
        )
        representative = sample_records[first_index_by_key[majority_answer]]
        majority_correct = bool(representative["is_correct"])

    return {
        "n_samples": total,
        "n_scored": total,
        "n_correct": n_correct,
        "sample_accuracy": (n_correct / total) if total else 0.0,
        "any_correct": any(sample["is_correct"] for sample in sample_records),
        "majority_answer": majority_answer,
        "majority_answer_count": majority_count,
        "majority_vote_correct": majority_correct,
        "reference_answer": reference_answer,
        "dataset_key": dataset_key,
    }


def build_sampling_params(config: Dict[str, Any], seed: int) -> SamplingParams:
    """Construct vLLM sampling params from config."""
    sampling_cfg = config["sampling"]
    return SamplingParams(
        n=int(config["global"]["samples_per_example"]),
        temperature=float(sampling_cfg["temperature"]),
        top_p=float(sampling_cfg["top_p"]),
        max_tokens=int(sampling_cfg["max_tokens"]),
        logprobs=int(sampling_cfg["logprobs"]),
        seed=seed,
    )


def reset_output_files(pair_dir: Path) -> None:
    """Remove old generated files when overwrite is requested."""
    for filename in ["samples.jsonl", "examples.jsonl", "summary.json"]:
        path = pair_dir / filename
        if path.exists():
            path.unlink()


def maybe_cleanup_vllm(llm: Optional[LLM]) -> None:
    """Best-effort cleanup between model runs."""
    if llm is None:
        return
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_sample(
    dataset_key: str,
    raw_text: str,
    parsed: Dict[str, Any],
    reference_answer: str,
    gsm_extract_answer: Callable[[str], str],
    gsm_invalid_answer: str,
) -> Dict[str, Any]:
    """Score one sampled completion."""
    if dataset_key == "gsm8k":
        predicted = gsm_extract_answer(raw_text)
        predicted_answer = None if predicted == gsm_invalid_answer else predicted
        reference = gsm_extract_answer(reference_answer)
        is_correct = (
            predicted_answer is not None
            and reference != gsm_invalid_answer
            and predicted_answer == reference
        )
        return {
            "answer_text": predicted_answer,
            "answer_key": predicted_answer,
            "reference_answer": reference,
            "is_correct": is_correct,
            "score_metadata": {
                "match_type": "grade_school_math_exact" if predicted_answer is not None else "missing_prediction",
                "normalized_prediction": predicted_answer,
                "normalized_reference": reference,
            },
        }

    if dataset_key == "math500":
        predicted_answer = parsed["answer_text"]
        is_correct, score_meta = compare_math_answers(predicted_answer, reference_answer)
        return {
            "answer_text": predicted_answer,
            "answer_key": math_answer_key(predicted_answer),
            "reference_answer": reference_answer,
            "is_correct": is_correct,
            "score_metadata": score_meta,
        }

    raise ValueError(f"Unsupported dataset key: {dataset_key}")


def parse_response(dataset_key: str, raw_text: str, gsm_extract_answer, gsm_invalid_answer: str) -> Dict[str, Any]:
    """Parse a completion into reasoning and answer fields."""
    if dataset_key == "gsm8k":
        return parse_gsm_response(raw_text, gsm_extract_answer, gsm_invalid_answer)
    if dataset_key == "math500":
        return parse_math_response(raw_text)
    raise ValueError(f"Unsupported dataset key: {dataset_key}")


def build_run_dir(config: Dict[str, Any], explicit_output_dir: Optional[Path]) -> Path:
    """Choose the run directory."""
    if explicit_output_dir is not None:
        return explicit_output_dir.resolve()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    experiment_name = slugify(config["experiment_name"])
    return Path(config["global"]["output_root"]) / f"{timestamp}_{experiment_name}"


def build_pair_dir(run_dir: Path, model_name: str, dataset_key: str) -> Path:
    """Build per-model/per-dataset output directory."""
    return run_dir / slugify(model_name) / dataset_key


def save_pair_summary(pair_dir: Path, summary: Dict[str, Any]) -> None:
    """Persist summary.json for one model/dataset pair."""
    save_json(summary, pair_dir / "summary.json")


def collect_pair_metrics(examples_path: Path) -> Dict[str, Any]:
    """Aggregate metrics from examples.jsonl."""
    example_rows = []
    if examples_path.exists():
        with open(examples_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    example_rows.append(json.loads(line))

    n_examples = len(example_rows)
    total_samples = sum(row.get("n_samples", 0) for row in example_rows)
    total_correct = sum(row.get("n_correct", 0) for row in example_rows)
    pass_count = sum(int(row.get("any_correct", False)) for row in example_rows)
    majority_count = sum(int(row.get("majority_vote_correct", False)) for row in example_rows)

    return {
        "n_examples": n_examples,
        "total_samples": total_samples,
        "total_correct_samples": total_correct,
        "sample_accuracy": (total_correct / total_samples) if total_samples else 0.0,
        "pass_at_k": (pass_count / n_examples) if n_examples else 0.0,
        "majority_vote_accuracy": (majority_count / n_examples) if n_examples else 0.0,
    }


def chunk_examples(examples: List[ExampleRecord], batch_size: int) -> List[List[ExampleRecord]]:
    """Chunk examples into prompt batches."""
    return [
        examples[idx:idx + batch_size]
        for idx in range(0, len(examples), batch_size)
    ]


def create_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description="Sample vLLM responses and evaluate them")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML or JSON config",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Override the list of model names",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["gsm8k", "math500"],
        default=None,
        help="Restrict the enabled datasets",
    )
    parser.add_argument(
        "--samples-per-example",
        type=int,
        default=None,
        help="Override global.samples_per_example",
    )
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=None,
        help="Number of questions to send in one vLLM generate call",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Limit examples per dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional run directory to reuse instead of timestamping a new one",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override sampling.temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Override sampling.top_p",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override sampling.max_tokens",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=None,
        help="Enable trust_remote_code in vLLM/tokenizer loading",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_false",
        dest="trust_remote_code",
        help="Disable trust_remote_code in vLLM/tokenizer loading",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite generated files in an existing pair directory",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log verbosity",
    )
    return parser


def main() -> None:
    """Entrypoint."""
    parser = create_arg_parser()
    args = parser.parse_args()

    config_data = build_default_config()
    if args.config.exists():
        config_data = deep_merge(config_data, load_config_file(args.config))
    config_data = apply_cli_overrides(config_data, args)
    config_data = normalize_config(config_data)

    run_dir = build_run_dir(config_data, args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    import logging

    log_level = logging.WARNING if args.quiet else logging.INFO
    logger = setup_logger(
        "reasoning_sampling",
        log_file=run_dir / "sample_vllm_eval.log",
        level=log_level,
    )
    logger.propagate = False

    logger.info("=" * 60)
    logger.info("REASONING vLLM SAMPLING + EVAL")
    logger.info("=" * 60)
    logger.info("Config: %s", args.config)
    logger.info("Run directory: %s", run_dir)

    resolved_config = deepcopy(config_data)
    resolved_config["runtime"] = {
        "run_dir": str(run_dir),
        "config_path": str(args.config.resolve()),
    }
    dump_yaml(resolved_config, run_dir / "resolved_config.yaml")

    external_cfg = config_data["external"]
    external_root = Path(external_cfg["root_dir"])
    auto_clone = bool(external_cfg.get("auto_clone", True))

    dataset_examples: Dict[str, List[ExampleRecord]] = {}
    for dataset_key, dataset_cfg in config_data["datasets"].items():
        if not dataset_cfg.get("enabled", False):
            continue
        dataset_examples[dataset_key] = load_examples(
            dataset_key,
            dataset_cfg,
            config_data["global"].get("max_examples"),
            logger,
        )

    grade_school_math_dir = None
    gsm_extract_answer = None
    gsm_invalid_answer = GSM_INVALID_ANSWER
    if "gsm8k" in dataset_examples:
        grade_school_math_dir = ensure_repo_cloned(
            external_root / "grade-school-math",
            external_cfg["grade_school_math_repo"],
            auto_clone,
            logger,
        )
        logger.info("Using grade-school-math repo at %s", grade_school_math_dir)
        gsm_extract_answer, gsm_invalid_answer = load_grade_school_math_extractor(grade_school_math_dir, logger)

    if "math500" in dataset_examples:
        simple_evals_dir = ensure_repo_cloned(
            external_root / "simple-evals",
            external_cfg["simple_evals_repo"],
            auto_clone,
            logger,
        )
        logger.info("Using simple-evals repo at %s", simple_evals_dir)
        logger.info("simple-evals is cloned for prompt/reference parity; MATH scoring stays rule-based")

    model_names = list(config_data["models"]["names"])
    pair_summaries: List[Dict[str, Any]] = []

    for model_name in model_names:
        logger.info("-" * 60)
        logger.info("Loading tokenizer for %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=bool(config_data["vllm"]["trust_remote_code"]),
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info("Initializing vLLM for %s", model_name)
        llm = LLM(
            model=model_name,
            dtype=config_data["models"]["dtype"],
            gpu_memory_utilization=float(config_data["vllm"]["gpu_memory_utilization"]),
            tensor_parallel_size=int(config_data["vllm"]["tensor_parallel_size"]),
            max_model_len=int(config_data["vllm"]["max_model_len"]),
            trust_remote_code=bool(config_data["vllm"]["trust_remote_code"]),
            seed=int(config_data["random_seed"]),
        )

        try:
            for dataset_key, examples in dataset_examples.items():
                pair_dir = build_pair_dir(run_dir, model_name, dataset_key)
                pair_dir.mkdir(parents=True, exist_ok=True)

                if args.overwrite:
                    reset_output_files(pair_dir)

                samples_path = pair_dir / "samples.jsonl"
                examples_path = pair_dir / "examples.jsonl"
                processed_ids = load_existing_example_ids(examples_path)

                logger.info(
                    "Running %s on %s (%d examples, %d already complete, prompt_batch_size=%d, samples_per_example=%d)",
                    model_name,
                    dataset_key,
                    len(examples),
                    len(processed_ids),
                    int(config_data["global"]["prompt_batch_size"]),
                    int(config_data["global"]["samples_per_example"]),
                )

                pending_examples = [
                    example for example in examples if example.example_id not in processed_ids
                ]
                progress = tqdm(total=len(pending_examples), desc=f"{slugify(model_name)}:{dataset_key}")

                for example_batch in chunk_examples(
                    pending_examples,
                    int(config_data["global"]["prompt_batch_size"]),
                ):
                    prompt_texts = []
                    for example in example_batch:
                        prompt = build_user_prompt(dataset_key, example.prompt_question, config_data)
                        prompt_texts.append(apply_chat_template(tokenizer, prompt))

                    sampling_params = build_sampling_params(
                        config_data,
                        seed=int(config_data["random_seed"]) + example_batch[0].index,
                    )
                    outputs = llm.generate(prompt_texts, sampling_params, use_tqdm=False)
                    if len(outputs) != len(example_batch):
                        raise RuntimeError(
                            f"Expected {len(example_batch)} outputs for {model_name} / {dataset_key}, got {len(outputs)}"
                        )

                    for example, prompt_text, request_output in zip(example_batch, prompt_texts, outputs):
                        sample_records: List[Dict[str, Any]] = []
                        for sample_idx, completion in enumerate(request_output.outputs):
                            raw_text = completion.text
                            parsed = parse_response(
                                dataset_key,
                                raw_text,
                                gsm_extract_answer,
                                gsm_invalid_answer,
                            )
                            scored = score_sample(
                                dataset_key,
                                raw_text,
                                parsed,
                                example.raw_reference,
                                gsm_extract_answer,
                                gsm_invalid_answer,
                            )
                            sample_record = {
                                "example_id": example.example_id,
                                "dataset_key": dataset_key,
                                "model": model_name,
                                "sample_index": sample_idx,
                                "raw_text": raw_text,
                                "reasoning_text": parsed["reasoning_text"],
                                "answer_text": scored["answer_text"],
                                "answer_source": parsed["answer_source"],
                                "answer_key": scored["answer_key"],
                                "reference_answer": scored["reference_answer"],
                                "is_correct": bool(scored["is_correct"]),
                                "score": float(bool(scored["is_correct"])),
                                "score_metadata": scored["score_metadata"],
                                "cumulative_logprob": float(completion.cumulative_logprob)
                                if completion.cumulative_logprob is not None
                                else None,
                                "token_ids": list(completion.token_ids),
                                "num_tokens": len(completion.token_ids),
                                "metadata": example.metadata,
                            }
                            sample_records.append(sample_record)
                            append_jsonl(samples_path, sample_record)

                        example_summary = summarize_example_scores(
                            sample_records,
                            sample_records[0]["reference_answer"],
                            dataset_key,
                        )
                        example_record = {
                            "example_id": example.example_id,
                            "dataset_key": dataset_key,
                            "model": model_name,
                            "question": example.prompt_question,
                            "prompt": prompt_text,
                            "metadata": example.metadata,
                            **example_summary,
                        }
                        append_jsonl(examples_path, example_record)
                        processed_ids.add(example.example_id)
                        progress.update(1)

                progress.close()

                pair_metrics = collect_pair_metrics(examples_path)
                pair_summary = {
                    "model": model_name,
                    "dataset_key": dataset_key,
                    "paths": {
                        "samples": str(samples_path),
                        "examples": str(examples_path),
                    },
                    **pair_metrics,
                }
                save_pair_summary(pair_dir, pair_summary)
                pair_summaries.append(pair_summary)
                logger.info(
                    "Completed %s / %s: %d examples, sample_accuracy=%.4f, pass@k=%.4f",
                    model_name,
                    dataset_key,
                    pair_metrics["n_examples"],
                    pair_metrics["sample_accuracy"],
                    pair_metrics["pass_at_k"],
                )
        finally:
            maybe_cleanup_vllm(llm)

    total_examples = sum(summary["n_examples"] for summary in pair_summaries)
    total_samples = sum(summary["total_samples"] for summary in pair_summaries)
    total_correct_samples = sum(summary["total_correct_samples"] for summary in pair_summaries)

    run_summary = {
        "experiment_name": config_data["experiment_name"],
        "run_dir": str(run_dir),
        "pair_summaries": pair_summaries,
        "overall": {
            "n_pairs": len(pair_summaries),
            "n_examples": total_examples,
            "total_samples": total_samples,
            "total_correct_samples": total_correct_samples,
            "sample_accuracy": (total_correct_samples / total_samples) if total_samples else 0.0,
        },
    }
    save_json(run_summary, run_dir / "run_summary.json")

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info("Run summary written to %s", run_dir / "run_summary.json")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
