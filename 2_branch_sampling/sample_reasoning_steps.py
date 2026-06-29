#!/usr/bin/env -S uv run python
"""Sample one-step reasoning branches while rolling out a fixed reasoning path."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

# Add project root to import path for direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.data_utils import save_json
from utils.logging_utils import setup_logger
from utils.reasoning_steps import (
    ReasoningStepCandidate,
    build_reasoning_branch_payload,
    deduplicate_candidates,
    normalize_step_text,
    select_top_confidence,
    sequence_probability,
    should_stop_after_step,
)


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "reasoning_qwen3_small.yaml"
FINAL_ANSWER_MARKERS = {
    "gsm8k": ["####"],
    "math500": ["Answer:"],
}


def load_config(config_path: Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as handle:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(handle) or {}
        if config_path.suffix.lower() == ".json":
            return json.load(handle)
    raise ValueError(f"Unsupported config type: {config_path}")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
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


def example_id_from_record(example: dict[str, object]) -> str:
    for key in ("example_id", "id", "unique_id"):
        value = example.get(key)
        if value is not None:
            return str(value)
    index = example.get("index", "unknown")
    return str(index)


def final_answer_markers_for_dataset(
    config: dict[str, Any],
    dataset_key: str,
) -> list[str]:
    reasoning_cfg = config.get("reasoning", {})
    marker_cfg = reasoning_cfg.get("final_answer_markers", {})
    configured_markers = marker_cfg.get(dataset_key)
    if configured_markers:
        return list(configured_markers)
    return list(FINAL_ANSWER_MARKERS.get(dataset_key, ["####", "Answer:"]))


def samples_per_step_from_config(config: dict[str, Any]) -> int:
    reasoning_cfg = config.get("reasoning", {})
    if reasoning_cfg.get("samples_per_step") is not None:
        return int(reasoning_cfg["samples_per_step"])
    return int(config["global"]["samples_per_example"])


def set_samples_per_step_override(config: dict[str, Any], samples_per_step: int) -> None:
    config.setdefault("reasoning", {})["samples_per_step"] = samples_per_step


def validate_cli_runtime_config(
    config: dict[str, Any],
    *,
    max_steps: int,
) -> tuple[str, list[str], int]:
    """Validate lightweight CLI invariants before loading model dependencies."""
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")

    model_names = list(config["models"]["names"])
    if len(model_names) != 1:
        raise ValueError(
            "Reasoning step sampling requires exactly one model. "
            "Pass --model or configure models.names with a single entry."
        )

    enabled_datasets = [
        dataset_key
        for dataset_key, dataset_cfg in config["datasets"].items()
        if dataset_cfg.get("enabled", False)
    ]
    if not enabled_datasets:
        raise ValueError("Reasoning step sampling requires at least one enabled dataset")

    samples_per_step = samples_per_step_from_config(config)
    if samples_per_step <= 0:
        raise ValueError("samples_per_step must be > 0")

    return model_names[0], enabled_datasets, samples_per_step


def rollout_example_with_sampler(
    *,
    dataset_key: str,
    example: dict[str, object],
    prompt_text: str,
    initial_prefix_text: str,
    initial_prefix_tokens_with_bos: list[int],
    model_name: str,
    max_steps: int,
    max_prefix_tokens: int | None = None,
    final_answer_markers: list[str],
    sample_step,
    encode_prefix,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Roll out one example by sampling branches at each reasoning step.

    The pure sampler callback returns candidates for the current prefix. The
    highest-confidence candidate is committed and becomes part of the next
    prefix, so every later branch file is anchored to a fixed former context.
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be > 0")

    example_id = example_id_from_record(example)
    prefix_text = initial_prefix_text
    prefix_tokens_with_bos = list(initial_prefix_tokens_with_bos)
    committed_steps: list[str] = []
    committed_step_spans: list[dict[str, int]] = []
    committed_step_metadata: list[dict[str, object]] = []
    branch_payloads: list[dict[str, object]] = []
    stop_reason = "max_steps"

    for step_index in range(1, max_steps + 1):
        if (
            max_prefix_tokens is not None
            and len(prefix_tokens_with_bos) >= max_prefix_tokens
        ):
            stop_reason = "max_context_length"
            break

        raw_candidates = list(sample_step(prefix_text, step_index))
        candidates = deduplicate_candidates(raw_candidates)
        if not candidates:
            stop_reason = "no_candidates"
            break

        selected_candidate_index, selected_candidate = select_top_confidence(candidates)
        branch_payloads.append(
            build_reasoning_branch_payload(
                dataset_key=dataset_key,
                example_id=example_id,
                prompt_text=prompt_text,
                prefix_text=prefix_text,
                prefix_tokens_with_bos=prefix_tokens_with_bos,
                committed_previous_steps=list(committed_steps),
                committed_previous_step_token_spans=list(committed_step_spans),
                step_index=step_index,
                candidates=candidates,
                selected_candidate_index=selected_candidate_index,
                model_name=model_name,
                extra_metadata={
                    "source": "reasoning_step_rollout",
                    "max_steps": max_steps,
                },
            )
        )

        previous_prefix_tokens_with_bos = prefix_tokens_with_bos
        next_prefix_text = prefix_text + selected_candidate.text
        next_prefix_tokens_with_bos = list(encode_prefix(next_prefix_text))
        previous_prefix_len = len(previous_prefix_tokens_with_bos)
        if (
            next_prefix_tokens_with_bos[:previous_prefix_len]
            != previous_prefix_tokens_with_bos
        ):
            raise ValueError(
                "prefix tokenization changed after appending committed step; "
                "reasoning step token spans cannot be trusted"
            )
        span = {
            "step_index": step_index,
            "start": previous_prefix_len,
            "end": len(next_prefix_tokens_with_bos),
        }

        committed_steps.append(selected_candidate.text)
        committed_step_spans.append(span)
        committed_step_metadata.append(
            {
                "step_index": step_index,
                "text": selected_candidate.text,
                "probability": selected_candidate.probability,
                "token_span": span,
            }
        )

        prefix_text = next_prefix_text
        prefix_tokens_with_bos = next_prefix_tokens_with_bos

        if should_stop_after_step(selected_candidate.text, final_answer_markers):
            stop_reason = "final_answer_marker"
            break

    rollout_metadata: dict[str, object] = {
        "dataset": dataset_key,
        "example_id": example_id,
        "model": model_name,
        "prompt_text": prompt_text,
        "initial_prefix_text": initial_prefix_text,
        "final_prefix_text": prefix_text,
        "max_steps": max_steps,
        "max_prefix_tokens": max_prefix_tokens,
        "final_prefix_token_length": len(prefix_tokens_with_bos),
        "completed": stop_reason in {"final_answer_marker", "max_steps"},
        "stop_reason": stop_reason,
        "num_steps": len(committed_steps),
        "committed_steps": committed_step_metadata,
        "committed_step_token_spans": committed_step_spans,
    }
    return branch_payloads, rollout_metadata


def apply_reasoning_chat_template(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def build_prefix_tokens_with_bos(prefix_text: str, tokenizer) -> list[int]:
    encoded = tokenizer(
        prefix_text,
        return_tensors=None,
        add_special_tokens=False,
    )
    token_ids = list(encoded["input_ids"])
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.pad_token_id
    if bos_id is None:
        bos_id = 0
    if token_ids and token_ids[0] == bos_id:
        return token_ids
    return [bos_id] + token_ids


def extract_completion_logprob(completion: Any) -> float:
    cumulative = getattr(completion, "cumulative_logprob", None)
    if cumulative is not None:
        return float(cumulative)

    logprobs = getattr(completion, "logprobs", None)
    if not logprobs:
        return float("-inf")

    total = 0.0
    seen = 0
    sampled_token_ids = list(getattr(completion, "token_ids", []) or [])
    for index, token_entry in enumerate(logprobs):
        if token_entry is None:
            continue
        if hasattr(token_entry, "logprob"):
            total += float(token_entry.logprob)
            seen += 1
            continue
        if isinstance(token_entry, dict):
            sampled_token_id = (
                sampled_token_ids[index] if index < len(sampled_token_ids) else None
            )
            logprob_item = token_entry.get(sampled_token_id)
            if logprob_item is None:
                values = list(token_entry.values())
                if not values:
                    continue
                logprob_item = max(values, key=logprob_value_for_sort)

            extracted = extract_logprob_value(logprob_item)
            if extracted is None:
                continue
            total += extracted
            seen += 1
    return total if seen else float("-inf")


def extract_logprob_value(item: Any) -> float | None:
    if hasattr(item, "logprob"):
        return float(item.logprob)
    if isinstance(item, dict):
        value = item.get("logprob", item.get("log_prob"))
        return float(value) if value is not None else None
    if isinstance(item, (int, float)):
        return float(item)
    return None


def logprob_value_for_sort(item: Any) -> float:
    return extract_logprob_value(item) or 0.0


def tokenize_step_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    if isinstance(encoded, dict):
        return list(encoded["input_ids"])
    if hasattr(encoded, "input_ids"):
        return list(encoded.input_ids)
    return list(encoded)


def candidate_from_vllm_completion(
    completion: Any,
    prefix_tokens_with_bos: list[int],
    tokenizer: Any | None = None,
) -> ReasoningStepCandidate:
    raw_text = getattr(completion, "text", "")
    text = normalize_step_text(raw_text)
    sampled_token_ids = list(getattr(completion, "token_ids", []) or [])
    probability_token_count = len(sampled_token_ids)
    if tokenizer is not None and text != raw_text:
        token_ids = tokenize_step_text(tokenizer, text)
    else:
        # Without a tokenizer, keep vLLM's sampled token IDs even if the
        # normalized text includes an added delimiter.
        token_ids = sampled_token_ids
    if probability_token_count <= 0:
        probability_token_count = len(token_ids)
    logprob = extract_completion_logprob(completion)
    return ReasoningStepCandidate(
        text=text,
        token_ids=token_ids,
        full_token_ids=list(prefix_tokens_with_bos) + token_ids,
        logprob=logprob,
        probability=sequence_probability(logprob, probability_token_count),
    )


def build_vllm_token_prompt(prefix_text: str, prefix_tokens_with_bos: list[int]) -> dict[str, Any]:
    return {
        "prompt": prefix_text,
        "prompt_token_ids": list(prefix_tokens_with_bos),
    }


def write_rollout_outputs(
    *,
    output_dir: Path,
    branch_payloads: list[dict[str, object]],
) -> list[str]:
    output_files: list[str] = []
    for payload in branch_payloads:
        path = output_dir / f"{payload['prefix_id']}_branches.json"
        save_json(payload, path)
        output_files.append(str(path))
    return output_files


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample reasoning-step branches with vLLM rollouts"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", choices=["gsm8k", "math500"], default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-step", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true", default=None)
    parser.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = create_arg_parser().parse_args()

    # Heavy dependencies stay inside CLI code so tests can import this module.
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from scripts.sample_reasoning_vllm_eval import (
        build_default_config,
        build_user_prompt,
        load_examples,
        normalize_config,
    )

    base_config = build_default_config()
    config = deep_merge(base_config, load_config(args.config))

    if args.model is not None:
        config["models"]["names"] = [args.model]
    if args.dataset is not None:
        for dataset_key in config["datasets"]:
            config["datasets"][dataset_key]["enabled"] = dataset_key == args.dataset
    if args.max_examples is not None:
        config["global"]["max_examples"] = args.max_examples
    if args.samples_per_step is not None:
        set_samples_per_step_override(config, args.samples_per_step)
    if args.temperature is not None:
        config["sampling"]["temperature"] = args.temperature
    if args.top_p is not None:
        config["sampling"]["top_p"] = args.top_p
    if args.max_tokens is not None:
        config["sampling"]["max_tokens"] = args.max_tokens
    if args.trust_remote_code is not None:
        config["vllm"]["trust_remote_code"] = args.trust_remote_code

    config = normalize_config(config)
    model_name, enabled_datasets, samples_per_step = validate_cli_runtime_config(
        config,
        max_steps=args.max_steps,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_level = logging.WARNING if args.quiet else logging.INFO
    logger = setup_logger(
        "reasoning_step_sampler",
        log_file=args.output_dir / "sample_reasoning_steps.log",
        level=log_level,
    )
    logger.propagate = False

    if args.dataset is not None and enabled_datasets != [args.dataset]:
        raise ValueError("--dataset must restrict execution to exactly one dataset")

    trust_remote_code = bool(config["vllm"].get("trust_remote_code", True))
    logger.info("Loading tokenizer for %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading vLLM for %s", model_name)
    max_model_len = int(config["vllm"].get("max_model_len", 4096))
    llm = LLM(
        model=model_name,
        dtype=config["models"].get("dtype", "bfloat16"),
        gpu_memory_utilization=float(config["vllm"].get("gpu_memory_utilization", 0.9)),
        tensor_parallel_size=int(config["vllm"].get("tensor_parallel_size", 1)),
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
    )

    sampling_cfg = config["sampling"]
    sampling_params = SamplingParams(
        temperature=float(sampling_cfg.get("temperature", 0.7)),
        top_p=float(sampling_cfg.get("top_p", 0.95)),
        max_tokens=int(sampling_cfg.get("max_tokens", 512)),
        n=samples_per_step,
        stop=["\n\n"],
        logprobs=int(sampling_cfg.get("logprobs", 1)),
    )

    all_output_files: list[str] = []
    rollout_records: list[dict[str, object]] = []

    def encode_prefix(prefix_text: str) -> list[int]:
        return build_prefix_tokens_with_bos(prefix_text, tokenizer)

    for dataset_key in enabled_datasets:
        dataset_cfg = config["datasets"][dataset_key]
        examples = load_examples(
            dataset_key,
            dataset_cfg,
            config["global"].get("max_examples"),
            logger,
        )
        final_answer_markers = final_answer_markers_for_dataset(config, dataset_key)

        for example_record in tqdm(examples, desc=f"{dataset_key} rollouts"):
            example = {
                "example_id": example_record.example_id,
                "index": example_record.index,
                "raw_reference": example_record.raw_reference,
                **example_record.metadata,
            }
            prompt_text = build_user_prompt(
                dataset_key,
                example_record.prompt_question,
                config,
            )
            initial_prefix_text = apply_reasoning_chat_template(tokenizer, prompt_text)
            initial_prefix_tokens_with_bos = encode_prefix(initial_prefix_text)
            current_prefix_tokens: list[int] = list(initial_prefix_tokens_with_bos)

            def sample_step(prefix_text: str, step_index: int):
                nonlocal current_prefix_tokens
                current_prefix_tokens = encode_prefix(prefix_text)
                prompt_input = build_vllm_token_prompt(prefix_text, current_prefix_tokens)
                outputs = llm.generate([prompt_input], sampling_params, use_tqdm=False)
                if not outputs:
                    return []
                return [
                    candidate_from_vllm_completion(
                        completion,
                        current_prefix_tokens,
                        tokenizer=tokenizer,
                    )
                    for completion in outputs[0].outputs
                ]

            branch_payloads, rollout_metadata = rollout_example_with_sampler(
                dataset_key=dataset_key,
                example=example,
                prompt_text=prompt_text,
                initial_prefix_text=initial_prefix_text,
                initial_prefix_tokens_with_bos=initial_prefix_tokens_with_bos,
                model_name=model_name,
                max_steps=args.max_steps,
                max_prefix_tokens=max_model_len,
                final_answer_markers=final_answer_markers,
                sample_step=sample_step,
                encode_prefix=encode_prefix,
            )
            if rollout_metadata["stop_reason"] == "max_context_length":
                logger.warning(
                    "Stopped %s example %s because prefix length %s reached "
                    "max_model_len=%s",
                    dataset_key,
                    example["example_id"],
                    rollout_metadata["final_prefix_token_length"],
                    max_model_len,
                )
            output_files = write_rollout_outputs(
                output_dir=args.output_dir,
                branch_payloads=branch_payloads,
            )
            all_output_files.extend(output_files)
            rollout_metadata["output_files"] = output_files
            rollout_records.append(rollout_metadata)

    branches_index = {
        "model": model_name,
        "datasets": enabled_datasets,
        "samples_per_step": samples_per_step,
        "max_steps": args.max_steps,
        "n_branch_files": len(all_output_files),
        "output_files": all_output_files,
    }
    save_json(branches_index, args.output_dir / "branches_index.json")

    rollout_index = {
        "model": model_name,
        "datasets": enabled_datasets,
        "samples_per_step": samples_per_step,
        "max_steps": args.max_steps,
        "n_examples": len(rollout_records),
        "examples": rollout_records,
    }
    save_json(rollout_index, args.output_dir / "reasoning_rollout_index.json")

    logger.info("Wrote %d branch files to %s", len(all_output_files), args.output_dir)


if __name__ == "__main__":
    main()
