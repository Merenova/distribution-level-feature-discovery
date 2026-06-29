#!/usr/bin/env -S uv run python
"""Sample continuations for prefixes using vLLM.

This script implements the branch sampling algorithm:
1. Sample continuations naturally from the prefix (top-p sampling)
2. Deduplicate continuations and record path probabilities

Note: In the latent_planning pipeline, this is Stage 2 (runs before attribution).
"""

import argparse
import sys
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import PathConfig, SamplingConfig
from utils.data_utils import load_json, save_json
from utils.logging_utils import setup_logger
from utils.manifest import filter_samples_by_manifest, update_manifest_with_results


class SkipPrefixError(Exception):
    """Raised when a prefix should be skipped (not a failure, just filtered out)."""
    pass


@dataclass
class Continuation:
    """Represents a unique continuation."""
    text: str
    token_ids: List[int]
    logprob: float
    probability: float

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)


def deduplicate_continuations(samples: List[Dict[str, Any]]) -> List[Continuation]:
    """Remove duplicate continuations by text and keep the highest-probability variant."""
    best_by_text: Dict[str, Continuation] = {}

    for sample in samples:
        text = sample.get("text", "").strip()
        if not text:
            continue

        token_ids = sample.get("token_ids", [])
        n_tokens = len(token_ids) if token_ids else 1
        logprob = float(sample.get("logprob", float("-inf")))

        # Length-normalize to avoid bias toward shorter continuations
        logprob_per_token = logprob / n_tokens if math.isfinite(logprob) else float("-inf")
        probability = math.exp(logprob_per_token) if math.isfinite(logprob_per_token) else 0.0

        cont = Continuation(
            text=text,
            token_ids=token_ids,
            logprob=logprob,
            probability=probability,
        )

        if text not in best_by_text or probability > best_by_text[text].probability:
            best_by_text[text] = cont

    unique = list(best_by_text.values())
    unique.sort(key=lambda c: c.probability, reverse=True)
    return unique

def build_prefix_tokens_with_bos(prefix: str, tokenizer) -> List[int]:
    """Build prefix token IDs including BOS at position 0."""
    tokens = tokenizer(prefix, return_tensors="pt", add_special_tokens=False).input_ids
    token_ids_list = tokens[0].tolist()

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.pad_token_id
    if bos_id is None:
        bos_id = 0  # Fallback

    if token_ids_list and token_ids_list[0] == bos_id:
        return token_ids_list
    return [bos_id] + token_ids_list


def build_vllm_token_prompt(prefix: str, prompt_token_ids: List[int]) -> Dict[str, Any]:
    """Build a vLLM token prompt that bypasses tokenizer special-token defaults."""
    return {
        "prompt": prefix,
        "prompt_token_ids": list(prompt_token_ids),
    }


def extract_logprob_for_token(logprob_entry: Any, token_id: int) -> Optional[float]:
    """Extract logprob for a specific token_id from a prompt_logprobs entry."""
    if logprob_entry is None:
        return None
    if isinstance(logprob_entry, dict):
        val = logprob_entry.get(token_id)
        if val is None:
            return None
        if hasattr(val, "logprob"):
            return float(val.logprob)
        if isinstance(val, dict):
            lp = val.get("logprob", val.get("log_prob"))
            return float(lp) if lp is not None else None
        if isinstance(val, (int, float)):
            return float(val)
        return None
    if hasattr(logprob_entry, "logprob"):
        return float(logprob_entry.logprob)
    return None


def compute_temp1_logprobs(
    llm: LLM,
    tokenizer: AutoTokenizer,
    prefix: str,
    continuations: List[Continuation],
    logger,
    batch_size: int,
    skip_rescore: bool = False,
    prefix_token_ids: Optional[List[int]] = None,
) -> None:
    """Rescore continuations with temperature=1.0 using prompt logprobs."""
    if not continuations:
        return

    # Skip rescoring if requested (saves significant time with newer vLLM)
    if skip_rescore:
        logger.info("Skipping temp=1.0 rescoring (--skip-rescore flag)")
        return

    try:
        scoring_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=1,  # vLLM requires at least 1; we only use prompt_logprobs
            prompt_logprobs=1,
        )
    except TypeError:
        logger.warning("prompt_logprobs not supported in this vLLM version; skipping temp=1.0 scoring")
        return

    prefix_ids = (
        list(prefix_token_ids)
        if prefix_token_ids is not None
        else build_prefix_tokens_with_bos(prefix, tokenizer)
    )
    score_batch_size = max(1, min(64, batch_size))

    for i in range(0, len(continuations), score_batch_size):
        batch = continuations[i:i + score_batch_size]
        prompts = [
            build_vllm_token_prompt(prefix + cont.text, prefix_ids + cont.token_ids)
            for cont in batch
        ]
        outputs = llm.generate(prompts, scoring_params, use_tqdm=False)

        for j, (cont, out) in enumerate(zip(batch, outputs)):
            prev_logprob = cont.logprob
            prev_prob = cont.probability
            prompt_logprobs = getattr(out, "prompt_logprobs", None)
            if prompt_logprobs is None:
                cont.logprob = float("-inf")
                cont.probability = 0.0
                continue

            full_ids = prefix_ids + cont.token_ids
            start_idx = len(prefix_ids)

            logprob_sum = 0.0
            n_tokens = 0
            # vLLM stores a placeholder for the first prompt token, then each
            # prompt_logprobs[i] corresponds to full_ids[i].
            max_idx = min(len(full_ids), len(prompt_logprobs))
            for idx in range(start_idx, max_idx):
                logprob_idx = idx
                if logprob_idx < 0 or logprob_idx >= len(prompt_logprobs):
                    continue
                lp = extract_logprob_for_token(prompt_logprobs[logprob_idx], full_ids[idx])
                if lp is None:
                    continue
                logprob_sum += lp
                n_tokens += 1

            cont.logprob = logprob_sum if n_tokens > 0 else float("-inf")
            if n_tokens > 0 and math.isfinite(cont.logprob):
                cont.probability = math.exp(cont.logprob / n_tokens)
            else:
                cont.probability = 0.0

            if i == 0 and j < 3:
                logger.info(
                    "Temp=1.0 rescore sample %d: logprob %+.4f -> %+.4f, prob %.4g -> %.4g",
                    j,
                    prev_logprob,
                    cont.logprob,
                    prev_prob,
                    cont.probability,
                )


def sample_continuations_natural(
    llm: LLM,
    prefix: str,
    prefix_token_ids: Optional[List[int]],
    sampling_config: SamplingConfig,
    max_total_continuations: Optional[int],
    max_batches: int,
    logger
) -> Tuple[List[Continuation], int]:
    """Sample continuations naturally from the prefix.

    Args:
        llm: vLLM LLM instance
        prefix: Prefix text
        prefix_token_ids: Tokenized prefix to pass directly to vLLM when available
        sampling_config: Sampling configuration
        max_total_continuations: Maximum number of distinct continuations to keep (None = no cap)
        max_batches: Maximum batches to sample
        logger: Logger instance

    Returns:
        Tuple of (unique continuations, total samples drawn)
    """
    all_samples: List[Dict[str, Any]] = []
    total_samples = 0
    batches_without_new = 0
    max_batches_without_new = 5  # Early stop if no new distinct in 5 batches
    prev_distinct_count = 0

    # Setup sampling parameters
    sampling_params = SamplingParams(
        temperature=sampling_config.temperature,
        top_p=sampling_config.nucleus_p,
        max_tokens=sampling_config.max_tokens,
        n=sampling_config.batch_size,  # Sample batch_size at a time
        stop=sampling_config.stop_tokens,
        logprobs=1,  # Minimal logprobs for fallback logprob calculation
    )
    prompt_input: Any = (
        build_vllm_token_prompt(prefix, prefix_token_ids)
        if prefix_token_ids is not None
        else prefix
    )

    for batch_idx in range(max_batches):
        # Generate batch
        outputs = llm.generate([prompt_input], sampling_params, use_tqdm=False)
        if not outputs:
            break

        request_output = outputs[0]
        batch_samples = []

        for completion in request_output.outputs:
            # Get cumulative logprob
            logprob = completion.cumulative_logprob
            if logprob is None and completion.logprobs is not None:
                # Fallback: sum token logprobs
                token_logprobs = [
                    tl.logprob for tl in completion.logprobs if tl is not None
                ]
                logprob = float(sum(token_logprobs))

            if completion.text.strip():  # Skip empty
                # We need token_ids for continuation tracking
                batch_samples.append({
                    "text": completion.text,
                    "token_ids": list(completion.token_ids),
                    "logprob": float(logprob) if logprob is not None else float("-inf"),
                })

        all_samples.extend(batch_samples)
        total_samples += len(batch_samples)

        # Deduplicate and check if we have enough distinct continuations
        continuations = deduplicate_continuations(all_samples)
        current_distinct = len(continuations)

        # Check stopping condition: enough distinct continuations
        if max_total_continuations is not None and current_distinct >= max_total_continuations:
            # Truncate to exactly max_total_continuations (keep top by probability)
            continuations = continuations[:max_total_continuations]
            logger.info(f"  Reached target: {len(continuations)} distinct continuations")
            return continuations, total_samples

        # Early stopping: if no new distinct sequences found for several batches, stop
        if current_distinct == prev_distinct_count:
            batches_without_new += 1
            if batches_without_new >= max_batches_without_new:
                logger.info(f"  Early stop: no new distinct in {max_batches_without_new} batches. Total: {current_distinct} distinct")
                return continuations, total_samples
        else:
            batches_without_new = 0
        prev_distinct_count = current_distinct

    # Return what we have even if we didn't reach the target count
    continuations = deduplicate_continuations(all_samples)
    if max_total_continuations is not None:
        continuations = continuations[:max_total_continuations]
        logger.info(f"  Max batches reached: {len(continuations)} distinct continuations (target: {max_total_continuations})")
    else:
        logger.info(f"  Max batches reached: {len(continuations)} distinct continuations (no target cap)")
    return continuations, total_samples


def continuations_to_payload(
    continuations: List[Continuation],
    prefix_tokens_with_bos: List[int],
) -> List[Dict[str, Any]]:
    """Convert Continuation objects to JSON-serializable payload.

    Args:
        continuations: List of Continuation objects (sorted by probability)
        prefix_tokens_with_bos: Prefix token IDs including BOS at position 0

    Returns:
        List of continuation dicts with full_token_ids
    """
    return [
        {
            "text": cont.text,
            "token_ids": cont.token_ids,
            "full_token_ids": prefix_tokens_with_bos + cont.token_ids,
            "num_tokens": cont.num_tokens,
            "logprob": cont.logprob,
            "probability": cont.probability,
        }
        for cont in continuations
    ]


def process_prefix(
    prefix: str,
    prefix_id: str,
    llm: LLM,
    tokenizer: AutoTokenizer,
    sampling_config: SamplingConfig,
    max_total_continuations: int,
    max_batches: int,
    output_dir: Path,
    logger,
    skip_rescore: bool = False,
) -> Tuple[Path, int]:
    """Process a single prefix: sample continuations from the prefix.

    Args:
        prefix: Prefix text
        prefix_id: Unique identifier
        llm: vLLM instance for continuation sampling
        tokenizer: Tokenizer
        sampling_config: Sampling configuration
        max_total_continuations: Maximum total continuations across all samples
        max_batches: Max batches for sampling
        output_dir: Output directory
        logger: Logger instance
        skip_rescore: Skip temperature=1.0 rescoring for speed

    Returns:
        Tuple of (Path to saved output file, number of continuations found)
    """
    logger.info(f"Processing prefix: {prefix_id}")
    logger.info(f"Prefix text: {prefix[:100]}...")

    prefix_tokens_with_bos = build_prefix_tokens_with_bos(prefix, tokenizer)
    logger.info(f"Prefix tokens: {len(prefix_tokens_with_bos)} tokens (including BOS)")

    # Step 2: Natural Sampling from Prefix
    
    logger.info(f"\nSampling continuations naturally from prefix (max_total={max_total_continuations})...")

    target_total = max_total_continuations if max_total_continuations and max_total_continuations > 0 else None
    continuations, num_samples = sample_continuations_natural(
        llm, prefix, prefix_tokens_with_bos, sampling_config, target_total, max_batches * 10, logger
    )
    
    logger.info(f"Sampled {len(continuations)} natural continuations")

    if sampling_config.temperature != 1.0 and not skip_rescore:
        logger.info("Rescoring continuations at temperature=1.0 for logprob stability...")
        compute_temp1_logprobs(
            llm=llm,
            tokenizer=tokenizer,
            prefix=prefix,
            continuations=continuations,
            logger=logger,
            batch_size=sampling_config.batch_size,
            skip_rescore=skip_rescore,
            prefix_token_ids=prefix_tokens_with_bos,
        )
    
    # Step 3: Construct Output Format
    continuation_payload = continuations_to_payload(continuations, prefix_tokens_with_bos)
    total_continuations_so_far = len(continuation_payload)

    # Create output data
    output_data = {
        "prefix_id": prefix_id,
        "prefix": prefix,
        "prefix_tokens_with_bos": prefix_tokens_with_bos,  # Token IDs with BOS at position 0
        "max_total_continuations": max_total_continuations,
        "total_continuations": total_continuations_so_far,
        "continuations": continuation_payload,
    }

    # Save output
    output_file = output_dir / f"{prefix_id}_branches.json"
    save_json(output_data, output_file)
    logger.info(f"\nSaved branch samples to: {output_file}")

    # Print statistics
    logger.info(f"Statistics for {prefix_id}:")
    logger.info(f"  Total continuations: {total_continuations_so_far} (max: {max_total_continuations})")

    return output_file, total_continuations_so_far


def main():
    parser = argparse.ArgumentParser(description="Sample branch continuations for prefixes")
    parser.add_argument(
        "--test-clozes",
        type=Path,
        required=True,
        help="Path to test clozes JSON file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model name or path for vLLM continuation sampling"
    )
    # Continuation sampling arguments
    parser.add_argument(
        "--max-total-continuations",
        type=int,
        default=10000,
        help="Maximum total continuations per prefix (distinct continuations kept)."
    )
    parser.add_argument(
        "--nucleus-p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter for continuations"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="Maximum tokens per continuation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for sampling continuations"
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=100,
        help="Maximum batches to sample"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: 2_branch_sampling/samples/)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM (0.0-1.0)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=2048,
        help="Maximum model sequence length"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code for model loading"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode (only progress bars)"
    )
    parser.add_argument(
        "--skip-rescore",
        action="store_true",
        help="Skip temperature=1.0 rescoring (faster, uses original sampling logprobs)"
    )
    parser.add_argument(
        "--max-complete",
        type=int,
        default=None,
        help="Stop after this many prefixes reach max_total_continuations (None = process all)"
    )
    args = parser.parse_args()

    # Setup paths
    paths = PathConfig()
    paths.ensure_dirs()

    if args.output_dir is None:
        args.output_dir = paths.branch_sampling / "samples"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    import logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logger = setup_logger(
        "branch_sampling",
        log_file=paths.branch_sampling / "sample_branches.log",
        level=log_level
    )

    logger.info("=" * 60)
    logger.info("BRANCH SAMPLING")
    logger.info("=" * 60)
    logger.info(f"Test clozes: {args.test_clozes}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Continuation sampling:")
    logger.info(f"  Max total continuations: {args.max_total_continuations}")
    logger.info(f"  Nucleus p: {args.nucleus_p}")
    logger.info(f"  Temperature: {args.temperature}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"Output directory: {args.output_dir}")

    # Load test clozes
    logger.info("\nLoading test clozes...")
    test_data = load_json(args.test_clozes)
    clozes = test_data["clozes"]
    logger.info(f"Loaded {len(clozes)} test clozes")

    # Filter samples based on Stage 1 manifest (data preparation)
    results_dir = paths.results
    all_cloze_ids = [c.get("cloze_id") or c.get("id") for c in clozes]
    available_ids, skipped_ids = filter_samples_by_manifest(
        all_cloze_ids, results_dir, "stage1", logger
    )
    # Filter clozes to only available ones
    available_id_set = set(available_ids)
    clozes = [c for c in clozes if (c.get("cloze_id") or c.get("id")) in available_id_set]
    logger.info(f"Processing {len(clozes)} available clozes (skipped {len(skipped_ids)})")

    # Setup sampling configuration
    sampling_config = SamplingConfig(
        n_samples=args.batch_size,  # Batch size
        nucleus_p=args.nucleus_p,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
    )

    # Initialize tokenizer (for decoding tokens in continuations)
    logger.info("\nInitializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer initialized")

    # Initialize vLLM for continuation sampling
    logger.info("\nInitializing vLLM for continuation sampling...")
    logger.info(f"  GPU memory utilization: {args.gpu_memory_utilization}")
    logger.info(f"  Tensor parallel size: {args.tensor_parallel_size}")
    logger.info(f"  Max model length: {args.max_model_len}")
    logger.info(f"  Trust remote code: {args.trust_remote_code}")

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
    )
    logger.info("vLLM initialized")

    # Process each prefix
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING PREFIXES")
    logger.info("=" * 60)

    output_files = []
    completed_ids = []
    failed_ids = []
    filtered_ids = []  # Prefixes filtered due to SkipPrefixError
    errors = {}
    
    for idx, cloze in enumerate(tqdm(clozes, desc="Processing prefixes")):
        prefix_id = cloze.get("id") or cloze.get("cloze_id") or f"cloze_{idx:03d}"

        # Extract prefix from cloze
        if isinstance(cloze, dict):
            prefix = cloze.get('prefix', cloze.get('text', str(cloze)))
        else:
            prefix = str(cloze)

        # Process prefix
        try:
            output_file, n_continuations = process_prefix(
                prefix, prefix_id, llm, tokenizer,
                sampling_config, args.max_total_continuations,
                args.max_batches,
                args.output_dir, logger,
                skip_rescore=args.skip_rescore,
            )
            output_files.append(str(output_file))
            
            # Check if we reached the target - treat as failed if not
            if n_continuations < args.max_total_continuations:
                logger.warning(f"Low diversity: {prefix_id} has only {n_continuations}/{args.max_total_continuations} continuations")
                failed_ids.append(prefix_id)
                errors[prefix_id] = f"Low diversity: only {n_continuations}/{args.max_total_continuations} continuations found"
            else:
                completed_ids.append(prefix_id)
                # Check if we've reached the max-complete target
                if args.max_complete and len(completed_ids) >= args.max_complete:
                    logger.info(f"Reached --max-complete target: {len(completed_ids)} complete prefixes")
                    break
        except SkipPrefixError as e:
        # Filtered out due to sample constraints
            logger.info(f"Skipping {prefix_id}: {str(e)}")
            filtered_ids.append(prefix_id)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Failed to process {prefix_id}: {error_msg}")
            import traceback
            logger.error(traceback.format_exc())
            failed_ids.append(prefix_id)
            errors[prefix_id] = error_msg

        logger.info("")

    # Save index of all output files
    index_data = {
        "model": args.model,
        "n_prefixes": len(clozes),
        "max_total_continuations": args.max_total_continuations,
        "sampling_config": {
            "nucleus_p": args.nucleus_p,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "batch_size": args.batch_size,
        },
        "output_files": output_files,
    }

    index_file = args.output_dir / "branches_index.json"
    save_json(index_data, index_file)

    # Combine skipped (from previous stage) and filtered (low diversity) for manifest
    # Both should be excluded from downstream processing
    all_skipped_ids = skipped_ids + filtered_ids
    update_manifest_with_results(
        results_dir=results_dir,
        stage_name="stage2",
        processed=completed_ids,
        failed=failed_ids,
        skipped=all_skipped_ids,
        logger=logger,
        errors=errors,
    )

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Processed {len(completed_ids)}/{len(clozes)} prefixes successfully")
    logger.info(f"  Completed: {len(completed_ids)} (reached {args.max_total_continuations} continuations)")
    logger.info(f"  Failed: {len(failed_ids)} (low diversity or error)")
    logger.info(f"  Filtered: {len(filtered_ids)}")
    logger.info(f"  Skipped (stage1): {len(skipped_ids)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Index file: {index_file}")


if __name__ == "__main__":
    main()
