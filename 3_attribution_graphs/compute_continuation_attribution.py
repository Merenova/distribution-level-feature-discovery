#!/usr/bin/env -S uv run python
"""Compute prefix-to-continuation attribution using attribute_prefix_to_continuations.

This is Stage 3 in the latent_planning pipeline. It:
1. Loads branch sampling results from Stage 2
2. Computes attribution from prefix components to continuation tokens
3. Aggregates attributions over the full continuation span
4. Saves prefix context and aggregated attributions for downstream stages
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory and circuit-tracer to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
CIRCUIT_TRACER_PATH = Path(__file__).resolve().parents[1] / "circuit-tracer"
sys.path.insert(0, str(CIRCUIT_TRACER_PATH))

from circuit_tracer import ReplacementModel
from circuit_tracer.attribution import (
    ContinuationTokenAttribution,
    attribute_prefix_to_continuations,
)

from utils.config import PathConfig
from utils.data_utils import load_json, save_json
from utils.logging_utils import setup_logger
from utils.manifest import filter_samples_by_manifest, update_manifest_with_results
from utils.model_backend import resolve_backend


def aggregate_attributions(
    token_attrs: List[ContinuationTokenAttribution],
    start: int,
    end: int,
) -> torch.Tensor:
    """Sum attribution scores across a span of tokens.

    Args:
        token_attrs: List of token attribution objects
        start: Start index (inclusive)
        end: End index (exclusive)

    Returns:
        Aggregated attribution tensor (n_prefix_sources,)
    """
    if start >= end or len(token_attrs) == 0:
        # Return zeros if span is empty
        return token_attrs[0].source_attribution.clone().zero_()

    # Sum attributions across the span
    return torch.stack([
        token_attrs[i].source_attribution for i in range(start, min(end, len(token_attrs)))
    ]).sum(dim=0)


def process_prefix(
    prefix_id: str,
    branches_data: Dict[str, Any],
    model: ReplacementModel,
    max_feature_nodes: int,
    batch_size: int,
    output_dir: Path,
    logger,
    store_all: bool = False,
) -> Tuple[Path, Path]:
    """Process all continuations for a single prefix.

    Args:
        prefix_id: Unique identifier for the prefix
        branches_data: Branch sampling data from Stage 2
        model: ReplacementModel for attribution
        max_feature_nodes: Maximum number of prefix features to include
        batch_size: How many continuation tokens to process per backward pass
        output_dir: Directory for output files
        logger: Logger instance
        store_all: If True, store all token-level attributions for later processing

    Returns:
        Tuple of (prefix_context_path, attribution_path)
    """
    logger.info(f"Processing prefix: {prefix_id}")

    # Extract prefix tokens (with BOS already included)
    prefix_tokens_with_bos = branches_data["prefix_tokens_with_bos"]
    logger.info(f"  Prefix length: {len(prefix_tokens_with_bos)} tokens (including BOS)")

    # Collect all continuations from branch sampling results
    all_continuations = []
    continuation_metadata = []

    for cont_idx, cont in enumerate(branches_data.get("continuations", [])):
        cont_tokens = cont.get("token_ids", [])
        if not cont_tokens:
            continue
        all_continuations.append(cont_tokens)
        continuation_metadata.append({
            "continuation_idx": cont_idx,
            "text": cont.get("text", ""),
        })

    n_continuations = len(all_continuations)
    logger.info(f"  Total continuations: {n_continuations}")

    if n_continuations == 0:
        logger.warning(f"  No continuations found for {prefix_id}, skipping")
        raise ValueError(f"No continuations found for {prefix_id}")

    
    # Memory debugging
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        logger.info(f"  GPU memory before attribution: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    logger.info(f"  Computing attribution (max_feature_nodes={max_feature_nodes}, batch_size={batch_size})...")
    
    # Determine verbose flag for circuit_tracer based on logger level
    # If logger is quiet (WARNING/ERROR), verbose should be False
    is_verbose = logger.getEffectiveLevel() <= logging.INFO
    
    result = attribute_prefix_to_continuations(
        prefix=prefix_tokens_with_bos,
        continuations=all_continuations,
        model=model,
        batch_size=batch_size,
        add_bos=False,  # Don't add BOS, use the one in prefix_tokens_with_bos
        max_feature_nodes=max_feature_nodes,
        verbose=is_verbose,
    )

    if torch.cuda.is_available():
        logger.info(f"  GPU memory after attribution: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"  GPU peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    logger.info(f"  Attribution computed:")
    logger.info(f"    Prefix features: {result.prefix_context.n_prefix_features}")
    logger.info(f"    Prefix errors: {result.prefix_context.n_prefix_errors}")
    logger.info(f"    Prefix tokens: {result.prefix_context.n_prefix_tokens}")
    logger.info(f"    Total prefix sources: {result.prefix_context.n_prefix_sources}")

    # Aggregate attributions by span mode (or store all for deferred processing)
    logger.info(f"  Processing attributions (span_mode=full, store_all={store_all})...")
    aggregated = []
    token_level_attributions = [] if store_all else None  # Only collect if store_all=True
    span_info = []

    for i, token_attrs in enumerate(result.continuation_attributions):
        start, end = 0, len(token_attrs)
        agg = aggregate_attributions(token_attrs, start, end)
        aggregated.append(agg)

        # Store individual token-level attributions only if store_all=True
        if store_all:
            # Shape: (n_tokens_in_continuation, n_prefix_sources)
            per_token_tensor = torch.stack([attr.source_attribution.cpu() for attr in token_attrs])
            token_level_attributions.append(per_token_tensor)

        span_info.append({
            "start": start,
            "end": end,
            "span_length": end - start,
            "continuation_length": len(all_continuations[i]),
        })

    # Stack aggregated attributions
    aggregated_tensor = torch.stack(aggregated)  # (n_continuations, n_prefix_sources)
    logger.info(f"  Aggregated attribution shape: {aggregated_tensor.shape}")

    if store_all:
        logger.info(f"  Token attributions: {len(token_level_attributions)} continuations, "
                    f"lengths {[t.shape[0] for t in token_level_attributions[:5]]}...")

    # Save prefix context for downstream stages (clustering, intervention)
    prefix_ctx = result.prefix_context
    context_data = {
        "prefix_tokens": result.prefix_tokens.tolist(),
        "activation_matrix": prefix_ctx.activation_matrix,  # sparse tensor
        "error_vectors": prefix_ctx.error_vectors,
        "token_vectors": prefix_ctx.token_vectors,
        "decoder_vecs": prefix_ctx.decoder_vecs,
        "encoder_vecs": prefix_ctx.encoder_vecs,
        "encoder_to_decoder_map": prefix_ctx.encoder_to_decoder_map,
        "decoder_locations": prefix_ctx.decoder_locations,
        "n_layers": prefix_ctx.n_layers,
        "selected_features": prefix_ctx.selected_features,
        "total_active_features": prefix_ctx.total_active_features,
        "n_prefix_features": prefix_ctx.n_prefix_features,
        "n_prefix_errors": prefix_ctx.n_prefix_errors,
        "n_prefix_tokens": prefix_ctx.n_prefix_tokens,
        "n_prefix_sources": prefix_ctx.n_prefix_sources,
        "prefix_length": prefix_ctx.prefix_length,
        # Attribution tensors
        "aggregated_attributions": aggregated_tensor,  # sum over selected continuation span
        "aggregated_attributions_pooling": "sum",
        "span_info": span_info,
        "store_all": store_all,  # Flag indicating if token_attributions is present
    }
    # Only store token-level attributions and continuation tokens if store_all=True
    if store_all:
        context_data["token_attributions"] = token_level_attributions  # list of (n_tokens, n_prefix_sources) tensors
        context_data["continuation_tokens"] = all_continuations  # list of token id lists (for recomputing spans)

    context_file = output_dir / f"{prefix_id}_prefix_context.pt"
    torch.save(context_data, context_file)
    logger.info(f"  Saved prefix context to: {context_file}")

    # Save metadata as JSON (for easy inspection, no large tensors)
    attribution_data = {
        "prefix_id": prefix_id,
        "span_mode": "full",
        "store_all": store_all,
        "n_continuations": n_continuations,
        "n_prefix_sources": prefix_ctx.n_prefix_sources,
        "n_prefix_features": prefix_ctx.n_prefix_features,
        "continuation_metadata": continuation_metadata,
        "span_info": span_info,
        "aggregated_attributions_pooling": "sum",
    }

    attribution_file = output_dir / f"{prefix_id}_attribution.json"
    save_json(attribution_data, attribution_file)
    logger.info(f"  Saved attribution metadata to: {attribution_file}")

    return context_file, attribution_file


def main():
    parser = argparse.ArgumentParser(
        description="Compute prefix-to-continuation attribution"
    )
    parser.add_argument(
        "--branches-dir",
        type=Path,
        required=True,
        help="Directory containing branch sampling results from Stage 2"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b",
        help="Base model name for ReplacementModel"
    )
    parser.add_argument(
        "--transcoder",
        type=str,
        default="mntss/gemma-scope-transcoders",
        help="Transcoder name for ReplacementModel"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model (bfloat16 uses half the memory of float32)"
    )
    parser.add_argument(
        "--max-feature-nodes",
        type=int,
        default=8192,
        help="Maximum number of prefix features to include in attribution"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="How many continuation tokens to process per backward pass (lower = less memory)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: results/3_attribution_graphs/)"
    )
    parser.add_argument(
        "--store-all",
        action="store_true",
        help="Store all token-level attributions (defer aggregation to Stage 5)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode (only progress bars)"
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "transformerlens", "nnsight"],
        default="auto",
        help="ReplacementModel backend. 'auto' picks nnsight for Gemma3 models.",
    )
    args = parser.parse_args()

    # Setup paths
    paths = PathConfig()
    paths.ensure_dirs()

    if args.output_dir is None:
        args.output_dir = paths.results / "3_attribution_graphs"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    import logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logger = setup_logger(
        "continuation_attribution",
        log_file=args.output_dir / "compute_continuation_attribution.log",
        level=log_level
    )

    logger.info("=" * 60)
    logger.info("CONTINUATION ATTRIBUTION (STAGE 3)")
    logger.info("=" * 60)
    logger.info(f"Branches dir: {args.branches_dir}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Transcoder: {args.transcoder}")
    logger.info(f"Dtype: {args.dtype}")
    logger.info("Span mode: full")
    logger.info(f"Store all: {args.store_all}")
    logger.info(f"Max feature nodes: {args.max_feature_nodes}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Output dir: {args.output_dir}")

    # Load branch sampling index
    branches_index_file = args.branches_dir / "branches_index.json"
    if not branches_index_file.exists():
        logger.error(f"Branches index not found: {branches_index_file}")
        sys.exit(1)

    branches_index = load_json(branches_index_file)
    logger.info(f"Loaded branches index: {len(branches_index['output_files'])} prefixes")

    # Filter by Stage 2 manifest
    # Derive results_dir from branches_dir to respect --output-dir
    # branches_dir is typically {output_dir}/results/2_branch_sampling/
    results_dir = args.branches_dir.parent
    all_prefix_ids = []
    branches_files = {}

    for file_path in branches_index["output_files"]:
        raw_path = Path(file_path)
        prefix_id = raw_path.stem.replace("_branches", "")
        all_prefix_ids.append(prefix_id)

        # Resolve path robustly:
        # - If absolute, use as-is.
        # - If it starts with "results/...", join with branches_dir.parent
        #   (so passing branches_dir=/.../results/2_branch_sampling works).
        # - Otherwise, join with branches_dir.
        if raw_path.is_absolute():
            resolved = raw_path
        elif raw_path.exists():
            resolved = raw_path
        else:
            if raw_path.parts and raw_path.parts[0] == "results":
                candidate = args.branches_dir.parent / Path(*raw_path.parts[1:])
            else:
                candidate = args.branches_dir / raw_path
            
            # If strictly constructed candidate doesn't exist, try just the filename in branches_dir
            if not candidate.exists() and (args.branches_dir / raw_path.name).exists():
                candidate = args.branches_dir / raw_path.name
                
            resolved = candidate

        branches_files[prefix_id] = resolved

    available_ids, skipped_ids = filter_samples_by_manifest(
        all_prefix_ids, results_dir, "stage2", logger
    )
    logger.info(f"Processing {len(available_ids)} prefixes (skipped {len(skipped_ids)})")

    # Initialize ReplacementModel
    logger.info("\nInitializing ReplacementModel...")
    # Convert dtype string to torch.dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    model_dtype = dtype_map[args.dtype]

    backend = resolve_backend(args.model, args.backend)
    logger.info("Using backend=%s for model=%s", backend, args.model)

    model = ReplacementModel.from_pretrained(
        args.model,
        args.transcoder,
        backend=backend,
        dtype=model_dtype,
    )
    logger.info("ReplacementModel initialized (backend=%s)", backend)

    # Process each prefix
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING PREFIXES")
    logger.info("=" * 60)

    completed_ids = []
    failed_ids = []
    errors = {}

    for prefix_id in tqdm(available_ids, desc="Computing attribution"):
        branches_file = branches_files.get(prefix_id)
        if branches_file is None or not Path(branches_file).exists():
            logger.warning(f"Branches file not found for {prefix_id}")
            failed_ids.append(prefix_id)
            errors[prefix_id] = "Branches file not found"
            continue

        try:
            branches_data = load_json(branches_file)
            process_prefix(
                prefix_id,
                branches_data,
                model,
                args.max_feature_nodes,
                args.batch_size,
                args.output_dir,
                logger,
                store_all=args.store_all,
            )
            completed_ids.append(prefix_id)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Failed to process {prefix_id}: {error_msg}")
            import traceback
            logger.error(traceback.format_exc())
            failed_ids.append(prefix_id)
            errors[prefix_id] = error_msg

    # Save attribution index
    index_data = {
        "model": args.model,
        "transcoder": args.transcoder,
        "span_mode": "full",
        "store_all": args.store_all,
        "max_feature_nodes": args.max_feature_nodes,
        "n_prefixes_processed": len(completed_ids),
        "prefixes": completed_ids,
    }

    index_file = args.output_dir / "attribution_index.json"
    save_json(index_data, index_file)

    # Write Stage 3 manifest
    update_manifest_with_results(
        results_dir=results_dir,
        stage_name="stage3",
        processed=completed_ids,
        failed=failed_ids,
        skipped=skipped_ids,
        logger=logger,
        errors=errors,
    )

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Processed {len(completed_ids)}/{len(available_ids)} prefixes")
    logger.info(f"Completed: {len(completed_ids)}, Failed: {len(failed_ids)}, Skipped: {len(skipped_ids)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Index file: {index_file}")


if __name__ == "__main__":
    main()
