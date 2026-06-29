#!/usr/bin/env -S uv run python
"""Compute reasoning step-pair attribution contexts.

This stage reuses the ordinary prefix-to-continuation attribution core for each
target reasoning step, then writes one zero-masked context per committed source
step ``i`` and sampled target step ``j``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE3_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(STAGE3_DIR))

from utils.data_utils import load_json, save_json
from utils.logging_utils import setup_logger
from utils.manifest import update_manifest_with_results
from utils.model_backend import resolve_backend
from utils.reasoning_steps import build_reasoning_pair_id


def _span_to_bounds(source_span: dict[str, Any]) -> tuple[int, int]:
    try:
        start = int(source_span["start"])
        end = int(source_span["end"])
    except KeyError as exc:
        raise ValueError("source_span must contain 'start' and 'end'") from exc
    if start > end:
        raise ValueError(f"source_span start must be <= end, got {start}>{end}")
    return start, end


def _positions_to_mask(
    positions: torch.Tensor,
    *,
    start: int,
    end: int,
) -> torch.Tensor:
    positions = positions.to(dtype=torch.long, device="cpu")
    return (positions >= start) & (positions < end)


def _attribution_width(context_data: dict[str, Any]) -> int:
    if "aggregated_attributions" not in context_data:
        raise ValueError("context_data is missing aggregated_attributions")
    aggregated = context_data["aggregated_attributions"]
    if not hasattr(aggregated, "ndim") or aggregated.ndim != 2:
        raise ValueError("aggregated_attributions must be a 2D tensor")
    return int(aggregated.shape[1])


def _decoder_locations(context_data: dict[str, Any]) -> torch.Tensor:
    if "decoder_locations" not in context_data:
        raise ValueError("context_data is missing decoder_locations")
    locations = torch.as_tensor(context_data["decoder_locations"])
    if locations.ndim != 2 or locations.shape[0] != 2:
        raise ValueError(
            f"decoder_locations must have shape (2, N), got {tuple(locations.shape)}"
        )
    return locations


def source_mask_from_context(
    context_data: dict[str, Any],
    source_span: dict[str, Any],
) -> torch.Tensor:
    """Build a full-width attribution source mask for one committed step span.

    The attribution width is ``features + errors + token sources``. Feature
    positions come from ``decoder_locations[1]``. Error positions are ordered as
    all prefix positions for each layer, then token sources are one per prefix
    position.
    """
    start, end = _span_to_bounds(source_span)
    n_sources = _attribution_width(context_data)
    decoder_locations = _decoder_locations(context_data)

    full_context_keys = {
        "n_prefix_features",
        "n_prefix_errors",
        "n_prefix_tokens",
        "prefix_length",
        "n_layers",
    }
    has_full_metadata = full_context_keys.issubset(context_data)
    if not has_full_metadata:
        if int(decoder_locations.shape[1]) != n_sources:
            raise ValueError(
                "context metadata is incomplete and decoder_locations width "
                f"({decoder_locations.shape[1]}) does not match attribution width "
                f"({n_sources})"
            )
        return _positions_to_mask(decoder_locations[1], start=start, end=end)

    n_prefix_features = int(context_data["n_prefix_features"])
    n_prefix_errors = int(context_data["n_prefix_errors"])
    n_prefix_tokens = int(context_data["n_prefix_tokens"])
    prefix_length = int(context_data["prefix_length"])
    n_layers = int(context_data["n_layers"])

    expected_sources = n_prefix_features + n_prefix_errors + n_prefix_tokens
    if expected_sources != n_sources:
        raise ValueError(
            "prefix source counts do not match attribution width: "
            f"{n_prefix_features}+{n_prefix_errors}+{n_prefix_tokens}="
            f"{expected_sources}, attribution width={n_sources}"
        )
    if (
        "n_prefix_sources" in context_data
        and int(context_data["n_prefix_sources"]) != n_sources
    ):
        raise ValueError(
            f"n_prefix_sources={context_data['n_prefix_sources']} does not match "
            f"attribution width={n_sources}"
        )
    if int(decoder_locations.shape[1]) != n_prefix_features:
        raise ValueError(
            f"decoder_locations has {decoder_locations.shape[1]} feature positions, "
            f"expected n_prefix_features={n_prefix_features}"
        )
    if n_prefix_errors != n_layers * prefix_length:
        raise ValueError(
            f"n_prefix_errors={n_prefix_errors} is incompatible with "
            f"n_layers * prefix_length={n_layers * prefix_length}"
        )
    if n_prefix_tokens != prefix_length:
        raise ValueError(
            f"n_prefix_tokens={n_prefix_tokens} is incompatible with "
            f"prefix_length={prefix_length}"
        )

    feature_mask = _positions_to_mask(
        decoder_locations[1],
        start=start,
        end=end,
    )
    error_positions = torch.arange(prefix_length, dtype=torch.long).repeat(n_layers)
    error_mask = _positions_to_mask(error_positions, start=start, end=end)
    token_positions = torch.arange(prefix_length, dtype=torch.long)
    token_mask = _positions_to_mask(token_positions, start=start, end=end)
    full_mask = torch.cat([feature_mask, error_mask, token_mask])
    if int(full_mask.numel()) != n_sources:
        raise ValueError(
            f"constructed mask has {full_mask.numel()} sources, expected {n_sources}"
        )
    return full_mask


def source_mask_from_decoder_locations(
    context_data: dict[str, Any],
    source_span: dict[str, Any],
) -> torch.Tensor:
    """Backward-compatible alias for tests and older plans."""
    return source_mask_from_context(context_data, source_span)


def _mask_tensor_last_dim(
    tensor: torch.Tensor,
    source_mask: torch.Tensor,
) -> torch.Tensor:
    if int(tensor.shape[-1]) != int(source_mask.numel()):
        raise ValueError(
            f"tensor source width {tensor.shape[-1]} does not match mask width "
            f"{source_mask.numel()}"
        )
    mask = source_mask.to(device=tensor.device, dtype=tensor.dtype)
    while mask.ndim < tensor.ndim:
        mask = mask.unsqueeze(0)
    return tensor.clone() * mask


def zero_mask_context_sources(
    context_data: dict[str, Any],
    source_mask: torch.Tensor,
    pair_metadata: dict[str, Any],
) -> dict[str, object]:
    """Return a context copy with non-source-step attribution columns zeroed."""
    if source_mask.dtype != torch.bool:
        source_mask = source_mask.to(dtype=torch.bool)
    n_sources = _attribution_width(context_data)
    if int(source_mask.numel()) != n_sources:
        raise ValueError(
            f"source_mask has {source_mask.numel()} entries, expected {n_sources}"
        )

    masked: dict[str, object] = dict(context_data)
    masked["aggregated_attributions"] = _mask_tensor_last_dim(
        context_data["aggregated_attributions"],
        source_mask,
    )
    if "token_attributions" in context_data:
        masked["token_attributions"] = [
            _mask_tensor_last_dim(token_attr, source_mask)
            for token_attr in context_data["token_attributions"]
        ]
    masked["reasoning_pair_metadata"] = deepcopy(pair_metadata)
    masked["source_mask"] = source_mask.clone()
    return masked


def build_pair_sample_payload(
    target_branch_data: dict[str, Any],
    pair_id: str,
    pair_metadata: dict[str, Any],
) -> dict[str, object]:
    """Build a branch-sampling payload for one reasoning source-target pair."""
    payload = {
        "prefix_id": pair_id,
        "prefix": target_branch_data["prefix"],
        "prefix_tokens_with_bos": deepcopy(
            target_branch_data["prefix_tokens_with_bos"]
        ),
        "continuations": deepcopy(target_branch_data.get("continuations", [])),
        "metadata": deepcopy(target_branch_data.get("metadata", {})),
        "reasoning_metadata": deepcopy(
            target_branch_data.get("reasoning_metadata", {})
        ),
        "reasoning_pair_metadata": deepcopy(pair_metadata),
    }
    return payload


def _resolve_branch_file(branches_dir: Path, file_path: str) -> Path:
    raw_path = Path(file_path)
    if raw_path.is_absolute() or raw_path.exists():
        return raw_path
    if raw_path.parts and raw_path.parts[0] == "results":
        candidate = branches_dir.parent / Path(*raw_path.parts[1:])
    else:
        candidate = branches_dir / raw_path
    if not candidate.exists() and (branches_dir / raw_path.name).exists():
        candidate = branches_dir / raw_path.name
    return candidate


def _target_prefix_id_from_path(file_path: Path) -> str:
    return file_path.stem.removesuffix("_branches")


def _build_pair_metadata(
    *,
    pair_id: str,
    target_prefix_id: str,
    source_span: dict[str, Any],
    target_reasoning_metadata: dict[str, Any],
    target_branch_file: Path,
) -> dict[str, object]:
    source_step_index = int(source_span["step_index"])
    target_step_index = int(target_reasoning_metadata["step_index"])
    return {
        "pair_id": pair_id,
        "target_prefix_id": target_prefix_id,
        "source_step_index": source_step_index,
        "target_step_index": target_step_index,
        "source_step_span": deepcopy(source_span),
        "target_reasoning_metadata": deepcopy(target_reasoning_metadata),
        "target_branch_file": str(target_branch_file),
    }


def _load_attribution_core():
    from compute_continuation_attribution import process_prefix
    from circuit_tracer import ReplacementModel

    return process_prefix, ReplacementModel


def write_stage3_pair_manifest(
    *,
    pair_samples_dir: Path,
    pair_records: list[dict[str, object]],
    failed_targets: dict[str, str],
    skipped_targets: list[str],
    logger,
) -> dict:
    """Write Stage 3 manifest where completed IDs are reasoning pair IDs."""
    completed_pair_ids = [str(record["pair_id"]) for record in pair_records]
    failed_ids = list(failed_targets.keys())
    return update_manifest_with_results(
        results_dir=pair_samples_dir.parent,
        stage_name="stage3",
        processed=completed_pair_ids,
        failed=failed_ids,
        skipped=skipped_targets,
        logger=logger,
        errors=failed_targets,
    )


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute reasoning step-pair attribution contexts"
    )
    parser.add_argument("--branches-dir", type=Path, required=True)
    parser.add_argument("--pair-samples-dir", type=Path, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--transcoder", type=str, required=True)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-feature-nodes", type=int, default=8192)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=["auto", "transformerlens", "nnsight"],
        default="auto",
    )
    parser.add_argument(
        "--store-all",
        action="store_true",
        help="Store and mask token-level attributions in each pair context",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = create_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.pair_samples_dir.mkdir(parents=True, exist_ok=True)

    log_level = logging.WARNING if args.quiet else logging.INFO
    logger = setup_logger(
        "reasoning_step_pair_attribution",
        log_file=args.output_dir / "compute_reasoning_step_pair_attribution.log",
        level=log_level,
    )

    branches_index_file = args.branches_dir / "branches_index.json"
    if not branches_index_file.exists():
        raise FileNotFoundError(f"Branches index not found: {branches_index_file}")
    branches_index = load_json(branches_index_file)
    output_files = list(branches_index.get("output_files", []))

    logger.info("=" * 60)
    logger.info("REASONING STEP-PAIR ATTRIBUTION (STAGE 3)")
    logger.info("=" * 60)
    logger.info("Branches dir: %s", args.branches_dir)
    logger.info("Pair samples dir: %s", args.pair_samples_dir)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("Model: %s", args.model)
    logger.info("Transcoder: %s", args.transcoder)
    logger.info("Loaded %d target-step branch files", len(output_files))

    process_prefix, ReplacementModel = _load_attribution_core()
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    backend = resolve_backend(args.model, args.backend)
    logger.info("Initializing ReplacementModel with backend=%s", backend)
    model = ReplacementModel.from_pretrained(
        args.model,
        args.transcoder,
        backend=backend,
        dtype=dtype_map[args.dtype],
    )

    pair_records: list[dict[str, object]] = []
    skipped_targets: list[str] = []
    failed_targets: dict[str, str] = {}

    for raw_file_path in tqdm(
        output_files,
        desc="Computing reasoning pair attribution",
    ):
        branch_file = _resolve_branch_file(args.branches_dir, raw_file_path)
        target_prefix_id = _target_prefix_id_from_path(branch_file)
        if not branch_file.exists():
            failed_targets[target_prefix_id] = f"branch file not found: {branch_file}"
            logger.warning(
                "Branch file not found for %s: %s",
                target_prefix_id,
                branch_file,
            )
            continue

        try:
            target_branch_data = load_json(branch_file)
            target_prefix_id = str(target_branch_data.get("prefix_id", target_prefix_id))
            reasoning_metadata = dict(target_branch_data.get("reasoning_metadata", {}))
            target_step_index = int(reasoning_metadata["step_index"])
            if target_step_index <= 1:
                skipped_targets.append(target_prefix_id)
                logger.info(
                    "Skipping %s because target step j=%d",
                    target_prefix_id,
                    target_step_index,
                )
                continue

            source_spans = [
                span
                for span in reasoning_metadata.get(
                    "committed_previous_step_token_spans", []
                )
                if int(span.get("step_index", 0)) < target_step_index
            ]
            if not source_spans:
                skipped_targets.append(target_prefix_id)
                logger.warning(
                    "Skipping %s because no committed source spans are available",
                    target_prefix_id,
                )
                continue

            with tempfile.TemporaryDirectory(
                prefix=f"{target_prefix_id}_",
                dir=args.output_dir,
            ) as temp_dir_name:
                temp_dir = Path(temp_dir_name)
                context_file, attribution_file = process_prefix(
                    target_prefix_id,
                    target_branch_data,
                    model,
                    args.max_feature_nodes,
                    args.batch_size,
                    temp_dir,
                    logger,
                    store_all=args.store_all,
                )
                context_data = torch.load(
                    context_file,
                    map_location="cpu",
                    weights_only=False,
                )
                attribution_data = load_json(attribution_file)

                for source_span in source_spans:
                    source_step_index = int(source_span["step_index"])
                    pair_id = build_reasoning_pair_id(
                        target_prefix_id,
                        source_step_index,
                        target_step_index,
                    )
                    pair_metadata = _build_pair_metadata(
                        pair_id=pair_id,
                        target_prefix_id=target_prefix_id,
                        source_span=source_span,
                        target_reasoning_metadata=reasoning_metadata,
                        target_branch_file=branch_file,
                    )
                    source_mask = source_mask_from_context(context_data, source_span)
                    masked_context = zero_mask_context_sources(
                        context_data,
                        source_mask,
                        pair_metadata,
                    )

                    pair_context_file = args.output_dir / f"{pair_id}_prefix_context.pt"
                    torch.save(masked_context, pair_context_file)

                    pair_attribution_data = dict(attribution_data)
                    pair_attribution_data["prefix_id"] = pair_id
                    pair_attribution_data["target_prefix_id"] = target_prefix_id
                    pair_attribution_data["reasoning_pair_metadata"] = pair_metadata
                    pair_attribution_file = args.output_dir / f"{pair_id}_attribution.json"
                    save_json(pair_attribution_data, pair_attribution_file)

                    pair_sample_payload = build_pair_sample_payload(
                        target_branch_data,
                        pair_id,
                        pair_metadata,
                    )
                    pair_sample_file = args.pair_samples_dir / f"{pair_id}_branches.json"
                    save_json(pair_sample_payload, pair_sample_file)

                    pair_record = {
                        **pair_metadata,
                        "context_file": str(pair_context_file),
                        "attribution_file": str(pair_attribution_file),
                        "pair_sample_file": str(pair_sample_file),
                    }
                    pair_records.append(pair_record)
        except Exception as exc:
            failed_targets[target_prefix_id] = f"{type(exc).__name__}: {exc}"
            logger.exception("Failed to process target prefix %s", target_prefix_id)

    index_data = {
        "model": args.model,
        "transcoder": args.transcoder,
        "backend": backend,
        "dtype": args.dtype,
        "max_feature_nodes": args.max_feature_nodes,
        "batch_size": args.batch_size,
        "store_all": args.store_all,
        "n_pairs": len(pair_records),
        "pair_ids": [record["pair_id"] for record in pair_records],
        "pairs": pair_records,
        "skipped_targets": skipped_targets,
        "failed_targets": failed_targets,
    }
    index_file = args.output_dir / "reasoning_pair_index.json"
    save_json(index_data, index_file)
    write_stage3_pair_manifest(
        pair_samples_dir=args.pair_samples_dir,
        pair_records=pair_records,
        failed_targets=failed_targets,
        skipped_targets=skipped_targets,
        logger=logger,
    )

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info("Generated %d reasoning pair attribution contexts", len(pair_records))
    logger.info("Skipped targets: %d", len(skipped_targets))
    logger.info("Failed targets: %d", len(failed_targets))
    logger.info("Index file: %s", index_file)


if __name__ == "__main__":
    main()
