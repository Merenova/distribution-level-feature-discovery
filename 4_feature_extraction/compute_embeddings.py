#!/usr/bin/env -S uv run python
"""Compute semantic embeddings for continuations.

This script computes CONTEXTUAL CONTINUATION EMBEDDINGS:
- Forward pass on full sequence (prefix + continuation) so continuation attends to prefix
- Extract hidden states for ONLY continuation tokens
- Mean pool continuation hidden states to get final embedding

This captures the semantic meaning of the continuation in the context of the prefix.
"""

import sys
from pathlib import Path

# Fix Python path issue - remove Python 3.12 global packages before importing
# This prevents version conflicts with packages in the venv
sys.path = [p for p in sys.path if 'python3.12' not in p]

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import List
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import PathConfig, EmbeddingConfig
from utils.data_utils import load_json, save_json
from utils.logging_utils import setup_logger
from utils.manifest import filter_samples_by_manifest, update_manifest_with_results


def embed_continuations_for_prefix(
    model: SentenceTransformer,
    prefix: str,
    continuations: List[str],
    batch_size: int,
    device: str,
    normalize: bool = True,
    logger=None
) -> np.ndarray:
    """Compute contextual embeddings for continuations given a shared prefix.

    Args:
        model: SentenceTransformer model (e.g., google/embeddinggemma-300m)
        prefix: Shared prefix text
        continuations: List of continuation texts
        batch_size: Batch size for processing
        device: Device for computation
        normalize: Whether to L2-normalize embeddings
        logger: Logger instance

    Returns:
        Embeddings array of shape (n_continuations, hidden_dim)
    """
    tokenizer = model.tokenizer

    # Tokenize prefix once to find the continuation boundary
    # add_special_tokens=False to get raw token count
    prefix_encoding = tokenizer.encode(prefix, add_special_tokens=False)
    prefix_len = len(prefix_encoding)

    if logger:
        logger.info(f"  Prefix length: {prefix_len} tokens")

    # Build full sequences
    full_sequences = [prefix + cont for cont in continuations]

    embeddings = []
    n_batches = (len(full_sequences) + batch_size - 1) // batch_size
    
    # Determine if we should show progress bar (quiet mode check via logger level)
    # If logger level is WARNING (30) or higher, it's quiet mode, so show tqdm
    show_pbar = False
    if logger and logger.getEffectiveLevel() >= 30: # logging.WARNING
        show_pbar = True
        
    iterator = range(n_batches)
    if show_pbar:
        iterator = tqdm(iterator, desc="  Computing embeddings", leave=False)

    for batch_idx in iterator:
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(full_sequences))
        batch_seqs = full_sequences[batch_start:batch_end]
        batch_conts = continuations[batch_start:batch_end]

        # Tokenize batch with padding
        inputs = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass through the transformer
        with torch.no_grad():
            # Access underlying transformer model
            # SentenceTransformer structure: model[0] is the Transformer module
            transformer = model[0].auto_model
            outputs = transformer(**inputs)
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)

        # For each sequence in batch, extract continuation tokens and pool
        for i in range(len(batch_seqs)):
            # Get actual token count for this sequence (without padding)
            # Tokenize the specific sequence to get its length
            seq_encoding = tokenizer.encode(batch_seqs[i], add_special_tokens=False)
            seq_len = len(seq_encoding)

            # Continuation spans from prefix_len to seq_len
            # (prefix tokens are 0:prefix_len, continuation is prefix_len:seq_len)
            cont_hidden = hidden_states[i, prefix_len:seq_len, :]

            if cont_hidden.shape[0] == 0:
                # Edge case: empty continuation (shouldn't happen but handle gracefully)
                if logger:
                    logger.warning(f"  Empty continuation hidden states for sequence {batch_start + i}")
                emb = torch.zeros(hidden_states.shape[-1], device=device)
            else:
                # Mean pool continuation hidden states
                emb = cont_hidden.mean(dim=0)

            # Normalize if requested
            if normalize:
                emb = F.normalize(emb, p=2, dim=0)

            embeddings.append(emb.cpu().numpy())

    return np.stack(embeddings)


def process_branch_file(
    branch_file: Path,
    embedding_model: SentenceTransformer,
    embedding_config: EmbeddingConfig,
    output_dir: Path,
    logger
) -> Path:
    """Process a branch sampling file and compute contextual continuation embeddings.

    For each continuation:
    - Forward pass on full sequence (prefix + continuation)
    - Extract hidden states for ONLY continuation tokens
    - Mean pool to get embedding that captures continuation meaning in prefix context

    Args:
        branch_file: Path to branch JSON file (from sample_branches.py)
        embedding_model: SentenceTransformer model
        embedding_config: Embedding configuration
        output_dir: Output directory
        logger: Logger instance

    Returns:
        Path to output file
    """
    logger.info(f"Processing: {branch_file.name}")

    # Load data
    data = load_json(branch_file)
    prefix_id = data["prefix_id"]
    prefix = data["prefix"]
    continuations_data = data.get("continuations", [])

    # Collect all continuations
    all_continuations = []
    all_metadata = []

    for cont in continuations_data:
        continuation = cont.get("text", "")
        all_continuations.append(continuation)
        all_metadata.append({
            "continuation_text": continuation,
            "token_ids": cont.get("token_ids", []),
            "probability": cont.get("probability", 0.0),
            "logprob": cont.get("logprob", 0.0),
            "num_tokens": cont.get("num_tokens", len(cont.get("token_ids", []))),
        })

    n_continuations = len(all_continuations)
    logger.info(f"Total continuations: {n_continuations}")

    if n_continuations == 0:
        logger.warning(f"No continuations in {branch_file.name}, skipping")
        return None

    # Compute contextual continuation embeddings
    logger.info(f"Computing contextual embeddings with {embedding_config.model_name}...")
    logger.info(f"  Method: Extract continuation tokens only (with prefix context via attention)")
    embeddings = embed_continuations_for_prefix(
        embedding_model,
        prefix,
        all_continuations,
        embedding_config.batch_size,
        embedding_config.device,
        normalize=True,
        logger=logger
    )

    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Save embeddings as numpy file
    output_file = output_dir / f"{prefix_id}_embeddings.npy"
    np.save(output_file, embeddings)
    logger.info(f"Saved embeddings to: {output_file}")

    # Save metadata
    # Build full sequences for reference (prefix + continuation)
    full_sequences = [prefix + cont for cont in all_continuations]

    metadata = {
        "prefix_id": prefix_id,
        "prefix": prefix,
        "n_continuations": n_continuations,
        "embedding_dim": embeddings.shape[1],
        "embedding_model": embedding_config.model_name,
        "embedding_method": "contextual_continuation",  # Extract continuation tokens with prefix context
        "continuations": all_continuations,
        "full_sequences": full_sequences,    # prefix + continuation
        "metadata": all_metadata,
    }

    metadata_file = output_dir / f"{prefix_id}_embeddings_meta.json"
    save_json(metadata, metadata_file)
    logger.info(f"Saved metadata to: {metadata_file}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Compute embeddings for continuations")
    parser.add_argument(
        "--samples-dir",
        type=Path,
        required=True,
        help="Directory containing branch sample files"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="google/embeddinggemma-300m",
        help="Sentence transformer model name (default: google/embeddinggemma-300m)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for embedding (cuda/cpu)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: 3_feature_extraction/embeddings/)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode (only progress bars)"
    )
    args = parser.parse_args()

    # Setup paths
    paths = PathConfig()
    paths.ensure_dirs()

    if args.output_dir is None:
        args.output_dir = paths.feature_extraction / "embeddings"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    import logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logger = setup_logger(
        "compute_embeddings",
        log_file=paths.feature_extraction / "compute_embeddings.log",
        level=log_level
    )

    logger.info("=" * 60)
    logger.info("COMPUTE EMBEDDINGS")
    logger.info("=" * 60)
    logger.info(f"Samples directory: {args.samples_dir}")
    logger.info(f"Embedding model: {args.embedding_model}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output directory: {args.output_dir}")

    # Setup embedding configuration
    embedding_config = EmbeddingConfig(
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"\nUsing device: {device}")

    # Load embedding model
    logger.info("\nLoading embedding model...")
    logger.info(f"Model: {args.embedding_model}")
    embedding_model = SentenceTransformer(args.embedding_model, device=str(device))
    logger.info(f"Model loaded successfully")

    # Find all branch sample files
    logger.info(f"\nFinding branch sample files in {args.samples_dir}...")
    branch_files = sorted(args.samples_dir.glob("*_branches.json"))
    logger.info(f"Found {len(branch_files)} branch files")

    if len(branch_files) == 0:
        logger.warning(f"No branch files found in {args.samples_dir}")
        logger.warning("Expected files matching pattern: *_branches.json")
        return

    # Filter branch files based on Stage 3 manifest
    # Derive results_dir from samples_dir to respect --output-dir
    # samples_dir is typically {output_dir}/results/2_branch_sampling/
    results_dir = args.samples_dir.parent
    all_prefix_ids = [f.stem.replace("_branches", "") for f in branch_files]
    available_ids, skipped_ids = filter_samples_by_manifest(
        all_prefix_ids, results_dir, "stage3", logger
    )
    # Filter branch files to only available ones
    available_id_set = set(available_ids)
    branch_files = [f for f in branch_files if f.stem.replace("_branches", "") in available_id_set]
    logger.info(f"Processing {len(branch_files)} available files (skipped {len(skipped_ids)})")

    # Process each file
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING FILES")
    logger.info("=" * 60)

    output_files = []
    completed_ids = []
    failed_ids = []
    errors = {}
    for branch_file in tqdm(branch_files, desc="Processing files"):
        prefix_id = branch_file.stem.replace("_branches", "")
        try:
            output_file = process_branch_file(
                branch_file, embedding_model, embedding_config, args.output_dir, logger
            )
            if output_file:
                output_files.append(str(output_file))
                completed_ids.append(prefix_id)
            else:
                # No continuations
                failed_ids.append(prefix_id)
                errors[prefix_id] = "No continuations found"
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Failed to process {prefix_id}: {error_msg}")
            import traceback
            logger.error(traceback.format_exc())
            failed_ids.append(prefix_id)
            errors[prefix_id] = error_msg
        logger.info("")

    # Save index
    index_data = {
        "embedding_model": args.embedding_model,
        "n_files": len(output_files),
        "output_files": output_files,
    }

    index_file = args.output_dir / "embeddings_index.json"
    save_json(index_data, index_file)

    # Write Stage 4a manifest (embeddings)
    update_manifest_with_results(
        results_dir=results_dir,
        stage_name="stage4a",
        processed=completed_ids,
        failed=failed_ids,
        skipped=skipped_ids,
        logger=logger,
        errors=errors,
    )

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Processed {len(completed_ids)} files")
    logger.info(f"Completed: {len(completed_ids)}, Failed: {len(failed_ids)}, Skipped: {len(skipped_ids)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Index file: {index_file}")


if __name__ == "__main__":
    main()
