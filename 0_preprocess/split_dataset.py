#!/usr/bin/env -S uv run python
"""
Dataset splitting script.

Randomly sample a subset from a chosen split of a HuggingFace dataset on disk,
and save ONLY that subset as a new dataset with a single split named "train".

Adapted from knowledge_attribution/scripts/1_data_generation/random_split_train.py
"""

import json
import sys
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.logging_utils import setup_logger


def _fix_features_dict(features_dict: Any) -> Any:
    """Recursively convert '_type': 'List' to '_type': 'Sequence' in a features dictionary."""
    if isinstance(features_dict, dict):
        if features_dict.get('_type') == 'List':
            features_dict['_type'] = 'Sequence'
        for k, v in list(features_dict.items()):
            features_dict[k] = _fix_features_dict(v)
    elif isinstance(features_dict, list):
        return [_fix_features_dict(v) for v in features_dict]
    return features_dict


def _patch_dataset_info_for_split(dataset_path: Path, split_name: str) -> bool:
    """Patch dataset_info.json for a specific split in-place if it uses 'List' types."""
    info_path = dataset_path / split_name / "dataset_info.json"
    if not info_path.exists():
        return False

    with open(info_path, "r") as f:
        info = json.load(f)

    original_info = json.dumps(info, sort_keys=True)
    if "features" in info:
        info["features"] = _fix_features_dict(info["features"])

    fixed_info = json.dumps(info, sort_keys=True)
    if fixed_info == original_info:
        return False

    backup_path = dataset_path / split_name / "dataset_info_original.json"
    info_path.rename(backup_path)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    return True


def _restore_dataset_info_for_split(dataset_path: Path, split_name: str) -> None:
    """Restore original dataset_info.json for the given split if we backed it up."""
    info_path = dataset_path / split_name / "dataset_info.json"
    backup_path = dataset_path / split_name / "dataset_info_original.json"
    if backup_path.exists():
        info_path.unlink(missing_ok=True)
        backup_path.rename(info_path)


def load_dataset_with_optional_fix(dataset_path: str, split_name: str) -> DatasetDict:
    """Load a dataset from disk, patching the specified split's dataset_info.json if needed."""
    ds_path = Path(dataset_path)
    patched = _patch_dataset_info_for_split(ds_path, split_name)
    try:
        ds = load_from_disk(str(ds_path))
    finally:
        if patched:
            _restore_dataset_info_for_split(ds_path, split_name)
    return ds


def sample_dataset(
    input_path: str,
    output_path: str,
    split_name: str = "train",
    keep_ratio: float = 0.1,
    seed: int = 42,
    shuffle: bool = True,
    logger=None,
    quiet: bool = False
) -> None:
    """
    Keep an exact ratio from the specified split and save the result as a new dataset
    containing ONLY a 'train' split.
    """
    log = logger.info if logger else print
    if quiet:
        def noop(*args, **kwargs): pass
        log = noop

    log(f"Loading dataset from {input_path}...")
    ds = load_dataset_with_optional_fix(input_path, split_name)

    if split_name not in ds:
        raise ValueError(
            f"Split '{split_name}' not found in dataset. Available splits: {list(ds.keys())}"
        )

    src = ds[split_name]
    n = len(src)
    log(f"Source split: '{split_name}' size = {n}")

    if shuffle:
        log(f"Shuffling with seed {seed}...")
        src = src.shuffle(seed=seed)

    keep_n = max(1, int(n * keep_ratio))
    if keep_n > n:
        keep_n = n
    log(f"Keeping {keep_n} / {n} samples ({keep_ratio*100:.1f}%). Discarding the rest.")

    subset = src.select(range(keep_n))

    new_ds = DatasetDict({"train": subset})

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    log(f"Saving new dataset (ONLY 'train') to: {out}")
    new_ds.save_to_disk(str(out))

    # Write metadata
    split_info = {
        "original_dataset": str(input_path),
        "sampled_from_split": split_name,
        "original_split_size": n,
        "keep_ratio": keep_ratio,
        "seed": seed,
        "shuffle": shuffle,
        "new_train_size": len(new_ds["train"]),
        "note": "Only the kept subset is saved as 'train'. Other splits not included.",
    }
    info_path = out / "split_info.json"
    with open(info_path, "w") as f:
        json.dump(split_info, f, indent=2)

    log("Done!")
    log(f"- Final 'train' size: {len(new_ds['train'])}")
    log(f"- Split information saved to: {info_path}")


def main():
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="Sample a proportion from a chosen split and save ONLY as 'train'.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to input dataset")
    parser.add_argument("--output", type=Path, required=True, help="Path to save sampled dataset")
    parser.add_argument("--split-name", type=str, default="train",
                        help="Name of the split to sample from")
    parser.add_argument("--keep-ratio", type=float, default=0.1,
                        help="Proportion of the chosen split to keep (0.0 < r < 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Do not shuffle before selecting")
    parser.add_argument("--log-dir", type=Path, default=None, help="Log directory")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    if not 0.0 < args.keep_ratio < 1.0:
        parser.error("--keep-ratio must be between 0.0 and 1.0 (exclusive).")

    log_level = logging.WARNING if args.quiet else logging.INFO
    logger = setup_logger("split_dataset",
                          log_file=args.log_dir / "split_dataset.log" if args.log_dir else None,
                          level=log_level)

    sample_dataset(
        input_path=str(args.input),
        output_path=str(args.output),
        split_name=args.split_name,
        keep_ratio=args.keep_ratio,
        seed=args.seed,
        shuffle=not args.no_shuffle,
        logger=logger,
        quiet=args.quiet
    )


if __name__ == "__main__":
    main()
