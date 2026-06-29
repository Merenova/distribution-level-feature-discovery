"""Manifest utilities for tracking pipeline sample completion across stages."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .data_utils import load_json, save_json


def load_manifest(results_dir: Path, stage_name: str) -> dict:
    """Load manifest for a stage.

    Args:
        results_dir: Base results directory
        stage_name: Stage identifier (e.g., "stage1", "stage2")

    Returns:
        Manifest dict with completed, failed, skipped lists
    """
    manifest_file = results_dir / f"manifest_{stage_name}.json"
    if manifest_file.exists():
        return load_json(manifest_file)
    return {"stage": stage_name, "completed": [], "failed": [], "skipped": []}


def save_manifest(
    results_dir: Path,
    stage_name: str,
    completed: List[str],
    failed: Optional[List[str]] = None,
    skipped: Optional[List[str]] = None,
    errors: Optional[Dict[str, str]] = None,
):
    """Save manifest for a stage.

    Args:
        results_dir: Base results directory
        stage_name: Stage identifier
        completed: List of successfully completed sample IDs
        failed: List of failed sample IDs
        skipped: List of skipped sample IDs
        errors: Dict mapping failed sample IDs to error messages
    """
    manifest = {
        "stage": stage_name,
        "timestamp": datetime.now().isoformat(),
        "completed": sorted(completed),
        "failed": sorted(failed) if failed else [],
        "skipped": sorted(skipped) if skipped else [],
        "n_completed": len(completed),
        "n_failed": len(failed) if failed else 0,
        "n_skipped": len(skipped) if skipped else 0,
        "errors": errors if errors else {},
    }
    manifest_file = results_dir / f"manifest_{stage_name}.json"
    save_json(manifest, manifest_file)
    return manifest


def get_available_samples(results_dir: Path, required_stage: str) -> List[str]:
    """Get samples that completed a required previous stage.

    Args:
        results_dir: Base results directory
        required_stage: Stage that must have completed

    Returns:
        List of sample IDs that completed the required stage
    """
    manifest = load_manifest(results_dir, required_stage)
    return manifest.get("completed", [])


def filter_samples_by_manifest(
    sample_ids: List[str],
    results_dir: Path,
    required_stage: str,
    logger=None,
) -> tuple:
    """Filter samples to only those that completed a previous stage.

    Args:
        sample_ids: Full list of sample IDs to potentially process
        results_dir: Base results directory
        required_stage: Stage that must have completed
        logger: Optional logger for info messages

    Returns:
        Tuple of (available_samples, skipped_samples)
    """
    available = set(get_available_samples(results_dir, required_stage))

    if not available:
        # No manifest exists - process all samples
        if logger:
            logger.info(f"No manifest for {required_stage}, processing all samples")
        return sample_ids, []

    available_samples = [s for s in sample_ids if s in available]
    skipped_samples = [s for s in sample_ids if s not in available]

    if logger and skipped_samples:
        logger.info(
            f"Skipping {len(skipped_samples)} samples not in {required_stage} manifest"
        )
        logger.info(f"Processing {len(available_samples)} available samples")

    return available_samples, skipped_samples


def update_manifest_with_results(
    results_dir: Path,
    stage_name: str,
    processed: List[str],
    failed: List[str],
    skipped: List[str],
    logger=None,
    errors: Optional[Dict[str, str]] = None,
):
    """Update and save manifest after processing.

    Args:
        results_dir: Base results directory
        stage_name: Stage identifier
        processed: Successfully processed sample IDs
        failed: Failed sample IDs
        skipped: Skipped sample IDs (from previous stage filter)
        logger: Optional logger
        errors: Dict mapping failed sample IDs to error messages
    """
    manifest = save_manifest(
        results_dir,
        stage_name,
        completed=processed,
        failed=failed,
        skipped=skipped,
        errors=errors,
    )

    if logger:
        logger.info(f"Manifest {stage_name}: {manifest['n_completed']} completed, "
                    f"{manifest['n_failed']} failed, {manifest['n_skipped']} skipped")

    return manifest
