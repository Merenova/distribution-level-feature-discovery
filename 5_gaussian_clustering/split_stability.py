"""Utilities for split-seed stability analysis."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np


def build_child_memberships(
    parent_global_indices: Sequence[int],
    child_local_indices: Sequence[Sequence[int]],
) -> List[List[int]]:
    """Map child memberships from local parent indices to global sample indices."""
    parent_indices = np.asarray(parent_global_indices, dtype=np.int32)
    memberships: List[List[int]] = []
    for child_local in child_local_indices:
        local_idx = np.asarray(child_local, dtype=np.int32)
        memberships.append(parent_indices[local_idx].astype(int).tolist())
    return memberships


def jaccard_similarity(indices_a: Iterable[int], indices_b: Iterable[int]) -> float:
    """Compute Jaccard similarity between two index collections."""
    set_a = set(int(idx) for idx in indices_a)
    set_b = set(int(idx) for idx in indices_b)
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def compute_label_invariant_split_jaccard(
    children_a: Sequence[Sequence[int]],
    children_b: Sequence[Sequence[int]],
) -> float:
    """Compare two binary splits up to a swap of child labels."""
    if len(children_a) != 2 or len(children_b) != 2:
        raise ValueError("Expected exactly two child memberships per split.")

    direct = 0.5 * (
        jaccard_similarity(children_a[0], children_b[0])
        + jaccard_similarity(children_a[1], children_b[1])
    )
    swapped = 0.5 * (
        jaccard_similarity(children_a[0], children_b[1])
        + jaccard_similarity(children_a[1], children_b[0])
    )
    return max(direct, swapped)


def summarize_values(values: Sequence[Optional[float]]) -> dict:
    """Summarize a list of optional numeric values."""
    valid = [float(v) for v in values if v is not None]
    if not valid:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(valid),
        "mean": float(np.mean(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
    }
