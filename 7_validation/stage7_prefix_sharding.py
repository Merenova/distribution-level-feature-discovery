"""Deterministic prefix sharding helpers for Stage 7 jobs."""

from typing import List, Sequence, TypeVar


T = TypeVar("T")


def select_prefix_shard(
    prefix_items: Sequence[T],
    shard_index: int,
    shard_count: int,
) -> List[T]:
    """Return the ordered subset assigned to a deterministic modulo shard."""
    if shard_count < 1:
        raise ValueError(f"shard_count must be >= 1, got {shard_count}")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"shard_index must satisfy 0 <= shard_index < shard_count, "
            f"got shard_index={shard_index}, shard_count={shard_count}"
        )

    return [
        item
        for position, item in enumerate(prefix_items)
        if position % shard_count == shard_index
    ]
