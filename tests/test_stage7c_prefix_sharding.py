import pytest

from import_helpers import load_module_from_path


sharding = load_module_from_path(
    "stage7_prefix_sharding_test",
    "7_validation/stage7_prefix_sharding.py",
)


def test_select_prefix_shard_is_deterministic_and_ordered():
    items = ["p0", "p1", "p2", "p3", "p4", "p5", "p6"]

    first = sharding.select_prefix_shard(items, shard_index=0, shard_count=3)
    second = sharding.select_prefix_shard(items, shard_index=0, shard_count=3)

    assert first == ["p0", "p3", "p6"]
    assert second == first


def test_select_prefix_shards_cover_all_items_without_overlap():
    items = [f"prefix_{idx}" for idx in range(17)]
    shards = [
        sharding.select_prefix_shard(items, shard_index=idx, shard_count=4)
        for idx in range(4)
    ]

    flattened = [item for shard in shards for item in shard]

    assert len(flattened) == len(items)
    assert set(flattened) == set(items)
    assert sum(len(shard) for shard in shards) == len(set(flattened))


@pytest.mark.parametrize(
    ("shard_index", "shard_count"),
    [
        (0, 0),
        (0, -1),
        (-1, 2),
        (2, 2),
    ],
)
def test_select_prefix_shard_rejects_invalid_args(shard_index, shard_count):
    with pytest.raises(ValueError):
        sharding.select_prefix_shard(["p0", "p1"], shard_index, shard_count)


def test_select_prefix_shard_default_count_returns_original_list():
    items = ["p0", "p1", "p2"]

    assert sharding.select_prefix_shard(items, shard_index=0, shard_count=1) == items
