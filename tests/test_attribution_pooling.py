from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from utils.attribution_pooling import load_pooled_attributions, pool_context_attributions
from utils.data_utils import save_json


def _span_info(lengths: list[int]) -> list[dict[str, int]]:
    return [
        {"start": 0, "end": length, "span_length": length, "continuation_length": length}
        for length in lengths
    ]


def _token_context() -> dict[str, object]:
    token_attributions = [
        torch.tensor([[1.0, 3.0], [3.0, 7.0]]),
        torch.tensor([[2.0, 4.0], [4.0, 8.0], [6.0, 12.0]]),
    ]
    return {
        "store_all": True,
        "token_attributions": token_attributions,
        "aggregated_attributions": torch.stack([tokens.sum(dim=0) for tokens in token_attributions]),
        "span_info": _span_info([2, 3]),
        "aggregated_attributions_pooling": "sum",
    }


def _compact_context() -> dict[str, object]:
    token_context = _token_context()
    return {
        "store_all": False,
        "aggregated_attributions": token_context["aggregated_attributions"],
        "span_info": _span_info([2, 3]),
        "aggregated_attributions_pooling": "sum",
    }


def test_store_all_false_mean_matches_store_all_true_mean():
    full = pool_context_attributions(_token_context(), pooling="mean")
    compact = pool_context_attributions(_compact_context(), pooling="mean")
    np.testing.assert_allclose(full.values, compact.values)
    np.testing.assert_allclose(compact.values, np.array([[2.0, 5.0], [4.0, 8.0]], dtype=np.float32))
    assert full.effective_pooling == "mean"
    assert compact.effective_pooling == "mean"
    assert compact.source == "aggregated_attributions_mean_from_sum"


def test_store_all_true_honors_sum_and_max_pooling():
    context = _token_context()
    summed = pool_context_attributions(context, pooling="sum")
    maxed = pool_context_attributions(context, pooling="max")
    np.testing.assert_allclose(summed.values, np.array([[4.0, 10.0], [12.0, 24.0]], dtype=np.float32))
    np.testing.assert_allclose(maxed.values, np.array([[3.0, 7.0], [6.0, 12.0]], dtype=np.float32))
    assert summed.source == "token_attributions"
    assert maxed.source == "token_attributions"


def test_store_all_true_uses_sidecar_span_info_and_slices_tokens():
    context = _token_context()
    context.pop("span_info")
    attr_meta = {
        "span_info": [
            {"start": 1, "end": 2, "span_length": 1, "continuation_length": 2},
            {"start": 1, "end": 3, "span_length": 2, "continuation_length": 3},
        ]
    }

    pooled = pool_context_attributions(context, attr_meta, pooling="sum")

    np.testing.assert_allclose(
        pooled.values,
        np.array([[3.0, 7.0], [10.0, 20.0]], dtype=np.float32),
    )
    assert pooled.source == "token_attributions"


def test_store_all_true_rejects_negative_span_bounds():
    context = _token_context()
    context["span_info"] = [
        {"start": -5, "end": -2},
        {"start": 0, "end": 3},
    ]

    with pytest.raises(ValueError, match=r"start=-5, end=-2.*continuation 0"):
        pool_context_attributions(context, pooling="sum")


def test_store_all_true_rejects_end_before_start():
    context = _token_context()
    context["span_info"] = [
        {"start": 1, "end": 2},
        {"start": 2, "end": 1},
    ]

    with pytest.raises(ValueError, match=r"start=2, end=1.*continuation 1"):
        pool_context_attributions(context, pooling="mean")


def test_store_all_true_clamps_end_past_token_length():
    context = _token_context()
    context["span_info"] = [
        {"start": 1, "end": 50},
        {"start": 2, "end": 50},
    ]

    pooled = pool_context_attributions(context, pooling="sum")

    np.testing.assert_allclose(
        pooled.values,
        np.array([[3.0, 7.0], [6.0, 12.0]], dtype=np.float32),
    )


def test_store_all_true_requires_span_info():
    context = _token_context()
    context.pop("span_info")

    with pytest.raises(ValueError, match="span_info.*required"):
        pool_context_attributions(context, pooling="mean")


def test_store_all_true_empty_context_span_info_does_not_fall_back_to_sidecar():
    context = _token_context()
    context["span_info"] = []
    attr_meta = {"span_info": _span_info([2, 3])}

    with pytest.raises(ValueError, match="span_info.*token_attributions"):
        pool_context_attributions(context, attr_meta, pooling="mean")


def test_store_all_true_rejects_truncated_span_info():
    context = _token_context()
    context["span_info"] = _span_info([2])

    with pytest.raises(ValueError, match="span_info.*token_attributions"):
        pool_context_attributions(context, pooling="mean")


@pytest.mark.parametrize("pooling", ["max"])
def test_store_all_false_rejects_pooling_that_requires_token_attributions(pooling: str):
    with pytest.raises(ValueError, match="requires token_attributions"):
        pool_context_attributions(_compact_context(), pooling=pooling)


def test_store_all_false_sum_pooling_passes_through_summed_rows():
    compact = pool_context_attributions(_compact_context(), pooling="sum")

    np.testing.assert_allclose(
        compact.values,
        np.array([[4.0, 10.0], [12.0, 24.0]], dtype=np.float32),
    )
    assert compact.values.dtype == np.float32
    assert compact.effective_pooling == "sum"
    assert compact.source == "aggregated_attributions_sum"


def test_store_all_false_sum_pooling_treats_missing_stored_pooling_as_sum():
    context = _compact_context()
    context.pop("aggregated_attributions_pooling")

    compact = pool_context_attributions(context, pooling="sum")

    np.testing.assert_allclose(
        compact.values,
        np.array([[4.0, 10.0], [12.0, 24.0]], dtype=np.float32),
    )
    assert compact.effective_pooling == "sum"
    assert compact.source == "aggregated_attributions_sum"


def test_store_all_false_stored_mean_rejects_requested_sum():
    context = {
        "store_all": False,
        "aggregated_attributions": torch.tensor([[2.0, 5.0]]),
        "aggregated_attributions_pooling": "mean",
    }

    with pytest.raises(ValueError, match="Cannot reconstruct sum|requires token_attributions"):
        pool_context_attributions(context, pooling="sum")


def test_store_all_false_requires_span_lengths_for_mean():
    context = {
        "store_all": False,
        "aggregated_attributions": torch.tensor([[4.0, 10.0]]),
        "aggregated_attributions_pooling": "sum",
    }
    with pytest.raises(ValueError, match="span_info"):
        pool_context_attributions(context, pooling="mean")


def test_store_all_false_mean_pooling_passes_through_float32():
    context = {
        "store_all": False,
        "aggregated_attributions": np.array([[1.0, 2.0]], dtype=np.float64),
        "aggregated_attributions_pooling": "mean",
    }

    pooled = pool_context_attributions(context, pooling="mean")

    np.testing.assert_allclose(pooled.values, np.array([[1.0, 2.0]], dtype=np.float32))
    assert pooled.values.dtype == np.float32
    assert pooled.source == "aggregated_attributions_mean"


def test_compact_context_span_info_can_live_inside_pt_without_sidecar():
    context = {
        "store_all": False,
        "aggregated_attributions": torch.tensor([[6.0, 12.0]]),
        "span_info": _span_info([3]),
        "aggregated_attributions_pooling": "sum",
    }

    pooled = pool_context_attributions(context, pooling="mean")

    np.testing.assert_allclose(
        pooled.values,
        np.array([[2.0, 4.0]], dtype=np.float32),
    )


def test_store_all_false_sum_pooling_allows_empty_compact_matrix():
    context = {
        "store_all": False,
        "aggregated_attributions": torch.empty((0, 2), dtype=torch.float64),
        "span_info": [],
        "aggregated_attributions_pooling": "sum",
    }

    pooled = pool_context_attributions(context, pooling="mean")

    assert pooled.values.shape == (0, 2)
    assert pooled.values.dtype == np.float32
    assert pooled.source == "aggregated_attributions_mean_from_sum"


def test_pooled_values_are_float32():
    full = pool_context_attributions(_token_context(), pooling="max")
    compact = pool_context_attributions(_compact_context(), pooling="mean")

    assert full.values.dtype == np.float32
    assert compact.values.dtype == np.float32


def test_load_pooled_attributions_reads_context_and_sidecar_metadata(tmp_path: Path):
    context_file = tmp_path / "prefix_a_prefix_context.pt"
    meta_file = tmp_path / "prefix_a_attribution.json"
    torch.save({
        "store_all": False,
        "aggregated_attributions": torch.tensor([[4.0, 10.0]]),
        "aggregated_attributions_pooling": "sum",
    }, context_file)
    save_json({"span_info": _span_info([2])}, meta_file)
    loaded = load_pooled_attributions(context_file, pooling="mean")
    np.testing.assert_allclose(loaded.values, np.array([[2.0, 5.0]], dtype=np.float32))
    assert loaded.context_file == context_file
    assert loaded.meta_file == meta_file
