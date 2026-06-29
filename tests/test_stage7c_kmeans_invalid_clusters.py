from __future__ import annotations

import numpy as np
import pytest

from import_helpers import load_module_from_path


baseline_kmeans = load_module_from_path(
    "baseline_kmeans_invalid_clusters_test",
    "7_validation/7c_baseline_kmeans.py",
)


@pytest.mark.parametrize(
    ("raw_k", "n_samples", "k_clamp", "expected"),
    [
        (None, 10, None, (None, "invalid_k_missing")),
        ("bad", 10, None, (None, "invalid_k_not_integer")),
        (1, 10, None, (None, "invalid_k_less_than_2")),
        (0, 10, None, (None, "invalid_k_less_than_2")),
        (5, 10, 4, (None, "invalid_k_above_clamp")),
        (6, 5, None, (None, "invalid_k_above_samples")),
        (np.int64(3), 5, 4, (3, None)),
        ("3", 5, 4, (3, None)),
    ],
)
def test_validate_kmeans_k_returns_stable_reasons(raw_k, n_samples, k_clamp, expected):
    assert baseline_kmeans.validate_kmeans_k(raw_k, n_samples, k_clamp) == expected


def test_record_skipped_kmeans_config_records_metadata():
    prefix_results = {"prefix_id": "prefix_a"}

    baseline_kmeans.record_skipped_kmeans_config(
        prefix_results=prefix_results,
        clustering_key="beta1.0_gamma0.5",
        raw_k=0,
        reason="invalid_k_less_than_2",
        n_samples=8,
        k_clamp=20,
    )

    assert prefix_results["clustering_runs"]["beta1.0_gamma0.5"] == {
        "skipped": True,
        "skip_reason": "invalid_k_less_than_2",
        "K": 0,
        "n_samples": 8,
        "K_clamp": 20,
        "results": {},
        "timing": {},
    }


def test_run_kmeans_clustering_rejects_k_zero_before_sklearn(monkeypatch):
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("sklearn KMeans should not be constructed")

    monkeypatch.setattr(baseline_kmeans, "KMeans", fail_if_called)

    with pytest.raises(ValueError, match="invalid_k_less_than_2"):
        baseline_kmeans.run_kmeans_clustering(np.zeros((4, 2), dtype=np.float32), 0)

    assert called is False


def test_select_prefix_shard_is_exposed_from_kmeans_baseline():
    assert baseline_kmeans.select_prefix_shard(["p0", "p1", "p2", "p3"], 1, 2) == ["p1", "p3"]
