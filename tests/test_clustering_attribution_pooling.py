from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from import_helpers import load_module_from_path
from utils.data_utils import save_json


cluster = load_module_from_path("cluster", "5_gaussian_clustering/cluster.py")


def test_cluster_pooling_resolution_prefers_cli_then_config():
    assert cluster._resolve_pooling("sum", {"pooling": "max"}) == "sum"
    assert cluster._resolve_pooling(None, {"pooling": "max"}) == "max"
    assert cluster._resolve_pooling(None, {}) == "mean"


def test_cluster_load_prefix_data_uses_mean_when_store_all_false(tmp_path: Path):
    prefix_id = "prefix_a"
    samples_dir = tmp_path / "2_branch_sampling"
    embeddings_dir = tmp_path / "4_feature_extraction" / "embeddings"
    attribution_dir = tmp_path / "3_attribution_graphs"
    samples_dir.mkdir(parents=True)
    embeddings_dir.mkdir(parents=True)
    attribution_dir.mkdir(parents=True)

    save_json(
        {
            "prefix": "Question",
            "continuations": [
                {"text": "a", "probability": 0.5},
                {"text": "b", "probability": 0.5},
            ],
        },
        samples_dir / f"{prefix_id}_branches.json",
    )
    np.save(embeddings_dir / f"{prefix_id}_embeddings.npy", np.eye(2, dtype=np.float32))
    torch.save(
        {
            "store_all": False,
            "aggregated_attributions": torch.tensor([[4.0, 10.0], [12.0, 24.0]]),
            "span_info": [
                {"start": 0, "end": 2, "span_length": 2, "continuation_length": 2},
                {"start": 0, "end": 3, "span_length": 3, "continuation_length": 3},
            ],
            "aggregated_attributions_pooling": "sum",
        },
        attribution_dir / f"{prefix_id}_prefix_context.pt",
    )
    save_json(
        {
            "span_info": [
                {"start": 0, "end": 2, "span_length": 2, "continuation_length": 2},
                {"start": 0, "end": 3, "span_length": 3, "continuation_length": 3},
            ]
        },
        attribution_dir / f"{prefix_id}_attribution.json",
    )

    data = cluster.load_prefix_data(
        prefix_id,
        embeddings_dir,
        attribution_dir,
        samples_dir,
        logging.getLogger("test_cluster_load_prefix_data"),
        pooling="mean",
        metric_a="l2",
    )

    np.testing.assert_allclose(
        data["attributions_a_original"],
        np.array([[2.0, 5.0], [4.0, 8.0]], dtype=np.float32),
    )
