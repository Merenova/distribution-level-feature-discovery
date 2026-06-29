from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
import torch

from import_helpers import load_module_from_path
from utils.data_utils import save_json


ROOT = Path(__file__).resolve().parents[1]

baseline_kmeans = load_module_from_path(
    "baseline_kmeans_pooling_test",
    "7_validation/7c_baseline_kmeans.py",
)
cross_silhouette = load_module_from_path(
    "cross_silhouette_pooling_test",
    "8_visualization/cross_silhouette_analysis.py",
)
extract_graphs = load_module_from_path(
    "extract_graphs_pooling_test",
    "6_semantic_graphs/extract_graphs.py",
)


def test_downstream_continuation_loaders_import_shared_pooling_helper():
    expected_files = [
        ROOT / "5_gaussian_clustering" / "cluster.py",
        ROOT / "6_semantic_graphs" / "extract_graphs.py",
        ROOT / "7_validation" / "7a_graph_validation.py",
        ROOT / "7_validation" / "7c_baseline_combined_medoid.py",
        ROOT / "7_validation" / "7c_baseline_kmeans.py",
        ROOT / "7_validation" / "7c_baseline_single.py",
        ROOT / "7_validation" / "extract_tokenwise_logit_diff.py",
        ROOT / "8_visualization" / "cross_silhouette_analysis.py",
        ROOT / "8_visualization" / "visualize.py",
    ]

    for path in expected_files:
        text = path.read_text()
        assert "utils.attribution_pooling" in text, path
        assert "load_pooled_attributions" in text, path


def test_main_pipeline_passes_pooling_to_stage6_and_stage7a():
    text = (ROOT / "scripts" / "run_pipeline.sh").read_text()
    stage6_command = text.split("uv run python 6_semantic_graphs/extract_graphs.py", 1)[1].split(
        "$QUIET_ARG", 1
    )[0]
    stage7a_command = text.split("uv run python 7_validation/7a_graph_validation.py", 1)[1].split(
        "$QUIET_ARG", 1
    )[0]
    assert "--pooling \"$CLUSTERING_POOLING\"" in stage6_command
    assert "--pooling \"$CLUSTERING_POOLING\"" in stage7a_command
    assert '"pooling": "$CLUSTERING_POOLING"' in text


def test_stage6_extract_semantic_graphs_uses_requested_pooling(tmp_path: Path):
    context_file = tmp_path / "prefix_a_prefix_context.pt"
    torch.save(
        {
            "store_all": False,
            "aggregated_attributions": torch.tensor([[4.0, 10.0], [12.0, 24.0]]),
            "span_info": [
                {"start": 0, "end": 2, "span_length": 2, "continuation_length": 2},
                {"start": 0, "end": 3, "span_length": 3, "continuation_length": 3},
            ],
            "aggregated_attributions_pooling": "sum",
            "n_prefix_features": 2,
            "n_prefix_errors": 0,
            "n_prefix_tokens": 0,
            "n_prefix_sources": 2,
            "decoder_locations": torch.tensor([[0, 0], [0, 1]]),
            "selected_features": torch.tensor([0, 1]),
            "activation_matrix": None,
        },
        context_file,
    )
    save_json(
        {
            "span_info": [
                {"start": 0, "end": 2, "span_length": 2, "continuation_length": 2},
                {"start": 0, "end": 3, "span_length": 3, "continuation_length": 3},
            ]
        },
        tmp_path / "prefix_a_attribution.json",
    )

    graphs = extract_graphs.extract_semantic_graphs(
        {
            "components": {"0": {"mu_a": [0.0, 0.0]}},
            "assignments": [0, 0],
            "H_0": None,
        },
        {
            "continuations": [
                {"probability": 0.5},
                {"probability": 0.5},
            ]
        },
        context_file,
        logging.getLogger("test_stage6_pooling"),
        pooling="sum",
    )

    assert graphs["component_ids"] == [0]
    assert graphs["n_features"] == 2
    assert graphs["active_features"].shape[0] == 2


def test_baseline_kmeans_pooling_resolution_prefers_cli_then_config():
    assert baseline_kmeans._resolve_pooling("sum", {"clustering": {"pooling": "max"}}) == "sum"
    assert baseline_kmeans._resolve_pooling(None, {"clustering": {"pooling": "max"}}) == "max"
    assert baseline_kmeans._resolve_pooling(None, {}) == "mean"


@pytest.mark.parametrize(
    ("config_pooling", "cli_pooling", "expected_pooling"),
    [
        ("sum", None, "sum"),
        ("sum", "max", "max"),
    ],
)
def test_baseline_kmeans_main_resolves_pooling_before_loading_prefix_data(
    monkeypatch,
    tmp_path: Path,
    config_pooling: str,
    cli_pooling: str | None,
    expected_pooling: str,
):
    samples_dir = tmp_path / "2_branch_sampling"
    embeddings_dir = tmp_path / "4_feature_extraction" / "embeddings"
    attribution_dir = tmp_path / "3_attribution_graphs"
    clustering_dir = tmp_path / "5_clustering"
    output_dir = tmp_path / "7_validation"
    for path in [samples_dir, embeddings_dir, attribution_dir, clustering_dir]:
        path.mkdir(parents=True)

    prefix_id = "prefix_a"
    np.save(embeddings_dir / f"{prefix_id}_embeddings.npy", np.zeros((2, 2), dtype=np.float32))
    save_json({"grid": []}, clustering_dir / f"{prefix_id}_sweep_results.json")
    config_file = tmp_path / "config.json"
    save_json(
        {
            "clustering": {"pooling": config_pooling},
            "stage_7c_steering": {"sweeps": []},
            "global": {"max_seq_len": 8},
            "model": {"base_model": "dummy-model", "transcoder": "dummy-transcoder"},
        },
        config_file,
    )
    seen = {}

    monkeypatch.setattr(
        baseline_kmeans.ReplacementModel,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: object()),
    )
    monkeypatch.setattr(baseline_kmeans, "get_model_device", lambda model, fallback=None: fallback)

    def fake_load_prefix_data_for_baseline(*args, **kwargs):
        seen["pooling"] = kwargs["pooling"]
        raise SystemExit("stop after pooling resolution")

    monkeypatch.setattr(
        baseline_kmeans,
        "load_prefix_data_for_baseline",
        fake_load_prefix_data_for_baseline,
    )

    argv = [
        "7c_baseline_kmeans.py",
        "--samples-dir",
        str(samples_dir),
        "--embeddings-dir",
        str(embeddings_dir),
        "--attribution-graphs-dir",
        str(attribution_dir),
        "--clustering-dir",
        str(clustering_dir),
        "--output-dir",
        str(output_dir),
        "--config",
        str(config_file),
        "--max-samples",
        "1",
    ]
    if cli_pooling is not None:
        argv.extend(["--pooling", cli_pooling])
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit, match="stop after pooling resolution"):
        baseline_kmeans.main()

    assert seen == {"pooling": expected_pooling}


def test_cross_silhouette_analyze_prefix_passes_pooling_to_loader(
    tmp_path: Path,
    monkeypatch,
):
    prefix_id = "prefix_a"
    clustering_dir = tmp_path / "5_clustering"
    embeddings_dir = tmp_path / "4_feature_extraction" / "embeddings"
    attribution_dir = tmp_path / "3_attribution_graphs"
    clustering_dir.mkdir(parents=True)
    embeddings_dir.mkdir(parents=True)
    attribution_dir.mkdir(parents=True)
    save_json(
        {
            "grid": [
                {
                    "beta": 1.0,
                    "gamma": 0.5,
                    "K": 2,
                    "sil_e": 0.1,
                    "sil_a": 0.2,
                    "harmonic": 0.15,
                }
            ]
        },
        clustering_dir / f"{prefix_id}_sweep_results.json",
    )
    np.save(
        embeddings_dir / f"{prefix_id}_embeddings.npy",
        np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    )
    seen = {}

    def fake_load_attributions(attribution_dir_arg, prefix_id_arg, pooling="mean"):
        seen["attribution_dir"] = attribution_dir_arg
        seen["prefix_id"] = prefix_id_arg
        seen["pooling"] = pooling
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(cross_silhouette, "load_attributions", fake_load_attributions)
    monkeypatch.setattr(cross_silhouette, "run_kmeans_sweep", lambda data, K_range: {})

    result = cross_silhouette.analyze_prefix(
        prefix_id,
        clustering_dir,
        embeddings_dir,
        attribution_dir,
        [2],
        logging.getLogger("test_cross_silhouette_pooling"),
        pooling="sum",
    )

    assert result is not None
    assert seen == {
        "attribution_dir": attribution_dir,
        "prefix_id": prefix_id,
        "pooling": "sum",
    }


def test_cross_silhouette_main_passes_cli_pooling_to_analyze_prefix(monkeypatch, tmp_path: Path):
    clustering_dir = tmp_path / "5_clustering"
    embeddings_dir = tmp_path / "embeddings"
    attribution_dir = tmp_path / "3_attribution_graphs"
    output_dir = tmp_path / "out"
    clustering_dir.mkdir()
    embeddings_dir.mkdir()
    attribution_dir.mkdir()
    save_json({"grid": [{"beta": 1.0, "gamma": 0.5}]}, clustering_dir / "prefix_a_sweep_results.json")
    seen = {}

    def fake_analyze_prefix(prefix_id, clustering_dir_arg, embeddings_dir_arg, attribution_dir_arg, K_range, logger, pooling="mean"):
        seen["prefix_id"] = prefix_id
        seen["pooling"] = pooling
        return {
            "best": {
                "rd": {"sil_e": -1, "sil_a": -1, "harmonic": -1},
                "kmeans_e_native": {"sil_e": -1, "sil_a": -1},
                "kmeans_e_harmonic": {"sil_e": -1, "sil_a": -1},
                "kmeans_a_native": {"sil_e": -1, "sil_a": -1},
                "kmeans_a_harmonic": {"sil_e": -1, "sil_a": -1},
            }
        }

    monkeypatch.setattr(cross_silhouette, "analyze_prefix", fake_analyze_prefix)
    monkeypatch.setattr(cross_silhouette, "print_summary_table", lambda agg, logger: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "cross_silhouette_analysis.py",
            "--clustering-dir",
            str(clustering_dir),
            "--embeddings-dir",
            str(embeddings_dir),
            "--attribution-dir",
            str(attribution_dir),
            "--output-dir",
            str(output_dir),
            "--pooling",
            "sum",
        ],
    )

    cross_silhouette.main()

    assert seen == {"prefix_id": "prefix_a", "pooling": "sum"}
