from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_selector_module():
    path = ROOT / "7_validation" / "select_stage7_clustering_manifest.py"
    spec = importlib.util.spec_from_file_location("stage7_manifest_selector", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


selector = load_selector_module()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_build_manifest_selects_top_k_per_prefix_from_sweep_and_aggregate(tmp_path: Path):
    stage5_dir = tmp_path / "5_clustering"
    write_json(
        stage5_dir / "cloze_a_sweep_results.json",
        {
            "prefix_id": "cloze_a",
            "grid": [
                {"beta": 1.0, "gamma": 0.5, "K": 2, "harmonic": 0.2},
                {"beta": 0.5, "gamma": 0.4, "K": 3, "harmonic": 0.9},
                {"beta": 2.0, "gamma": 0.7, "K": 4, "harmonic": 0.8},
            ],
        },
    )
    write_json(
        stage5_dir / "aggregate_summary.json",
        {
            "results": [
                {
                    "prefix_id": "cloze_b",
                    "grid": [
                        {"beta": 4.0, "gamma": 0.1, "K": 2, "harmonic": 0.4},
                        {"beta": 3.0, "gamma": 0.2, "K": 5, "harmonic": 0.6},
                    ],
                }
            ]
        },
    )

    manifest = selector.build_manifest(stage5_dir, top_k=2)

    assert [
        (entry["prefix_id"], entry["beta"], entry["gamma"], entry["K"], entry["score"])
        for entry in manifest
    ] == [
        ("cloze_a", 0.5, 0.4, 3, 0.9),
        ("cloze_a", 2.0, 0.7, 4, 0.8),
        ("cloze_b", 3.0, 0.2, 5, 0.6),
        ("cloze_b", 4.0, 0.1, 2, 0.4),
    ]
    assert all(entry["cloze_id"] == entry["prefix_id"] for entry in manifest)
    assert all(entry["score_key"] == "harmonic" for entry in manifest)
    assert all(entry["source_path"].endswith(".json") for entry in manifest)
    assert all(entry["source_ref"].startswith("$.") for entry in manifest)


def test_build_manifest_applies_k_min_and_max_filters(tmp_path: Path):
    stage5_dir = tmp_path / "5_clustering"
    write_json(
        stage5_dir / "cloze_a_sweep_results.json",
        {
            "grid": [
                {"beta": 1.0, "gamma": 0.1, "K": 1, "harmonic": 0.99},
                {"beta": 2.0, "gamma": 0.1, "K": 2, "harmonic": 0.5},
                {"beta": 3.0, "gamma": 0.1, "K": 4, "harmonic": 0.7},
                {"beta": 4.0, "gamma": 0.1, "K": 6, "harmonic": 0.95},
            ]
        },
    )

    manifest = selector.build_manifest(stage5_dir, top_k=10, min_k=2, max_k=4)

    assert [(entry["K"], entry["score"]) for entry in manifest] == [(4, 0.7), (2, 0.5)]


def test_build_manifest_skips_entries_with_missing_scores(tmp_path: Path):
    stage5_dir = tmp_path / "5_clustering"
    write_json(
        stage5_dir / "cloze_a_sweep_results.json",
        {
            "grid": [
                {"beta": 1.0, "gamma": 0.1, "K": 2},
                {"beta": 2.0, "gamma": 0.1, "K": 2, "harmonic": None},
                {"beta": 3.0, "gamma": 0.1, "K": 2, "harmonic": "nan"},
                {
                    "beta": 4.0,
                    "gamma": 0.1,
                    "components": {"0": {}, "1": {}},
                    "metrics": {"harmonic": 0.4},
                },
            ]
        },
    )

    manifest = selector.build_manifest(stage5_dir)

    assert [(entry["beta"], entry["K"], entry["score"]) for entry in manifest] == [
        (4.0, 2, 0.4)
    ]


def test_build_manifest_uses_deterministic_tie_breaking(tmp_path: Path):
    stage5_dir = tmp_path / "5_clustering"
    write_json(
        stage5_dir / "cloze_a_sweep_results.json",
        {
            "grid": [
                {"beta": 2.0, "gamma": 0.4, "K": 2, "harmonic": 0.5},
                {"beta": 1.0, "gamma": 0.6, "K": 2, "harmonic": 0.5},
                {"beta": 1.0, "gamma": 0.4, "K": 3, "harmonic": 0.5},
                {"beta": 1.0, "gamma": 0.4, "K": 2, "harmonic": 0.5},
            ]
        },
    )

    manifest = selector.build_manifest(stage5_dir, top_k=4)

    assert [(entry["beta"], entry["gamma"], entry["K"]) for entry in manifest] == [
        (1.0, 0.4, 2),
        (1.0, 0.4, 3),
        (1.0, 0.6, 2),
        (2.0, 0.4, 2),
    ]


def test_build_manifest_can_limit_scanned_prefix_files(tmp_path: Path):
    stage5_dir = tmp_path / "5_clustering"
    for prefix in ["cloze_a", "cloze_b", "cloze_c"]:
        write_json(
            stage5_dir / f"{prefix}_sweep_results.json",
            {"grid": [{"beta": 1.0, "gamma": 0.1, "K": 2, "harmonic": 0.5}]},
        )

    manifest = selector.build_manifest(stage5_dir, max_prefixes=2)

    assert [entry["prefix_id"] for entry in manifest] == ["cloze_a", "cloze_b"]
