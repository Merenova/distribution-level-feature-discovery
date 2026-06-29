from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "publication" / "plot_reasoning_fixed_cluster_effects.py"
ALLUVIAL_SCRIPT = ROOT / "scripts" / "publication" / "plot_reasoning_cluster_alluvial.py"
LEGACY_SCRIPT = ROOT / "scripts" / "publication" / "plot_reasoning_cluster_effects.py"


def load_plot_module():
    scripts_dir = str(ROOT / "scripts" / "publication")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("plot_reasoning_fixed_cluster_effects", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_alluvial_module():
    scripts_dir = str(ROOT / "scripts" / "publication")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("plot_reasoning_cluster_alluvial", ALLUVIAL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def continuation(text: str, probability: float, token_ids: list[int]) -> dict[str, Any]:
    return {"text": text, "probability": probability, "token_ids": token_ids}


def write_synthetic_reasoning_case(tmp_path: Path) -> None:
    lane = "synthetic_lane"
    prefixes = [
        {
            "prefix_id": "synthetic_i1_j3",
            "source_step": 1,
            "target_step": 3,
            "K": 2,
            "mean_abs_rho_s": 0.3,
            "max_abs_rho_s": 0.8,
        },
        {
            "prefix_id": "synthetic_i2_j3",
            "source_step": 2,
            "target_step": 3,
            "K": 2,
            "mean_abs_rho_s": 0.7,
            "max_abs_rho_s": 0.9,
        },
    ]
    write_json(
        tmp_path / "selected_cases.json",
        {
            "selected_cases": [
                {
                    "dataset": "gsm 8k/math",
                    "lane": lane,
                    "example_id": "synthetic case/1",
                    "target_step": 3,
                    "prefixes": prefixes,
                    "score": 1.0,
                }
            ]
        },
    )

    continuations = [
        continuation("A direct answer", 0.50, [1]),
        continuation("A longer explanation", 0.30, [2]),
        continuation("A corrected answer", 0.20, [3]),
    ]
    for prefix in prefixes:
        prefix_id = prefix["prefix_id"]
        write_json(
            tmp_path / "selected_case_inputs" / lane / f"{prefix_id}_branches.json",
            {
                "prefix": "Solve the following problem.\n\nWhat is 1+1?",
                "continuations": continuations,
                "reasoning_metadata": {
                    "prompt_text": "Solve the following problem.\n\nWhat is 1+1?",
                    "committed_previous_steps": ["Add one.", "Get two."],
                },
            },
        )

    write_json(
        tmp_path / "selected_case_inputs" / lane / "synthetic_i1_j3_sweep_results.json",
        {
            "grid": [
                {"beta": 0.1, "gamma": 0.2, "K": 2, "assignments": [99, 99, 99]},
                {"beta": 0.75, "gamma": 0.7, "K": 2, "assignments": [10, 10, 11]},
            ]
        },
    )
    write_json(
        tmp_path / "selected_case_inputs" / lane / "synthetic_i2_j3_sweep_results.json",
        {
            "grid": [
                {"beta": 0.1, "gamma": 0.2, "K": 2, "assignments": [99, 99, 99]},
                {"beta": 0.75, "gamma": 0.7, "K": 2, "assignments": [2, 3, 3]},
            ]
        },
    )

    for prefix in prefixes:
        prefix_id = prefix["prefix_id"]
        write_json(
            tmp_path
            / "runs"
            / lane
            / "7_validation/7c_combined_medoid/H4a_combined_medoid"
            / f"{prefix_id}_sweep_results.json",
            {
                "clustering_runs": {
                    "beta0.75_gamma0.7": {
                        "results": {
                            "sign_full_B5": {
                                "per_cluster_logit": {
                                    "2": {"centered_logit_spearman": 0.1},
                                    "3": {"centered_logit_spearman": -0.2},
                                    "10": {"centered_logit_spearman": 0.8},
                                    "11": {"centered_logit_spearman": -0.4},
                                }
                            }
                        }
                    }
                }
            },
        )


def test_fixed_cluster_effect_plot_writes_png_pdf_and_metadata(tmp_path: Path):
    write_synthetic_reasoning_case(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--selected-cases",
            "selected_cases.json",
            "--case-root",
            ".",
            "--stage7-root",
            "runs",
            "--output-root",
            "figures",
            "--case-index",
            "1",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output_base = tmp_path / "figures" / "fixed_cluster_effects_gsm_8k_math_synthetic_case_1_j03"
    assert output_base.with_suffix(".png").exists()
    assert output_base.with_suffix(".pdf").exists()
    metadata_path = output_base.with_suffix(".metadata.json")
    assert metadata_path.exists()
    tex_path = tmp_path / "figures" / "reasoning_fixed_cluster_effect_figures.tex"
    aggregate_path = tmp_path / "figures" / "reasoning_fixed_cluster_effect_figures.json"
    assert tex_path.exists()
    assert aggregate_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["figure"] == {
        "png": "figures/fixed_cluster_effects_gsm_8k_math_synthetic_case_1_j03.png",
        "pdf": "figures/fixed_cluster_effects_gsm_8k_math_synthetic_case_1_j03.pdf",
    }
    assert "fixed_clusters" in metadata["case"]
    assert "per_source_effects" in metadata["case"]
    assert [round(cluster["mass_frac"], 3) for cluster in metadata["case"]["fixed_clusters"]] == [
        0.5,
        0.5,
    ]
    source1_effects = metadata["case"]["per_source_effects"][0]["fixed_cluster_effects"]
    source2_effects = metadata["case"]["per_source_effects"][1]["fixed_cluster_effects"]
    assert round(source1_effects["F1"]["rho_s"], 3) == 0.8
    assert round(source1_effects["F2"]["rho_s"], 3) == 0.32
    assert round(source2_effects["F1"]["rho_s"], 3) == 0.1
    assert round(source2_effects["F2"]["rho_s"], 3) == -0.2

    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["figures"][0]["metadata_json"] == (
        "figures/fixed_cluster_effects_gsm_8k_math_synthetic_case_1_j03.metadata.json"
    )
    assert aggregate["figures"][0]["figure"]["pdf"] == metadata["figure"]["pdf"]
    assert aggregate["figures"][0]["case"]["example_id"] == "synthetic case/1"

    tex = tex_path.read_text(encoding="utf-8")
    assert "figures/fixed_cluster_effects_gsm_8k_math_synthetic_case_1_j03.pdf" in tex
    assert str(tmp_path) not in tex


def test_alluvial_plot_writes_png_pdf_and_metadata(tmp_path: Path):
    write_synthetic_reasoning_case(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ALLUVIAL_SCRIPT),
            "--selected-cases",
            "selected_cases.json",
            "--case-root",
            ".",
            "--stage7-root",
            "runs",
            "--output-root",
            "figures",
            "--case-index",
            "1",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    output_base = tmp_path / "figures" / "alluvial_cluster_dynamics_gsm_8k_math_synthetic_case_1_j03"
    assert output_base.with_suffix(".png").exists()
    assert output_base.with_suffix(".pdf").exists()
    metadata_path = output_base.with_suffix(".metadata.json")
    assert metadata_path.exists()
    tex_path = tmp_path / "figures" / "reasoning_alluvial_cluster_figures.tex"
    aggregate_path = tmp_path / "figures" / "reasoning_alluvial_cluster_figures.json"
    assert tex_path.exists()
    assert aggregate_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["figure"] == {
        "png": "figures/alluvial_cluster_dynamics_gsm_8k_math_synthetic_case_1_j03.png",
        "pdf": "figures/alluvial_cluster_dynamics_gsm_8k_math_synthetic_case_1_j03.pdf",
    }
    assert "adjacent_cluster_flows" in metadata["case"]
    assert "local_cluster_partitions" in metadata["case"]
    assert metadata["case"]["adjacent_cluster_flows"][0]["flows"]
    assert metadata["case"]["local_cluster_partitions"][0]["members"]
    assert metadata["render"]["script"] == "plot_reasoning_cluster_alluvial.py"
    assert metadata["render"]["run_key"] == "beta0.75_gamma0.7"
    assert metadata["render"]["sweep_key"] == "sign_full_B5"
    assert metadata["render"]["case_index"] == 1
    diagnostics = metadata["render"]["ribbon_diagnostics"]
    assert diagnostics["n_segments"] == 3
    assert diagnostics["n_skipped_flows"] == 0
    assert diagnostics["max_source_overflow"] <= diagnostics["epsilon"]
    assert diagnostics["max_target_overflow"] <= diagnostics["epsilon"]

    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["figures"][0]["metadata_json"] == (
        "figures/alluvial_cluster_dynamics_gsm_8k_math_synthetic_case_1_j03.metadata.json"
    )
    assert aggregate["figures"][0]["figure"]["pdf"] == metadata["figure"]["pdf"]

    tex = tex_path.read_text(encoding="utf-8")
    assert "figures/alluvial_cluster_dynamics_gsm_8k_math_synthetic_case_1_j03.pdf" in tex
    assert str(tmp_path) not in tex


def test_alluvial_helpers_scale_flows_to_gapped_node_heights():
    alluvial = load_alluvial_module()
    partitions = [
        {
            "source_step": 1,
            "members": {
                "A": {"a": 0.75},
                "B": {"b": 0.25},
            },
        },
        {
            "source_step": 2,
            "members": {
                "X": {"x": 0.50},
                "Y": {"y": 0.50},
            },
        },
    ]
    layouts = alluvial.layout_nodes(partitions, gap=0.10)
    assert layouts[0]["A"]["height"] == pytest.approx(0.675)
    assert layouts[0]["B"]["height"] == pytest.approx(0.225)
    assert layouts[1]["X"]["height"] == pytest.approx(0.45)
    assert layouts[1]["Y"]["height"] == pytest.approx(0.45)

    segments, diagnostics = alluvial.compute_ribbon_segments(
        [
            {
                "source_step": 1,
                "target_step": 2,
                "flows": [
                    {"source_cluster_id": "A", "target_cluster_id": "Y", "mass": 0.25},
                    {"source_cluster_id": "B", "target_cluster_id": "Y", "mass": 0.25},
                    {"source_cluster_id": "A", "target_cluster_id": "X", "mass": 0.50},
                ],
            }
        ],
        layouts,
        [1, 2],
    )

    assert diagnostics["n_segments"] == 3
    assert diagnostics["n_skipped_flows"] == 0
    by_pair = {
        (segment["source_cluster_id"], segment["target_cluster_id"]): segment
        for segment in segments
    }
    a_to_x = by_pair[("A", "X")]
    a_to_y = by_pair[("A", "Y")]
    b_to_y = by_pair[("B", "Y")]

    assert a_to_x["source_y0"] == pytest.approx(layouts[0]["A"]["y0"])
    assert a_to_x["source_visual_height"] == pytest.approx(0.45)
    assert a_to_y["source_y0"] == pytest.approx(a_to_x["source_y1"])
    assert a_to_y["source_visual_height"] == pytest.approx(0.225)
    assert a_to_y["source_y1"] == pytest.approx(layouts[0]["A"]["y1"])

    assert a_to_y["target_y0"] == pytest.approx(layouts[1]["Y"]["y0"])
    assert a_to_y["target_visual_height"] == pytest.approx(0.225)
    assert b_to_y["target_y0"] == pytest.approx(a_to_y["target_y1"])
    assert b_to_y["target_visual_height"] == pytest.approx(0.225)
    assert b_to_y["target_y1"] == pytest.approx(layouts[1]["Y"]["y1"])


def test_alluvial_helpers_raise_when_flow_exceeds_node_height():
    alluvial = load_alluvial_module()
    layouts = {
        0: {"A": {"y0": 0.0, "y1": 0.5, "height": 0.5, "mass": 0.5}},
        1: {"B": {"y0": 0.0, "y1": 1.0, "height": 1.0, "mass": 1.0}},
    }

    with pytest.raises(ValueError, match="ribbon flow exceeds visual node height"):
        alluvial.compute_ribbon_segments(
            [
                {
                    "source_step": 1,
                    "target_step": 2,
                    "flows": [
                        {"source_cluster_id": "A", "target_cluster_id": "B", "mass": 0.6},
                    ],
                }
            ],
            layouts,
            [1, 2],
        )


def test_alluvial_script_rejects_zero_selected_cases(tmp_path: Path):
    write_synthetic_reasoning_case(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            str(ALLUVIAL_SCRIPT),
            "--selected-cases",
            "selected_cases.json",
            "--case-root",
            ".",
            "--stage7-root",
            "runs",
            "--output-root",
            "figures",
            "--example-id",
            "missing",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "No selected cases matched the requested filters." in result.stderr
    assert not (tmp_path / "figures" / "reasoning_alluvial_cluster_figures.json").exists()


def test_legacy_cluster_effect_script_delegates_to_fixed_cluster_script():
    result = subprocess.run(
        [
            sys.executable,
            str(LEGACY_SCRIPT),
            "--help",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "fixed target-cluster" in result.stdout


def test_select_cases_filters_before_case_index(tmp_path: Path):
    plot = load_plot_module()
    write_json(
        tmp_path / "selected_cases.json",
        {
            "selected_cases": [
                {"example_id": "keep", "target_step": 3, "case": "first"},
                {"example_id": "drop", "target_step": 3, "case": "wrong-example"},
                {"example_id": "keep", "target_step": 4, "case": "wrong-step"},
                {"example_id": "keep", "target_step": 3, "case": "second"},
            ]
        },
    )

    selected = plot.select_cases(
        tmp_path / "selected_cases.json",
        case_index=2,
        example_id="keep",
        target_step=3,
        max_cases=10,
    )

    assert [case["case"] for case in selected] == ["second"]


def test_select_cases_rejects_max_cases_less_than_one(tmp_path: Path):
    plot = load_plot_module()
    write_json(tmp_path / "selected_cases.json", {"selected_cases": []})

    with pytest.raises(SystemExit, match="--max-cases must be >= 1"):
        plot.select_cases(
            tmp_path / "selected_cases.json",
            case_index=None,
            example_id=None,
            target_step=None,
            max_cases=0,
        )
