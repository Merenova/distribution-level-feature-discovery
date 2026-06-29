from __future__ import annotations

import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module_from_path(module_name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


dyn = load_module_from_path(
    "reasoning_dynamics_data",
    "scripts/publication/reasoning_dynamics_data.py",
)


def continuation(text: str, probability: float, token_ids: list[int]):
    return {"text": text, "probability": probability, "token_ids": token_ids}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_continuation_key_prefers_token_ids():
    row = continuation(" same text ", 0.5, [10, 20, 30])

    assert dyn.continuation_key(row) == "tok:10,20,30"


def test_continuation_key_falls_back_to_normalized_text():
    row = {"text": " same\n\n text\t", "probability": 0.5, "token_ids": []}

    assert dyn.continuation_key(row) == "txt:same text"


def test_fixed_cluster_basis_has_constant_mass_across_source_steps():
    continuations = [
        continuation("A final subtraction", 0.50, [1]),
        continuation("B compact answer", 0.30, [2]),
        continuation("C restated answer", 0.20, [3]),
    ]
    reference_assignments = [2, 3, 3]

    fixed = dyn.build_fixed_clusters_from_reference(
        continuations=continuations,
        assignments=reference_assignments,
    )

    assert [cluster["fixed_cluster_id"] for cluster in fixed] == ["F1", "F2"]
    assert [round(cluster["mass_frac"], 3) for cluster in fixed] == [0.5, 0.5]
    assert fixed[0]["continuation_keys"] == ["tok:1"]
    assert fixed[1]["continuation_keys"] == ["tok:2", "tok:3"]


def test_nonnumeric_probability_counts_as_zero_mass():
    continuations = [
        {"text": "bad probability", "probability": "not-a-number", "token_ids": [1]},
        continuation("valid", 0.75, [2]),
    ]

    fixed = dyn.build_fixed_clusters_from_reference(
        continuations=continuations,
        assignments=[2, 3],
    )

    assert [cluster["reference_cluster_id"] for cluster in fixed] == ["3", "2"]
    assert fixed[0]["mass"] == 0.75
    assert fixed[1]["mass"] == 0.0


def test_project_effects_to_fixed_clusters_uses_overlap_mass():
    continuations = [
        continuation("A", 0.50, [1]),
        continuation("B", 0.30, [2]),
        continuation("C", 0.20, [3]),
    ]
    fixed = dyn.build_fixed_clusters_from_reference(
        continuations=continuations,
        assignments=[2, 3, 3],
    )
    local_assignments = [10, 10, 11]
    local_stats = {
        "10": {"centered_logit_spearman": 0.8},
        "11": {"centered_logit_spearman": -0.4},
    }

    projected = dyn.project_local_effects_to_fixed_clusters(
        fixed_clusters=fixed,
        continuations=continuations,
        local_assignments=local_assignments,
        local_cluster_stats=local_stats,
    )

    assert round(projected["F1"]["rho_s"], 3) == 0.8
    assert round(projected["F1"]["overlap_mass_frac"], 3) == 0.5
    assert round(projected["F2"]["rho_s"], 3) == 0.32
    assert round(projected["F2"]["overlap_mass_frac"], 3) == 0.5
    assert round(projected["F2"]["metric_overlap_mass_frac"], 3) == 0.5


def test_project_effects_separates_total_and_metric_overlap_mass():
    continuations = [
        continuation("A", 0.50, [1]),
        continuation("B", 0.30, [2]),
        continuation("C", 0.20, [3]),
    ]
    fixed = dyn.build_fixed_clusters_from_reference(
        continuations=continuations,
        assignments=[2, 3, 3],
    )
    local_stats = {
        "10": {"centered_logit_spearman": 0.8},
    }

    projected = dyn.project_local_effects_to_fixed_clusters(
        fixed_clusters=fixed,
        continuations=continuations,
        local_assignments=[10, 10, 11],
        local_cluster_stats=local_stats,
    )

    assert round(projected["F2"]["overlap_mass"], 3) == 0.5
    assert round(projected["F2"]["metric_overlap_mass"], 3) == 0.3
    assert round(projected["F2"]["overlap_mass_frac"], 3) == 0.5
    assert round(projected["F2"]["metric_overlap_mass_frac"], 3) == 0.3
    assert round(projected["F2"]["rho_s"], 3) == 0.8


def test_duplicate_continuation_keys_do_not_overwrite_mass_in_projection():
    continuations = [
        continuation("duplicate", 0.20, [1]),
        continuation("duplicate", 0.30, [1]),
        continuation("other", 0.50, [2]),
    ]
    fixed = dyn.build_fixed_clusters_from_reference(
        continuations=continuations,
        assignments=[7, 7, 8],
    )

    assert fixed[0]["continuation_keys"] == ["tok:1"]
    assert fixed[0]["continuation_key_mass"] == {"tok:1": 0.5}

    projected = dyn.project_local_effects_to_fixed_clusters(
        fixed_clusters=fixed,
        continuations=continuations,
        local_assignments=[10, 11, 12],
        local_cluster_stats={
            "10": {"centered_logit_spearman": 1.0},
            "11": {"centered_logit_spearman": -1.0},
            "12": {"centered_logit_spearman": 0.2},
        },
    )

    assert round(projected["F1"]["overlap_mass"], 3) == 0.5
    assert round(projected["F1"]["rho_s"], 3) == -0.2
    assert projected["F1"]["local_cluster_overlap_mass"] == {"10": 0.2, "11": 0.3}


def test_alluvial_flows_use_continuation_overlap_mass_between_adjacent_sources():
    continuations = [
        continuation("A", 0.50, [1]),
        continuation("B", 0.30, [2]),
        continuation("C", 0.20, [3]),
    ]
    left = dyn.local_cluster_members(continuations, [2, 2, 3])
    right = dyn.local_cluster_members(continuations, [5, 6, 6])

    flows = dyn.compute_adjacent_cluster_flows(left, right)

    assert flows == [
        {"source_cluster_id": "2", "target_cluster_id": "5", "mass": 0.5},
        {"source_cluster_id": "2", "target_cluster_id": "6", "mass": 0.3},
        {"source_cluster_id": "3", "target_cluster_id": "6", "mass": 0.2},
    ]


def test_duplicate_continuation_keys_do_not_overwrite_mass_in_alluvial_flows():
    continuations = [
        continuation("duplicate", 0.20, [1]),
        continuation("duplicate", 0.30, [1]),
        continuation("other", 0.50, [2]),
    ]
    left = dyn.local_cluster_members(continuations, [2, 2, 3])
    right = dyn.local_cluster_members(continuations, [5, 6, 6])

    flows = dyn.compute_adjacent_cluster_flows(left, right)

    assert flows == [
        {"source_cluster_id": "2", "target_cluster_id": "5", "mass": 0.2},
        {"source_cluster_id": "2", "target_cluster_id": "6", "mass": 0.3},
        {"source_cluster_id": "3", "target_cluster_id": "6", "mass": 0.5},
    ]


def test_alluvial_flows_use_minimum_overlap_when_partition_masses_differ():
    left = {
        "2": {"tok:1": 0.8, "tok:2": 0.4},
        "3": {"tok:3": 0.6},
    }
    right = {
        "5": {"tok:1": 0.3},
        "6": {"tok:2": 0.9, "tok:3": 0.2},
    }

    flows = dyn.compute_adjacent_cluster_flows(left, right)

    assert flows == [
        {"source_cluster_id": "2", "target_cluster_id": "5", "mass": 0.3},
        {"source_cluster_id": "2", "target_cluster_id": "6", "mass": 0.4},
        {"source_cluster_id": "3", "target_cluster_id": "6", "mass": 0.2},
    ]


def test_stage7_cluster_stats_does_not_fallback_to_unrequested_run_or_sweep():
    stage7_prefix = {
        "clustering_runs": {
            "other_run": {
                "results": {
                    "other_sweep": {
                        "per_cluster_logit": {
                            "2": {"centered_logit_spearman": 0.9},
                        }
                    }
                }
            },
            "beta0.75_gamma0.7": {
                "results": {
                    "other_sweep": {
                        "per_cluster_logit": {
                            "3": {"centered_logit_spearman": -0.9},
                        }
                    }
                }
            },
        }
    }

    assert dyn.stage7_cluster_stats(stage7_prefix, "missing_run", "sign_full_B5") == {}
    assert dyn.stage7_cluster_stats(stage7_prefix, "beta0.75_gamma0.7", "sign_full_B5") == {}


def test_build_fixed_cluster_case_loads_selected_prefixes_and_projects_effects(tmp_path):
    lane = "synthetic_lane"
    prefixes = [
        {
            "prefix_id": "case_i1_j3",
            "source_step": 1,
            "target_step": 3,
            "K": 2,
            "mean_abs_rho_s": 0.2,
            "max_abs_rho_s": 0.4,
        },
        {
            "prefix_id": "case_i2_j3",
            "source_step": 2,
            "target_step": 3,
            "K": 2,
            "mean_abs_rho_s": 0.7,
            "max_abs_rho_s": 0.8,
        },
    ]
    case = {
        "dataset": "gsm8k",
        "lane": lane,
        "example_id": "ex-1",
        "target_step": 3,
        "prefixes": prefixes,
    }
    continuations = [
        continuation("A", 0.50, [1]),
        continuation("B", 0.30, [2]),
        continuation("C", 0.20, [3]),
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
                    "committed_previous_steps": ["step one", "step two"],
                },
            },
        )

    write_json(
        tmp_path
        / "selected_case_inputs"
        / lane
        / "case_i1_j3_sweep_results.json",
        {
            "grid": [
                {"beta": 0.1, "gamma": 0.2, "K": 2, "assignments": [99, 99, 99]},
                {"beta": 0.75, "gamma": 0.7, "K": 3, "assignments": [88, 88, 88]},
                {"beta": 0.75, "gamma": 0.7, "K": 2, "assignments": [10, 10, 11]},
            ]
        },
    )
    write_json(
        tmp_path
        / "selected_case_inputs"
        / lane
        / "case_i2_j3_sweep_results.json",
        {
            "grid": [
                {"beta": 0.1, "gamma": 0.2, "K": 2, "assignments": [99, 99, 99]},
                {"beta": 0.75, "gamma": 0.7, "K": 3, "assignments": [88, 88, 88]},
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

    case_data = dyn.build_fixed_cluster_case(
        case,
        case_root=tmp_path,
        stage7_root=tmp_path / "runs",
    )

    assert case_data["reference_prefix_id"] == "case_i2_j3"
    assert [cluster["fixed_cluster_id"] for cluster in case_data["fixed_clusters"]] == ["F1", "F2"]
    assert case_data["per_source_effects"][0]["source_step"] == 1
    assert case_data["per_source_effects"][0]["stage5_grid_entry"] == {
        "beta": 0.75,
        "gamma": 0.7,
        "K": 2,
    }
    assert set(case_data["per_source_effects"][0]["fixed_cluster_effects"]) == {"F1", "F2"}
    source1_effects = case_data["per_source_effects"][0]["fixed_cluster_effects"]
    assert round(source1_effects["F1"]["rho_s"], 3) == 0.8
    assert round(source1_effects["F2"]["rho_s"], 3) == 0.32
    source2_effects = case_data["per_source_effects"][1]["fixed_cluster_effects"]
    assert round(source2_effects["F1"]["rho_s"], 3) == 0.1
    assert round(source2_effects["F2"]["rho_s"], 3) == -0.2
    assert case_data["per_source_effects"][1]["source_step"] == 2
    assert case_data["adjacent_cluster_flows"] == [
        {
            "source_step": 1,
            "target_step": 2,
            "flows": [
                {"source_cluster_id": "10", "target_cluster_id": "2", "mass": 0.5},
                {"source_cluster_id": "10", "target_cluster_id": "3", "mass": 0.3},
                {"source_cluster_id": "11", "target_cluster_id": "3", "mass": 0.2},
            ],
        }
    ]


def test_build_fixed_cluster_case_raises_for_missing_selected_input(tmp_path):
    case = {
        "dataset": "gsm8k",
        "lane": "synthetic_lane",
        "example_id": "ex-1",
        "target_step": 3,
        "prefixes": [
            {
                "prefix_id": "missing_i1_j3",
                "source_step": 1,
                "target_step": 3,
                "K": 2,
                "mean_abs_rho_s": 0.5,
            }
        ],
    }

    try:
        dyn.build_fixed_cluster_case(case, case_root=tmp_path, stage7_root=tmp_path / "runs")
    except FileNotFoundError as exc:
        assert "missing branch file" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_build_fixed_cluster_case_raises_when_stage5_grid_has_no_exact_match(tmp_path):
    lane = "synthetic_lane"
    prefix_id = "case_i1_j3"
    case = {
        "dataset": "gsm8k",
        "lane": lane,
        "example_id": "ex-1",
        "target_step": 3,
        "prefixes": [
            {
                "prefix_id": prefix_id,
                "source_step": 1,
                "target_step": 3,
                "K": 2,
                "mean_abs_rho_s": 0.5,
            }
        ],
    }
    write_json(
        tmp_path / "selected_case_inputs" / lane / f"{prefix_id}_branches.json",
        {"continuations": [continuation("A", 0.6, [1]), continuation("B", 0.4, [2])]},
    )
    write_json(
        tmp_path / "selected_case_inputs" / lane / f"{prefix_id}_sweep_results.json",
        {
            "grid": [
                {"beta": 0.1, "gamma": 0.2, "K": 2, "assignments": [2, 3]},
                {"beta": 0.75, "gamma": 0.7, "K": 3, "assignments": [2, 3]},
            ]
        },
    )

    try:
        dyn.build_fixed_cluster_case(case, case_root=tmp_path, stage7_root=tmp_path / "runs")
    except ValueError as exc:
        message = str(exc)
        assert "no exact Stage 5 grid entry" in message
        assert "beta=0.75" in message
        assert "gamma=0.7" in message
        assert "K=2" in message
    else:
        raise AssertionError("expected ValueError")


def test_build_fixed_cluster_case_raises_when_assignment_count_mismatches_continuations(tmp_path):
    lane = "synthetic_lane"
    prefix_id = "case_i1_j3"
    case = {
        "dataset": "gsm8k",
        "lane": lane,
        "example_id": "ex-1",
        "target_step": 3,
        "prefixes": [
            {
                "prefix_id": prefix_id,
                "source_step": 1,
                "target_step": 3,
                "K": 2,
                "mean_abs_rho_s": 0.5,
            }
        ],
    }
    write_json(
        tmp_path / "selected_case_inputs" / lane / f"{prefix_id}_branches.json",
        {"continuations": [continuation("A", 0.6, [1]), continuation("B", 0.4, [2])]},
    )
    write_json(
        tmp_path / "selected_case_inputs" / lane / f"{prefix_id}_sweep_results.json",
        {"grid": [{"beta": 0.75, "gamma": 0.7, "K": 2, "assignments": [2]}]},
    )

    try:
        dyn.build_fixed_cluster_case(case, case_root=tmp_path, stage7_root=tmp_path / "runs")
    except ValueError as exc:
        assert "assignment length does not match continuation count" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_build_fixed_cluster_case_raises_for_missing_stage7_file(tmp_path):
    lane = "synthetic_lane"
    prefix_id = "case_i1_j3"
    case = {
        "dataset": "gsm8k",
        "lane": lane,
        "example_id": "ex-1",
        "target_step": 3,
        "prefixes": [
            {
                "prefix_id": prefix_id,
                "source_step": 1,
                "target_step": 3,
                "K": 2,
                "mean_abs_rho_s": 0.5,
            }
        ],
    }
    write_json(
        tmp_path / "selected_case_inputs" / lane / f"{prefix_id}_branches.json",
        {"continuations": [continuation("A", 1.0, [1])]},
    )
    write_json(
        tmp_path / "selected_case_inputs" / lane / f"{prefix_id}_sweep_results.json",
        {"grid": [{"beta": 0.75, "gamma": 0.7, "K": 2, "assignments": [2]}]},
    )

    try:
        dyn.build_fixed_cluster_case(case, case_root=tmp_path, stage7_root=tmp_path / "runs")
    except FileNotFoundError as exc:
        assert "missing stage7 validation file" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_build_fixed_cluster_case_raises_when_stage7_run_or_sweep_missing(tmp_path):
    lane = "synthetic_lane"
    prefix_id = "case_i1_j3"
    case = {
        "dataset": "gsm8k",
        "lane": lane,
        "example_id": "ex-1",
        "target_step": 3,
        "prefixes": [
            {
                "prefix_id": prefix_id,
                "source_step": 1,
                "target_step": 3,
                "K": 2,
                "mean_abs_rho_s": 0.5,
            }
        ],
    }
    write_json(
        tmp_path / "selected_case_inputs" / lane / f"{prefix_id}_branches.json",
        {"continuations": [continuation("A", 1.0, [1])]},
    )
    write_json(
        tmp_path / "selected_case_inputs" / lane / f"{prefix_id}_sweep_results.json",
        {"grid": [{"beta": 0.75, "gamma": 0.7, "K": 2, "assignments": [2]}]},
    )
    write_json(
        tmp_path
        / "runs"
        / lane
        / "7_validation/7c_combined_medoid/H4a_combined_medoid"
        / f"{prefix_id}_sweep_results.json",
        {
            "clustering_runs": {
                "other_run": {
                    "results": {
                        "other_sweep": {
                            "per_cluster_logit": {
                                "2": {"centered_logit_spearman": 0.9},
                            }
                        }
                    }
                }
            }
        },
    )

    try:
        dyn.build_fixed_cluster_case(case, case_root=tmp_path, stage7_root=tmp_path / "runs")
    except ValueError as exc:
        message = str(exc)
        assert "missing Stage 7 cluster stats" in message
        assert "beta0.75_gamma0.7" in message
        assert "sign_full_B5" in message
    else:
        raise AssertionError("expected ValueError")
