import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_pipeline_uses_clean_stage7c_baselines():
    script = (ROOT / "scripts" / "run_pipeline.sh").read_text()

    assert "normalize_stage_name()" in script
    assert 'normalized_stage=$(normalize_stage_name "$stage")' in script

    assert "7_validation/7c_baseline_combined_medoid.py" in script
    assert "7_validation/7c_baseline_single.py" in script
    assert "7_validation/7c_baseline_kmeans.py" in script

    assert "7c_combined_medoid" in script
    assert "7c_single" in script
    assert "7c_kmeans" in script

    assert "7_validation/7c_hypotheses.py" not in script
    assert "7c1" not in script
    assert "7c2" not in script


def test_stage_aliases_are_normalized_for_stage_lists():
    script = (ROOT / "scripts" / "run_pipeline.sh").read_text()

    assert 'combined_medoid) echo "7c_combined_medoid"' in script
    assert 'single) echo "7c_single"' in script
    assert 'kmeans) echo "7c_kmeans"' in script
    assert 'if [[ "$s_normalized" == "$normalized_stage" ]]; then' in script


def test_pipeline_helper_scripts_do_not_call_removed_stage7_aliases():
    script_paths = [
        ROOT / "scripts" / "remote" / "run_remote_mmlu_rd_worker.sh",
        ROOT / "scripts" / "remote" / "sync_and_run_baseline_remote.sh",
        ROOT / "scripts" / "pipeline" / "rerun_from_stage5.sh",
        ROOT / "scripts" / "remote" / "sync_ambigqa_baseline_remote.sh",
    ]
    combined = "\n".join(path.read_text() for path in script_paths)

    assert "--only 7c1" not in combined
    assert "--only 7c2" not in combined
    assert "--stages 5,6,7c1,7c2" not in combined
    assert "--stages 7c2" not in combined
    assert "--only 7c_combined_medoid" in combined
    assert "--only 7c_kmeans" in combined
    assert "--stages 5,6,7c" in combined


def test_paper_correlation_summary_uses_existing_aggregation():
    metrics = load_module(ROOT / "7_validation" / "7c_metrics.py", "stage7c_metrics")

    sweep_key = metrics.generate_sweep_key("sign", "full", 5)
    sweep_tuple = metrics.SweepKey.from_string(sweep_key).to_tuple()
    all_results = {
        "prefix_a": {
            "results": {
                sweep_key: {
                    "mean_logit_corr": 0.25,
                    "mean_logit_spearman": 0.5,
                }
            }
        },
        "prefix_b": {
            "results": {
                sweep_key: {
                    "mean_logit_corr": 0.75,
                    "mean_logit_spearman": 1.0,
                }
            }
        },
    }

    aggregated = metrics.aggregate_results_across_prefixes(all_results, [sweep_tuple])
    rows = metrics.build_paper_correlation_summary_rows(aggregated)

    assert rows == [
        {
            "steering_method": "sign",
            "h_c_selection": "full",
            "top_B": 5,
            "centered_logit_spearman_mean": 0.75,
            "centered_logit_corr_mean": 0.5,
            "n_prefixes": 2,
        }
    ]
    assert all("sum_diff" not in key for key in rows[0])
    assert all(not key.startswith("mean_") for key in rows[0])


def test_stage7_analysis_exports_only_paper_correlation_metrics():
    analyzer = (ROOT / "7_validation" / "analyze_steering_methods.py").read_text()
    table = (ROOT / "scripts" / "tables" / "csv_to_latex_table.py").read_text()
    steering_table = (ROOT / "scripts" / "tables" / "csv_to_latex_steering.py").read_text()

    assert "sum_diff_eps" not in analyzer
    assert "dose_response" not in analyzer
    assert "win_rate" not in analyzer

    assert "sum_diff_eps" not in table
    assert "steering effect at strength" not in table

    assert "sum_diff_eps" not in steering_table
    assert r"\epsilon_{" not in steering_table


def test_stage7_analysis_outputs_standard_json_for_empty_metric_groups():
    analyzer = load_module(
        ROOT / "7_validation" / "analyze_steering_methods.py",
        "stage7_analyze_standard_json",
    )

    summary = analyzer.summarize_metric([], "centered_logit_corr")

    assert summary == {"mean": None, "std": None, "median": None, "n": 0}


def test_stage7_analysis_loads_nested_h4a_result_dirs(tmp_path):
    analyzer = load_module(
        ROOT / "7_validation" / "analyze_steering_methods.py",
        "stage7_analyze_steering_methods",
    )
    method_dir = tmp_path / "7c_combined_medoid"
    leaf_dir = method_dir / "H4a_combined_medoid"
    leaf_dir.mkdir(parents=True)
    result_path = leaf_dir / "cloze_0000_sweep_results.json"
    result_path.write_text('{"prefix_id": "cloze_0000", "clustering_runs": {}}')

    loaded = analyzer.load_results_from_dir(method_dir)

    assert loaded == {"cloze_0000": {"prefix_id": "cloze_0000", "clustering_runs": {}}}


def test_legacy_stage7c_direct_path_carries_logit_spearman():
    hypotheses = (ROOT / "7_validation" / "7c_hypotheses.py").read_text()

    assert "'mean_logit_spearman': computed_metrics['mean_logit_spearman']" in hypotheses
    assert "'logit_spearman': []" in hypotheses
    assert "['logit_spearman'].append(result.get('mean_logit_spearman', 0.0))" in hypotheses
    assert "'mean_r2' in result" not in hypotheses


def test_compute_steering_metrics_exposes_only_logit_metrics():
    metrics = load_module(ROOT / "7_validation" / "7c_metrics.py", "stage7c_metrics_minimal")

    result = metrics.compute_steering_metrics(
        centered_logits_steered={
            0: {-1.0: [0.0, 0.0], 0.0: [1.0, 1.0], 1.0: [2.0, 2.0]},
            1: {-1.0: [1.0, 1.0], 0.0: [2.0, 2.0], 1.0: [3.0, 3.0]},
        },
        centered_logits_original={
            0: {-1.0: [0.0, 0.0], 0.0: [0.0, 0.0], 1.0: [0.0, 0.0]},
            1: {-1.0: [1.0, 1.0], 0.0: [1.0, 1.0], 1.0: [1.0, 1.0]},
        },
        epsilons=[-1.0, 0.0, 1.0],
    )

    assert result["mean_logit_corr"] == 1.0
    assert result["mean_logit_spearman"] == 1.0
    assert set(result) == {
        "per_cluster_logit",
        "per_cluster_demeaned_logit",
        "mean_logit_corr",
        "mean_logit_spearman",
        "mean_demeaned_logit_corr",
        "mean_demeaned_logit_spearman",
    }

    forbidden = {
        "per_cluster",
        "per_cluster_mass",
        "mean_r2",
        "mean_corr",
        "mean_spearman",
        "mean_win_r2",
        "mean_win_corr",
        "mean_win_spearman",
        "n_clusters_with_effect",
        "mean_logit_r2",
        "mean_sum_logit_r2",
        "mean_sum_logit_corr",
        "mean_sum_logit_spearman",
    }
    assert forbidden.isdisjoint(result)


def test_aggregate_results_uses_logit_metrics_as_validity_gate():
    metrics = load_module(ROOT / "7_validation" / "7c_metrics.py", "stage7c_metrics_aggregate_minimal")

    sweep_key = metrics.generate_sweep_key("sign", "full", 5)
    sweep_tuple = metrics.SweepKey.from_string(sweep_key).to_tuple()
    all_results = {
        "prefix_a": {
            "results": {
                sweep_key: {
                    "mean_logit_corr": 0.25,
                    "mean_logit_spearman": 0.5,
                }
            }
        },
        "prefix_b": {
            "results": {
                sweep_key: {
                    "mean_logit_corr": 0.75,
                    "mean_logit_spearman": 1.0,
                }
            }
        },
        "prefix_c": {
            "results": {
                sweep_key: {
                    "mean_corr": 100.0,
                    "mean_spearman": 100.0,
                }
            }
        },
        "prefix_d": {
            "results": {
                sweep_key: {
                    "mean_logit_corr": 0.125,
                }
            }
        },
    }

    aggregated = metrics.aggregate_results_across_prefixes(all_results, [sweep_tuple])

    assert aggregated == {
        sweep_tuple: {
            "logit_corr": [0.25, 0.75, 0.125],
            "logit_spearman": [0.5, 1.0],
        }
    }


def test_prefix_batched_stage7c_results_keep_sweep_metadata():
    hypotheses = (ROOT / "7_validation" / "7c_hypotheses.py").read_text()
    prefix_batch_result_block = hypotheses.split("def run_steering_sweep_prefix_batch", 1)[1]
    prefix_batch_result_block = prefix_batch_result_block.split("def run_sweep_mode", 1)[0]

    assert 'result.update({' in prefix_batch_result_block
    assert '"steering_method": steering_method' in prefix_batch_result_block
    assert '"hc_selection": hc_selection' in prefix_batch_result_block
    assert '"top_B": top_B' in prefix_batch_result_block
    assert '"epsilon_values": epsilons' in prefix_batch_result_block
