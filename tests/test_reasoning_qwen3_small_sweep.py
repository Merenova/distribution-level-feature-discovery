from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "pipeline" / "run_reasoning_qwen3_small_sweep.sh"


def run_sweep(*args: str):
    env = os.environ.copy()
    env.pop("REASONING_TRANSCODER", None)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dry_run_prints_all_four_model_dataset_runs(tmp_path: Path):
    result = run_sweep("--output-root", str(tmp_path), "--dry-run")

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("sample_reasoning_steps.py") == 4
    assert result.stdout.count("compute_reasoning_step_pair_attribution.py") == 4
    assert result.stdout.count("compute_embeddings.py") == 4
    assert result.stdout.count("5_gaussian_clustering/cluster.py") == 4

    for model in ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]:
        assert model in result.stdout
    for dataset in ["gsm8k", "math500"]:
        assert f"--dataset {dataset}" in result.stdout

    assert "mwhanna/qwen3-0.6b-transcoders-lowl0" in result.stdout
    assert "mwhanna/qwen3-1.7b-transcoders-lowl0" in result.stdout

    for suffix in [
        "qwen3_0_6b_gsm8k",
        "qwen3_0_6b_math500",
        "qwen3_1_7b_gsm8k",
        "qwen3_1_7b_math500",
    ]:
        assert str(tmp_path / suffix) in result.stdout
        assert (tmp_path / suffix / "runtime_clustering_config.json").exists()


def test_only_filters_to_requested_combination(tmp_path: Path):
    result = run_sweep(
        "--output-root",
        str(tmp_path),
        "--only",
        "qwen3_1_7b:math500",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("sample_reasoning_steps.py") == 1
    assert "Qwen/Qwen3-1.7B" in result.stdout
    assert "--dataset math500" in result.stdout
    assert "mwhanna/qwen3-1.7b-transcoders-lowl0" in result.stdout
    assert str(tmp_path / "qwen3_1_7b_math500") in result.stdout
    assert "Qwen/Qwen3-0.6B" not in result.stdout
    assert "--dataset gsm8k" not in result.stdout


def test_pooling_override_is_forwarded_to_each_reasoning_run(tmp_path: Path):
    result = run_sweep(
        "--output-root",
        str(tmp_path),
        "--only",
        "qwen3_0_6b:gsm8k,qwen3_1_7b:math500",
        "--pooling",
        "sum",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("--pooling sum") == 4
    for suffix in ["qwen3_0_6b_gsm8k", "qwen3_1_7b_math500"]:
        runtime_config = json.loads(
            (tmp_path / suffix / "runtime_clustering_config.json").read_text()
        )
        assert runtime_config["clustering"]["pooling"] == "sum"


def test_only_accepts_comma_separated_combinations(tmp_path: Path):
    result = run_sweep(
        "--output-root",
        str(tmp_path),
        "--only",
        "qwen3_0_6b:gsm8k,qwen3_1_7b:math500",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.count("sample_reasoning_steps.py") == 2
    assert str(tmp_path / "qwen3_0_6b_gsm8k") in result.stdout
    assert str(tmp_path / "qwen3_1_7b_math500") in result.stdout
    assert str(tmp_path / "qwen3_0_6b_math500") not in result.stdout
    assert str(tmp_path / "qwen3_1_7b_gsm8k") not in result.stdout


def test_datasets_are_assigned_to_distinct_gpus_by_default(tmp_path: Path):
    result = run_sweep("--output-root", str(tmp_path), "--dry-run")

    assert result.returncode == 0, result.stderr
    for model_key in ["qwen3_0_6b", "qwen3_1_7b"]:
        assert (
            f"=== reasoning {model_key}:gsm8k on CUDA_VISIBLE_DEVICES=0 ==="
            in result.stdout
        )
        assert (
            f"=== reasoning {model_key}:math500 on CUDA_VISIBLE_DEVICES=1 ==="
            in result.stdout
        )


def test_only_subset_uses_dataset_gpu_assignment(tmp_path: Path):
    result = run_sweep(
        "--output-root",
        str(tmp_path),
        "--only",
        "qwen3_1_7b:math500",
        "--dry-run",
    )

    assert result.returncode == 0, result.stderr
    assert "=== reasoning qwen3_1_7b:math500 on CUDA_VISIBLE_DEVICES=1 ===" in (
        result.stdout
    )
    assert "CUDA_VISIBLE_DEVICES=0" not in result.stdout


def test_rejects_unknown_only_model_key():
    result = run_sweep("--only", "qwen3_4b:gsm8k", "--dry-run")

    assert result.returncode != 0
    assert "unknown model key in --only: qwen3_4b" in result.stderr
    assert "unbound variable" not in result.stderr


def test_rejects_unknown_only_dataset():
    result = run_sweep("--only", "qwen3_0_6b:mmlu", "--dry-run")

    assert result.returncode != 0
    assert "unknown dataset in --only: mmlu" in result.stderr
    assert "unbound variable" not in result.stderr


def test_rejects_only_entry_with_empty_model_key():
    result = run_sweep("--only", ":gsm8k", "--dry-run")

    assert result.returncode != 0
    assert "--only entry has an empty model key: :gsm8k" in result.stderr
    assert "bad array subscript" not in result.stderr
    assert "unbound variable" not in result.stderr


def test_rejects_only_entry_with_empty_dataset():
    result = run_sweep("--only", "qwen3_0_6b:", "--dry-run")

    assert result.returncode != 0
    assert "--only entry has an empty dataset: qwen3_0_6b:" in result.stderr
    assert "bad array subscript" not in result.stderr
    assert "unbound variable" not in result.stderr


def test_rejects_only_list_with_empty_entry_between_commas():
    result = run_sweep(
        "--only",
        "qwen3_0_6b:gsm8k,,qwen3_1_7b:math500",
        "--dry-run",
    )

    assert result.returncode != 0
    assert "--only contains an empty entry" in result.stderr
    assert "bad array subscript" not in result.stderr
    assert "unbound variable" not in result.stderr


def test_missing_option_value_reports_controlled_error():
    result = run_sweep("--output-root")

    assert result.returncode != 0
    assert "--output-root requires a value" in result.stderr
    assert "unbound variable" not in result.stderr
