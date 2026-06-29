from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "pipeline" / "run_reasoning_qwen_pipeline.sh"
CONFIG = ROOT / "configs" / "reasoning_qwen3_small.yaml"


def run_pipeline(*args: str, output_root: Path, env: dict[str, str] | None = None):
    command = [
        "bash",
        str(SCRIPT),
        "--config",
        str(CONFIG),
        "--model",
        "Qwen/Qwen3-0.6B",
        "--dataset",
        "gsm8k",
        "--output-root",
        str(output_root),
        *args,
    ]
    merged_env = os.environ.copy()
    merged_env.pop("REASONING_TRANSCODER", None)
    if env:
        merged_env.update(env)
    return subprocess.run(
        command,
        cwd=ROOT,
        env=merged_env,
        text=True,
        capture_output=True,
        check=False,
    )


def run_script(*args: str):
    merged_env = os.environ.copy()
    merged_env.pop("REASONING_TRANSCODER", None)
    return subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=ROOT,
        env=merged_env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dry_run_requires_transcoder(tmp_path: Path):
    result = run_pipeline("--dry-run", output_root=tmp_path)

    assert result.returncode != 0
    assert "transcoder" in result.stderr.lower()


def test_missing_option_value_reports_controlled_error():
    result = run_script("--config")

    assert result.returncode != 0
    assert "--config requires a value" in result.stderr
    assert "unbound variable" not in result.stderr


def test_invalid_yaml_config_reports_config_read_failure(tmp_path: Path):
    invalid_config = tmp_path / "invalid_reasoning.yaml"
    invalid_config.write_text("reasoning:\n  samples_per_step: [\n")

    result = run_script(
        "--config",
        str(invalid_config),
        "--model",
        "Qwen/Qwen3-0.6B",
        "--dataset",
        "gsm8k",
        "--transcoder",
        "local/test-transcoder",
        "--output-root",
        str(tmp_path / "output"),
        "--dry-run",
    )

    assert result.returncode != 0
    assert "failed to read reasoning config" in result.stderr.lower()
    assert "unbound variable" not in result.stderr


def test_dry_run_writes_runtime_clustering_config_and_prints_commands(
    tmp_path: Path,
):
    result = run_pipeline(
        "--transcoder",
        "local/test-transcoder",
        "--dry-run",
        output_root=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    assert "sample_reasoning_steps.py" in result.stdout
    assert "compute_reasoning_step_pair_attribution.py" in result.stdout
    assert "runtime_clustering_config.json" in result.stdout
    for expected_dir in [
        "2_reasoning_pair_samples",
        "3_attribution_graphs",
        "4_feature_extraction/embeddings",
        "5_gaussian_clustering",
    ]:
        assert expected_dir in result.stdout
    assert "--save-intermediate" in result.stdout
    assert "--skip-existing" in result.stdout

    runtime_config = json.loads(
        (tmp_path / "runtime_clustering_config.json").read_text()
    )
    assert runtime_config["clustering"]["K_max"] == 30
    assert runtime_config["clustering"]["sweeps"]["beta_values"] == [
        0.5,
        0.75,
        1.0,
        1.25,
    ]
    assert runtime_config["clustering"]["sweeps"]["gamma_values"] == [
        0.3,
        0.5,
        0.7,
    ]
    assert runtime_config["clustering"]["pooling"] == "mean"


def test_pooling_override_updates_runtime_config_and_stage5_command(tmp_path: Path):
    result = run_pipeline(
        "--transcoder",
        "local/test-transcoder",
        "--pooling",
        "sum",
        "--dry-run",
        output_root=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    stage5_command = next(
        line for line in result.stdout.splitlines() if "5_gaussian_clustering/cluster.py" in line
    )
    assert "--pooling sum" in stage5_command

    runtime_config = json.loads(
        (tmp_path / "runtime_clustering_config.json").read_text()
    )
    assert runtime_config["clustering"]["pooling"] == "sum"


def test_compact_storage_rejects_max_pooling_override(tmp_path: Path):
    result = run_pipeline(
        "--transcoder",
        "local/test-transcoder",
        "--pooling",
        "max",
        "--dry-run",
        output_root=tmp_path,
    )

    assert result.returncode != 0
    assert "pooling=max requires attribution.store_all=true" in result.stderr


def test_reasoning_config_uses_long_enough_vllm_context_for_math_rollouts():
    config = yaml.safe_load(CONFIG.read_text())

    assert config["vllm"]["max_model_len"] >= 8192


def test_reasoning_config_uses_compact_mean_equivalent_attribution_storage():
    config = yaml.safe_load(CONFIG.read_text())

    assert config["attribution"]["store_all"] is False
    assert config["clustering"]["pooling"] == "mean"


def test_reasoning_dry_run_omits_store_all_when_compact_storage_is_configured(
    tmp_path: Path,
):
    result = run_pipeline(
        "--transcoder",
        "local/test-transcoder",
        "--dry-run",
        output_root=tmp_path,
    )

    assert result.returncode == 0, result.stderr
    stage3_command = next(
        line
        for line in result.stdout.splitlines()
        if "compute_reasoning_step_pair_attribution.py" in line
    )
    assert "--store-all" not in stage3_command
