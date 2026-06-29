from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "scripts" / "remote" / "run_scale_experiments_8gpu.sh"
SEND_SCRIPT = ROOT / "scripts" / "remote" / "send_scale_experiments_8gpu.sh"
PULL_SCRIPT = ROOT / "scripts" / "remote" / "pull_scale_experiments_8gpu_results.sh"
FAST_GEMMA_OVERLAY = ROOT / "configs" / "stage7_fast_gemma_overlay.json"


def _text(path: Path) -> str:
    return path.read_text()


def test_run_script_assigns_non_gemma1b_combos_to_gpus_0_through_6():
    text = _text(RUN_SCRIPT)

    assignments = re.findall(r'"([0-6]):([^"]+)"', text)
    assert assignments == [
        ("0", "AmbigQA_Gemma3-4B-it"),
        ("1", "MMLU_Qwen3-8B"),
        ("2", "MMLU_Qwen3-4B"),
        ("3", "MMLU_Gemma3-4B-it"),
        ("4", "HarmBench_Qwen3-8B"),
        ("5", "HarmBench_Qwen3-4B"),
        ("6", "HarmBench_Gemma3-4B-it"),
    ]
    assert "run_lane \"$gpu_id\" \"$tag\" &" in text


def test_run_script_runs_gemma1b_combos_sequentially_on_gpu_7():
    text = _text(RUN_SCRIPT)

    assert 'GEMMA3_1B_GPU="${GEMMA3_1B_GPU:-7}"' in text
    sequential = re.findall(r'"(AmbigQA_Gemma3-1B-it|MMLU_Gemma3-1B-it|HarmBench_Gemma3-1B-it)"', text)
    assert sequential == [
        "AmbigQA_Gemma3-1B-it",
        "MMLU_Gemma3-1B-it",
        "HarmBench_Gemma3-1B-it",
    ]
    assert 'for tag in "${GEMMA3_1B_COMBOS[@]}"; do' in text
    assert 'run_lane "$GEMMA3_1B_GPU" "$tag"' in text


def test_run_script_has_opt_in_accelerated_stage7_sharding_mode():
    text = _text(RUN_SCRIPT)

    assert 'STAGE7_ACCELERATED="false"' in text
    assert 'STAGE7_SHARD_COUNT="${STAGE7_SHARD_COUNT:-8}"' in text
    assert 'STAGE7_BASELINES="${STAGE7_BASELINES:-combined_medoid}"' in text
    assert 'STAGE7_CONFIG_OVERLAY="${STAGE7_CONFIG_OVERLAY:-configs/stage7_fast_gemma_overlay.json}"' in text
    assert "--stage7-accelerated" in text
    assert "--stage7-shard-count" in text
    assert "--stage7-baselines" in text
    assert "--stage7-config-overlay" in text

    assert "run_stage7_baseline_shard()" in text
    assert "run_stage7_accelerated()" in text
    assert "stage7_config_for_tag()" in text
    assert "stage7_manifest_for_tag()" in text
    assert "select_stage7_clustering_manifest.py" in text
    assert "--prefix-shard-index" in text
    assert "--prefix-shard-count" in text
    assert "--clustering-manifest" in text
    assert "stdout_stderr.log" in text

    for baseline_script in [
        "7_validation/7c_baseline_combined_medoid.py",
        "7_validation/7c_baseline_single.py",
        "7_validation/7c_baseline_kmeans.py",
    ]:
        assert baseline_script in text

    accelerated_branch = 'if [[ "$STAGE7_ACCELERATED" == "true" ]]; then\n  run_stage7_accelerated'
    normal_launch = 'declare -a PIDS=()\ndeclare -a LABELS=()'
    assert text.index(accelerated_branch) < text.index(normal_launch)


def test_fast_gemma_stage7_overlay_is_reduced_and_manifest_driven():
    config = json.loads(_text(FAST_GEMMA_OVERLAY))
    steering = config["stage_7c_steering"]

    assert steering["baselines"] == ["combined_medoid"]
    assert steering["max_cluster_samples"] == 5
    assert steering["cross_prefix_batching"] is False
    assert steering["prefix_batch_size"] == 1
    assert steering["clustering_top_k"] == 1
    assert steering["clustering_score_key"] == "harmonic"
    assert steering["clustering_score_order"] == "desc"
    assert steering["clustering_min_k"] == 2
    assert steering["clustering_max_k"] == 10
    assert steering["clustering_selection"] == {
        "top_k": 1,
        "score_key": "harmonic",
        "score_order": "desc",
        "min_k": 2,
        "max_k": 10,
    }
    assert steering["sweeps"] == [
        {
            "name": "sign_fast_full",
            "steering_method": "sign",
            "h_c_selections": ["full"],
            "top_B": [5],
            "epsilon_values": [-0.5, 0.0, 0.5],
        }
    ]


def test_send_script_defaults_to_current_vastai_endpoint_without_endpoint_literals():
    text = _text(SEND_SCRIPT)
    new_public_ip = ".".join(["95", "3", "33", "46"])
    public_ip = "109." + "231." + "106." + "68"
    root_at = "ro" + "ot@"
    root_user_assignment = 'USER="' + "ro" + "ot" + '"'

    assert 'REMOTE_ALIAS="${REMOTE_ALIAS:-vastai_8}"' in text
    assert "DEFAULT_HOST_OCTETS=(95 3 33 46)" in text
    assert 'SSH_PORT="${SSH_PORT:-45561}"' in text
    assert 'SSH_USER="${SSH_USER:-$DEFAULT_SSH_USER}"' in text
    assert "StrictHostKeyChecking=accept-new" in text
    assert "UserKnownHostsFile=" in text
    assert "--host-name HOST" in text
    assert "--port PORT" in text
    assert "--user USER" in text
    assert 'SSH_TARGET="$REMOTE_ALIAS"' in text
    assert 'RSYNC_RSH="${SSH_CMD[*]}"' in text
    assert new_public_ip not in text
    assert public_ip not in text
    assert root_at not in text
    assert root_user_assignment not in text


def test_send_script_excludes_generated_results_and_large_artifacts():
    text = _text(SEND_SCRIPT)

    required_excludes = [
        "--exclude=.git/",
        "--exclude=.venv/",
        "--exclude=__pycache__/",
        "--exclude=*.pyc",
        "--exclude=*.log",
        "--exclude=logs/",
        "--exclude=figures/",
        "--exclude=archive/",
        "--exclude=*results*/",
        "--exclude=smoke_single_example*/",
        "--exclude=AmbigQA_Gemma3-*/",
        "--exclude=MMLU_*/",
        "--exclude=HarmBench_*/",
        "--exclude=*.pt",
        "--exclude=*.safetensors",
    ]
    for pattern in required_excludes:
        assert pattern in text

    required_includes = [
        "--include=/data/",
        "--include=/data/cloze_llm_improved_split_ratio_0.1/***",
        "--include=/data/mmlu_cais_validation_7subjects/***",
        "--include=/data/harmbench_walledai_standard/***",
        "--include=/scripts/remote/run_scale_experiments_8gpu.sh",
    ]
    for pattern in required_includes:
        assert pattern in text


def test_pull_script_defaults_to_current_vastai_endpoint_without_endpoint_literals():
    text = _text(PULL_SCRIPT)
    new_public_ip = ".".join(["95", "3", "33", "46"])
    root_at = "ro" + "ot@"

    assert "DEFAULT_HOST_OCTETS=(95 3 33 46)" in text
    assert 'SSH_PORT="${SSH_PORT:-45561}"' in text
    assert 'SSH_USER="${SSH_USER:-$DEFAULT_SSH_USER}"' in text
    assert "StrictHostKeyChecking=accept-new" in text
    assert "UserKnownHostsFile=" in text
    assert "--host-name HOST" in text
    assert "--local-base DIR" in text
    assert "--only TAG1,TAG2" in text
    assert new_public_ip not in text
    assert root_at not in text


def test_pull_script_pulls_analysis_results_logs_and_skips_large_artifacts():
    text = _text(PULL_SCRIPT)

    for tag in [
        "AmbigQA_Gemma3-1B-it",
        "AmbigQA_Gemma3-4B-it",
        "MMLU_Qwen3-8B",
        "MMLU_Qwen3-4B",
        "MMLU_Gemma3-1B-it",
        "MMLU_Gemma3-4B-it",
        "HarmBench_Qwen3-8B",
        "HarmBench_Qwen3-4B",
        "HarmBench_Gemma3-1B-it",
        "HarmBench_Gemma3-4B-it",
    ]:
        assert f'"{tag}"' in text

    assert "--include=/results/*.json" in text
    assert "--include=/results/configs/***" in text
    assert "--include=/results/2_branch_sampling/***" in text
    assert "--include=/results/5_clustering/*.json" in text
    assert "--include=/results/6_semantic_graphs/***" in text
    assert "--include=/results/7_validation/***" in text
    assert "--include=/results/8_visualization/***" in text
    assert "--include=/logs/***" in text
    assert "--exclude=/results/3_attribution_graphs/***" in text
    assert "--exclude=/results/4_feature_extraction/***" in text
    assert "--exclude=/results/5_clustering/intermediate/***" in text
    assert "--exclude=*.pt" in text
    assert "--exclude=*.pth" in text
    assert "--exclude=*.safetensors" in text
    assert "--exclude=*.ckpt" in text
    assert "--info=progress2,stats2" in text
    assert "--full" in text
    assert "--include-tensors" in text
    assert "rsync" in text
    assert "--dry-run --itemize-changes" in text
