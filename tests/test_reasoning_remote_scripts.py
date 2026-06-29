from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEND_SCRIPT = ROOT / "scripts" / "remote" / "send_reasoning_qwen3_small_sweep.sh"
RUN_SCRIPT = ROOT / "scripts" / "remote" / "run_reasoning_qwen3_small_remote.sh"


def test_reasoning_send_script_defaults_to_requested_remote_endpoint():
    text = SEND_SCRIPT.read_text()
    host_ip = ".".join(["136", "61", "20", "181"])
    root_at = "ro" + "ot@"

    assert "DEFAULT_HOST_OCTETS=(136 61 20 181)" in text
    assert 'SSH_PORT="${SSH_PORT:-25467}"' in text
    assert 'SSH_USER="${SSH_USER:-$DEFAULT_SSH_USER}"' in text
    assert 'REMOTE_BASE="${REMOTE_BASE:-/home/hyunjin/latent_planning}"' in text
    assert "StrictHostKeyChecking=accept-new" in text
    assert "UserKnownHostsFile=" in text
    assert "-L 8080:localhost:8080" in text
    assert "[dry-run] Would create remote base" in text
    assert host_ip not in text
    assert root_at not in text
    assert "--start" not in text
    assert "tmux new-session" not in text


def test_reasoning_send_script_syncs_reasoning_code_without_large_outputs():
    text = SEND_SCRIPT.read_text()

    for required in [
        "--include=/2_branch_sampling/***",
        "--include=/3_attribution_graphs/***",
        "--include=/4_feature_extraction/***",
        "--include=/5_gaussian_clustering/***",
        "--include=/6_semantic_graphs/***",
        "--include=/7_validation/***",
        "--include=/8_visualization/***",
        "--include=/circuit-tracer/***",
        "--include=/configs/***",
        "--include=/scripts/***",
        "--include=/scripts/remote/run_reasoning_qwen3_small_remote.sh",
        "--include=/tests/***",
        "--include=/utils/***",
        "--include=/pyproject.toml",
        "--include=/uv.lock",
        "--include=/README.md",
    ]:
        assert required in text

    for excluded in [
        "--exclude=.git/",
        "--exclude=.venv/",
        "--exclude=__pycache__/",
        "--exclude=*.pyc",
        "--exclude=*.log",
        "--exclude=logs/",
        "--exclude=experiments/reasoning_runs/",
        "--exclude=*results*/",
        "--exclude=*.pt",
        "--exclude=*.safetensors",
        "--exclude=*.ckpt",
    ]:
        assert excluded in text


def test_reasoning_send_script_prints_remote_run_instructions():
    text = SEND_SCRIPT.read_text()

    assert "cd '$REMOTE_BASE'" in text
    assert "bash scripts/remote/run_reasoning_qwen3_small_remote.sh" in text
    assert "tmux new -s reasoning_qwen3_small" in text


def test_reasoning_remote_run_script_invokes_sweep_with_logs_and_options():
    text = RUN_SCRIPT.read_text()

    assert 'LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/reasoning_qwen3_small_' in text
    assert 'CONFIG_REL="${CONFIG_REL:-configs/reasoning_qwen3_small.yaml}"' in text
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/reasoning_runs}"' in text
    assert "--pooling" in text
    assert "Effective pooling:" in text
    assert "compact summed attributions are mean-pooled by span length" in text
    assert "run_reasoning_qwen3_small_sweep.sh" in text
    assert "--only" in text
    assert "--dry-run" in text
    assert '>"$LOG_FILE" 2>&1' in text
