from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PipelineRunnerTests(unittest.TestCase):
    def run_pipeline(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "scripts/run_paper_pipeline.sh", *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_default_dry_run_uses_ambigqa_qwen3_8b(self) -> None:
        result = self.run_pipeline(
            "--dry-run",
            "--prefixes-file",
            "inputs/prefixes.example.json",
            "--output-dir",
            "/tmp/lp_default_dry_run",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Config: configs/default.json", result.stdout)
        self.assertIn("Model: Qwen/Qwen3-8B", result.stdout)
        self.assertIn("Stage 2: branch sampling", result.stdout)
        self.assertIn("--model Qwen/Qwen3-8B", result.stdout)
        self.assertIn("--config /tmp/lp_default_dry_run/results/resolved_config.json", result.stdout)

    def test_qwen4_preset_dry_run_overrides_model(self) -> None:
        result = self.run_pipeline(
            "--dry-run",
            "--config",
            "configs/presets/ambigqa_qwen3_4b.json",
            "--prefixes-file",
            "inputs/prefixes.example.json",
            "--output-dir",
            "/tmp/lp_qwen4_dry_run",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Model: Qwen/Qwen3-4B", result.stdout)
        self.assertIn("--model Qwen/Qwen3-4B", result.stdout)
        self.assertIn("mwhanna/qwen3-4b-transcoders", result.stdout)

    def test_dataset_mode_dry_run_uses_ambigqa_preparation(self) -> None:
        result = self.run_pipeline(
            "--dry-run",
            "--dataset-dir",
            "/tmp/missing_dataset",
            "--output-dir",
            "/tmp/lp_dataset_dry_run",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Stage 0: AmbigQA question preparation", result.stdout)


if __name__ == "__main__":
    unittest.main()
