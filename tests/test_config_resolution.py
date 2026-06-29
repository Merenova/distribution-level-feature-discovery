from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from utils.config import config_to_shell_exports, deep_merge, load_config, write_resolved_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigResolutionTests(unittest.TestCase):
    def test_deep_merge_preserves_base_and_overrides_nested_values(self) -> None:
        base = {
            "model": {"base_model": "Qwen/Qwen3-8B", "dtype": "bfloat16"},
            "sampling": {"max_tokens": 20, "batch_size": 32},
        }
        overlay = {
            "model": {"base_model": "Qwen/Qwen3-4B"},
            "sampling": {"batch_size": 16},
        }

        merged = deep_merge(base, overlay)

        self.assertEqual(merged["model"]["base_model"], "Qwen/Qwen3-4B")
        self.assertEqual(merged["model"]["dtype"], "bfloat16")
        self.assertEqual(merged["sampling"]["max_tokens"], 20)
        self.assertEqual(merged["sampling"]["batch_size"], 16)
        self.assertEqual(base["model"]["base_model"], "Qwen/Qwen3-8B")

    def test_default_config_resolves_to_ambigqa_qwen3_8b(self) -> None:
        config = load_config(ROOT / "configs/default.json")

        self.assertEqual(config["experiment_name"], "ambigqa_qwen3_8b")
        self.assertEqual(config["data"]["dataset"], "ambigqa")
        self.assertEqual(config["data"]["adapter"], "ambigqa")
        self.assertEqual(config["model"]["base_model"], "Qwen/Qwen3-8B")
        self.assertEqual(config["model"]["transcoder"], "mwhanna/qwen3-8b-transcoders")
        self.assertEqual(config["stage_7c_steering"]["max_batch_size"], 256)

    def test_qwen4_preset_resolves_replication_overrides(self) -> None:
        config = load_config(ROOT / "configs/presets/ambigqa_qwen3_4b.json")

        self.assertEqual(config["experiment_name"], "ambigqa_qwen3_4b")
        self.assertEqual(config["model"]["base_model"], "Qwen/Qwen3-4B")
        self.assertEqual(config["model"]["transcoder"], "mwhanna/qwen3-4b-transcoders")
        self.assertEqual(config["stage_7c_steering"]["max_batch_size"], 512)
        self.assertEqual(config["sampling"]["max_total_continuations"], 500)

    def test_shell_exports_include_runner_keys(self) -> None:
        config = load_config(ROOT / "configs/default.json")
        exports = config_to_shell_exports(config)

        self.assertEqual(exports["MODEL_NAME"], "Qwen/Qwen3-8B")
        self.assertEqual(exports["TRANSCODER_NAME"], "mwhanna/qwen3-8b-transcoders")
        self.assertEqual(exports["DATA_ADAPTER"], "ambigqa")
        self.assertEqual(exports["MAX_TOTAL_CONTINUATIONS"], "500")
        self.assertEqual(exports["TRUST_REMOTE_CODE"], "true")

    def test_write_resolved_config_removes_extends_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "resolved.json"
            resolved = write_resolved_config(ROOT / "configs/default.json", output)
            on_disk = json.loads(output.read_text())

        self.assertNotIn("extends", on_disk)
        self.assertEqual(on_disk, resolved)
        self.assertEqual(on_disk["model"]["base_model"], "Qwen/Qwen3-8B")


if __name__ == "__main__":
    unittest.main()
