from __future__ import annotations

import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.audit.necessity_inventory import build_inventory


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def working_directory(path: Path):
    original = Path.cwd()
    try:
        import os

        os.chdir(path)
        yield
    finally:
        os.chdir(original)


class NecessityInventoryTests(unittest.TestCase):
    def test_default_inventory_includes_pipeline_stages(self) -> None:
        inventory = build_inventory(ROOT, [ROOT / "configs/default.json"])
        required = set(inventory["required_files"])

        self.assertIn("scripts/run_paper_pipeline.sh", required)
        self.assertIn("0_preprocess/prepare_ambigqa_questions.py", required)
        self.assertIn("1_data_preparation/format_ambigqa_questions.py", required)
        self.assertIn("2_branch_sampling/sample_branches.py", required)
        self.assertIn("3_attribution_graphs/compute_continuation_attribution.py", required)
        self.assertIn("4_feature_extraction/compute_embeddings.py", required)
        self.assertIn("5_gaussian_clustering/cluster.py", required)
        self.assertIn("6_semantic_graphs/extract_graphs.py", required)
        self.assertIn("7_validation/rd_medoid.py", required)
        self.assertIn("7_validation/km_sem.py", required)
        self.assertIn("7_validation/single.py", required)

    def test_inventory_reports_supported_presets(self) -> None:
        inventory = build_inventory(
            ROOT,
            [
                ROOT / "configs/default.json",
                ROOT / "configs/presets/ambigqa_qwen3_4b.json",
            ],
        )

        self.assertEqual(inventory["unsupported_presets"], [])
        self.assertEqual(
            inventory["presets"],
            [
                "configs/default.json",
                "configs/presets/ambigqa_qwen3_4b.json",
            ],
        )

    def test_generated_and_cache_files_are_not_required(self) -> None:
        inventory = build_inventory(ROOT, [ROOT / "configs/default.json"])
        required = set(inventory["required_files"])

        self.assertNotIn("utils/__pycache__/config.cpython-311.pyc", required)
        self.assertNotIn("results", required)
        self.assertNotIn("logs", required)

    def test_inventory_includes_transitive_config_dependencies(self) -> None:
        inventory = build_inventory(ROOT, [ROOT / "configs/default.json"])
        required = set(inventory["required_files"])

        self.assertIn("configs/default.json", required)
        self.assertIn("configs/presets/ambigqa_qwen3_8b.json", required)
        self.assertIn("configs/base.json", required)

    def test_relative_config_paths_resolve_against_root(self) -> None:
        with TemporaryDirectory() as tmp:
            with working_directory(Path(tmp)):
                inventory = build_inventory(ROOT, [Path("configs/default.json")])

        self.assertIn("configs/default.json", inventory["presets"])

    def test_unused_retained_presets_are_unclassified(self) -> None:
        inventory = build_inventory(ROOT, [ROOT / "configs/default.json"])
        unclassified = set(inventory["unclassified_tracked_files"])

        self.assertIn("configs/presets/ambigqa_qwen3_4b.json", unclassified)
        self.assertIn("configs/presets/integrity_qwen3_4b_single.json", unclassified)


if __name__ == "__main__":
    unittest.main()
