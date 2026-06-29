from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_scale_experiments.sh"
PIPELINE_SCRIPT = ROOT / "scripts" / "run_pipeline.sh"
MMLU_CLOZE_DIR = "data/mmlu_cais_validation_7subjects"
MMLU_SUBJECTS = (
    "international_law",
    "logical_fallacies",
    "moral_disputes",
    "philosophy",
    "professional_psychology",
    "sociology",
    "world_religions",
)


EXPECTED_COMBOS = {
    "AmbigQA_Gemma3-1B-it": (
        "configs/ambigqa_gemma3_1b_it_config.json",
        "data/cloze_llm_improved_split_ratio_0.1",
    ),
    "AmbigQA_Gemma3-4B-it": (
        "configs/ambigqa_gemma3_4b_it_config.json",
        "data/cloze_llm_improved_split_ratio_0.1",
    ),
    "MMLU_Qwen3-8B": (
        "configs/mmlu_qwen3_8b_config.json",
        MMLU_CLOZE_DIR,
    ),
    "MMLU_Qwen3-4B": (
        "configs/mmlu_qwen3_4b_config.json",
        MMLU_CLOZE_DIR,
    ),
    "MMLU_Gemma3-1B-it": (
        "configs/mmlu_gemma3_1b_it_config.json",
        MMLU_CLOZE_DIR,
    ),
    "MMLU_Gemma3-4B-it": (
        "configs/mmlu_gemma3_4b_it_config.json",
        MMLU_CLOZE_DIR,
    ),
    "HarmBench_Qwen3-8B": (
        "configs/harmbench_qwen3_8b_config.json",
        "data/harmbench_walledai_standard",
    ),
    "HarmBench_Qwen3-4B": (
        "configs/harmbench_qwen3_4b_config.json",
        "data/harmbench_walledai_standard",
    ),
    "HarmBench_Gemma3-1B-it": (
        "configs/harmbench_gemma3_1b_it_config.json",
        "data/harmbench_walledai_standard",
    ),
    "HarmBench_Gemma3-4B-it": (
        "configs/harmbench_gemma3_4b_it_config.json",
        "data/harmbench_walledai_standard",
    ),
}


MMLU_CONFIGS = [
    ROOT / "configs" / "mmlu_qwen3_4b_config.json",
    ROOT / "configs" / "mmlu_qwen3_8b_config.json",
    ROOT / "configs" / "mmlu_gemma3_1b_it_config.json",
    ROOT / "configs" / "mmlu_gemma3_4b_it_config.json",
]

SCALE_CONFIGS = [ROOT / config for config, _ in EXPECTED_COMBOS.values()]
CONFIG_FILES = sorted((ROOT / "configs").glob("*.json"))
MMLU_NAMED_CONFIGS = [
    path for path in CONFIG_FILES if "mmlu" in path.name
]


def _script_combos() -> dict[str, tuple[str, str]]:
    text = SCRIPT.read_text()
    match = re.search(r"COMBOS_RAW='(?P<body>.*?)'", text, re.S)
    assert match is not None

    combos: dict[str, tuple[str, str]] = {}
    for raw_line in match.group("body").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        tag, config, cloze_dir = line.split()
        combos[tag] = (config, cloze_dir)
    return combos


def test_scale_script_lists_publication_experiment_matrix():
    combos = _script_combos()

    assert combos == EXPECTED_COMBOS
    assert "AmbigQA_Qwen3-4B" not in combos
    assert "AmbigQA_Qwen3-8B" not in combos


def test_scale_script_prepares_expected_dataset_sources():
    text = SCRIPT.read_text()

    assert "ensure_ambigqa_real_dataset" in text
    assert "ensure_mmlu_prepped" in text
    assert "ensure_harmbench_prepped" in text
    assert "ambigqa_smoke" in text
    assert "scripts/prepare_mmlu_questions.py" in text
    assert "scripts/prepare_harmbench_questions.py" in text
    assert ",".join(MMLU_SUBJECTS) in text
    assert MMLU_CLOZE_DIR in text


def test_mmlu_configs_use_publication_subset_dataset():
    for path in MMLU_CONFIGS:
        config = json.loads(path.read_text())

        assert config["data"]["cloze_dir"] == MMLU_CLOZE_DIR
        assert config["data"]["split"] == "validation"
        assert config["stage_1_data_prep"]["mode"] == "question"
        assert config["stage_1_data_prep"]["n_groups"] == 209
        assert config["sampling"]["max_tokens"] == 64


def test_mmlu_named_configs_limit_sampling_tokens():
    for path in MMLU_NAMED_CONFIGS:
        config = json.loads(path.read_text())

        assert config["sampling"]["max_tokens"] == 64


def test_configs_limit_vllm_context_length():
    for path in CONFIG_FILES:
        config = json.loads(path.read_text())

        assert config["vllm"]["max_model_len"] == 256


def test_configs_disable_stage5_intermediate_snapshots():
    for path in CONFIG_FILES:
        config = json.loads(path.read_text())
        clustering = config.get("clustering", {})

        if "save_intermediate" in clustering:
            assert clustering["save_intermediate"] is False


def test_pipeline_respects_stage5_intermediate_config():
    text = PIPELINE_SCRIPT.read_text()
    stage5_block = text.split("# Stage 5: Gaussian Clustering", 1)[1].split(
        "# Stage 6: Semantic Graph Extraction", 1
    )[0]

    assert 'CLUSTERING_SAVE_INTERMEDIATE" = "true"' in stage5_block
    assert "--save-intermediate \\" not in stage5_block


def test_pipeline_forwards_stage7_acceleration_config():
    text = PIPELINE_SCRIPT.read_text()
    stage7_block = text.split("# Stage 7c: Paper steering baselines", 1)[1].split(
        "# Stage 8: Visualization", 1
    )[0]

    for exported in [
        "STEERING_MAX_SAMPLES",
        "STEERING_MAX_CLUSTER_SAMPLES",
        "STEERING_BETA_VALUES",
        "STEERING_GAMMA_VALUES",
        "STEERING_CLUSTERING_TOP_K",
        "STEERING_CLUSTERING_SCORE_KEY",
        "STEERING_CLUSTERING_SCORE_ORDER",
        "STEERING_CLUSTERING_MIN_K",
        "STEERING_CLUSTERING_MAX_K",
        "STEERING_BASELINES",
        "STEERING_PREFIX_SHARD_INDEX",
        "STEERING_PREFIX_SHARD_COUNT",
    ]:
        assert exported in text

    for forwarded in [
        "--max-samples",
        "--max-cluster-samples",
        "--beta-values",
        "--gamma-values",
        "--prefix-shard-index",
        "--prefix-shard-count",
        "--clustering-manifest",
        '--pooling "$CLUSTERING_POOLING"',
    ]:
        assert forwarded in stage7_block

    assert "7_validation/select_stage7_clustering_manifest.py" in stage7_block
    assert 'stage7_baseline_enabled "combined_medoid"' in stage7_block
    assert 'stage7_baseline_enabled "single"' in stage7_block
    assert 'stage7_baseline_enabled "kmeans"' in stage7_block
