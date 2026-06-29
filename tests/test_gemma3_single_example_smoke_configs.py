from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_gemma3_4b_single_example_smoke.sh"

EXPECTED = {
    "AmbigQA_Gemma3-4B-it_single": (
        ROOT / "configs" / "ambigqa_gemma3_4b_it_config.json",
        ROOT / "configs" / "smoke_ambigqa_gemma3_4b_it_config.json",
        "data/cloze_llm_improved_split_ratio_0.1",
    ),
    "MMLU_Gemma3-4B-it_single": (
        ROOT / "configs" / "mmlu_gemma3_4b_it_config.json",
        ROOT / "configs" / "smoke_mmlu_gemma3_4b_it_config.json",
        "data/mmlu_cais_validation_7subjects",
    ),
    "HarmBench_Gemma3-4B-it_single": (
        ROOT / "configs" / "harmbench_gemma3_4b_it_config.json",
        ROOT / "configs" / "smoke_harmbench_gemma3_4b_it_config.json",
        "data/harmbench_walledai_standard",
    ),
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _script_combos() -> dict[str, tuple[Path, str]]:
    text = SCRIPT.read_text()
    match = re.search(r"COMBOS_RAW='(?P<body>.*?)'", text, re.S)
    assert match is not None
    combos: dict[str, tuple[Path, str]] = {}
    for raw_line in match.group("body").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        tag, config, cloze_dir = line.split()
        combos[tag] = (ROOT / config, cloze_dir)
    return combos


def test_single_example_smoke_script_references_expected_configs():
    combos = _script_combos()

    assert combos == {
        tag: (config_path, cloze_dir)
        for tag, (_, config_path, cloze_dir) in EXPECTED.items()
    }


def test_single_example_smoke_script_prepares_missing_datasets():
    text = SCRIPT.read_text()

    assert "ensure_ambigqa_smoke_prepped" in text
    assert "ensure_mmlu_prepped" in text
    assert "ensure_harmbench_prepped" in text
    assert "data/cloze_llm_improved_split_ratio_0.1" in text
    assert "data/mmlu_cais_validation_7subjects" in text
    assert "data/harmbench_walledai_standard" in text
    assert 'DatasetDict({"train": Dataset.from_list(rows)}).save_to_disk' in text
    assert "scripts/prepare_mmlu_questions.py" in text
    assert "scripts/prepare_harmbench_questions.py" in text
    assert "capital of France" not in text
    assert "Give one unusual but plausible reason" in text


def test_single_example_configs_keep_one_prefix_with_many_rollouts():
    for base_path, smoke_path, cloze_dir in EXPECTED.values():
        base = _load_json(base_path)
        smoke = _load_json(smoke_path)

        assert smoke["data"]["cloze_dir"] == cloze_dir
        assert smoke["model"]["base_model"] == "google/gemma-3-4b-it"
        assert smoke["attribution"]["backend"] == "nnsight"
        assert smoke["stage_7c_steering"]["backend"] == "nnsight"
        assert smoke["stage_7c_steering"]["store_sequence_logit_values"] is True
        assert smoke["stage_1_data_prep"]["n_groups"] == 1
        assert smoke["sampling"]["max_total_continuations"] == 300
        assert smoke["sampling"]["batch_size"] == 64
        assert smoke["sampling"]["max_complete"] == 1
        assert smoke["sampling"]["temperature"] > base["sampling"]["temperature"]
        assert smoke["sampling"]["nucleus_p"] >= base["sampling"]["nucleus_p"]
        assert smoke["vllm"]["gpu_memory_utilization"] == 0.6
        assert smoke["clustering"]["sweeps"]["beta_values"] == [
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.3,
        ]

        comparable_base = json.loads(json.dumps(base))
        comparable_smoke = json.loads(json.dumps(smoke))
        comparable_base["experiment_name"] = comparable_smoke["experiment_name"]
        comparable_base["stage_1_data_prep"]["n_groups"] = 1
        comparable_base["sampling"]["max_total_continuations"] = 300
        comparable_base["sampling"]["batch_size"] = 64
        comparable_base["sampling"]["temperature"] = comparable_smoke["sampling"]["temperature"]
        comparable_base["sampling"]["nucleus_p"] = comparable_smoke["sampling"]["nucleus_p"]
        comparable_base["sampling"]["max_complete"] = 1
        comparable_base["vllm"]["gpu_memory_utilization"] = 0.6
        comparable_base["clustering"]["sweeps"]["beta_values"] = comparable_smoke["clustering"]["sweeps"]["beta_values"]
        comparable_base["stage_7c_steering"]["store_sequence_logit_values"] = True

        assert comparable_smoke == comparable_base
