from __future__ import annotations

import torch

from import_helpers import load_module_from_path
from utils.data_utils import load_json, save_json
from utils.reasoning_steps import build_reasoning_pair_id


pair_attr = load_module_from_path(
    "compute_reasoning_step_pair_attribution",
    "3_attribution_graphs/compute_reasoning_step_pair_attribution.py",
)


def test_source_mask_supports_feature_only_simplified_context():
    context_data = {
        "decoder_locations": torch.tensor(
            [
                [0, 1, 1, 2],
                [1, 2, 4, 5],
            ]
        ),
        "aggregated_attributions": torch.zeros(2, 4),
    }

    mask = pair_attr.source_mask_from_context(
        context_data,
        {"step_index": 1, "start": 2, "end": 5},
    )

    assert mask.dtype == torch.bool
    assert mask.tolist() == [False, True, True, False]


def test_source_mask_covers_features_errors_and_token_sources():
    context_data = {
        "decoder_locations": torch.tensor(
            [
                [0, 1, 1],
                [0, 2, 3],
            ]
        ),
        "aggregated_attributions": torch.zeros(2, 15),
        "n_layers": 2,
        "n_prefix_features": 3,
        "n_prefix_errors": 8,
        "n_prefix_tokens": 4,
        "n_prefix_sources": 15,
        "prefix_length": 4,
    }

    mask = pair_attr.source_mask_from_context(
        context_data,
        {"step_index": 2, "start": 2, "end": 4},
    )

    assert mask.tolist() == [
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
    ]


def test_zero_mask_context_sources_preserves_shapes_and_masks_token_attributions():
    context_data = {
        "aggregated_attributions": torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ]
        ),
        "token_attributions": [
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                ]
            ),
            torch.tensor([[9.0, 10.0, 11.0, 12.0]]),
        ],
        "store_all": True,
    }
    pair_metadata = {"pair_id": "pair-a", "source_step_index": 1}

    masked = pair_attr.zero_mask_context_sources(
        context_data,
        torch.tensor([False, True, False, True]),
        pair_metadata,
    )

    assert masked["aggregated_attributions"].shape == (2, 4)
    assert masked["aggregated_attributions"].tolist() == [
        [0.0, 2.0, 0.0, 4.0],
        [0.0, 6.0, 0.0, 8.0],
    ]
    assert [tensor.shape for tensor in masked["token_attributions"]] == [
        torch.Size([2, 4]),
        torch.Size([1, 4]),
    ]
    assert masked["token_attributions"][0].tolist() == [
        [0.0, 2.0, 0.0, 4.0],
        [0.0, 6.0, 0.0, 8.0],
    ]
    assert masked["token_attributions"][1].tolist() == [[0.0, 10.0, 0.0, 12.0]]
    assert masked["reasoning_pair_metadata"] == pair_metadata
    assert context_data["aggregated_attributions"].tolist()[0] == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]


def test_pair_sample_payload_uses_pair_id_and_preserves_target_branch_data():
    target_branch_data = {
        "prefix_id": "gsm8k_42_step_03",
        "prefix": "question\nstep 1\nstep 2\n",
        "prefix_tokens_with_bos": [0, 10, 11, 12],
        "continuations": [
            {"text": "target A", "token_ids": [1], "probability": 0.7},
            {"text": "target B", "token_ids": [2], "probability": 0.3},
        ],
        "metadata": {"dataset": "gsm8k", "task_type": "reasoning_step"},
        "reasoning_metadata": {"step_index": 3},
    }
    pair_metadata = {
        "pair_id": "gsm8k_42_step_03_src_01_tgt_03",
        "source_step_index": 1,
        "target_step_index": 3,
    }

    payload = pair_attr.build_pair_sample_payload(
        target_branch_data,
        "gsm8k_42_step_03_src_01_tgt_03",
        pair_metadata,
    )

    assert payload["prefix_id"] == "gsm8k_42_step_03_src_01_tgt_03"
    assert payload["prefix"] == target_branch_data["prefix"]
    assert (
        payload["prefix_tokens_with_bos"]
        == target_branch_data["prefix_tokens_with_bos"]
    )
    assert payload["continuations"] == target_branch_data["continuations"]
    assert payload["reasoning_metadata"] == target_branch_data["reasoning_metadata"]
    assert payload["reasoning_pair_metadata"] == pair_metadata


def test_pair_id_naming_uses_reasoning_pair_helper():
    assert build_reasoning_pair_id("math500_level_1_step_03", 1, 3) == (
        "math500_level_1_step_03_src_01_tgt_03"
    )


def test_stage3_pair_manifest_is_written_for_pair_sample_root(tmp_path):
    output_root = tmp_path / "reasoning_run"
    pair_samples_dir = output_root / "2_reasoning_pair_samples"
    pair_samples_dir.mkdir(parents=True)
    save_json(
        {
            "stage": "stage3",
            "completed": ["stale_target_step_id"],
            "failed": [],
            "skipped": [],
        },
        output_root / "manifest_stage3.json",
    )

    manifest = pair_attr.write_stage3_pair_manifest(
        pair_samples_dir=pair_samples_dir,
        pair_records=[
            {"pair_id": "gsm8k_42_step_02_src_01_tgt_02"},
            {"pair_id": "gsm8k_42_step_03_src_01_tgt_03"},
        ],
        failed_targets={"gsm8k_42_step_04": "RuntimeError: boom"},
        skipped_targets=["gsm8k_42_step_01"],
        logger=None,
    )

    saved_manifest = load_json(output_root / "manifest_stage3.json")
    assert manifest["completed"] == [
        "gsm8k_42_step_02_src_01_tgt_02",
        "gsm8k_42_step_03_src_01_tgt_03",
    ]
    assert saved_manifest["completed"] == manifest["completed"]
    assert saved_manifest["failed"] == ["gsm8k_42_step_04"]
    assert saved_manifest["skipped"] == ["gsm8k_42_step_01"]
    assert saved_manifest["errors"] == {
        "gsm8k_42_step_04": "RuntimeError: boom",
    }
