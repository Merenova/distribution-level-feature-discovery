# RD Clustering Paper-Clean Implementation

This repository is the cleaned reproduction code for the rate-distortion clustering experiments. The default configuration is AmbigQA with Qwen3-8B.

## Default Run

```bash
bash scripts/run_paper_pipeline.sh \
  --dataset-dir /path/to/ambigqa_dataset \
  --output-dir runs/ambigqa_qwen3_8b
```

This uses `configs/default.json`, which resolves to `configs/presets/ambigqa_qwen3_8b.json`.

## Presets

Available tracked presets:

- `configs/default.json`: AmbigQA + Qwen3-8B default.
- `configs/presets/ambigqa_qwen3_8b.json`: explicit default preset.
- `configs/presets/ambigqa_qwen3_4b.json`: AmbigQA + Qwen3-4B replication.
- `configs/presets/integrity_qwen3_4b_single.json`: single-prefix integrity fixture.

Run a preset with:

```bash
bash scripts/run_paper_pipeline.sh \
  --config configs/presets/ambigqa_qwen3_4b.json \
  --dataset-dir /path/to/ambigqa_dataset \
  --output-dir runs/ambigqa_qwen3_4b
```

Direct prefix input is available for smoke checks:

```bash
bash scripts/run_paper_pipeline.sh \
  --prefixes-file inputs/prefixes.example.json \
  --output-dir runs/prefix_smoke
```

Print commands without launching model work:

```bash
bash scripts/run_paper_pipeline.sh \
  --dry-run \
  --prefixes-file inputs/prefixes.example.json \
  --output-dir /tmp/lp_dry_run
```

## Outputs

Stage outputs are written under `<output-dir>/results`:

- `0_preprocess`
- `1_data_preparation`
- `2_branch_sampling`
- `3_attribution_graphs`
- `4_feature_extraction`
- `5_clustering`
- `6_semantic_graphs`
- `7_validation`

The resolved config used by a run is written to `<output-dir>/results/resolved_config.json`.

## Integrity Audit

The integrity audit compares a minimal clean run against the original repository:

```bash
bash scripts/run_integrity_compare.sh \
  --original-root /home/hyunjin/latent_planning \
  --clean-root /home/hyunjin/latent_planning_paper_clean \
  --dataset-dir /home/hyunjin/latent_planning/data/cloze_llm_improved_split_ratio_0.1 \
  --config configs/presets/integrity_qwen3_4b_single.json
```

The audit writes JSON and Markdown reports under its temporary workdir.

## Necessity Audit

Explain which tracked files are required by the retained presets:

```bash
uv run python scripts/audit/necessity_inventory.py \
  --config configs/default.json \
  --config configs/presets/ambigqa_qwen3_4b.json
```

Generated outputs, logs, caches, virtual environments, and large result directories are intentionally not tracked.
