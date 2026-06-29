# Distribution-Level Feature Discovery

This repository contains the cleaned reproduction code for **Shared Semantics, Divergent Mechanisms: Unsupervised Feature Discovery by Aligning Semantics and Mechanisms**. The paper studies distribution-level feature discovery for language model continuations: instead of explaining one hand-picked target answer, it samples many possible continuations for a prompt, represents each continuation with both semantic embeddings and sequence-level mechanistic attribution signatures, and clusters them with a rate-distortion objective.

The default reproduction setting is **AmbigQA + Qwen3-8B**. The configuration system is flexible, so AmbigQA and Qwen3-8B are defaults rather than hard-coded assumptions.

## Links

- [Paper](https://arxiv.org/abs/2606.08236)
- [Project page](https://merenova.github.io/distribution-level-feature-discovery/)

## What This Repository Reproduces

- The paper-clean AmbigQA + Qwen3-8B pipeline.
- A Qwen3-4B preset for the smaller replication setting.
- A single-prefix integrity fixture for comparing this cleaned repository against the original research code.
- Scripts for end-to-end runs, dry runs, integrity comparison, and necessity auditing.

The retained code is intentionally narrow: it focuses on the stages needed for the paper reproduction rather than preserving every exploratory utility from the original project.

## Method At A Glance

1. **Sample continuations** for each prompt so the analysis works over a model's continuation distribution, not a single selected answer.
2. **Build two views** of each continuation: a semantic embedding and a mechanistic attribution signature over prefix-to-continuation effects.
3. **Cluster with a rate-distortion objective** that trades off semantic coherence, mechanistic consistency, and cluster granularity.
4. **Validate cluster-derived features** with downstream analysis such as steering comparisons when the required artifacts are available.

## Quick Start

Install or sync the Python environment from the repository root:

```bash
uv sync
```

Before launching model work, check the resolved commands with a dry run:

```bash
bash scripts/run_paper_pipeline.sh \
  --dry-run \
  --prefixes-file inputs/prefixes.example.json \
  --output-dir /tmp/lp_dry_run
```

Run the default AmbigQA + Qwen3-8B reproduction:

```bash
bash scripts/run_paper_pipeline.sh \
  --dataset-dir /path/to/ambigqa_dataset \
  --output-dir runs/ambigqa_qwen3_8b
```

Run the Qwen3-4B preset:

```bash
bash scripts/run_paper_pipeline.sh \
  --config configs/presets/ambigqa_qwen3_4b.json \
  --dataset-dir /path/to/ambigqa_dataset \
  --output-dir runs/ambigqa_qwen3_4b
```

For a lightweight smoke check, run from explicit prefixes instead of a dataset directory:

```bash
bash scripts/run_paper_pipeline.sh \
  --prefixes-file inputs/prefixes.example.json \
  --output-dir runs/prefix_smoke
```

## Configuration

The default config is:

- `configs/default.json`

It resolves to:

- `configs/presets/ambigqa_qwen3_8b.json`

Tracked presets:

- `configs/presets/ambigqa_qwen3_8b.json`: explicit AmbigQA + Qwen3-8B default.
- `configs/presets/ambigqa_qwen3_4b.json`: AmbigQA + Qwen3-4B replication.
- `configs/presets/integrity_qwen3_4b_single.json`: single-prefix fixture for integrity checks.

Use `--config` to select a preset. Use `--dataset-dir` for AmbigQA-style dataset input, or `--prefixes-file` for direct prompt/prefix input.

## Outputs

Stage outputs are written under:

```text
<output-dir>/results
```

The retained stages are:

- `0_preprocess`
- `1_data_preparation`
- `2_branch_sampling`
- `3_attribution_graphs`
- `4_feature_extraction`
- `5_clustering`
- `6_semantic_graphs`
- `7_validation`

The resolved config for each run is saved as:

```text
<output-dir>/results/resolved_config.json
```

## Stages

- `0_preprocess`: loads AmbigQA data and writes grouped question records for the selected split. This runs only in `--dataset-dir` mode.
- `1_data_preparation`: formats the grouped AmbigQA questions into model-ready prefixes. This also runs only in `--dataset-dir` mode.
- `2_branch_sampling`: samples continuations from the configured language model for each prefix.
- `3_attribution_graphs`: computes continuation-level mechanistic attribution signatures with the configured transcoder.
- `4_feature_extraction`: computes semantic embeddings for the sampled continuations.
- `5_clustering`: runs rate-distortion clustering over the semantic and attribution views.
- `6_semantic_graphs`: extracts cluster-level semantic graph summaries from the clustering outputs.
- `7_validation`: runs steering and baseline analyses, then aggregates validation summaries.

## Project Layout

- `configs/`: base config, default config, and tracked reproduction presets.
- `scripts/`: pipeline runner, integrity comparison, and audit utilities.
- `0_preprocess/` through `7_validation/`: retained reproduction stages.
- `5_gaussian_clustering/`: rate-distortion clustering implementation.
- `inputs/`: small example prefix input for smoke checks.
- `tests/`: unit tests for config resolution, pipeline command construction, and necessity inventory.
- `docs/`: project page assets preserved from the deployment repository.

## Attribution

Some attribution and circuit-tracing code was adapted from [decoderesearch/circuit-tracer](https://github.com/decoderesearch/circuit-tracer). Please refer to the upstream project for its original implementation and license.
