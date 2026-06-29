# Friendly README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the root README so it introduces the paper, explains the cleaned reproduction repository, and keeps reproduction commands easy to run.

**Architecture:** This is a documentation-only change. `README.md` remains the single public entry point, while the existing configs, scripts, stages, tests, and `docs/` project page are referenced but not modified.

**Tech Stack:** Markdown, existing shell scripts, `uv`, Python `unittest`.

---

## File Structure

- Modify: `README.md`
  - Responsibility: public-facing overview, quick start, configuration, outputs, reproducibility checks, and paper status.
- Leave unchanged: `configs/`, `scripts/`, stage directories, `tests/`, `docs/`, and all Python code.
  - Responsibility: existing implementation and project page behavior.

## Constraints

- Keep AmbigQA and Qwen3-8B as the documented defaults.
- Present dataset/model choices as flexible configuration, not hard-coded assumptions.
- Mention the paper title and authors without claiming final acceptance metadata.
- Preserve the existing operational commands: dry run, default run, preset run, integrity audit, and necessity audit.
- Do not add new dependencies or hidden prerequisites.

### Task 1: Rewrite Root README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read the current README**

Run:

```bash
sed -n '1,240p' README.md
```

Expected: the current README starts with `# RD Clustering Paper-Clean Implementation` and contains sections for default run, presets, outputs, integrity audit, and necessity audit.

- [ ] **Step 2: Replace `README.md` with the friendly paper-first version**

Use this exact content:

```markdown
# Distribution-Level Feature Discovery

This repository contains the cleaned reproduction code for **Shared Semantics, Divergent Mechanisms: Unsupervised Feature Discovery by Aligning Semantics and Mechanisms**. The paper studies distribution-level feature discovery for language model continuations: instead of explaining one hand-picked target answer, it samples many possible continuations for a prompt, represents each continuation with both semantic embeddings and sequence-level mechanistic attribution signatures, and clusters them with a rate-distortion objective.

The default reproduction setting is **AmbigQA + Qwen3-8B**. The configuration system is flexible, so AmbigQA and Qwen3-8B are defaults rather than hard-coded assumptions.

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

## Reproducibility Checks

Compare a minimal clean run against the original repository:

```bash
bash scripts/run_integrity_compare.sh \
  --original-root /home/hyunjin/latent_planning \
  --clean-root /home/hyunjin/latent_planning_paper_clean \
  --dataset-dir /home/hyunjin/latent_planning/data/cloze_llm_improved_split_ratio_0.1 \
  --config configs/presets/integrity_qwen3_4b_single.json
```

The integrity script writes JSON and Markdown reports under its temporary work directory.

Explain which tracked files are required by the retained presets:

```bash
uv run python scripts/audit/necessity_inventory.py \
  --config configs/default.json \
  --config configs/presets/ambigqa_qwen3_4b.json
```

Generated outputs, logs, caches, virtual environments, and large result directories are intentionally not tracked.

## Project Layout

- `configs/`: base config, default config, and tracked reproduction presets.
- `scripts/`: pipeline runner, integrity comparison, and audit utilities.
- `0_preprocess/` through `7_validation/`: retained reproduction stages.
- `5_gaussian_clustering/`: rate-distortion clustering implementation.
- `inputs/`: small example prefix input for smoke checks.
- `tests/`: unit tests for config resolution, pipeline command construction, and necessity inventory.
- `docs/`: project page assets preserved from the deployment repository.

## Paper Status

This repository accompanies the ICML 2026 paper draft:

**Shared Semantics, Divergent Mechanisms: Unsupervised Feature Discovery by Aligning Semantics and Mechanisms**

Hyunjin Cho, Youngji Roh, and Jaehyung Kim

The final citation metadata can be added once the paper record is finalized.
```

- [ ] **Step 3: Review the rewritten README**

Run:

```bash
sed -n '1,260p' README.md
```

Expected:

- The title is `# Distribution-Level Feature Discovery`.
- The first section mentions the paper title.
- The quick start contains dry-run, default, Qwen3-4B preset, and prefix smoke-check commands.
- The README still documents integrity and necessity audits.

### Task 2: Verify Paths, Commands, And Tests

**Files:**
- Read: `README.md`
- Read: existing referenced files and directories

- [ ] **Step 1: Check referenced repository paths exist**

Run:

```bash
for path in \
  configs/default.json \
  configs/presets/ambigqa_qwen3_8b.json \
  configs/presets/ambigqa_qwen3_4b.json \
  configs/presets/integrity_qwen3_4b_single.json \
  scripts/run_paper_pipeline.sh \
  scripts/run_integrity_compare.sh \
  scripts/audit/necessity_inventory.py \
  inputs/prefixes.example.json \
  tests \
  docs \
  0_preprocess \
  1_data_preparation \
  2_branch_sampling \
  3_attribution_graphs \
  4_feature_extraction \
  5_gaussian_clustering \
  6_semantic_graphs \
  7_validation
do
  test -e "$path" || { echo "missing $path"; exit 1; }
done
```

Expected: command exits with status 0 and prints nothing.

- [ ] **Step 2: Run the dry-run command documented in the README**

Run:

```bash
bash scripts/run_paper_pipeline.sh \
  --dry-run \
  --prefixes-file inputs/prefixes.example.json \
  --output-dir /tmp/lp_dry_run
```

Expected: command exits with status 0 and prints planned stage commands without launching model work.

- [ ] **Step 3: Run the existing test suite**

Run:

```bash
uv run python -m unittest discover -s tests -v
```

Expected: all existing tests pass.

### Task 3: Commit And Push README Work

**Files:**
- Modify: `README.md`
- Create: `docs/superpowers/plans/2026-06-29-friendly-readme.md`

- [ ] **Step 1: Check the final diff**

Run:

```bash
git diff -- README.md docs/superpowers/plans/2026-06-29-friendly-readme.md
```

Expected: diff only contains the README rewrite and this implementation plan.

- [ ] **Step 2: Commit the README update and plan**

Run:

```bash
git add README.md docs/superpowers/plans/2026-06-29-friendly-readme.md
git commit -m "Improve README for paper reproduction"
```

Expected: commit succeeds.

- [ ] **Step 3: Push to origin**

Run:

```bash
git push origin main
```

Expected: push succeeds and includes the previous design-spec commit plus the README update commit.
