# Friendly README Design

## Goal

Rewrite the repository README so it is useful to both external researchers and future collaborators. The README should introduce the paper briefly, explain what this cleaned repository reproduces, and keep the exact reproduction commands easy to find.

## Audience

- External researchers who find the repository from the paper or project page and need to understand the method before running the code.
- Future maintainers and collaborators who need to reproduce, audit, or extend the cleaned implementation without rediscovering the repository structure.

## Paper Introduction

The README will introduce the accompanying paper:

> Shared Semantics, Divergent Mechanisms: Unsupervised Feature Discovery by Aligning Semantics and Mechanisms

The summary will stay short and factual. It will say that the paper studies distribution-level feature discovery for language model continuations: instead of explaining one hand-picked target answer, it samples many continuations for a prompt, represents each continuation with both semantic embeddings and sequence-level mechanistic attribution signatures, and clusters them with a rate-distortion objective. The purpose is to reveal continuation modes that are coherent both in meaning and in internal model mechanisms.

## README Structure

The README will use this section order:

1. Title: `Distribution-Level Feature Discovery`
2. One-paragraph paper and repository overview.
3. `What this repository reproduces`
   - Default AmbigQA + Qwen3-8B reproduction.
   - Flexible dataset/model configuration, with AmbigQA and Qwen3-8B as defaults.
   - Qwen3-4B preset and integrity fixture.
   - Integrity comparison and necessity audit tooling.
4. `Method at a glance`
   - Sample continuations.
   - Build semantic and mechanistic views.
   - Run rate-distortion clustering.
   - Optionally validate cluster-derived features with steering analyses.
5. `Quick start`
   - Install or sync dependencies.
   - Dry-run command for smoke checking.
   - Default command.
   - Preset command.
6. `Configuration`
   - Explain `configs/default.json` and preset overrides.
   - State that AmbigQA and Qwen3-8B are defaults, not hard-coded assumptions.
7. `Outputs`
   - Explain `<output-dir>/results`.
   - List the stage directories.
   - Mention `resolved_config.json`.
8. `Reproducibility checks`
   - Integrity comparison against the original repository.
   - Necessity inventory for tracked files required by retained presets.
9. `Project layout`
   - Map `configs/`, `scripts/`, `stages/`, `tests/`, `docs/`, and `inputs/`.
10. `Paper status`
   - Name the ICML 2026 paper draft and authors.
   - Leave citation lightweight until the final citation is available.

## Content Constraints

- Keep the tone friendly, direct, and reproducibility-oriented.
- Do not overstate paper acceptance, benchmark status, or result coverage.
- Keep commands exact and runnable from the repository root.
- Preserve current operational information: default run, preset run, dry run, outputs, integrity audit, and necessity audit.
- Do not add unrelated installation claims or hidden prerequisites that are not reflected in the repository.
- Prefer short paragraphs and command blocks over long prose.

## Testing And Review

README implementation will be checked by:

- Reviewing the rendered Markdown structure for clarity.
- Running the existing unit test suite to ensure documentation edits did not disturb tracked code or configuration.
- Checking that all paths and command names mentioned in the README exist in the repository.

## Out Of Scope

- Changing pipeline behavior.
- Renaming config files or stage directories.
- Adding new experiments.
- Rewriting the project page under `docs/`.
- Adding a final BibTeX citation before the paper metadata is finalized.
