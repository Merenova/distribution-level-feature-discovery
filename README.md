# Latent Planning: Rate-Distortion Clustering

Implementation of **Rate-Distortion two-view clustering** for analyzing semantic-attribution structure in language model continuations. This approach discovers latent semantic components by jointly optimizing over semantic embeddings and attribution features using rate-distortion theory.

## Overview

Given a prefix (e.g., "The capital of France is"), this pipeline:
1. Samples diverse continuations from the language model
2. Computes circuit-tracer attributions from prefix features to continuation tokens
3. Extracts semantic embeddings for each continuation
4. Clusters continuations using a Rate-Distortion objective over both views
5. Validates discovered clusters via steering interventions

**Key features:**
- **Rate-Distortion objective**: Information-theoretic clustering with principled trade-offs
- **Two-view clustering**: Joint optimization over semantic (embedding) and mechanistic (attribution) spaces
- **Probability weighting**: Path probabilities weight all component statistics
- **Steering validation**: Causal validation via feature steering interventions
- **Flexible attribution spans**: Full continuation, distinguishing token, or post-LCS attribution (Default: full)

## Project Structure

```
latent_planning/
├── 0_preprocess/                    # AmbigQA cloze prep (optional)
├── 1_data_preparation/              # Stage 1: select_test_clozes.py
├── 2_branch_sampling/               # Stage 2: sample_branches.py + sample_reasoning_steps.py
├── 3_attribution_graphs/            # Stage 3: prefix→continuation + reasoning step-pair attribution
├── 4_feature_extraction/            # Stage 4: EmbeddingGemma continuation embeddings
├── 5_gaussian_clustering/           # Stage 5: Rate-Distortion clustering core
│   ├── cluster.py em_loop.py rd_objective.py
│   ├── adaptive_control.py initialize.py split_stability.py
│   └── sweep_utils.py gpu_utils.py
├── 6_semantic_graphs/               # Stage 6: H_c / token-score extraction (input to Stage 8)
├── 7_validation/                    # Stage 7: validation
│   ├── 7a_graph_validation.py
│   ├── 7c_{hypotheses,steering,graph,metrics,utils}.py
│   ├── 7c_baseline_{kmeans,single,combined_medoid}.py    # paper §5 baselines (KM-Sem, Single)
│   ├── 7c_cluster_analysis.py
│   ├── analyze_steering_methods.py extract_tokenwise_logit_diff.py select_h4c_manifest.py
├── 8_visualization/                 # Stage 8: paper figures
│   ├── make_rd_table.py rd_curve_compare_kmeans.py rd_curve_sweep_k.py
│   ├── cross_silhouette_{analysis,histogram}.py jaccard_matrix_relabel.py
│   ├── anneal_beta_exact_k.py find_minimal_beta.py
│   ├── beta_similarity_heatmaps.py cluster_plots.py parameter_sweep_plots.py
│   ├── sankey_plots.py semantic_graph_plots.py summary_plots.py tsne_plots.py
│   ├── verify_d_gamma_full.py visualize_{logit_heatmaps,medoid_farthest,text_attribution}.py
│   └── compare_rd_coreg_fixed_rate.py                   # Co-Reg baseline
├── circuit-tracer/                  # vendored attribution library
├── configs/                         # paper-aligned and extended experiment configs
│   ├── default_config.json
│   ├── beta_gamma_scaled_config.json     # Qwen3-8B β/γ sweep (§4–§5)
│   ├── beta_gamma_scaled_qwen4.json      # Qwen3-4B replication (App.)
│   ├── {ambigqa,harmbench,mmlu}_{qwen3,gemma3}_*_config.json
│   ├── smoke_{ambigqa,harmbench,mmlu}_gemma3_4b_it_config.json
│   └── reasoning_qwen3_small.yaml   # Qwen3-0.6B/1.7B × MATH-500/GSM8K
├── scripts/
│   ├── run_pipeline.sh                   # stage dispatcher
│   ├── run_scale_experiments.sh          # scale experiment driver
│   ├── prepare_harmbench_questions.py
│   ├── prepare_mmlu_questions.py
│   ├── sample_reasoning_vllm_eval.py      # MATH-500 / GSM8K reasoning runner
│   ├── pipeline/                         # end-to-end pipeline runners
│   │   ├── run_qwen_8b.sh                # paper Qwen3-8B primary
│   │   ├── run_qwen_4b.sh                # paper Qwen3-4B appendix
│   │   ├── run_beta_gamma_scaled_split.sh
│   │   ├── rerun_from_stage5.sh
│   │   ├── run_gemma3_4b.sh
│   │   ├── run_mmlu_qwen_pipeline.sh
│   │   └── run_reasoning_qwen_pipeline.sh
│   ├── validation/                       # Stage-7 sanity / steering
│   │   ├── run_7c_validation.sh run_attr_steer_all.sh
│   │   ├── run_stage7c_scaled_split.sh run_viz_minus_random.sh
│   │   ├── run_clustering_seed_variance.py run_split_seed_stability.py
│   │   ├── benchmark_clustering_runtime_by_k.py
│   │   ├── centered_logit_alignment_multi.py
│   │   ├── check_median_vs_weighted_mean_sign.py
│   │   ├── reality_check_single_continuation_steer.py
│   │   ├── sign_consistency_report.py
│   │   ├── steering_mass_monotonicity_report.py
│   │   └── compare_rd_vs_kmeans_first5.py
│   ├── token_attr/                       # Appendix C continuation-level attribution
│   ├── tables/                           # paper LaTeX-table builders
│   │   ├── csv_to_latex_table.py csv_to_latex_steering.py
│   │   ├── generate_rd_table_with_dgamma.py mass_within_config_corr_to_tex.py
│   │   └── run_all_paper_tables.sh
│   ├── remote/                           # distributed-run infra
│   └── misc/install_deps.sh + mmlu_remote_baseline_lib.sh
├── utils/                                # shared config/data/logging/manifest helpers
├── tests/
├── docs/superpowers/plans/               # implementation plans + notes
└── pyproject.toml, uv.lock, .gitignore, README.md
```

### Reasoning benchmarks

Reasoning benchmarks use step-rollout branch sampling and pairwise step attribution rather than ordinary prefix-continuation sampling. For each target step `j`, the sampler draws `k` current-step traces and commits the top-confidence trace for future prefixes. The reasoning stage 3 emits `(i, j)` pair artifacts from each committed source step `i < j` to sampled target step candidates.

Run the full small-Qwen reasoning sweep with:

```bash
bash scripts/pipeline/run_reasoning_qwen3_small_sweep.sh --output-root experiments/reasoning_runs
```

The sweep runs `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-1.7B` on `gsm8k` and `math500`, using these transcoders:

| Model | Transcoder |
| --- | --- |
| `Qwen/Qwen3-0.6B` | `mwhanna/qwen3-0.6b-transcoders-lowl0` |
| `Qwen/Qwen3-1.7B` | `mwhanna/qwen3-1.7b-transcoders-lowl0` |

For each model, the sweep launches the two datasets concurrently with `gsm8k` on `CUDA_VISIBLE_DEVICES=0` and `math500` on `CUDA_VISIBLE_DEVICES=1`, then waits before starting the next model.

To validate the generated commands and runtime clustering configs without launching the full run:

```bash
bash scripts/pipeline/run_reasoning_qwen3_small_sweep.sh \
  --output-root /tmp/reasoning_qwen3_small_dry_run \
  --dry-run
```

To run a subset, pass comma-separated `model_key:dataset` combinations:

```bash
bash scripts/pipeline/run_reasoning_qwen3_small_sweep.sh \
  --output-root experiments/reasoning_runs \
  --only qwen3_1_7b:math500
```

For lower-level control over a single combination, call the underlying runner directly:

```bash
bash scripts/pipeline/run_reasoning_qwen_pipeline.sh \
  --config configs/reasoning_qwen3_small.yaml \
  --model Qwen/Qwen3-0.6B \
  --dataset gsm8k \
  --transcoder mwhanna/qwen3-0.6b-transcoders-lowl0 \
  --output-root experiments/reasoning_runs/qwen3_0_6b_gsm8k
```

## Rate-Distortion Objective

The algorithm optimizes:

```
L_RD = H(C) + β_e · D^(e) + β_a · D^(a)
```

Where:
- **H(C)** = -Σ_c P̄_c · log(P̄_c) — entropy (rate/compression term)
- **D^(e)** = Σ_n P_n · ||e_n - μ_{c(n)}^(e)||² / d_e — semantic distortion (L2 loss)
- **D^(a)** = Σ_n P_n · ||a_n - μ_{c(n)}^(a)|| / d_a — attribution distortion (L1 loss)
- **β = β_e + β_a** — total precision (inverse temperature)
- **γ = β_e / β** — view ratio (semantic vs attribution weighting)

### Key Parameters

| Parameter | Config Key | Description |
|-----------|------------|-------------|
| **β** | `clustering.beta` | Total precision. Higher = more clusters |
| **γ** | `clustering.gamma` | View ratio. 0.5 = equal weight; >0.5 favors semantic |
| **K_max** | `clustering.K_max` | Maximum number of components |
| **pooling** | `clustering.pooling` | Attribution pooling: `mean`, `max`, `sum` |
| **span_mode** | `attribution.span_mode` | Attribution span: `full`, `lcs_plus_one`, `post_lcs` |
