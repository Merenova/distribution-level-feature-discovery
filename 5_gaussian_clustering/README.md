# Rate-Distortion Gaussian Clustering

This module implements **rate-distortion based two-view Gaussian clustering** for semantic and attribution embeddings. It uses principled information-theoretic criteria with MSE-normalized distances.

## Overview

The algorithm optimizes the rate-distortion objective:

```
L_RD = H(C) + β_e · D^(e) + β_a · D^(a)
```

Where:
- **H(C)** = -Σ_c P̄_c · log(P̄_c) — entropy (rate/compression term)
- **D^(e)** = Σ_n P_n · ||e_n - μ_c^(e)||² / d_e — semantic distortion (MSE-normalized)
- **D^(a)** = Σ_n P_n · ||a_n - μ_c^(a)||² / d_a — attribution distortion (MSE-normalized)

The MSE normalization (dividing by dimensionality) ensures comparable scales across embedding spaces.

## Key Features

| Aspect | Description |
|--------|-------------|
| **Initialization** | Single component (K=1); structure emerges from split operations |
| **Split trigger** | R-D criterion: β_e·dD_e + β_a·dD_a > P_c·H_binary(α) |
| **Split method** | 2-means with combined distance: β_e·\|\|e\|\|²/d_e + β_a·\|\|a\|\|²/d_a |
| **Junk criterion** | R-D: \|dH\| > β_e·dD_e + β_a·dD_a (with reassignment) |
| **Convergence** | Relative L_RD change < threshold |

## Module Structure

```
5_gaussian_clustering/
├── cluster.py              # Main orchestrator and CLI
├── rd_objective.py         # R-D objective computation + MSE helpers
├── em_loop.py              # E-step (R-D assignment) and M-step
├── adaptive_control.py     # Split/Junk operations (exact R-D criteria)
├── initialize.py           # Single-component initialization
├── fit_logit_templates.py  # Post-hoc logit template fitting
└── README.md
```

## Configuration Parameters

Set in `configs/default_config.json`:

```json
{
  "clustering": {
    "beta": 20000.0,
    "gamma": 0.5,
    "K_max": 30,
    "K_clamp": 10,
    "max_iterations": 50,
    "convergence_threshold": 1e-3
  }
}
```

### Parameter Descriptions

| Parameter | Description |
|-----------|-------------|
| **beta** | Total precision β = β_e + β_a. Higher values prioritize reconstruction over compression. |
| **gamma** | View ratio γ = β_e/β. Controls semantic vs attribution weighting. γ=0.5 gives equal weight. |
| **K_max** | (Deprecated for clustering) Stored in sweep results for backward compatibility. Clustering now converges naturally based on R-D criterion without a hard cap. |
| **K_clamp** | Maximum K for downstream tasks (e.g., steering). Configs with K > K_clamp are skipped in downstream stages. |
| **max_iterations** | Maximum EM iterations per clustering run. |
| **convergence_threshold** | Relative change in L_RD for convergence detection. |

## Usage

### Command Line

```bash
uv run python 5_gaussian_clustering/cluster.py \
    --embeddings-dir results/4_feature_extraction/embeddings/ \
    --attribution-graphs-dir results/2_attribution_graphs/ \
    --logits-dir results/4_feature_extraction/logits/ \
    --samples-dir results/3_branch_sampling/ \
    --beta 20000.0 \
    --gamma 0.5 \
    --K-max 20 \
    --output-dir results/5_clustering/ \
    --log-dir logs/
```

### Using Config File

```bash
uv run python 5_gaussian_clustering/cluster.py \
    --embeddings-dir results/4_feature_extraction/embeddings/ \
    --attribution-graphs-dir results/2_attribution_graphs/ \
    --logits-dir results/4_feature_extraction/logits/ \
    --samples-dir results/3_branch_sampling/ \
    --config configs/default_config.json \
    --output-dir results/5_clustering/
```

## Algorithm Details

### Initialization

All samples start in a single component (K=1). The number of components emerges organically through split operations based on R-D criteria.

### E-Step (Assignment)

Each sample is assigned to the cluster minimizing the R-D cost:

```
c(n) = argmin_c [ -log(P̄_c) + β_e·||e_n - μ_c^(e)||²/d_e + β_a·||a_n - μ_c^(a)||²/d_a ]
```

The `-log(P̄_c)` term acts as a rate regularizer, discouraging assignment to rare clusters.

### M-Step (Update)

Cluster centers are updated with probability-weighted means:

```
μ_c^(e) = Σ_{n: c(n)=c} P_n · e_n / W_c
μ_c^(a) = Σ_{n: c(n)=c} P_n · a_n / W_c
P̄_c = W_c / W_total
```

Where W_c = Σ_{n: c(n)=c} P_n is the probability mass of cluster c.

### Adaptive Control

After each EM iteration:

1. **Split**: Components with high internal variance are split using 2-means on the combined distance. A split is accepted if:
   ```
   β_e·dD_e + β_a·dD_a > P_c·H_binary(α)
   ```
   where α is the split ratio and H_binary is the binary entropy function.

2. **Junk**: Components are removed if their rate savings exceed the reassignment distortion cost:
   ```
   |dH| > β_e·dD_e + β_a·dD_a
   ```
   Points are reassigned to the nearest remaining component (not discarded).

### MSE Distance Computation

The module uses MSE-normalized distances to handle different embedding dimensions:

```python
# From rd_objective.py
def compute_mse_distances(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Compute MSE (mean squared error) distances from data points to centers."""
    diff = data[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sum(diff ** 2, axis=2) / data.shape[1]  # Divide by dimension
```

## Output Format

The output JSON files contain:

```json
{
  "prefix_id": "cloze_0042",
  "prefix": "The capital of France is",
  "n_components": 4,
  "converged": true,
  "n_iterations": 12,
  "assignments": [1, 1, 2, 1, 0, 3, ...],
  "H_0": [...],
  "v_0": 12.34,
  "components": {
    "0": {
      "mu_e": [...],
      "mu_a": [...],
      "Delta_v": 0.0,
      "W_c": 0.05,
      "n_samples": 15,
      "is_junk": true
    },
    "1": {
      "mu_e": [...],
      "mu_a": [...],
      "Delta_v": 0.82,
      "W_c": 0.35,
      "n_samples": 120,
      "is_junk": false
    }
  },
  "rd_objective": {
    "L_RD": 1.234,
    "H": 1.5,
    "D_e": 0.02,
    "D_a": 0.03,
    "beta_e": 10000.0,
    "beta_a": 10000.0
  },
  "config": {
    "beta": 20000.0,
    "gamma": 0.5,
    "K_max": 20
  }
}
```

### Component Fields

| Field | Description |
|-------|-------------|
| `mu_e` | Semantic centroid (d_e dimensional) |
| `mu_a` | Attribution centroid / ΔH_c (d_a dimensional) |
| `Delta_v` | Logit template delta from baseline v_0 |
| `W_c` | Probability mass (sum of P_n for assigned samples) |
| `n_samples` | Number of samples assigned to this cluster |
| `is_junk` | Whether this is the junk cluster (cluster 0) |

### Cluster 0 (Junk)

Cluster 0 is the "junk" cluster containing low-probability or poorly-fit samples. It is preserved in the output but typically excluded from downstream analysis (semantic graph extraction, steering).

## Logit Template Fitting

After clustering converges, logit templates are fit using the `fit_logit_templates.py` module:

```python
from fit_logit_templates import compute_token_scores

# Compute π-scores for each token across clusters
token_scores = compute_token_scores(
    assignments=assignments,
    first_token_ids=first_token_ids,
    probabilities=probabilities,
    n_clusters=n_clusters
)
# Returns: {token_id: {cluster_id: π_score}}
```

The token scores π_{s,c} represent the affinity of token s for cluster c, computed as the probability-weighted frequency of token s in cluster c.

## References

- **Method specification**: `rate_distortion.tex`
- **Validation specification**: `validation.tex`
- Shannon rate-distortion theory (1959)
- 3D Gaussian Splatting probability weighting (Kerbl et al., 2023)
