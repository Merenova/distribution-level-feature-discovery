# Stage 7 Legacy Archive

This directory contains Stage 7 files that are no longer part of the active
rebuttal pipeline.

Active Stage 7 entrypoints live in `../../7_validation` and are limited to:

- `7a_graph_validation.py`
- `7c_hypotheses.py`
- `7c_baseline_combined_medoid.py`
- `7c_baseline_single.py`
- `7c_baseline_kmeans.py`

Archived files:

- `7c_baseline_medoid.py`: older L1-medoid baseline entrypoint superseded by
  the clean combined-medoid and single-feature baselines.
- `7c_contrastive_hc.py`: experimental contrastive H_c path with no active
  production caller.

These files are kept for historical reference only. Do not add new pipeline
dependencies on files in this archive.
