#!/usr/bin/env bash
# Regenerate every LaTeX table under ${PAPER_DIR:-paper}/tables.
#
# Usage:
#   scripts/tables/run_all_paper_tables.sh <STAGE_RESULTS_DIR_8B> <STAGE_RESULTS_DIR_4B> <OUT_TEX_DIR>
#
# Example:
#   scripts/tables/run_all_paper_tables.sh AmbigQA_Qwen3-8B AmbigQA_Qwen3-4B /tmp/paper_tables
#
# Builder map (see docs/superpowers/plans/notes/2026-05-23-cleanup-inventory.md Table E):
#   csv_to_latex_table.py             -> steering_table.tex  (paper §5 main multirow)
#   csv_to_latex_steering.py          -> steering_full.tex, steering_full_qwen4.tex  (App. longtable)
#   generate_rd_table_with_dgamma.py  -> RD_table.tex  (paper §4 — hard-codes AmbigQA_Qwen3-8B input)
#   mass_within_config_corr_to_tex.py -> method_comparison_mass_within_config_corr_tables.tex
#
# Paper tables that have no in-tree builder (the user must regenerate by hand or
# from the AmbigQA_Qwen3-4B/ outputs by editing the relevant script):
#   RD_table_qwen4.tex   (Qwen3-4B replication of RD_table)
#   RD_table_by_K.tex    (no builder found under scripts/tables/)

set -euo pipefail

STAGE_8B="${1:?stage results dir Qwen3-8B (e.g. AmbigQA_Qwen3-8B)}"
STAGE_4B="${2:?stage results dir Qwen3-4B (e.g. AmbigQA_Qwen3-4B)}"
OUT_DIR="${3:?output tex dir}"
mkdir -p "$OUT_DIR"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# --- §5 main steering table (multirow form) -----------------------------------
# csv_to_latex_table.py reads a CSV with steering metrics and writes a multirow
# table. Input CSV path is the standard plots location under the stage dir.
STEER_CSV_8B="${STAGE_8B}/results/plots/steer_results.csv"
if [[ -f "$STEER_CSV_8B" ]]; then
  uv run python scripts/tables/csv_to_latex_table.py \
      "$STEER_CSV_8B" \
      --output "$OUT_DIR/steering_table.tex" \
      --breakdown by_config
else
  echo "skip steering_table.tex: $STEER_CSV_8B not found"
fi

# --- App. longtable steering tables (8B and 4B) ------------------------------
# csv_to_latex_steering.py takes positional input_csv + output_tex + --model-name.
STEER_COMP_8B="${STAGE_8B}/results/plots/steering_comparison.csv"
STEER_COMP_4B="${STAGE_4B}/results/plots/steering_comparison.csv"
if [[ -f "$STEER_COMP_8B" ]]; then
  uv run python scripts/tables/csv_to_latex_steering.py \
      "$STEER_COMP_8B" "$OUT_DIR/steering_full.tex" \
      --model-name "Qwen3-8B"
else
  echo "skip steering_full.tex: $STEER_COMP_8B not found"
fi
if [[ -f "$STEER_COMP_4B" ]]; then
  uv run python scripts/tables/csv_to_latex_steering.py \
      "$STEER_COMP_4B" "$OUT_DIR/steering_full_qwen4.tex" \
      --model-name "Qwen3-4B"
else
  echo "skip steering_full_qwen4.tex: $STEER_COMP_4B not found"
fi

# --- §4 RD table with D_gamma column -----------------------------------------
# generate_rd_table_with_dgamma.py has hard-coded input path
#   ${ROOT_DIR}/AmbigQA_Qwen3-8B/results/plots/rd_sweep_table.csv
# and writes the .tex next to it. We invoke it then copy the result into $OUT_DIR.
# For RD_table_qwen4.tex (4B), the script needs editing OR a temporary symlink — TODO.
RD_8B_CSV="${ROOT_DIR}/AmbigQA_Qwen3-8B/results/plots/rd_sweep_table.csv"
RD_8B_TEX="${ROOT_DIR}/AmbigQA_Qwen3-8B/results/plots/rd_sweep_table.tex"
if [[ -f "$RD_8B_CSV" ]]; then
  uv run python scripts/tables/generate_rd_table_with_dgamma.py
  cp "$RD_8B_TEX" "$OUT_DIR/RD_table.tex"
else
  echo "skip RD_table.tex: $RD_8B_CSV not found (this script hard-codes the 8B path)"
fi
# TODO: produce RD_table_qwen4.tex — either modify generate_rd_table_with_dgamma.py
# to accept a --results-dir flag, or temporarily symlink AmbigQA_Qwen3-4B's
# plots/rd_sweep_table.csv into the 8B path and re-run.

# --- Extended mass-within-config correlation tables --------------------------
# mass_within_config_corr_to_tex.py has reasonable defaults (input + output under
# AmbigQA_Qwen3-8B/results/plots/). We invoke with explicit paths and copy
# the output into $OUT_DIR for completeness.
MASS_CSV="${ROOT_DIR}/AmbigQA_Qwen3-8B/results/plots/method_comparison_mass_within_config_corr.csv"
MASS_OUT_TEX="$OUT_DIR/method_comparison_mass_within_config_corr_tables.tex"
if [[ -f "$MASS_CSV" ]]; then
  uv run python scripts/tables/mass_within_config_corr_to_tex.py \
      --input "$MASS_CSV" \
      --output "$MASS_OUT_TEX"
else
  echo "skip method_comparison_mass_within_config_corr_tables.tex: $MASS_CSV not found"
fi

echo "Tables written to $OUT_DIR"
