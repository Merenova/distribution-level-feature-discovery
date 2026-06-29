#!/usr/bin/env python3
"""
Generate prettified LaTeX tables (booktabs + multirow + multicolumn) from:
  results/plots/method_comparison_mass_within_config_corr.csv

Tables are produced per (metric_name, sign_type, topB) with rows beta×gamma and
columns grouped by method, each showing Spearman/Pearson P50.

Example output: method_comparison_mass_within_config_corr_tables.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


METHOD_ORDER = ["steering", "single", "kmeans"]
SIGN_ORDER = ["full", "negative", "positive"]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLOTS_DIR = REPO_ROOT / "AmbigQA_Qwen3-8B" / "results" / "plots"


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{float(x):.3f}"


def _latex_escape(s: str) -> str:
    return (
        str(s)
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


def _metric_pretty(metric_name: str) -> str:
    mapping = {
        "delta_log_mass": r"$\Delta \log \mathrm{mass}$",
        "cluster_relative_diff": r"cluster\_relative\_diff",
        "cluster_win_rate": r"cluster\_win\_rate",
    }
    return mapping.get(metric_name, _latex_escape(metric_name))


def build_table_block(
    df: pd.DataFrame,
    metric_name: str,
    sign_type: str,
    topB: int,
    betas: List[float],
    gammas: List[float],
    min_n: int,
    caption_prefix: str,
    label_prefix: str,
) -> str:
    sub = df[
        (df["metric_name"] == metric_name)
        & (df["sign_type"] == sign_type)
        & (df["topB"] == topB)
    ].copy()
    if sub.empty:
        return ""

    # Keep only relevant columns
    need_cols = [
        "method",
        "beta",
        "gamma",
        "spearman_p50",
        "pearson_p50",
        "spearman_n",
        "pearson_n",
    ]
    sub = sub[need_cols]

    # Wide pivot: index=(beta,gamma), columns=method, values=stat
    piv_s = sub.pivot_table(index=["beta", "gamma"], columns="method", values="spearman_p50")
    piv_p = sub.pivot_table(index=["beta", "gamma"], columns="method", values="pearson_p50")
    piv_n = sub.pivot_table(index=["beta", "gamma"], columns="method", values="spearman_n")

    # Ensure method columns exist in order
    for m in METHOD_ORDER:
        if m not in piv_s.columns:
            piv_s[m] = np.nan
            piv_p[m] = np.nan
            piv_n[m] = np.nan
    piv_s = piv_s[METHOD_ORDER]
    piv_p = piv_p[METHOD_ORDER]
    piv_n = piv_n[METHOD_ORDER]

    lines: List[str] = []
    metric_tex = _metric_pretty(metric_name)

    caption = (
        f"{caption_prefix} {metric_tex} correlations vs $\\epsilon$ "
        f"(median across cluster series). sign={_latex_escape(sign_type)}, B={topB}. "
        f"Cells with n<{min_n} are shown but may be noisy."
    )
    label = f"{label_prefix}:{metric_name}:{sign_type}:B{topB}".replace("_", "-")

    # Table header
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    # 2 row keys + 3 methods * 2 stats
    colspec = "ll" + "cc" * len(METHOD_ORDER)
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")
    header_methods = " & ".join([rf"\multicolumn{{2}}{{c}}{{{m}}}" for m in METHOD_ORDER])
    lines.append(rf"\multirow{{2}}{{*}}{{$\beta$}} & \multirow{{2}}{{*}}{{$\gamma$}} & {header_methods} \\")
    cmid = []
    start = 3
    for _m in METHOD_ORDER:
        cmid.append(rf"\cmidrule(lr){{{start}-{start+1}}}")
        start += 2
    lines.append(" ".join(cmid))
    subhdr = " & ".join(["Spearman" + " & " + "Pearson"] * len(METHOD_ORDER))
    lines.append(rf" &  & {subhdr} \\")
    lines.append(r"\midrule")

    # Body with multirow beta groups
    for b in betas:
        rows_for_b = []
        for g in gammas:
            idx = (b, g)
            if idx not in piv_s.index:
                continue
            row_cells = []
            for m in METHOD_ORDER:
                s = piv_s.loc[idx, m]
                p = piv_p.loc[idx, m]
                n = piv_n.loc[idx, m]
                # Optionally mark low-n cells with a dagger
                mark = ""
                try:
                    if np.isfinite(n) and float(n) < min_n:
                        mark = r"$^\dagger$"
                except Exception:
                    pass
                row_cells.append(_fmt(s) + mark)
                row_cells.append(_fmt(p) + mark)
            rows_for_b.append((g, row_cells))

        if not rows_for_b:
            continue

        lines.append(rf"\multirow{{{len(rows_for_b)}}}{{*}}{{{_fmt(b)}}} & {_fmt(rows_for_b[0][0])} & " + " & ".join(rows_for_b[0][1]) + r" \\")
        for g, cells in rows_for_b[1:]:
            lines.append(rf" & {_fmt(g)} & " + " & ".join(cells) + r" \\")
        lines.append(r"\addlinespace")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default=str(DEFAULT_PLOTS_DIR / "method_comparison_mass_within_config_corr.csv"),
        help="Path to method_comparison_mass_within_config_corr.csv",
    )
    ap.add_argument(
        "--output",
        default=str(DEFAULT_PLOTS_DIR / "method_comparison_mass_within_config_corr_tables.tex"),
        help="Output .tex path",
    )
    ap.add_argument(
        "--metrics",
        default="delta_log_mass,cluster_win_rate",
        help="Comma-separated metric_name values to include (e.g., delta_log_mass,cluster_win_rate,cluster_relative_diff)",
    )
    ap.add_argument("--min-n", type=int, default=100, help="Mark cells with n < min-n using a dagger.")
    ap.add_argument("--caption-prefix", default="Within-config mass-pooled", help="Caption prefix text.")
    ap.add_argument("--label-prefix", default="tab:mass-within-config", help="LaTeX label prefix.")
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp)
    # Normalize ordering/typing
    df["topB"] = pd.to_numeric(df["topB"], errors="coerce").astype(int)
    df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    df["gamma"] = pd.to_numeric(df["gamma"], errors="coerce")

    # Use only known methods (still show in fixed order)
    df = df[df["method"].isin(METHOD_ORDER)].copy()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    betas = sorted(df["beta"].unique().tolist())
    gammas = sorted(df["gamma"].unique().tolist())

    blocks: List[str] = []
    # Helpful LaTeX preamble note
    blocks.append("% Auto-generated. Requires: \\usepackage{booktabs,multirow}")
    blocks.append("% Each table shows median Spearman/Pearson correlation vs epsilon across (prefix,cluster) series.")
    blocks.append("")

    for metric in metrics:
        for sign in SIGN_ORDER:
            for B in [5, 10]:
                blk = build_table_block(
                    df=df,
                    metric_name=metric,
                    sign_type=sign,
                    topB=B,
                    betas=betas,
                    gammas=gammas,
                    min_n=int(args.min_n),
                    caption_prefix=args.caption_prefix,
                    label_prefix=args.label_prefix,
                )
                if blk:
                    blocks.append(blk)

    out.write_text("\n".join(blocks) + "\n")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()

