#!/usr/bin/env -S uv run python
"""Create a table for beta/gamma sweeps with D_e, D_a, H per method."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm

try:
    import ijson  # type: ignore
except Exception:  # pragma: no cover
    ijson = None


def _accumulate(agg: Dict[Tuple[float, float], Dict[str, float]], beta: float, gamma: float, h: float, de: float, da: float):
    key = (beta, gamma)
    if key not in agg:
        agg[key] = {"H": 0.0, "D_e": 0.0, "D_a": 0.0, "n": 0.0}
    agg[key]["H"] += h
    agg[key]["D_e"] += de
    agg[key]["D_a"] += da
    agg[key]["n"] += 1.0


def _finalize(agg: Dict[Tuple[float, float], Dict[str, float]]) -> Dict[Tuple[float, float], Dict[str, float]]:
    out = {}
    for (beta, gamma), vals in agg.items():
        n = max(vals["n"], 1.0)
        out[(beta, gamma)] = {
            "H": vals["H"] / n,
            "D_e": vals["D_e"] / n,
            "D_a": vals["D_a"] / n,
            "n": n,
        }
    return out


def load_rd_means(sweep_dir: Path, max_clozes: int | None) -> Dict[Tuple[float, float], Dict[str, float]]:
    if ijson is None:
        raise SystemExit("ijson is required: uv pip install ijson")
    agg: Dict[Tuple[float, float], Dict[str, float]] = {}
    for idx, path in enumerate(tqdm(sorted(sweep_dir.glob("*_sweep_results.json")), desc="RD sweeps")):
        if max_clozes is not None and idx >= max_clozes:
            break
        with path.open("rb") as f:
            for entry in ijson.items(f, "grid.item"):
                try:
                    beta = float(entry.get("beta"))
                    gamma = float(entry.get("gamma"))
                    h = float(entry.get("H"))
                    de = float(entry.get("D_e"))
                    da = float(entry.get("D_a"))
                except Exception:
                    continue
                _accumulate(agg, beta, gamma, h, de, da)
    return _finalize(agg)


def load_kmeans_means(cache_path: Path, max_clozes: int | None) -> Dict[str, Dict[Tuple[float, float], Dict[str, float]]]:
    data = json.loads(cache_path.read_text())
    methods = {"kmeans_semantic": {}, "kmeans_attribution": {}}
    aggs = {"kmeans_semantic": {}, "kmeans_attribution": {}}
    for idx, entry in enumerate(data.get("prefixes", {}).values()):
        if max_clozes is not None and idx >= max_clozes:
            break
        for method in methods.keys():
            for r in entry.get(method, []):
                try:
                    beta = float(r.get("beta"))
                    gamma = float(r.get("gamma"))
                    h = float(r.get("H"))
                    de = float(r.get("D_e"))
                    da = float(r.get("D_a"))
                except Exception:
                    continue
                _accumulate(aggs[method], beta, gamma, h, de, da)
    return {m: _finalize(aggs[m]) for m in aggs}


def _compute_d_gamma(gamma: float, d_e: float, d_a: float) -> float:
    """Compute combined distortion: D_gamma = gamma * D_e + (1 - gamma) * D_a"""
    return gamma * d_e + (1 - gamma) * d_a


def _load_dims_from_sweep(sweep_dir: Path) -> Tuple[float | None, float | None]:
    if ijson is None:
        return None, None
    for path in sorted(sweep_dir.glob("*_sweep_results.json")):
        with path.open("rb") as f:
            d_e = None
            d_a = None
            for prefix, event, value in ijson.parse(f):
                if prefix == "sweep_config.d_e" and event == "number":
                    d_e = float(value)
                if prefix == "sweep_config.d_a" and event == "number":
                    d_a = float(value)
                if d_e is not None and d_a is not None:
                    return d_e, d_a
    return None, None


def _write_tex_grouped(
    rows: List[Dict[str, str]],
    tex_path: Path,
    d_e_dim: float | None,
    d_a_dim: float | None,
    normalized: bool,
) -> None:
    # Layout: rows = method, subrows = beta; columns = gamma with subcolumns (H, D_e, D_a, D_gamma)
    betas = sorted({float(r["beta"]) for r in rows})
    gamma_values = sorted({float(r["gamma"]) for r in rows})
    method_order = ["rd", "kmeans_semantic", "kmeans_attribution"]
    method_label = {"rd": "RD", "kmeans_semantic": "KM-S", "kmeans_attribution": "KM-A"}

    def fmt(x: str) -> str:
        return f"{float(x):.2f}"

    # Build a lookup: method -> beta -> gamma -> row
    lookup: Dict[str, Dict[float, Dict[float, Dict[str, str]]]] = {}
    for r in rows:
        m = r["method"]
        b = float(r["beta"])
        g = float(r["gamma"])
        lookup.setdefault(m, {}).setdefault(b, {})[g] = r

    lines = []
    lines.append("% requires in preamble:")
    lines.append("% \\usepackage{booktabs}")
    lines.append("% \\usepackage{multirow}")
    lines.append("")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\renewcommand{\\arraystretch}{0.90}")
    lines.append("\\setlength{\\tabcolsep}{3.2pt}")
    lines.append("")

    # Column spec: Method + Beta + (4 columns per gamma: H, D_e, D_a, D_gamma)
    n_gamma = len(gamma_values)
    col_spec = "l c " + " ".join(["c c c c"] * n_gamma)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    
    # Header row 1: gamma label spanning all columns
    n_data_cols = n_gamma * 4
    lines.append(f"\\multicolumn{{2}}{{c}}{{}} & \\multicolumn{{{n_data_cols}}}{{c}}{{$\\gamma$}} \\\\")
    
    # cmidrule for gamma header
    lines.append(f"\\cmidrule(lr){{3-{2 + n_data_cols}}}")
    
    # Header row 2: Method, beta, then gamma values
    header2_parts = ["Method", "$\\beta$"]
    for g in gamma_values:
        header2_parts.append(f"\\multicolumn{{4}}{{c}}{{{fmt(str(g))}}}")
    lines.append(" & ".join(header2_parts) + " \\\\")
    
    # cmidrules for each gamma group
    cmidrules = []
    for i, g in enumerate(gamma_values):
        start = 3 + i * 4
        end = start + 3
        cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
    lines.append("".join(cmidrules))
    
    # Header row 3: H, D_e, D_a, D_gamma for each gamma
    header3_parts = [" ", " "]
    for _ in gamma_values:
        header3_parts.extend(["$H$", "$D_e$", "$D_a$", "$D_{\\gamma}$"])
    lines.append(" & ".join(header3_parts) + " \\\\")
    lines.append("\\midrule")
    lines.append("")

    for method in method_order:
        method_rows = lookup.get(method, {})
        if not method_rows:
            continue
        lines.append(f"\\multirow{{{len(betas)}}}{{*}}{{\\textbf{{{method_label[method]}}}}} & {fmt(str(betas[0]))}")
        
        for i, beta in enumerate(betas):
            row_vals = []
            if i > 0:
                row_vals.append(f"& {fmt(str(beta))}")
            for g in gamma_values:
                entry = method_rows.get(beta, {}).get(g)
                if entry:
                    d_e = float(entry["D_e"])
                    d_a = float(entry["D_a"])
                    if normalized and d_e_dim and d_a_dim:
                        d_e = d_e / (d_e_dim ** 0.5)
                        d_a = d_a / d_a_dim
                    d_gamma = _compute_d_gamma(g, d_e, d_a)
                    row_vals.append(fmt(entry["H"]))
                    row_vals.append(fmt(str(d_e)))
                    row_vals.append(fmt(str(d_a)))
                    row_vals.append(fmt(str(d_gamma)))
                else:
                    row_vals.extend(["--", "--", "--", "--"])
            if i == 0:
                lines.append(" & ".join(row_vals) + " \\\\")
            else:
                lines.append(" & ".join(row_vals) + " \\\\")
        lines.append("\\midrule")
        lines.append("")

    # Remove the last \midrule and add bottomrule
    while lines and lines[-1].strip() in ("", "\\midrule"):
        lines.pop()
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")
    caption = "RD sweep evaluation (means over clozes). "
    if normalized:
        caption += "Distortions are dimension-normalized ($D_e/\\sqrt{d_e}$, $D_a/d_a$). "
    caption += "We additionally report $D_{\\gamma}=\\gamma D_e + (1-\\gamma)D_a$. "
    caption += "\\textbf{KM-S}: K-means semantic; \\textbf{KM-A}: K-means attribution."
    lines.append(f"\\caption{{{caption}}}")
    lines.append("\\label{tab:rd-sweep}")
    lines.append("\\end{table*}")
    tex_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True, help="Results dir containing 5_clustering")
    ap.add_argument("--baseline-cache", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--max-clozes", type=int, default=300, help="Limit number of prefixes/clozes (default: 300)")
    ap.add_argument("--gamma-values", type=str, default=None, help="Comma-separated gamma values to keep")
    args = ap.parse_args()

    sweep_dir = args.results_dir / "5_clustering"
    rd_means = load_rd_means(sweep_dir, args.max_clozes)
    kmeans_means = load_kmeans_means(args.baseline_cache, args.max_clozes)
    d_e_dim, d_a_dim = _load_dims_from_sweep(sweep_dir)

    gamma_filter = None
    if args.gamma_values:
        gamma_filter = {float(x) for x in args.gamma_values.split(",") if x.strip() != ""}

    rows: List[Dict[str, object]] = []
    all_keys = sorted(rd_means.keys())
    for beta, gamma in all_keys:
        if gamma_filter is not None and gamma not in gamma_filter:
            continue
        rd = rd_means.get((beta, gamma))
        if rd:
            d_gamma = _compute_d_gamma(gamma, rd["D_e"], rd["D_a"])
            rows.append(
                {
                    "method": "rd",
                    "beta": beta,
                    "gamma": gamma,
                    "H": rd["H"],
                    "D_e": rd["D_e"],
                    "D_a": rd["D_a"],
                    "D_gamma": d_gamma,
                    "n": rd["n"],
                }
            )
        for method in ("kmeans_semantic", "kmeans_attribution"):
            km = kmeans_means.get(method, {}).get((beta, gamma))
            if km:
                d_gamma = _compute_d_gamma(gamma, km["D_e"], km["D_a"])
                rows.append(
                    {
                        "method": method,
                        "beta": beta,
                        "gamma": gamma,
                        "H": km["H"],
                        "D_e": km["D_e"],
                        "D_a": km["D_a"],
                        "D_gamma": d_gamma,
                        "n": km["n"],
                    }
                )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "beta", "gamma", "H", "D_e", "D_a", "D_gamma", "n"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out_csv}")

    # Also write grouped LaTeX tables (raw + normalized) next to CSV.
    tex_path_raw = args.out_csv.with_suffix(".tex")
    _write_tex_grouped(rows, tex_path_raw, d_e_dim, d_a_dim, normalized=False)
    print(f"Wrote {tex_path_raw}")
    tex_path_norm = args.out_csv.with_name(args.out_csv.stem + "_norm.tex")
    _write_tex_grouped(rows, tex_path_norm, d_e_dim, d_a_dim, normalized=True)
    print(f"Wrote {tex_path_norm}")


if __name__ == "__main__":
    main()
