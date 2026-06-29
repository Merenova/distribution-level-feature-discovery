#!/usr/bin/env python
"""Generate LaTeX table with D_gamma from existing CSV data."""

import csv
from pathlib import Path
from typing import Dict


def compute_d_gamma(gamma: float, d_e: float, d_a: float) -> float:
    """Compute combined distortion: D_gamma = gamma * D_e + (1 - gamma) * D_a"""
    return gamma * d_e + (1 - gamma) * d_a


def main():
    csv_path = Path(__file__).parent.parent / "AmbigQA_Qwen3-8B/results/plots/rd_sweep_table.csv"
    tex_path = csv_path.with_suffix(".tex")
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Build lookup: method -> beta -> gamma -> data
    lookup: Dict[str, Dict[float, Dict[float, dict]]] = {}
    for r in rows:
        m = r["method"]
        b = float(r["beta"])
        g = float(r["gamma"])
        lookup.setdefault(m, {}).setdefault(b, {})[g] = {
            "H": float(r["H"]),
            "D_e": float(r["D_e"]),
            "D_a": float(r["D_a"]),
        }
    
    betas = sorted({float(r["beta"]) for r in rows})
    gamma_values = sorted({float(r["gamma"]) for r in rows})
    method_order = ["rd", "kmeans_semantic", "kmeans_attribution"]
    method_label = {"rd": "RD", "kmeans_semantic": "KM-S", "kmeans_attribution": "KM-A"}
    
    def fmt(x: float) -> str:
        return f"{x:.2f}"
    
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
        header2_parts.append(f"\\multicolumn{{4}}{{c}}{{{fmt(g)}}}")
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
        
        for i, beta in enumerate(betas):
            row_vals = []
            if i == 0:
                row_vals.append(f"\\multirow{{6}}{{*}}{{\\textbf{{{method_label[method]}}}}}")
            else:
                row_vals.append("")
            row_vals.append(fmt(beta))
            
            for g in gamma_values:
                entry = method_rows.get(beta, {}).get(g)
                if entry:
                    d_e = entry["D_e"]
                    d_a = entry["D_a"]
                    d_gamma = compute_d_gamma(g, d_e, d_a)
                    row_vals.append(fmt(entry["H"]))
                    row_vals.append(fmt(d_e))
                    row_vals.append(fmt(d_a))
                    row_vals.append(fmt(d_gamma))
                else:
                    row_vals.extend(["--", "--", "--", "--"])
            lines.append(" & ".join(row_vals) + " \\\\")
        lines.append("\\midrule")
        lines.append("")
    
    # Remove the last \midrule and add bottomrule
    while lines and lines[-1].strip() in ("", "\\midrule"):
        lines.pop()
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("")
    lines.append("\\caption{RD sweep evaluation (means over clozes). We additionally report the combined "
                 "distortion $D_{\\gamma}=\\gamma D_e + (1-\\gamma)D_a$. "
                 "\\textbf{KM-S}: K-means semantic; \\textbf{KM-A}: K-means attribution.}")
    lines.append("\\label{tab:rd-sweep}")
    lines.append("\\end{table*}")
    
    tex_content = "\n".join(lines) + "\n"
    tex_path.write_text(tex_content)
    print(f"Wrote {tex_path}")
    print()
    print("Generated table preview:")
    print("=" * 80)
    print(tex_content)


if __name__ == "__main__":
    main()
