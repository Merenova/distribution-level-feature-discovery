#!/usr/bin/env python3
"""Convert K-sweep cache JSON to LaTeX table format.

Generates a table similar to RD_table.tex but indexed by K instead of beta.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_cache(cache_path: Path) -> Dict[Tuple[int, float], Dict[str, Dict[str, float]]]:
    """Load cache and convert string keys to (K, gamma) tuples."""
    data = json.loads(cache_path.read_text())
    results = {}
    for key, value in data.items():
        # Parse key like "K=2,gamma=0.1"
        parts = key.split(",")
        k = int(parts[0].split("=")[1])
        gamma = float(parts[1].split("=")[1])
        results[(k, gamma)] = value
    return results


def generate_latex_table(
    results: Dict[Tuple[int, float], Dict[str, Dict[str, float]]],
    k_values: List[int],
    gamma_values: List[float],
    output_path: Path,
):
    """Generate LaTeX table from results."""
    
    methods = [
        ("rd", "RD"),
        ("kmeans_semantic", "KM-S"),
        ("kmeans_attribution", "KM-A"),
    ]
    
    # Build LaTeX content
    lines = []
    lines.append(r"% requires in preamble:")
    lines.append(r"% \usepackage{booktabs}")
    lines.append(r"% \usepackage{multirow}")
    lines.append("")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\renewcommand{\arraystretch}{0.90}")
    lines.append(r"\setlength{\tabcolsep}{3.2pt}")
    lines.append("")
    
    # Header row
    n_gamma = len(gamma_values)
    n_cols = 2 + n_gamma * 4  # Method, K, then 4 metrics per gamma
    
    col_spec = "l c " + " ".join(["c c c c"] * n_gamma)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    
    # Gamma header
    gamma_cols = 2 + n_gamma * 4
    lines.append(r"\multicolumn{2}{c}{} & \multicolumn{" + str(n_gamma * 4) + r"}{c}{$\gamma$} \\")
    lines.append(r"\cmidrule(lr){3-" + str(gamma_cols) + "}")
    
    # Sub-header for each gamma
    gamma_header = "Method & $K$ "
    for g in gamma_values:
        gamma_header += f"& \\multicolumn{{4}}{{c}}{{{g:.2f}}} "
    gamma_header += r"\\"
    lines.append(gamma_header)
    
    # Cmidrule for each gamma group
    cmidrules = []
    col_start = 3
    for i, g in enumerate(gamma_values):
        col_end = col_start + 3
        cmidrules.append(f"\\cmidrule(lr){{{col_start}-{col_end}}}")
        col_start = col_end + 1
    lines.append("".join(cmidrules))
    
    # Metric names header
    metric_header = "  &   "
    for g in gamma_values:
        metric_header += r"& $H$ & $D_e$ & $D_a$ & $D_{\gamma}$ "
    metric_header += r"\\"
    lines.append(metric_header)
    lines.append(r"\midrule")
    lines.append("")
    
    # Data rows
    for method_key, method_label in methods:
        n_k = len(k_values)
        lines.append(f"\\multirow{{{n_k}}}{{*}}{{\\textbf{{{method_label}}}}}")
        
        for i, k in enumerate(k_values):
            if i == 0:
                row = f" & {k}"
            else:
                row = f"& {k}"
            
            for gamma in gamma_values:
                key = (k, gamma)
                if key in results and method_key in results[key]:
                    data = results[key][method_key]
                    H = data.get("H", 0)
                    D_e = data.get("D_e", 0)
                    D_a = data.get("D_a", 0)
                    D_gamma = gamma * D_e + (1 - gamma) * D_a
                    row += f" & {H:.2f} & {D_e:.2f} & {D_a:.2f} & {D_gamma:.2f}"
                else:
                    row += " & -- & -- & -- & --"
            
            row += r" \\"
            lines.append(row)
        
        lines.append(r"\midrule")
        lines.append("")
    
    # Remove last midrule, replace with bottomrule
    while lines[-1] == "" or lines[-1] == r"\midrule":
        lines.pop()
    lines.append(r"\bottomrule")
    
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"\caption{RD sweep evaluation by $K$ (means over clozes). $D_{\gamma}=\gamma D_e + (1-\gamma)D_a$. \textbf{KM-S}: K-means semantic; \textbf{KM-A}: K-means attribution.}")
    lines.append(r"\label{tab:rd-sweep-by-k}")
    lines.append(r"\end{table*}")
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Saved LaTeX table: {output_path}")
    
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Convert cache JSON to LaTeX table")
    ap.add_argument("--cache", type=Path, required=True, help="Path to cache.json")
    ap.add_argument("--output", type=Path, required=True, help="Output .tex file")
    ap.add_argument("--K-values", type=str, default="2,3,4,5,6,7,8,9",
                    help="Comma-separated K values to include")
    ap.add_argument("--gamma-values", type=str, default="0.1,0.3,0.5,0.7,0.9",
                    help="Comma-separated gamma values to include")
    args = ap.parse_args()
    
    k_values = [int(k.strip()) for k in args.K_values.split(",")]
    gamma_values = [float(g.strip()) for g in args.gamma_values.split(",")]
    
    print(f"Loading cache: {args.cache}")
    results = load_cache(args.cache)
    print(f"Found {len(results)} entries")
    
    latex = generate_latex_table(results, k_values, gamma_values, args.output)
    
    # Also print to console
    print("\n" + "=" * 80)
    print("GENERATED LATEX TABLE")
    print("=" * 80)
    print(latex)


if __name__ == "__main__":
    main()

