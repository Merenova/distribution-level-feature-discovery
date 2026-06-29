#!/usr/bin/env python3
"""
Convert steering comparison CSV to LaTeX longtable format.

Usage:
    python csv_to_latex_steering.py input.csv output.tex [--model-name "Model Name"]
"""

import argparse
import pandas as pd
from pathlib import Path


def csv_to_latex(input_csv: str, output_tex: str, model_name: str = "Model"):
    """Convert steering CSV to LaTeX longtable."""
    
    df = pd.read_csv(input_csv)
    
    # Map method names
    method_map = {
        'combined_medoid': 'RD',
        'default': 'KM-Sem',
        'single': 'Single'
    }
    
    # Map h_c_selection to Sign
    sign_map = {
        'full': 'Full',
        'positive': 'Positive',
        'negative': 'Negative'
    }
    
    # Sort by method, sign, top_B, beta, gamma
    method_order = ['combined_medoid', 'default', 'single']
    sign_order = ['full', 'positive', 'negative']
    
    df['method_order'] = df['method'].map({m: i for i, m in enumerate(method_order)})
    df['sign_order'] = df['h_c_selection'].map({s: i for i, s in enumerate(sign_order)})
    df = df.sort_values(['method_order', 'sign_order', 'top_B', 'beta', 'gamma'])
    
    # Build LaTeX content
    lines = []
    
    # Header
    lines.append(r"\begin{longtable}{lll cc | cc}")
    lines.append(r"\caption{Complete Steering Results: Method $\times$ Sign $\times$ Beam $\times$ $\beta$ $\times$ $\gamma$ (" + model_name + r")} \label{tab:full_steering_" + model_name.lower().replace("-", "_").replace(" ", "_") + r"} \\")
    lines.append(r"\toprule")
    lines.append(r"Method & Sign & B & $\beta$ & $\gamma$ & $\rho_s$ & $\rho$ \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"Method & Sign & B & $\beta$ & $\gamma$ & $\rho_s$ & $\rho$ \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{r}{Continued on next page} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\endfoot")
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")
    
    # Track previous values for hierarchical display
    prev_method = None
    prev_sign = None
    prev_B = None
    prev_beta = None
    
    for idx, row in df.iterrows():
        method = row['method']
        sign = row['h_c_selection']
        B = int(row['top_B'])
        beta = row['beta']
        gamma = row['gamma']
        
        # Get metrics
        rho_s = row['centered_logit_spearman_mean']
        rho = row['centered_logit_corr_mean']
        
        # Build row
        parts = []
        
        # Method column (with \midrule between methods)
        if method != prev_method:
            if prev_method is not None:
                lines.append(r"\midrule")
            parts.append(r"\textbf{" + method_map.get(method, method) + "}")
            prev_sign = None
            prev_B = None
            prev_beta = None
        else:
            parts.append("")
        
        # Sign column (with \cmidrule between signs)
        if sign != prev_sign:
            if prev_sign is not None and method == prev_method:
                lines.append(r"\cmidrule(lr){2-7}")
            parts.append(sign_map.get(sign, sign))
            prev_B = None
            prev_beta = None
        else:
            parts.append("")
        
        # B column (with \cmidrule between B values within same sign)
        if B != prev_B:
            if prev_B is not None and sign == prev_sign and method == prev_method:
                lines.append(r"\cmidrule(lr){3-7}")
            parts.append(str(B))
            prev_beta = None
        else:
            parts.append("")
        
        # Beta column
        if beta != prev_beta:
            parts.append(f"{beta:.2f}")
        else:
            parts.append("")
        
        # Gamma column (always show)
        parts.append(f"{gamma:.1f}")
        
        # Metrics with formatting
        parts.append(r"\cellcolor{gray!15}" + f"{rho_s:.2f}")
        parts.append(r"\cellcolor{gray!15}" + f"{rho:.2f}")
        
        lines.append(" & ".join(parts) + r" \\")
        
        # Update previous values
        prev_method = method
        prev_sign = sign
        prev_B = B
        prev_beta = beta
    
    # Footer
    lines.append(r"\label{tab:" + model_name.lower().replace("-", "_").replace(" ", "_") + r"-steering-full}")
    lines.append(r"\end{longtable}")
    
    # Write output
    with open(output_tex, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"LaTeX table written to: {output_tex}")
    print(f"Total rows: {len(df)}")


def main():
    parser = argparse.ArgumentParser(description="Convert steering CSV to LaTeX longtable")
    parser.add_argument("input_csv", help="Input CSV file")
    parser.add_argument("output_tex", help="Output LaTeX file")
    parser.add_argument("--model-name", default="Model", help="Model name for caption")
    
    args = parser.parse_args()
    csv_to_latex(args.input_csv, args.output_tex, args.model_name)


if __name__ == "__main__":
    main()
