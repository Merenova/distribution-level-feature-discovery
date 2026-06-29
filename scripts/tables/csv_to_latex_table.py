#!/usr/bin/env python3
"""
Convert steer_results.csv to a LaTeX multirow table similar to the example format.
"""

import pandas as pd
import argparse
from pathlib import Path


def parse_config(config_str: str) -> tuple[float, float] | None:
    """Parse config string like 'beta0.75_gamma0.3' into (beta, gamma) tuple."""
    if config_str == "all":
        return None
    try:
        parts = config_str.split("_")
        beta = float(parts[0].replace("beta", ""))
        gamma = float(parts[1].replace("gamma", ""))
        return beta, gamma
    except (IndexError, ValueError):
        return None


def format_value(val: float, precision: int = 2) -> str:
    """Format a float value for LaTeX display."""
    if pd.isna(val):
        return "-"
    return f"{val:.{precision}f}"


# Shorter metric names for headers
METRIC_SHORT_NAMES = {
    "centered_logit_spearman_mean": r"$\rho_s$",
    "centered_logit_corr_mean": r"$\rho$",
}

# Metrics that are correlation (for gray shading)
CORRELATION_METRICS = {
    "centered_logit_spearman_mean",
    "centered_logit_corr_mean",
}

# Method display names
METHOD_NAMES = {
    "combined_medoid": "Combined",
    "default": "Default",
    "single": "Single",
}

# Paper-facing Stage 7 metrics.
DEFAULT_METRICS = [
    "centered_logit_spearman_mean",
    "centered_logit_corr_mean",
]


def generate_transposed_table(
    df_filtered, methods, betas, gammas, metrics, precision, output_path
) -> str:
    """
    Generate transposed table: methods with metric sub-rows, beta/gamma as columns.
    
    Structure:
    - Rows: Method (multirow) -> Metric sub-rows
    - Columns: Beta (header) -> Gamma sub-columns
    """
    num_gammas = len(gammas)
    num_betas = len(betas)
    num_metrics = len(metrics)
    total_data_cols = num_betas * num_gammas

    lines = []
    
    # Preamble
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\renewcommand{\arraystretch}{0.90}")
    lines.append(r"\setlength{\tabcolsep}{3.2pt}")
    lines.append("")
    
    # Column specification: Method | Metric | data columns...
    col_spec = "l l " + " ".join(["c" * num_gammas for _ in betas])
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    
    # Beta header row
    lines.append(r"\multicolumn{2}{c}{} & " + f"\\multicolumn{{{total_data_cols}}}{{c}}{{$\\beta$}} \\\\")
    lines.append(f"\\cmidrule(lr){{3-{total_data_cols + 2}}}")
    
    # Beta values row with gamma sub-headers
    beta_headers = []
    for beta in betas:
        beta_headers.append(f"\\multicolumn{{{num_gammas}}}{{c}}{{{beta:.2f}}}")
    lines.append(r"Method & Metric & " + " & ".join(beta_headers) + r" \\")
    
    # Cmidrules for each beta group
    cmidrules = []
    start_col = 3
    for _ in betas:
        end_col = start_col + num_gammas - 1
        cmidrules.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
        start_col = end_col + 1
    lines.append("".join(cmidrules))
    
    # Gamma values row (sub-header under each beta)
    gamma_headers = []
    for _ in betas:
        for gamma in gammas:
            gamma_headers.append(f"{gamma:.1f}")
    lines.append(r"  & $\gamma$ & " + " & ".join(gamma_headers) + r" \\")
    lines.append(r"\midrule")
    lines.append("")
    
    # Data rows: for each method, iterate through paper metrics
    for method_idx, method in enumerate(methods):
        method_display = METHOD_NAMES.get(method, method)

        for metric_idx, metric in enumerate(metrics):
            row_values = []

            # Get data for this method and metric across all beta/gamma combinations
            for beta in betas:
                for gamma in gammas:
                    mask = (df_filtered["method"] == method) & \
                           (df_filtered["beta"] == beta) & \
                           (df_filtered["gamma"] == gamma)
                    row_data = df_filtered[mask]
                    
                    if len(row_data) > 0:
                        val = row_data[metric].values[0]
                        row_values.append(f"\\cellcolor{{gray!15}}{format_value(val, precision)}")
                    else:
                        row_values.append(r"\cellcolor{gray!15}-")

            # Build row string with cellcolor for metric and data cells (not method cell)
            if metric_idx == 0:
                method_cell = f"\\multirow{{{num_metrics}}}{{*}}{{\\textbf{{{method_display}}}}}"
            else:
                method_cell = ""

            metric_name = METRIC_SHORT_NAMES.get(metric, metric[:8])
            row_str = f"{method_cell} & \\cellcolor{{gray!15}}{metric_name} & " + " & ".join(row_values) + r" \\"
            lines.append(row_str)

        # Add midrule between methods (except after last)
        if method_idx < len(methods) - 1:
            lines.append(r"\midrule")
            lines.append("")
    
    # Footer
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    
    # Caption
    lines.append(r"\caption{Steering results by $\beta$ and $\gamma$ configuration. " +
                r"Rows show methods with metric sub-rows. " +
                r"Rows report demeaned-logit correlation metrics. " +
                r"\textbf{Combined}: Combined medoid; \textbf{Default}: Default steering; " +
                r"\textbf{Single}: Single direction.}")
    lines.append(r"\label{tab:steer-results-transposed}")
    lines.append(r"\end{table*}")
    
    latex_str = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(latex_str)
        print(f"LaTeX table saved to: {output_path}")
    
    return latex_str


def generate_by_config_table(
    df_filtered, methods, betas, gammas, metrics, precision, output_path
) -> str:
    """
    Generate standard table: methods with gamma sub-rows, beta as columns with metric sub-columns.
    """
    num_metrics = len(metrics)
    num_betas = len(betas)
    total_cols = num_metrics * num_betas
    
    lines = []
    
    # Preamble
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\renewcommand{\arraystretch}{0.90}")
    lines.append(r"\setlength{\tabcolsep}{3.2pt}")
    lines.append("")
    
    # Column specification
    col_spec = "l c " + " ".join([f"*{{{num_metrics}}}{{c}}" for _ in betas])
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    
    # Beta header row
    beta_cols = total_cols + 2  # +2 for Method and gamma columns
    lines.append(r"\multicolumn{2}{c}{} & " + f"\\multicolumn{{{total_cols}}}{{c}}{{$\\beta$}} \\\\")
    
    # Cmidrule for beta header
    lines.append(f"\\cmidrule(lr){{3-{beta_cols}}}")
    
    # Beta values row
    beta_headers = []
    for beta in betas:
        beta_headers.append(f"\\multicolumn{{{num_metrics}}}{{c}}{{{beta:.2f}}}")
    lines.append(r"Method & $\gamma$ & " + " & ".join(beta_headers) + r" \\")
    
    # Cmidrules for each beta group
    cmidrules = []
    start_col = 3
    for _ in betas:
        end_col = start_col + num_metrics - 1
        cmidrules.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
        start_col = end_col + 1
    lines.append("".join(cmidrules))
    
    # Metric names row
    metric_headers = []
    for _ in betas:
        for m in metrics:
            metric_headers.append(METRIC_SHORT_NAMES.get(m, m[:8]))
    lines.append(r"  & & " + " & ".join(metric_headers) + r" \\")
    lines.append(r"\midrule")
    lines.append("")
    
    # Data rows
    for method_idx, method in enumerate(methods):
        method_display = METHOD_NAMES.get(method, method)
        num_gammas = len(gammas)
        
        for gamma_idx, gamma in enumerate(gammas):
            row_values = []
            
            # Get data for this method and gamma across all betas
            for beta in betas:
                mask = (df_filtered["method"] == method) & \
                       (df_filtered["beta"] == beta) & \
                       (df_filtered["gamma"] == gamma)
                row_data = df_filtered[mask]
                
                if len(row_data) > 0:
                    for m in metrics:
                        val = row_data[m].values[0]
                        row_values.append(format_value(val, precision))
                else:
                    row_values.extend(["-"] * num_metrics)
            
            # Build row string
            if gamma_idx == 0:
                method_cell = f"\\multirow{{{num_gammas}}}{{*}}{{\\textbf{{{method_display}}}}}"
            else:
                method_cell = ""
            
            row_str = f"{method_cell} & {gamma:.1f} & " + " & ".join(row_values) + r" \\"
            lines.append(row_str)
        
        # Add midrule between methods (except after last)
        if method_idx < len(methods) - 1:
            lines.append(r"\midrule")
            lines.append("")
    
    # Footer
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    
    # Caption
    metric_desc = ", ".join([METRIC_SHORT_NAMES.get(m, m) for m in metrics])
    lines.append(r"\caption{Steering results by $\beta$ and $\gamma$ configuration. " +
                f"Metrics: {metric_desc}. " +
                r"\textbf{Combined}: Combined medoid; \textbf{Default}: Default steering; " +
                r"\textbf{Single}: Single direction.}")
    lines.append(r"\label{tab:steer-results}")
    lines.append(r"\end{table*}")
    
    latex_str = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(latex_str)
        print(f"LaTeX table saved to: {output_path}")
    
    return latex_str


def generate_transposed_sweep_table(df, metrics, precision, output_path) -> str:
    """
    Generate transposed table for by_sweep: methods with metric sub-rows, sign/batch as columns.
    
    Structure:
    - Rows: Method (multirow) -> Metric sub-rows
    - Columns: Sign (header) -> Batch sub-columns
    """
    df_sweep = df[df["breakdown"] == "by_sweep"].copy()
    
    # Parse sweep config
    df_sweep["sign"] = df_sweep["config"].apply(lambda x: x.rsplit("_", 1)[0].replace("sign_", ""))
    df_sweep["batch"] = df_sweep["config"].apply(lambda x: x.rsplit("_", 1)[1])
    
    methods = df_sweep["method"].unique().tolist()
    # Order signs as: full, positive, negative
    signs = ["full", "positive", "negative"]
    batches = sorted(df_sweep["batch"].unique())
    
    if metrics is None:
        metrics = DEFAULT_METRICS
    
    num_metrics = len(metrics)
    num_signs = len(signs)
    num_batches = len(batches)
    total_data_cols = num_signs * num_batches
    
    # Sign display names
    sign_display_names = {"full": "Full", "negative": "Neg", "positive": "Pos"}
    
    lines = []
    
    # Preamble
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\renewcommand{\arraystretch}{0.90}")
    lines.append(r"\setlength{\tabcolsep}{3.2pt}")
    lines.append("")
    
    # Column specification: Method | Metric | data columns...
    col_spec = "l l " + " ".join(["c" * num_batches for _ in signs])
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    
    # Sign header row
    lines.append(r"\multicolumn{2}{c}{} & " + f"\\multicolumn{{{total_data_cols}}}{{c}}{{Sign Type}} \\\\")
    lines.append(f"\\cmidrule(lr){{3-{total_data_cols + 2}}}")
    
    # Sign values row with batch sub-headers
    sign_headers = []
    for sign in signs:
        sign_headers.append(f"\\multicolumn{{{num_batches}}}{{c}}{{{sign_display_names.get(sign, sign)}}}")
    lines.append(r"Method & Metric & " + " & ".join(sign_headers) + r" \\")
    
    # Cmidrules for each sign group
    cmidrules = []
    start_col = 3
    for _ in signs:
        end_col = start_col + num_batches - 1
        cmidrules.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
        start_col = end_col + 1
    lines.append("".join(cmidrules))
    
    # Batch values row (sub-header under each sign)
    batch_headers = []
    for _ in signs:
        for batch in batches:
            batch_num = batch.replace("B", "")
            batch_headers.append(f"{batch_num}")
    lines.append(r"  & Top $B$ & " + " & ".join(batch_headers) + r" \\")
    lines.append(r"\midrule")
    lines.append("")
    
    # Data rows: for each method, iterate through paper metrics
    for method_idx, method in enumerate(methods):
        method_display = METHOD_NAMES.get(method, method)

        for metric_idx, metric in enumerate(metrics):
            row_values = []

            # Get data for this method and metric across all sign/batch combinations
            for sign in signs:
                for batch in batches:
                    mask = (df_sweep["method"] == method) & \
                           (df_sweep["batch"] == batch) & \
                           (df_sweep["sign"] == sign)
                    row_data = df_sweep[mask]
                    
                    if len(row_data) > 0:
                        val = row_data[metric].values[0]
                        row_values.append(f"\\cellcolor{{gray!15}}{format_value(val, precision)}")
                    else:
                        row_values.append(r"\cellcolor{gray!15}-")

            # Build row string with cellcolor for metric and data cells (not method cell)
            if metric_idx == 0:
                method_cell = f"\\multirow{{{num_metrics}}}{{*}}{{\\textbf{{{method_display}}}}}"
            else:
                method_cell = ""

            metric_name = METRIC_SHORT_NAMES.get(metric, metric[:8])
            row_str = f"{method_cell} & \\cellcolor{{gray!15}}{metric_name} & " + " & ".join(row_values) + r" \\"
            lines.append(row_str)

        # Add midrule between methods (except after last)
        if method_idx < len(methods) - 1:
            lines.append(r"\midrule")
            lines.append("")
    
    # Footer
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    
    # Caption
    lines.append(r"\caption{Steering results by sign type and top $B$. " +
                r"Rows show methods with metric sub-rows. " +
                r"Rows report demeaned-logit correlation metrics. " +
                r"\textbf{Combined}: Combined medoid; \textbf{Default}: Default steering; " +
                r"\textbf{Single}: Single direction. " +
                r"\textbf{Full}: all directions; \textbf{Pos}: positive only; \textbf{Neg}: negative only.}")
    lines.append(r"\label{tab:steer-sweep-transposed}")
    lines.append(r"\end{table*}")
    
    latex_str = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(latex_str)
        print(f"LaTeX table saved to: {output_path}")
    
    return latex_str


def generate_by_sweep_table(df, metrics, precision, output_path) -> str:
    """
    Generate table for by_sweep breakdown.
    """
    df_sweep = df[df["breakdown"] == "by_sweep"].copy()
    
    # Parse sweep config
    df_sweep["sign"] = df_sweep["config"].apply(lambda x: x.rsplit("_", 1)[0])
    df_sweep["batch"] = df_sweep["config"].apply(lambda x: x.rsplit("_", 1)[1])
    
    methods = df_sweep["method"].unique().tolist()
    signs = sorted(df_sweep["sign"].unique())
    batches = sorted(df_sweep["batch"].unique())
    
    if metrics is None:
        metrics = DEFAULT_METRICS
    
    num_metrics = len(metrics)
    num_batches = len(batches)
    total_cols = num_metrics * num_batches
    
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\renewcommand{\arraystretch}{0.90}")
    lines.append(r"\setlength{\tabcolsep}{3.2pt}")
    lines.append("")
    
    col_spec = "l c " + " ".join([f"*{{{num_metrics}}}{{c}}" for _ in batches])
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    
    batch_cols = total_cols + 2
    lines.append(r"\multicolumn{2}{c}{} & " + f"\\multicolumn{{{total_cols}}}{{c}}{{Batch Size}} \\\\")
    lines.append(f"\\cmidrule(lr){{3-{batch_cols}}}")
    
    batch_headers = []
    for batch in batches:
        batch_headers.append(f"\\multicolumn{{{num_metrics}}}{{c}}{{{batch}}}")
    lines.append(r"Method & Sign & " + " & ".join(batch_headers) + r" \\")
    
    cmidrules = []
    start_col = 3
    for _ in batches:
        end_col = start_col + num_metrics - 1
        cmidrules.append(f"\\cmidrule(lr){{{start_col}-{end_col}}}")
        start_col = end_col + 1
    lines.append("".join(cmidrules))
    
    metric_headers = []
    for _ in batches:
        for m in metrics:
            metric_headers.append(METRIC_SHORT_NAMES.get(m, m[:8]))
    lines.append(r"  & & " + " & ".join(metric_headers) + r" \\")
    lines.append(r"\midrule")
    lines.append("")
    
    for method_idx, method in enumerate(methods):
        method_display = METHOD_NAMES.get(method, method)
        num_signs = len(signs)
        
        for sign_idx, sign in enumerate(signs):
            row_values = []
            
            for batch in batches:
                mask = (df_sweep["method"] == method) & \
                       (df_sweep["sign"] == sign) & \
                       (df_sweep["batch"] == batch)
                row_data = df_sweep[mask]
                
                if len(row_data) > 0:
                    for m in metrics:
                        val = row_data[m].values[0]
                        row_values.append(format_value(val, precision))
                else:
                    row_values.extend(["-"] * num_metrics)
            
            if sign_idx == 0:
                method_cell = f"\\multirow{{{num_signs}}}{{*}}{{\\textbf{{{method_display}}}}}"
            else:
                method_cell = ""
            
            sign_display = sign.replace("_", " ").replace("sign ", "")
            row_str = f"{method_cell} & {sign_display} & " + " & ".join(row_values) + r" \\"
            lines.append(row_str)
        
        if method_idx < len(methods) - 1:
            lines.append(r"\midrule")
            lines.append("")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"\caption{Steering results by sweep configuration. " +
                r"\textbf{Combined}: Combined medoid; \textbf{Default}: Default steering; " +
                r"\textbf{Single}: Single direction.}")
    lines.append(r"\label{tab:steer-sweep-results}")
    lines.append(r"\end{table*}")
    
    latex_str = "\n".join(lines)
    
    if output_path:
        with open(output_path, "w") as f:
            f.write(latex_str)
        print(f"LaTeX table saved to: {output_path}")
    
    return latex_str


def generate_latex_table(
    csv_path: str,
    output_path: str | None = None,
    breakdown: str = "by_config",
    metrics: list[str] | None = None,
    precision: int = 2,
    transpose: bool = False,
) -> str:
    """
    Generate LaTeX table from steer_results.csv.
    
    Args:
        csv_path: Path to the input CSV file
        output_path: Path to save the LaTeX output (optional)
        breakdown: Which breakdown to use ('by_config' or 'by_sweep')
        metrics: List of metrics to include (default: all numeric columns)
        precision: Decimal precision for values
        transpose: If True, metrics as rows, beta/gamma as columns
    
    Returns:
        LaTeX table string
    """
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Use default metrics if not specified
    if metrics is None:
        metrics = DEFAULT_METRICS
    else:
        metrics = [metric for metric in metrics if metric in CORRELATION_METRICS]
    
    if breakdown == "by_config":
        # Filter by breakdown
        df_filtered = df[df["breakdown"] == breakdown].copy()
        
        # Parse beta and gamma from config
        parsed = df_filtered["config"].apply(parse_config)
        df_filtered = df_filtered[parsed.notna()].copy()
        df_filtered["beta"] = parsed[parsed.notna()].apply(lambda x: x[0])
        df_filtered["gamma"] = parsed[parsed.notna()].apply(lambda x: x[1])
        
        # Get unique values
        methods = df_filtered["method"].unique().tolist()
        betas = sorted(df_filtered["beta"].unique())
        gammas = sorted(df_filtered["gamma"].unique())
        
        if transpose:
            return generate_transposed_table(
                df_filtered, methods, betas, gammas, metrics, precision, output_path
            )
        else:
            return generate_by_config_table(
                df_filtered, methods, betas, gammas, metrics, precision, output_path
            )
    
    elif breakdown == "by_sweep":
        if transpose:
            return generate_transposed_sweep_table(df, metrics, precision, output_path)
        else:
            return generate_by_sweep_table(df, metrics, precision, output_path)
    
    else:
        raise ValueError(f"Unknown breakdown: {breakdown}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert steer_results.csv to LaTeX multirow table"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save the LaTeX output (optional, prints to stdout if not specified)",
    )
    parser.add_argument(
        "-b", "--breakdown",
        type=str,
        default="by_config",
        choices=["by_config", "by_sweep"],
        help="Which breakdown to use (default: by_config)",
    )
    parser.add_argument(
        "-m", "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="List of metrics to include (default: all key metrics)",
    )
    parser.add_argument(
        "-p", "--precision",
        type=int,
        default=2,
        help="Decimal precision for values (default: 2)",
    )
    parser.add_argument(
        "-t", "--transpose",
        action="store_true",
        help="Transpose table: metrics as sub-rows, beta/gamma as columns",
    )
    
    args = parser.parse_args()
    
    latex_table = generate_latex_table(
        csv_path=args.csv_path,
        output_path=args.output,
        breakdown=args.breakdown,
        metrics=args.metrics,
        precision=args.precision,
        transpose=args.transpose,
    )
    
    if not args.output:
        print(latex_table)


if __name__ == "__main__":
    main()
