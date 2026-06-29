#!/usr/bin/env -S uv run python
"""Full verification of D_gamma = gamma * D_e + (1-gamma) * D_a against LaTeX table values."""

import csv
from pathlib import Path


def compute_d_gamma(gamma: float, d_e: float, d_a: float) -> float:
    """Compute combined distortion: D_gamma = gamma * D_e + (1 - gamma) * D_a"""
    return gamma * d_e + (1 - gamma) * d_a


# Expected D_gamma values from the user's LaTeX table
# Format: (method, beta, gamma): D_gamma
LATEX_TABLE_VALUES = {
    # RD method
    ("rd", 0.50, 0.10): 9.06,
    ("rd", 0.50, 0.30): 8.04,
    ("rd", 0.50, 0.50): 6.88,
    ("rd", 0.50, 0.70): 5.00,
    ("rd", 0.50, 0.90): 2.11,
    ("rd", 0.75, 0.10): 7.51,
    ("rd", 0.75, 0.30): 6.51,
    ("rd", 0.75, 0.50): 5.59,
    ("rd", 0.75, 0.70): 4.39,
    ("rd", 0.75, 0.90): 2.04,
    ("rd", 1.00, 0.10): 6.96,
    ("rd", 1.00, 0.30): 5.82,
    ("rd", 1.00, 0.50): 4.82,
    ("rd", 1.00, 0.70): 3.82,
    ("rd", 1.00, 0.90): 1.99,
    ("rd", 1.25, 0.10): 6.67,
    ("rd", 1.25, 0.30): 5.46,
    ("rd", 1.25, 0.50): 4.38,
    ("rd", 1.25, 0.70): 3.41,
    ("rd", 1.25, 0.90): 1.93,
    ("rd", 1.50, 0.10): 6.57,
    ("rd", 1.50, 0.30): 5.27,
    ("rd", 1.50, 0.50): 4.09,
    ("rd", 1.50, 0.70): 3.13,
    ("rd", 1.50, 0.90): 1.84,
    ("rd", 2.00, 0.10): 6.49,
    ("rd", 2.00, 0.30): 5.15,
    ("rd", 2.00, 0.50): 3.83,
    ("rd", 2.00, 0.70): 2.72,
    ("rd", 2.00, 0.90): 1.71,
    # KM-S method
    ("kmeans_semantic", 0.50, 0.10): 12.23,
    ("kmeans_semantic", 0.50, 0.30): 10.27,
    ("kmeans_semantic", 0.50, 0.50): 8.13,
    ("kmeans_semantic", 0.50, 0.70): 5.47,
    ("kmeans_semantic", 0.50, 0.90): 1.98,
    ("kmeans_semantic", 0.75, 0.10): 11.05,
    ("kmeans_semantic", 0.75, 0.30): 9.10,
    ("kmeans_semantic", 0.75, 0.50): 7.19,
    ("kmeans_semantic", 0.75, 0.70): 4.99,
    ("kmeans_semantic", 0.75, 0.90): 1.83,
    ("kmeans_semantic", 1.00, 0.10): 10.64,
    ("kmeans_semantic", 1.00, 0.30): 8.55,
    ("kmeans_semantic", 1.00, 0.50): 6.62,
    ("kmeans_semantic", 1.00, 0.70): 4.67,
    ("kmeans_semantic", 1.00, 0.90): 1.84,
    ("kmeans_semantic", 1.25, 0.10): 10.44,
    ("kmeans_semantic", 1.25, 0.30): 8.31,
    ("kmeans_semantic", 1.25, 0.50): 6.23,
    ("kmeans_semantic", 1.25, 0.70): 4.34,
    ("kmeans_semantic", 1.25, 0.90): 1.99,
    ("kmeans_semantic", 1.50, 0.10): 10.36,
    ("kmeans_semantic", 1.50, 0.30): 8.20,
    ("kmeans_semantic", 1.50, 0.50): 6.05,
    ("kmeans_semantic", 1.50, 0.70): 4.11,
    ("kmeans_semantic", 1.50, 0.90): 1.94,
    ("kmeans_semantic", 2.00, 0.10): 10.31,
    ("kmeans_semantic", 2.00, 0.30): 8.08,
    ("kmeans_semantic", 2.00, 0.50): 5.87,
    ("kmeans_semantic", 2.00, 0.70): 3.82,
    ("kmeans_semantic", 2.00, 0.90): 1.86,
    # KM-A method
    ("kmeans_attribution", 0.50, 0.10): 8.35,
    ("kmeans_attribution", 0.50, 0.30): 7.61,
    ("kmeans_attribution", 0.50, 0.50): 6.78,
    ("kmeans_attribution", 0.50, 0.70): 4.89,
    ("kmeans_attribution", 0.50, 0.90): 1.98,
    ("kmeans_attribution", 0.75, 0.10): 6.91,
    ("kmeans_attribution", 0.75, 0.30): 5.95,
    ("kmeans_attribution", 0.75, 0.50): 5.22,
    ("kmeans_attribution", 0.75, 0.70): 4.34,
    ("kmeans_attribution", 0.75, 0.90): 1.83,
    ("kmeans_attribution", 1.00, 0.10): 6.46,
    ("kmeans_attribution", 1.00, 0.30): 5.35,
    ("kmeans_attribution", 1.00, 0.50): 4.44,
    ("kmeans_attribution", 1.00, 0.70): 3.58,
    ("kmeans_attribution", 1.00, 0.90): 1.82,
    ("kmeans_attribution", 1.25, 0.10): 6.22,
    ("kmeans_attribution", 1.25, 0.30): 5.03,
    ("kmeans_attribution", 1.25, 0.50): 4.00,
    ("kmeans_attribution", 1.25, 0.70): 3.19,
    ("kmeans_attribution", 1.25, 0.90): 1.83,
    ("kmeans_attribution", 1.50, 0.10): 6.13,
    ("kmeans_attribution", 1.50, 0.30): 4.92,
    ("kmeans_attribution", 1.50, 0.50): 3.78,
    ("kmeans_attribution", 1.50, 0.70): 2.87,
    ("kmeans_attribution", 1.50, 0.90): 1.81,
    ("kmeans_attribution", 2.00, 0.10): 6.06,
    ("kmeans_attribution", 2.00, 0.30): 4.80,
    ("kmeans_attribution", 2.00, 0.50): 3.56,
    ("kmeans_attribution", 2.00, 0.70): 2.47,
    ("kmeans_attribution", 2.00, 0.90): 1.68,
}


def main():
    csv_path = Path(__file__).parent.parent / "AmbigQA_Qwen3-8B/results/plots/rd_sweep_table.csv"
    
    print("=" * 100)
    print("Full verification: D_gamma = gamma * D_e + (1 - gamma) * D_a")
    print("=" * 100)
    print()
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    method_labels = {
        "rd": "RD",
        "kmeans_semantic": "KM-S", 
        "kmeans_attribution": "KM-A"
    }
    
    # Build lookup from CSV data
    csv_lookup = {}
    for row in rows:
        method = row["method"]
        beta = float(row["beta"])
        gamma = float(row["gamma"])
        d_e = float(row["D_e"])
        d_a = float(row["D_a"])
        csv_lookup[(method, beta, gamma)] = {"D_e": d_e, "D_a": d_a}
    
    print(f"{'Method':<8} {'β':>6} {'γ':>6} | {'D_e':>8} {'D_a':>8} | "
          f"{'D_γ computed':>14} {'D_γ table':>12} {'Δ':>8} {'Status':<6}")
    print("-" * 100)
    
    total_checks = 0
    passed_checks = 0
    large_diffs = []
    
    for (method, beta, gamma), expected_d_gamma in sorted(LATEX_TABLE_VALUES.items()):
        key = (method, beta, gamma)
        if key not in csv_lookup:
            print(f"WARNING: Key {key} not found in CSV!")
            continue
        
        d_e = csv_lookup[key]["D_e"]
        d_a = csv_lookup[key]["D_a"]
        computed_d_gamma = compute_d_gamma(gamma, d_e, d_a)
        computed_rounded = round(computed_d_gamma, 2)
        
        diff = abs(computed_rounded - expected_d_gamma)
        total_checks += 1
        
        if diff <= 0.02:  # Allow small rounding differences
            status = "✓"
            passed_checks += 1
        else:
            status = "✗"
            large_diffs.append((method, beta, gamma, computed_rounded, expected_d_gamma, diff))
        
        print(f"{method_labels.get(method, method):<8} {beta:>6.2f} {gamma:>6.2f} | "
              f"{d_e:>8.4f} {d_a:>8.2f} | "
              f"{computed_rounded:>14.2f} {expected_d_gamma:>12.2f} {diff:>8.2f} {status:<6}")
    
    print()
    print("=" * 100)
    print(f"SUMMARY: {passed_checks}/{total_checks} values match (within 0.02 tolerance)")
    print("=" * 100)
    
    if large_diffs:
        print()
        print("Values with larger discrepancies (> 0.02):")
        print("-" * 80)
        for method, beta, gamma, computed, expected, diff in large_diffs:
            print(f"  {method_labels.get(method, method)} β={beta:.2f} γ={gamma:.2f}: "
                  f"computed={computed:.2f}, table={expected:.2f}, diff={diff:.2f}")
    else:
        print()
        print("✓ All D_gamma values computed correctly!")
    
    return 0 if not large_diffs else 1


if __name__ == "__main__":
    exit(main())
