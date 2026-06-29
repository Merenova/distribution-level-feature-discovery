#!/usr/bin/env python3
"""
Anneal beta to find exact K clusters, then compare with K-means.
Outputs a LaTeX table.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm

# Add clustering module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "5_gaussian_clustering"))

from cluster import load_prefix_data
from rd_objective import compute_full_rd_statistics, compute_component_masses, compute_component_variance
from em_loop import weighted_median, run_em_iteration
from initialize import initialize_single_component
from adaptive_control import apply_adaptive_control
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("anneal_beta")


def check_convergence(L_prev, L_curr, threshold=1e-4):
    return abs(L_prev - L_curr) < threshold


def run_clustering_get_K(data, beta, gamma, max_iterations=50, metric_a="l1", 
                         normalize_dims=False, use_gpu=False):
    """Run RD clustering and return K and stats."""
    embeddings_e = data['embeddings_e']
    attributions_a = data['attributions_a']
    path_probs = data['path_probs']
    
    n_samples = len(path_probs)
    if n_samples < 2:
        return None
    
    # Get dimensions for normalization
    d_e = embeddings_e.shape[1]
    d_a = attributions_a.shape[1]
    
    # Compute beta_e, beta_a with optional dimension normalization
    if normalize_dims:
        beta_e = gamma * beta / np.sqrt(d_e)
        beta_a = (1 - gamma) * beta / d_a
    else:
        beta_e = gamma * beta
        beta_a = (1 - gamma) * beta
    
    # Initialize with single component (K=1)
    components, assignments = initialize_single_component(
        embeddings_e, attributions_a, path_probs, metric_a=metric_a
    )
    
    next_component_id = max(components.keys()) + 1 if components else 2
    L_RD_prev = np.inf
    convergence_threshold = 1e-4
    
    rd_stats = None
    
    # EM loop
    for iteration in range(max_iterations):
        assignments, components, rd_stats = run_em_iteration(
            embeddings_e, attributions_a, path_probs,
            components, beta_e, beta_a,
            metric_a=metric_a,
            use_gpu=use_gpu,
        )
        
        L_RD_curr = rd_stats['L_RD']
        
        # Adaptive control (split/junk)
        P_bar = rd_stats['P_bar']
        Var_e = rd_stats.get('Var_e', {})
        Var_a = rd_stats.get('Var_a', {})
        
        if not Var_e or not Var_a:
            W_c, _ = compute_component_masses(assignments, path_probs, list(components.keys()))
            for c, comp in components.items():
                indices = [i for i, a in enumerate(assignments) if a == c]
                W_c_val = W_c.get(c, 0)
                if not indices:
                    Var_e[c] = 0.0
                    Var_a[c] = 0.0
                    continue
                Var_e[c] = compute_component_variance(
                    embeddings_e[indices], comp['mu_e'], path_probs[indices], W_c_val, "l2"
                )
                Var_a[c] = compute_component_variance(
                    attributions_a[indices], comp['mu_a'], path_probs[indices], W_c_val, metric_a
                )
        
        components, assignments, next_component_id = apply_adaptive_control(
            embeddings_e, attributions_a, path_probs,
            assignments, components, P_bar, Var_e, Var_a,
            beta_e, beta_a,
            next_component_id=next_component_id,
            metric_a=metric_a,
        )
        
        if len(components) > 0:
            rd_stats = compute_full_rd_statistics(
                embeddings_e, attributions_a, assignments, path_probs,
                components, beta_e, beta_a,
                metric_a=metric_a,
            )
            L_RD_curr = rd_stats['L_RD']
        
        if check_convergence(L_RD_prev, L_RD_curr, convergence_threshold):
            break
        
        L_RD_prev = L_RD_curr
    
    K = len(components)
    if K < 2 or rd_stats is None:
        return None
    
    return {
        'K': K,
        'H': float(rd_stats['H']),
        'D_e': float(rd_stats['D_e']),
        'D_a': float(rd_stats['D_a']),
        'L_RD': float(rd_stats['L_RD']),
        'beta': beta
    }


def binary_search_exact_K(data, target_K, gamma, beta_min=0.01, beta_max=5.0, 
                          max_iterations=20, metric_a="l1", normalize_dims=False):
    """
    Binary search to find beta that gives exact K clusters.
    Higher beta -> fewer clusters, lower beta -> more clusters.
    Returns result with exact K match, or None if not found.
    """
    best_match = None
    lo, hi = beta_min, beta_max
    
    # First check boundaries
    result_hi = run_clustering_get_K(data, hi, gamma, metric_a=metric_a, normalize_dims=normalize_dims)
    result_lo = run_clustering_get_K(data, lo, gamma, metric_a=metric_a, normalize_dims=normalize_dims)
    
    if result_hi is None or result_lo is None:
        return None
    
    K_hi, K_lo = result_hi['K'], result_lo['K']
    
    # Check if target_K is in range
    if target_K < K_hi or target_K > K_lo:
        return None  # Target K not achievable in this beta range
    
    # Check boundaries for exact match
    if K_hi == target_K:
        return result_hi
    if K_lo == target_K:
        best_match = result_lo
    
    # Binary search
    for _ in range(max_iterations):
        if hi - lo < 0.01:
            break
            
        mid = (lo + hi) / 2
        result = run_clustering_get_K(data, mid, gamma, metric_a=metric_a, normalize_dims=normalize_dims)
        
        if result is None:
            hi = mid
            continue
        
        K_mid = result['K']
        
        if K_mid == target_K:
            # Found exact match - try to find higher beta with same K
            best_match = result
            lo = mid  # Search higher betas
        elif K_mid < target_K:
            # Too few clusters, need lower beta
            hi = mid
        else:
            # Too many clusters, need higher beta
            lo = mid
    
    return best_match


def anneal_beta_for_exact_K(data, target_K, gamma, beta_min=0.01, beta_max=5.0, 
                            n_steps=30, metric_a="l1", normalize_dims=False):
    """
    Hybrid approach: binary search first, then local refinement.
    """
    # Try binary search first (fast)
    result = binary_search_exact_K(data, target_K, gamma, beta_min, beta_max, 
                                   max_iterations=15, metric_a=metric_a, 
                                   normalize_dims=normalize_dims)
    
    if result is not None:
        return result
    
    # Fallback: sparse linear search if binary search failed
    exact_matches = []
    betas = np.linspace(beta_max, beta_min, n_steps)
    
    for beta in betas:
        result = run_clustering_get_K(data, beta, gamma, metric_a=metric_a, 
                                      normalize_dims=normalize_dims)
        if result is None:
            continue
        
        if result['K'] == target_K:
            exact_matches.append(result)
            break  # Early stop on first match (highest beta due to descending order)
    
    if not exact_matches:
        return None
    
    return exact_matches[0]


def compute_kmeans_stats(embeddings_e, attributions_a, path_probs, K, use_semantic=True):
    """Run K-means and compute H, D_e, D_a."""
    n_samples = len(path_probs)
    if n_samples < K:
        return None
    
    features = embeddings_e if use_semantic else attributions_a
    
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    # Build components
    components = {}
    for c in range(K):
        mask = labels == c
        if not np.any(mask):
            continue
        W_c = np.sum(path_probs[mask])
        if W_c > 0:
            mu_e = np.average(embeddings_e[mask], weights=path_probs[mask], axis=0)
            mu_a = weighted_median(attributions_a[mask], path_probs[mask])
            components[c] = {"mu_e": mu_e, "mu_a": mu_a, "W_c": W_c}
    
    if len(components) < 2:
        return None
    
    assignments = labels.tolist()
    rd_stats = compute_full_rd_statistics(
        embeddings_e, attributions_a, assignments, path_probs,
        components, 1.0, 1.0, metric_a="l1"
    )
    
    return {
        "H": float(rd_stats["H"]),
        "D_e": float(rd_stats["D_e"]),
        "D_a": float(rd_stats["D_a"])
    }


def process_prefix(args):
    """Process a single prefix for all (K, gamma) combinations."""
    prefix_id, embeddings_dir, attribution_graphs_dir, samples_dir, target_Ks, gammas, beta_min, beta_max, n_steps = args
    
    try:
        data = load_prefix_data(
            prefix_id, embeddings_dir, attribution_graphs_dir, samples_dir,
            logger, metric_a="l1"
        )
    except Exception as e:
        return prefix_id, {'error': str(e)}
    
    results = {}
    
    for gamma in gammas:
        gamma_key = f"{gamma:.1f}"
        results[gamma_key] = {}
        
        for target_K in target_Ks:
            K_key = str(target_K)
            
            # Anneal beta for RD
            rd_result = anneal_beta_for_exact_K(
                data, target_K, gamma, 
                beta_min=beta_min, beta_max=beta_max, n_steps=n_steps
            )
            
            # K-means baselines
            km_sem = compute_kmeans_stats(
                data['embeddings_e'], data['attributions_a'], 
                data['path_probs'], target_K, use_semantic=True
            )
            km_attr = compute_kmeans_stats(
                data['embeddings_e'], data['attributions_a'], 
                data['path_probs'], target_K, use_semantic=False
            )
            
            results[gamma_key][K_key] = {
                'rd': rd_result,
                'km_sem': km_sem,
                'km_attr': km_attr
            }
    
    return prefix_id, results


def generate_latex_table(aggregated, target_Ks, gammas, output_path):
    """Generate LaTeX table from aggregated results."""
    
    tex = r"""% requires in preamble:
% \usepackage{booktabs}
% \usepackage{multirow}

\begin{table*}[t]
\centering
\scriptsize
\renewcommand{\arraystretch}{0.90}
\setlength{\tabcolsep}{3.2pt}

\begin{tabular}{l c c c c c c c c c c c c c c c c c c c c c}
\toprule
\multicolumn{2}{c}{} & \multicolumn{20}{c}{$\gamma$} \\
\cmidrule(lr){3-22}
Method & $K$ & \multicolumn{4}{c}{0.10} & \multicolumn{4}{c}{0.30} & \multicolumn{4}{c}{0.50} & \multicolumn{4}{c}{0.70} & \multicolumn{4}{c}{0.90} \\
\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}\cmidrule(lr){15-18}\cmidrule(lr){19-22}
  &   & $H$ & $D_e$ & $D_a$ & $D_{\gamma}$ & $H$ & $D_e$ & $D_a$ & $D_{\gamma}$ & $H$ & $D_e$ & $D_a$ & $D_{\gamma}$ & $H$ & $D_e$ & $D_a$ & $D_{\gamma}$ & $H$ & $D_e$ & $D_a$ & $D_{\gamma}$ \\
\midrule

"""
    
    # RD rows
    tex += r"\multirow{" + str(len(target_Ks)) + r"}{*}{\textbf{RD}}" + "\n"
    for K in target_Ks:
        K_key = str(K)
        row = f" & {K}"
        for gamma in gammas:
            gamma_key = f"{gamma:.1f}"
            d = aggregated.get(gamma_key, {}).get(K_key, {}).get('rd', {})
            if d and d.get('n', 0) > 0:
                D_gamma = gamma * d['D_e'] + (1 - gamma) * d['D_a']
                row += f" & {d['H']:.2f} & {d['D_e']:.2f} & {d['D_a']:.2f} & {D_gamma:.2f}"
            else:
                row += " & -- & -- & -- & --"
        row += r" \\" + "\n"
        tex += row
    
    tex += r"\midrule" + "\n\n"
    
    # KM-S rows
    tex += r"\multirow{" + str(len(target_Ks)) + r"}{*}{\textbf{KM-S}}" + "\n"
    for K in target_Ks:
        K_key = str(K)
        row = f" & {K}"
        for gamma in gammas:
            gamma_key = f"{gamma:.1f}"
            d = aggregated.get(gamma_key, {}).get(K_key, {}).get('km_sem', {})
            if d and d.get('n', 0) > 0:
                D_gamma = gamma * d['D_e'] + (1 - gamma) * d['D_a']
                row += f" & {d['H']:.2f} & {d['D_e']:.2f} & {d['D_a']:.2f} & {D_gamma:.2f}"
            else:
                row += " & -- & -- & -- & --"
        row += r" \\" + "\n"
        tex += row
    
    tex += r"\midrule" + "\n\n"
    
    # KM-A rows
    tex += r"\multirow{" + str(len(target_Ks)) + r"}{*}{\textbf{KM-A}}" + "\n"
    for K in target_Ks:
        K_key = str(K)
        row = f" & {K}"
        for gamma in gammas:
            gamma_key = f"{gamma:.1f}"
            d = aggregated.get(gamma_key, {}).get(K_key, {}).get('km_attr', {})
            if d and d.get('n', 0) > 0:
                D_gamma = gamma * d['D_e'] + (1 - gamma) * d['D_a']
                row += f" & {d['H']:.2f} & {d['D_e']:.2f} & {d['D_a']:.2f} & {D_gamma:.2f}"
            else:
                row += " & -- & -- & -- & --"
        row += r" \\" + "\n"
        tex += row
    
    tex += r"""\bottomrule
\end{tabular}

\caption{RD clustering (annealed $\beta$ for exact $K$) vs.\ K-means comparison. $D_{\gamma}=\gamma D_e + (1-\gamma)D_a$. \textbf{KM-S}: K-means semantic; \textbf{KM-A}: K-means attribution.}
\label{tab:rd-kmeans-annealed-exact-k}
\end{table*}
"""
    
    with open(output_path, 'w') as f:
        f.write(tex)
    
    return tex


def main():
    parser = argparse.ArgumentParser(description="Anneal beta to find exact K")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--target-K", type=str, default="2,4,6,8,10")
    parser.add_argument("--gamma-values", type=str, default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--beta-min", type=float, default=0.01)
    parser.add_argument("--beta-max", type=float, default=5.0)
    parser.add_argument("--n-steps", type=int, default=30, help="Number of beta steps for fallback linear search")
    parser.add_argument("--n-workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_Ks = [int(k) for k in args.target_K.split(",")]
    gammas = [float(g) for g in args.gamma_values.split(",")]
    
    # Find prefixes
    embeddings_dir = results_dir / "4_feature_extraction" / "embeddings"
    attribution_graphs_dir = results_dir / "3_attribution_graphs"
    samples_dir = results_dir / "2_branch_sampling"
    
    # Find prefixes from branch files
    branch_files = sorted(samples_dir.glob("*_branches.json"))
    prefix_ids = [f.stem.replace("_branches", "") for f in branch_files]
    if args.limit:
        prefix_ids = prefix_ids[:args.limit]
    
    print(f"Processing {len(prefix_ids)} prefixes")
    print(f"Target K: {target_Ks}")
    print(f"Gamma values: {gammas}")
    print(f"Beta range: [{args.beta_min}, {args.beta_max}] with {args.n_steps} steps")
    print(f"Using {args.n_workers} workers")
    print()
    
    # Prepare tasks
    tasks = [
        (pid, embeddings_dir, attribution_graphs_dir, samples_dir, 
         target_Ks, gammas, args.beta_min, args.beta_max, args.n_steps)
        for pid in prefix_ids
    ]
    
    # Process in parallel with progress bar
    all_results = {}
    
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(process_prefix, task): task[0] for task in tasks}
        
        with tqdm(total=len(prefix_ids), desc="Processing prefixes") as pbar:
            for future in as_completed(futures):
                prefix_id = futures[future]
                try:
                    pid, result = future.result()
                    all_results[pid] = result
                except Exception as e:
                    print(f"\n  Error processing {prefix_id}: {e}")
                    all_results[prefix_id] = {'error': str(e)}
                pbar.update(1)
    
    print(f"\nCompleted all {len(all_results)} prefixes")
    
    # Aggregate results
    aggregated = defaultdict(lambda: defaultdict(lambda: {
        'rd': {'H': [], 'D_e': [], 'D_a': [], 'beta': [], 'n': 0},
        'km_sem': {'H': [], 'D_e': [], 'D_a': [], 'n': 0},
        'km_attr': {'H': [], 'D_e': [], 'D_a': [], 'n': 0}
    }))
    
    for prefix_id, result in all_results.items():
        if 'error' in result:
            continue
        
        for gamma_key, gamma_data in result.items():
            for K_key, K_data in gamma_data.items():
                agg = aggregated[gamma_key][K_key]
                
                # RD
                if K_data.get('rd'):
                    rd = K_data['rd']
                    agg['rd']['H'].append(rd['H'])
                    agg['rd']['D_e'].append(rd['D_e'])
                    agg['rd']['D_a'].append(rd['D_a'])
                    agg['rd']['beta'].append(rd['beta'])
                    agg['rd']['n'] += 1
                
                # K-means Semantic
                if K_data.get('km_sem'):
                    km = K_data['km_sem']
                    agg['km_sem']['H'].append(km['H'])
                    agg['km_sem']['D_e'].append(km['D_e'])
                    agg['km_sem']['D_a'].append(km['D_a'])
                    agg['km_sem']['n'] += 1
                
                # K-means Attribution
                if K_data.get('km_attr'):
                    km = K_data['km_attr']
                    agg['km_attr']['H'].append(km['H'])
                    agg['km_attr']['D_e'].append(km['D_e'])
                    agg['km_attr']['D_a'].append(km['D_a'])
                    agg['km_attr']['n'] += 1
    
    # Compute means
    for gamma_key in aggregated:
        for K_key in aggregated[gamma_key]:
            for method in ['rd', 'km_sem', 'km_attr']:
                d = aggregated[gamma_key][K_key][method]
                if d['n'] > 0:
                    d['H'] = np.mean(d['H'])
                    d['D_e'] = np.mean(d['D_e'])
                    d['D_a'] = np.mean(d['D_a'])
                    if 'beta' in d and d['beta']:
                        d['beta'] = np.mean(d['beta'])
    
    # Print summary
    print("\n" + "="*120)
    print("SUMMARY: Annealed Beta for Exact K")
    print("="*120)
    print(f"\n{'K':<4} {'γ':<5} {'n_rd':<6} {'β_avg':<8} | {'RD H':<8} {'RD D_e':<10} {'RD D_a':<10} | {'KM-S H':<8} {'KM-A H':<8}")
    print("-"*100)
    
    for K in target_Ks:
        K_key = str(K)
        for gamma in gammas:
            gamma_key = f"{gamma:.1f}"
            agg = aggregated.get(gamma_key, {}).get(K_key, {})
            rd = agg.get('rd', {})
            km_sem = agg.get('km_sem', {})
            km_attr = agg.get('km_attr', {})
            
            n_rd = rd.get('n', 0)
            if n_rd > 0:
                print(f"{K:<4} {gamma:<5.1f} {n_rd:<6} {rd.get('beta', 0):<8.3f} | "
                      f"{rd['H']:<8.4f} {rd['D_e']:<10.4f} {rd['D_a']:<10.4f} | "
                      f"{km_sem.get('H', 0):<8.4f} {km_attr.get('H', 0):<8.4f}")
            else:
                print(f"{K:<4} {gamma:<5.1f} 0      --       | --       --         --         | --       --")
    
    # Save JSON
    json_path = output_dir / "annealed_beta_results.json"
    
    # Convert defaultdict to regular dict for JSON
    json_data = {
        'config': {
            'target_K': target_Ks,
            'gammas': gammas,
            'beta_min': args.beta_min,
            'beta_max': args.beta_max,
            'n_steps': args.n_steps,
            'n_prefixes': len(prefix_ids)
        },
        'aggregated': {gk: dict(gv) for gk, gv in aggregated.items()},
        'per_prefix': all_results
    }
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=float)
    print(f"\nSaved JSON to: {json_path}")
    
    # Save CSV
    csv_path = output_dir / "annealed_beta_comparison.csv"
    with open(csv_path, 'w') as f:
        f.write("K,gamma,n,beta_avg,RD_H,RD_D_e,RD_D_a,KM_Sem_H,KM_Sem_D_e,KM_Sem_D_a,KM_Attr_H,KM_Attr_D_e,KM_Attr_D_a\n")
        for K in target_Ks:
            K_key = str(K)
            for gamma in gammas:
                gamma_key = f"{gamma:.1f}"
                agg = aggregated.get(gamma_key, {}).get(K_key, {})
                rd = agg.get('rd', {})
                km_sem = agg.get('km_sem', {})
                km_attr = agg.get('km_attr', {})
                
                if rd.get('n', 0) > 0:
                    f.write(f"{K},{gamma},{rd['n']},{rd.get('beta', 0):.4f},"
                            f"{rd['H']:.4f},{rd['D_e']:.4f},{rd['D_a']:.4f},"
                            f"{km_sem.get('H', 0):.4f},{km_sem.get('D_e', 0):.4f},{km_sem.get('D_a', 0):.4f},"
                            f"{km_attr.get('H', 0):.4f},{km_attr.get('D_e', 0):.4f},{km_attr.get('D_a', 0):.4f}\n")
    print(f"Saved CSV to: {csv_path}")
    
    # Generate LaTeX
    tex_path = output_dir / "annealed_beta_table.tex"
    tex = generate_latex_table(aggregated, target_Ks, gammas, tex_path)
    print(f"Saved LaTeX to: {tex_path}")
    
    paper_tex_path = Path(args.results_dir).parent / "paper" / "tables" / "RD_kmeans_annealed_exact_K.tex"
    if paper_tex_path.parent.exists():
        with open(paper_tex_path, 'w') as f:
            f.write(tex)
        print(f"Saved LaTeX to: {paper_tex_path}")


if __name__ == "__main__":
    main()
