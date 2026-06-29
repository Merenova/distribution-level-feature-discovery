#!/usr/bin/env python3
"""
Smart RD vs K-means comparison with conditional annealing.

Algorithm:
1. Start from beta values in existing sweep results for each (K, gamma)
2. Check condition: RD's D_e < KM-Attr's D_e AND RD's D_a < KM-Sem's D_a
3. If condition met → done, use this result
4. If not → increase beta and retry until K breaks or condition met
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent / "5_gaussian_clustering"))

from cluster import load_prefix_data
from rd_objective import compute_full_rd_statistics, compute_component_masses, compute_component_variance
from em_loop import weighted_median, run_em_iteration
from initialize import initialize_single_component
from adaptive_control import apply_adaptive_control
from sklearn.cluster import KMeans

import logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("rd_vs_kmeans")


def run_clustering_get_stats(data, beta, gamma, max_iterations=50, metric_a="l1"):
    """Run RD clustering and return K and stats."""
    embeddings_e = data['embeddings_e']
    attributions_a = data['attributions_a']
    path_probs = data['path_probs']
    
    n_samples = len(path_probs)
    if n_samples < 2:
        return None
    
    beta_e = gamma * beta
    beta_a = (1 - gamma) * beta
    
    components, assignments = initialize_single_component(
        embeddings_e, attributions_a, path_probs, metric_a=metric_a
    )
    
    next_component_id = max(components.keys()) + 1 if components else 2
    L_RD_prev = np.inf
    rd_stats = None
    
    for iteration in range(max_iterations):
        assignments, components, rd_stats = run_em_iteration(
            embeddings_e, attributions_a, path_probs,
            components, beta_e, beta_a,
            metric_a=metric_a, use_gpu=False,
        )
        
        L_RD_curr = rd_stats['L_RD']
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
                components, beta_e, beta_a, metric_a=metric_a,
            )
            L_RD_curr = rd_stats['L_RD']
        
        if abs(L_RD_prev - L_RD_curr) < 1e-4:
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
        'beta': beta
    }


def compute_kmeans_stats(embeddings_e, attributions_a, path_probs, K, use_semantic=True):
    """Run K-means and compute H, D_e, D_a."""
    n_samples = len(path_probs)
    if n_samples < K:
        return None
    
    features = embeddings_e if use_semantic else attributions_a
    
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
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


def compute_objective(stats, beta_e, beta_a):
    """Compute full RD objective: L = H + β_e·D_e + β_a·D_a"""
    return stats['H'] + beta_e * stats['D_e'] + beta_a * stats['D_a']


def check_win_condition(rd_stats, km_sem, km_attr, beta_e, beta_a):
    """Check if RD beats both K-means baselines on full objective."""
    L_rd = compute_objective(rd_stats, beta_e, beta_a)
    L_km_sem = compute_objective(km_sem, beta_e, beta_a)
    L_km_attr = compute_objective(km_attr, beta_e, beta_a)
    return L_rd < L_km_sem and L_rd < L_km_attr


def anneal_until_condition_or_break(data, start_beta, target_K, gamma, km_sem, km_attr,
                                     beta_step=0.1, max_beta=10.0):
    """
    Starting from start_beta, increase beta until:
    - Condition met (RD objective < both KM objectives), or
    - K changes (breaks)
    
    Returns (success, result)
    """
    beta = start_beta
    
    while beta <= max_beta:
        result = run_clustering_get_stats(data, beta, gamma)
        
        if result is None:
            return False, None
        
        if result['K'] != target_K:
            # K broke, stop
            return False, None
        
        beta_e = gamma * beta
        beta_a = (1 - gamma) * beta
        if check_win_condition(result, km_sem, km_attr, beta_e, beta_a):
            # Condition met!
            return True, result
        
        # Increase beta
        beta += beta_step
    
    return False, None


def process_prefix(args):
    """Process a single prefix."""
    (prefix_id, embeddings_dir, attribution_graphs_dir, samples_dir, sweep_dir,
     target_Ks, gammas, beta_step, max_beta) = args
    
    # Load sweep results
    sweep_file = sweep_dir / f"{prefix_id}_sweep_results.json"
    if not sweep_file.exists():
        return prefix_id, {'error': 'No sweep file'}
    
    try:
        with open(sweep_file) as f:
            sweep_data = json.load(f)
    except:
        return prefix_id, {'error': 'Failed to load sweep'}
    
    # Load prefix data
    try:
        data = load_prefix_data(
            prefix_id, embeddings_dir, attribution_graphs_dir, samples_dir,
            logger, metric_a="l1"
        )
    except Exception as e:
        return prefix_id, {'error': str(e)}
    
    # Index existing RD results by (K, gamma)
    rd_by_k_gamma = defaultdict(list)
    for entry in sweep_data.get('grid', []):
        K = int(entry.get('K', 0))
        gamma = float(entry.get('gamma', -1))
        if K in target_Ks:
            for target_gamma in gammas:
                if abs(gamma - target_gamma) < 0.01:
                    rd_by_k_gamma[(K, target_gamma)].append({
                        'H': float(entry.get('H', 0)),
                        'D_e': float(entry.get('D_e', 0)),
                        'D_a': float(entry.get('D_a', 0)),
                        'beta': float(entry.get('beta', 0))
                    })
                    break
    
    # Compute K-means baselines once per K
    km_cache = {}
    for K in target_Ks:
        km_sem = compute_kmeans_stats(
            data['embeddings_e'], data['attributions_a'],
            data['path_probs'], K, use_semantic=True
        )
        km_attr = compute_kmeans_stats(
            data['embeddings_e'], data['attributions_a'],
            data['path_probs'], K, use_semantic=False
        )
        km_cache[K] = {'sem': km_sem, 'attr': km_attr}
    
    results = {}
    
    for K in target_Ks:
        km_sem = km_cache[K]['sem']
        km_attr = km_cache[K]['attr']
        
        if km_sem is None or km_attr is None:
            continue
        
        for gamma in gammas:
            key = f"{K}_{gamma}"
            rd_entries = rd_by_k_gamma.get((K, gamma), [])
            
            if not rd_entries:
                results[key] = {'status': 'no_rd_data'}
                continue
            
            # Sort by beta descending (try highest compression first)
            rd_entries = sorted(rd_entries, key=lambda x: -x['beta'])
            
            # Check if any existing result meets condition
            found = False
            for rd in rd_entries:
                beta_e = gamma * rd['beta']
                beta_a = (1 - gamma) * rd['beta']
                if check_win_condition(rd, km_sem, km_attr, beta_e, beta_a):
                    results[key] = {
                        'status': 'found_existing',
                        'rd': rd,
                        'km_sem': km_sem,
                        'km_attr': km_attr,
                        'annealed': False
                    }
                    found = True
                    break
            
            if found:
                continue
            
            # Need to anneal - start from highest beta we have
            start_beta = rd_entries[0]['beta']
            success, result = anneal_until_condition_or_break(
                data, start_beta, K, gamma, km_sem, km_attr,
                beta_step=beta_step, max_beta=max_beta
            )
            
            if success and result:
                results[key] = {
                    'status': 'found_annealed',
                    'rd': result,
                    'km_sem': km_sem,
                    'km_attr': km_attr,
                    'annealed': True
                }
            else:
                # Use best existing result even though condition not met
                best_rd = rd_entries[0]  # Highest beta
                results[key] = {
                    'status': 'condition_not_met',
                    'rd': best_rd,
                    'km_sem': km_sem,
                    'km_attr': km_attr,
                    'annealed': False
                }
    
    return prefix_id, results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--target-K", type=str, default="2,4,6,8,10")
    parser.add_argument("--gamma-values", type=str, default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--beta-step", type=float, default=0.1)
    parser.add_argument("--max-beta", type=float, default=10.0)
    parser.add_argument("--n-workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    target_Ks = [int(k) for k in args.target_K.split(",")]
    gammas = [float(g) for g in args.gamma_values.split(",")]
    
    sweep_dir = results_dir / "5_clustering"
    embeddings_dir = results_dir / "4_feature_extraction" / "embeddings"
    attribution_graphs_dir = results_dir / "3_attribution_graphs"
    samples_dir = results_dir / "2_branch_sampling"
    
    # Find prefixes
    branch_files = sorted(samples_dir.glob("*_branches.json"))
    prefix_ids = [f.stem.replace("_branches", "") for f in branch_files]
    if args.limit:
        prefix_ids = prefix_ids[:args.limit]
    
    print(f"Processing {len(prefix_ids)} prefixes")
    print(f"Target K: {target_Ks}, Gammas: {gammas}")
    print(f"Beta step: {args.beta_step}, Max beta: {args.max_beta}")
    print()
    
    # Prepare tasks
    tasks = [
        (pid, embeddings_dir, attribution_graphs_dir, samples_dir, sweep_dir,
         target_Ks, gammas, args.beta_step, args.max_beta)
        for pid in prefix_ids
    ]
    
    # Process in parallel
    all_results = {}
    
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(process_prefix, task): task[0] for task in tasks}
        
        with tqdm(total=len(prefix_ids), desc="Processing") as pbar:
            for future in as_completed(futures):
                prefix_id = futures[future]
                try:
                    pid, result = future.result()
                    all_results[pid] = result
                except Exception as e:
                    all_results[prefix_id] = {'error': str(e)}
                pbar.update(1)
    
    # Aggregate results
    agg = defaultdict(lambda: {
        'rd': {'D_e': [], 'D_a': [], 'H': [], 'beta': []},
        'km_sem': {'D_e': [], 'D_a': [], 'H': []},
        'km_attr': {'D_e': [], 'D_a': [], 'H': []},
        'found_existing': 0,
        'found_annealed': 0,
        'condition_not_met': 0,
        'n': 0
    })
    
    for prefix_id, results in all_results.items():
        if 'error' in results:
            continue
        
        for key, data in results.items():
            if isinstance(data, dict) and 'status' in data:
                K, gamma = key.split('_')
                agg_key = (int(K), float(gamma))
                cell = agg[agg_key]
                
                if data['status'] in ['found_existing', 'found_annealed', 'condition_not_met']:
                    rd = data['rd']
                    km_sem = data['km_sem']
                    km_attr = data['km_attr']
                    
                    cell['rd']['D_e'].append(rd['D_e'])
                    cell['rd']['D_a'].append(rd['D_a'])
                    cell['rd']['H'].append(rd['H'])
                    cell['rd']['beta'].append(rd['beta'])
                    cell['km_sem']['D_e'].append(km_sem['D_e'])
                    cell['km_sem']['D_a'].append(km_sem['D_a'])
                    cell['km_sem']['H'].append(km_sem['H'])
                    cell['km_attr']['D_e'].append(km_attr['D_e'])
                    cell['km_attr']['D_a'].append(km_attr['D_a'])
                    cell['km_attr']['H'].append(km_attr['H'])
                    cell[data['status']] += 1
                    cell['n'] += 1
    
    # Print results
    print("\n" + "="*150)
    print("SMART ANNEALING RESULTS")
    print("Condition: L_RD < L_KM-Sem AND L_RD < L_KM-Attr  (where L = H + β_e·D_e + β_a·D_a)")
    print("="*150)
    
    print(f"\n{'K':<4} {'γ':<5} {'n':<5} | {'Existing':<8} {'Annealed':<8} {'NotMet':<8} | {'L_RD':<10} {'L_KM-S':<10} {'L_KM-A':<10} | {'β_avg':<8}")
    print("-"*130)
    
    results_table = {}
    for K in target_Ks:
        for gamma in gammas:
            cell = agg.get((K, gamma))
            if cell is None or cell['n'] == 0:
                print(f"{K:<4} {gamma:<5.1f} 0")
                continue
            
            n = cell['n']
            rd_De = np.mean(cell['rd']['D_e'])
            rd_Da = np.mean(cell['rd']['D_a'])
            rd_H = np.mean(cell['rd']['H'])
            rd_beta = np.mean(cell['rd']['beta'])
            km_attr_De = np.mean(cell['km_attr']['D_e'])
            km_sem_Da = np.mean(cell['km_sem']['D_a'])
            km_sem_De = np.mean(cell['km_sem']['D_e'])
            km_attr_Da = np.mean(cell['km_attr']['D_a'])
            km_sem_H = np.mean(cell['km_sem']['H'])
            km_attr_H = np.mean(cell['km_attr']['H'])
            
            # Compute objectives: L = H + β_e·D_e + β_a·D_a
            beta_e = gamma * rd_beta
            beta_a = (1 - gamma) * rd_beta
            L_rd = rd_H + beta_e * rd_De + beta_a * rd_Da
            L_km_sem = km_sem_H + beta_e * km_sem_De + beta_a * km_sem_Da
            L_km_attr = km_attr_H + beta_e * km_attr_De + beta_a * km_attr_Da
            
            # Compute standard deviations
            rd_De_std = np.std(cell['rd']['D_e']) if n > 1 else 0
            rd_Da_std = np.std(cell['rd']['D_a']) if n > 1 else 0
            km_sem_De_std = np.std(cell['km_sem']['D_e']) if n > 1 else 0
            km_sem_Da_std = np.std(cell['km_sem']['D_a']) if n > 1 else 0
            km_attr_De_std = np.std(cell['km_attr']['D_e']) if n > 1 else 0
            km_attr_Da_std = np.std(cell['km_attr']['D_a']) if n > 1 else 0
            
            print(f"{K:<4} {gamma:<5.1f} {n:<5} | {cell['found_existing']:<8} {cell['found_annealed']:<8} {cell['condition_not_met']:<8} | "
                  f"{L_rd:<10.4f} {L_km_sem:<10.4f} {L_km_attr:<10.4f} | {rd_beta:<8.3f}")
            
            results_table[(K, gamma)] = {
                'n': n,
                'found_existing': cell['found_existing'],
                'found_annealed': cell['found_annealed'],
                'condition_not_met': cell['condition_not_met'],
                'RD_H': rd_H, 'RD_D_e': rd_De, 'RD_D_a': rd_Da, 'RD_beta': rd_beta,
                'RD_D_e_std': rd_De_std, 'RD_D_a_std': rd_Da_std,
                'KM_Sem_H': km_sem_H, 'KM_Sem_D_e': km_sem_De, 'KM_Sem_D_a': km_sem_Da,
                'KM_Sem_D_e_std': km_sem_De_std, 'KM_Sem_D_a_std': km_sem_Da_std,
                'KM_Attr_H': km_attr_H, 'KM_Attr_D_e': km_attr_De, 'KM_Attr_D_a': km_attr_Da,
                'KM_Attr_D_e_std': km_attr_De_std, 'KM_Attr_D_a_std': km_attr_Da_std,
                'L_RD': L_rd, 'L_KM_Sem': L_km_sem, 'L_KM_Attr': L_km_attr,
            }
    
    # Save CSV
    csv_path = output_dir / "rd_vs_kmeans_smart.csv"
    with open(csv_path, 'w') as f:
        f.write("K,gamma,n,found_existing,found_annealed,condition_not_met,RD_H,RD_D_e,RD_D_a,RD_beta,KM_Sem_H,KM_Sem_D_e,KM_Sem_D_a,KM_Attr_H,KM_Attr_D_e,KM_Attr_D_a\n")
        for (K, gamma), r in results_table.items():
            f.write(f"{K},{gamma},{r['n']},{r['found_existing']},{r['found_annealed']},{r['condition_not_met']},"
                    f"{r['RD_H']:.4f},{r['RD_D_e']:.4f},{r['RD_D_a']:.4f},{r['RD_beta']:.4f},"
                    f"{r['KM_Sem_H']:.4f},{r['KM_Sem_D_e']:.4f},{r['KM_Sem_D_a']:.4f},"
                    f"{r['KM_Attr_H']:.4f},{r['KM_Attr_D_e']:.4f},{r['KM_Attr_D_a']:.4f}\n")
    print(f"\nSaved CSV to: {csv_path}")
    
    # Generate LaTeX - main table
    tex = generate_latex_table(results_table, target_Ks, gammas)
    tex_path = output_dir / "rd_vs_kmeans_smart_table.tex"
    with open(tex_path, 'w') as f:
        f.write(tex)
    print(f"Saved LaTeX to: {tex_path}")
    
    # Generate LaTeX - variance table
    tex_std = generate_latex_table_with_std(results_table, target_Ks, gammas)
    tex_std_path = output_dir / "rd_vs_kmeans_variance_table.tex"
    with open(tex_std_path, 'w') as f:
        f.write(tex_std)
    print(f"Saved variance table to: {tex_std_path}")
    
    # Print variance summary
    print("\n" + "="*120)
    print("DISTORTION VARIANCE SUMMARY (mean±std)")
    print("="*120)
    print(f"{'K':<4} {'γ':<5} {'n':<5} | {'RD D_e':<15} {'RD D_a':<15} | {'KM-S D_e':<15} {'KM-S D_a':<15} | {'KM-A D_e':<15} {'KM-A D_a':<15}")
    print("-"*120)
    for K in target_Ks:
        for gamma in gammas:
            r = results_table.get((K, gamma))
            if r is None or r['n'] == 0:
                continue
            print(f"{K:<4} {gamma:<5.1f} {r['n']:<5} | "
                  f"{r['RD_D_e']:.2f}±{r.get('RD_D_e_std',0):.2f}      {r['RD_D_a']:.1f}±{r.get('RD_D_a_std',0):.1f}      | "
                  f"{r['KM_Sem_D_e']:.2f}±{r.get('KM_Sem_D_e_std',0):.2f}      {r['KM_Sem_D_a']:.1f}±{r.get('KM_Sem_D_a_std',0):.1f}      | "
                  f"{r['KM_Attr_D_e']:.2f}±{r.get('KM_Attr_D_e_std',0):.2f}      {r['KM_Attr_D_a']:.1f}±{r.get('KM_Attr_D_a_std',0):.1f}")
    
    paper_path = Path(args.results_dir).parent / "paper" / "tables" / "RD_vs_kmeans_smart.tex"
    if paper_path.parent.exists():
        with open(paper_path, 'w') as f:
            f.write(tex)
        print(f"\nSaved LaTeX to: {paper_path}")
        
        paper_var_path = Path(args.results_dir).parent / "paper" / "tables" / "RD_vs_kmeans_variance.tex"
        with open(paper_var_path, 'w') as f:
            f.write(tex_std)
        print(f"Saved variance table to: {paper_var_path}")


def generate_latex_table(results, target_Ks, gammas):
    """Generate LaTeX table comparing full RD objectives."""
    tex = r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{3pt}

\begin{tabular}{c c c | c c c c | c c c c | c c c c | c}
\toprule
& & & \multicolumn{4}{c|}{RD Clustering} & \multicolumn{4}{c|}{KM-Semantic} & \multicolumn{4}{c|}{KM-Attribution} & \\
\cmidrule(lr){4-7}\cmidrule(lr){8-11}\cmidrule(lr){12-15}
$K$ & $\gamma$ & $\beta$ & $H$ & $D_e$ & $D_a$ & $\mathcal{L}$ & $H$ & $D_e$ & $D_a$ & $\mathcal{L}$ & $H$ & $D_e$ & $D_a$ & $\mathcal{L}$ & Win\% \\
\midrule
"""
    
    for K in target_Ks:
        for gamma in gammas:
            r = results.get((K, gamma))
            if r is None or r['n'] == 0:
                continue
            
            win_pct = 100 * (r['found_existing'] + r['found_annealed']) / r['n']
            
            # Bold the best objective
            L_rd = r.get('L_RD', r['RD_H'] + gamma * r['RD_beta'] * r['RD_D_e'] + (1-gamma) * r['RD_beta'] * r['RD_D_a'])
            L_km_sem = r.get('L_KM_Sem', r['KM_Sem_H'] + gamma * r['RD_beta'] * r['KM_Sem_D_e'] + (1-gamma) * r['RD_beta'] * r['KM_Sem_D_a'])
            L_km_attr = r.get('L_KM_Attr', r['KM_Attr_H'] + gamma * r['RD_beta'] * r['KM_Attr_D_e'] + (1-gamma) * r['RD_beta'] * r['KM_Attr_D_a'])
            
            L_rd_str = f"{L_rd:.2f}"
            L_km_sem_str = f"{L_km_sem:.2f}"
            L_km_attr_str = f"{L_km_attr:.2f}"
            
            # Bold the minimum objective
            min_L = min(L_rd, L_km_sem, L_km_attr)
            if L_rd == min_L:
                L_rd_str = r"\textbf{" + L_rd_str + "}"
            if L_km_sem == min_L:
                L_km_sem_str = r"\textbf{" + L_km_sem_str + "}"
            if L_km_attr == min_L:
                L_km_attr_str = r"\textbf{" + L_km_attr_str + "}"
            
            tex += f"{K} & {gamma:.1f} & {r['RD_beta']:.2f} & "
            tex += f"{r['RD_H']:.2f} & {r['RD_D_e']:.2f} & {r['RD_D_a']:.2f} & {L_rd_str} & "
            tex += f"{r['KM_Sem_H']:.2f} & {r['KM_Sem_D_e']:.2f} & {r['KM_Sem_D_a']:.2f} & {L_km_sem_str} & "
            tex += f"{r['KM_Attr_H']:.2f} & {r['KM_Attr_D_e']:.2f} & {r['KM_Attr_D_a']:.2f} & {L_km_attr_str} & "
            tex += f"{win_pct:.0f}\\% \\\\\n"
        
        if K != target_Ks[-1]:
            tex += r"\midrule" + "\n"
    
    tex += r"""\bottomrule
\end{tabular}

\caption{RD vs.\ K-means comparison using full objective $\mathcal{L} = H + \beta_e D_e + \beta_a D_a$ (with $\beta_e = \gamma\beta$, $\beta_a = (1{-}\gamma)\beta$). Win\% = fraction where $\mathcal{L}_{\text{RD}} < \mathcal{L}_{\text{KM-Sem}}$ \textbf{and} $\mathcal{L}_{\text{RD}} < \mathcal{L}_{\text{KM-Attr}}$. Bold indicates lowest objective.}
\label{tab:rd-vs-kmeans-objective}
\end{table*}
"""
    return tex


def generate_latex_table_with_std(results, target_Ks, gammas):
    """Generate LaTeX table with mean±std for distortions."""
    tex = r"""\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{2pt}

\begin{tabular}{c c | c c | c c | c c | c}
\toprule
& & \multicolumn{2}{c|}{RD Clustering} & \multicolumn{2}{c|}{KM-Semantic} & \multicolumn{2}{c|}{KM-Attribution} & \\
\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}
$K$ & $\gamma$ & $D_e$ & $D_a$ & $D_e$ & $D_a$ & $D_e$ & $D_a$ & n \\
\midrule
"""
    
    for K in target_Ks:
        for gamma in gammas:
            r = results.get((K, gamma))
            if r is None or r['n'] == 0:
                continue
            
            # Format mean±std
            rd_De = f"{r['RD_D_e']:.2f}±{r.get('RD_D_e_std', 0):.2f}"
            rd_Da = f"{r['RD_D_a']:.1f}±{r.get('RD_D_a_std', 0):.1f}"
            km_sem_De = f"{r['KM_Sem_D_e']:.2f}±{r.get('KM_Sem_D_e_std', 0):.2f}"
            km_sem_Da = f"{r['KM_Sem_D_a']:.1f}±{r.get('KM_Sem_D_a_std', 0):.1f}"
            km_attr_De = f"{r['KM_Attr_D_e']:.2f}±{r.get('KM_Attr_D_e_std', 0):.2f}"
            km_attr_Da = f"{r['KM_Attr_D_a']:.1f}±{r.get('KM_Attr_D_a_std', 0):.1f}"
            
            tex += f"{K} & {gamma:.1f} & {rd_De} & {rd_Da} & {km_sem_De} & {km_sem_Da} & {km_attr_De} & {km_attr_Da} & {r['n']} \\\\\n"
        
        if K != target_Ks[-1]:
            tex += r"\midrule" + "\n"
    
    tex += r"""\bottomrule
\end{tabular}

\caption{Distortion comparison with standard deviations (mean±std). $D_e$: semantic distortion, $D_a$: attribution distortion.}
\label{tab:distortion-variance}
\end{table*}
"""
    return tex


if __name__ == "__main__":
    main()
