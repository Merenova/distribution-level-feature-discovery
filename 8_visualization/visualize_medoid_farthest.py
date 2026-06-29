#!/usr/bin/env python3
"""
Visualize medoid and farthest samples for each cluster with steering effects.
Shows question and cluster answer summaries at the top.
"""

import json
import numpy as np
import argparse
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_color(value, max_abs=3.0):
    """Get color based on logit difference value."""
    value = max(-max_abs, min(max_abs, value))
    norm = value / max_abs
    
    if norm > 0:
        # Positive: terracotta/rust
        r = 200
        g = int(180 - norm * 100)
        b = int(160 - norm * 120)
        return f"rgb({r}, {g}, {b})"
    else:
        # Negative: sage/muted green
        r = int(180 + norm * 60)
        g = int(190 + norm * 20)
        b = int(170 + norm * 50)
        return f"rgb({r}, {g}, {b})"


def create_visualization(
    prefix_id: str,
    question: str,
    selected_samples: dict,
    heatmap_data: dict,
    output_path: str
):
    """Create HTML visualization with medoid and farthest samples."""
    
    html_parts = []
    
    # Header - Minimal ivory/wood design
    html_parts.append(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{prefix_id}</title>
    <style>
        body {{
            font-family: 'Georgia', serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #FFFFF0;
            color: #3d3226;
            line-height: 1.6;
        }}
        h1 {{
            color: #5c4a3d;
            font-weight: normal;
            font-size: 1.4em;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #6b5344;
            font-weight: normal;
            font-size: 1.1em;
            margin-top: 40px;
            border-bottom: 1px solid #d4c4b0;
            padding-bottom: 8px;
        }}
        .question-box {{
            background: #faf6f0;
            padding: 20px;
            margin-bottom: 30px;
            border-left: 3px solid #b89f7d;
        }}
        .question {{
            font-size: 1.1em;
            color: #4a3f35;
        }}
        .note {{
            font-size: 0.8em;
            color: #8a7a6a;
            margin-bottom: 30px;
            line-height: 1.5;
        }}
        .cluster {{
            margin-bottom: 35px;
        }}
        .sample {{
            background: #faf8f4;
            padding: 15px;
            margin: 12px 0;
            border: 1px solid #e8dfd3;
        }}
        .sample-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            font-size: 0.85em;
        }}
        .sample-type {{
            padding: 2px 8px;
            font-size: 0.8em;
        }}
        .medoid {{
            background: #d4c4a8;
            color: #3d3226;
        }}
        .farthest {{
            background: #c9b896;
            color: #3d3226;
        }}
        .epsilon-section {{
            margin: 10px 0;
            padding: 10px;
            background: #fffdf8;
        }}
        .epsilon-label {{
            font-size: 0.85em;
            margin-bottom: 8px;
            color: #6b5a4a;
        }}
        .token {{
            display: inline;
            padding: 1px 3px;
            margin: 1px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        .stats {{
            font-size: 0.8em;
            color: #8a7a6a;
        }}
    </style>
</head>
<body>
    <h1>{prefix_id}</h1>
    
    <div class="question-box">
        <div class="question">{question}</div>
    </div>
    
    <div class="note">
        Distance: L2 distance from cluster centroid in embedding space.<br>
        Medoid: closest to centroid. Farthest: most distant from centroid.
    </div>
""")
    
    html_parts.append("")
    
    # Add clusters
    clusters = heatmap_data.get('clusters', {})
    
    for cid_str, cluster_data in sorted(clusters.items(), key=lambda x: int(x[0])):
        cid = int(cid_str)
        if cid not in selected_samples:
            continue
        
        sel = selected_samples[cid]
        branches = cluster_data.get('branches', [])
        
        # Find medoid and farthest branches
        medoid_branch = None
        farthest_branch = None
        
        for branch in branches:
            sample_idx = branch.get('sample_idx', -1)
            if sample_idx == sel['medoid_idx']:
                medoid_branch = branch
            elif sample_idx == sel['farthest_idx']:
                farthest_branch = branch
        
        # If we don't have sample_idx, use first and last
        if medoid_branch is None and branches:
            medoid_branch = branches[0]
        if farthest_branch is None and len(branches) > 1:
            farthest_branch = branches[-1]
        
        html_parts.append(f"""
    <div class="cluster">
        <h2>Cluster {cid}</h2>
""")
        
        # Render medoid and farthest
        for branch, sample_type, idx, dist in [
            (medoid_branch, 'medoid', sel['medoid_idx'], sel['medoid_dist']),
            (farthest_branch, 'farthest', sel['farthest_idx'], sel['farthest_dist'])
        ]:
            if branch is None:
                continue
            
            tokens = branch.get('continuation_tokens', [])
            eps_results = branch.get('epsilon_results', {})
            
            html_parts.append(f"""
        <div class="sample">
            <div class="sample-header">
                <span class="sample-type {sample_type}">{sample_type.upper()}</span>
                <span class="stats">Index: {idx} | Distance: {dist:.3f}</span>
            </div>
""")
            
            # Epsilon -1.0
            if '-1.0' in eps_results:
                eps_data = eps_results['-1.0']
                diffs = eps_data.get('diff', {}).get('centered_logit_diffs', [])
                total = eps_data.get('diff', {}).get('total_log_prob', 0)
                
                html_parts.append(f"""
            <div class="epsilon-section">
                <div class="epsilon-label">ε = −1 · Δ = {total:+.1f}</div>
                <div>
""")
                for i, tok in enumerate(tokens):
                    diff = diffs[i] if i < len(diffs) else 0
                    color = get_color(diff)
                    tok_display = tok.replace('Ġ', ' ').replace('Ċ', '↵')
                    html_parts.append(f'<span class="token" style="background:{color};" title="Δ={diff:.2f}">{tok_display}</span>')
                
                html_parts.append("""
                </div>
            </div>
""")
            
            # Epsilon +1.0
            if '1.0' in eps_results:
                eps_data = eps_results['1.0']
                diffs = eps_data.get('diff', {}).get('centered_logit_diffs', [])
                total = eps_data.get('diff', {}).get('total_log_prob', 0)
                
                html_parts.append(f"""
            <div class="epsilon-section">
                <div class="epsilon-label">ε = +1 · Δ = {total:+.1f}</div>
                <div>
""")
                for i, tok in enumerate(tokens):
                    diff = diffs[i] if i < len(diffs) else 0
                    color = get_color(diff)
                    tok_display = tok.replace('Ġ', ' ').replace('Ċ', '↵')
                    html_parts.append(f'<span class="token" style="background:{color};" title="Δ={diff:.2f}">{tok_display}</span>')
                
                html_parts.append("""
                </div>
            </div>
""")
            
            html_parts.append("        </div>")
        
        html_parts.append("    </div>")
    
    # Footer
    html_parts.append("""
</body>
</html>
""")
    
    with open(output_path, 'w') as f:
        f.write(''.join(html_parts))
    
    print(f"Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cloze', required=True)
    parser.add_argument('--config', default='beta0.75_gamma0.7')
    parser.add_argument('--data-dir', default='AmbigQA_Qwen3-8B/results/7_validation/heatmap_data')
    parser.add_argument('--output-dir', default='AmbigQA_Qwen3-8B/results/7_validation/heatmap_plots')
    args = parser.parse_args()
    
    # Load selected samples
    selected_file = os.path.join(args.data_dir, f'{args.cloze}_selected_samples.json')
    with open(selected_file) as f:
        selected_samples = {int(k): v for k, v in json.load(f).items()}
    
    # Load heatmap data
    heatmap_file = os.path.join(args.data_dir, f'{args.cloze}_{args.config}.json')
    with open(heatmap_file) as f:
        heatmap_data = json.load(f)
    
    # Get question from embeddings meta
    meta_file = f'AmbigQA_Qwen3-8B/results/4_feature_extraction/embeddings/{args.cloze}_embeddings_meta.json'
    question = "Who pays for the renovations on Holmes Next Generation?"
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            meta = json.load(f)
            question = meta.get('question', question)
    
    output_path = os.path.join(args.output_dir, f'{args.cloze}_medoid_farthest.html')
    
    create_visualization(
        args.cloze,
        question,
        selected_samples,
        heatmap_data,
        output_path
    )


if __name__ == '__main__':
    main()

