#!/usr/bin/env python3
"""Visualize token-wise logit differences as colored text (transformers-interpret style).

Creates HTML visualizations with tokens colored by their logit change.

Usage:
    python visualize_text_attribution.py --input heatmap_data.json --output visualization.html
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import html

import numpy as np


def get_color_for_value(value: float, vmin: float, vmax: float, colormap: str = "RdBu_r") -> str:
    """Get RGB color for a value using a diverging colormap."""
    if vmax == vmin:
        return "rgb(255, 255, 255)"
    
    # Normalize to [0, 1]
    normalized = (value - vmin) / (vmax - vmin)
    normalized = max(0, min(1, normalized))  # Clamp
    
    if colormap == "RdBu_r":
        # Red (high) -> White (zero) -> Blue (low)
        if normalized > 0.5:
            # Red side
            t = (normalized - 0.5) * 2
            r = int(255)
            g = int(255 * (1 - t * 0.7))
            b = int(255 * (1 - t * 0.85))
        else:
            # Blue side
            t = (0.5 - normalized) * 2
            r = int(255 * (1 - t * 0.85))
            g = int(255 * (1 - t * 0.7))
            b = int(255)
    else:
        # Default grayscale
        v = int(255 * (1 - normalized))
        r, g, b = v, v, v
    
    return f"rgb({r}, {g}, {b})"


def clean_token(token: str) -> str:
    """Clean token for display."""
    # Handle common byte-level tokens
    token = token.replace('Ġ', ' ')
    token = token.replace('Ċ', '\n')
    token = token.replace('\u010a', '\n')
    token = token.replace('\u0120', ' ')
    return token


def create_token_html(
    token: str,
    value: float,
    vmin: float,
    vmax: float,
    show_value: bool = True,
) -> str:
    """Create HTML span for a single token."""
    color = get_color_for_value(value, vmin, vmax)
    clean = html.escape(clean_token(token))
    
    # Determine text color based on background brightness
    # Simple heuristic: use white text for darker backgrounds
    normalized = (value - vmin) / (vmax - vmin) if vmax != vmin else 0.5
    text_color = "#000" if 0.3 < normalized < 0.7 else "#000"
    
    if show_value:
        title = f"Δlogit: {value:.3f}"
    else:
        title = ""
    
    return f'<span style="background-color: {color}; color: {text_color}; padding: 2px 4px; margin: 1px; border-radius: 3px; display: inline-block;" title="{title}">{clean}</span>'


def create_visualization_html(
    tokens: List[str],
    values: List[float],
    title: str = "",
    subtitle: str = "",
    symmetric_scale: bool = True,
) -> str:
    """Create HTML visualization for a sequence of tokens."""
    if not values:
        return "<p>No data</p>"
    
    # Determine color scale
    if symmetric_scale:
        abs_max = max(abs(min(values)), abs(max(values)), 0.01)
        vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax = min(values), max(values)
    
    # Create token spans
    token_spans = []
    for i, (token, value) in enumerate(zip(tokens[:len(values)], values)):
        token_spans.append(create_token_html(token, value, vmin, vmax))
    
    html_content = f"""
    <div style="margin: 10px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;">
        {f'<h4 style="margin: 0 0 10px 0; color: #333;">{html.escape(title)}</h4>' if title else ''}
        {f'<p style="margin: 0 0 10px 0; color: #666; font-size: 0.9em;">{html.escape(subtitle)}</p>' if subtitle else ''}
        <div style="line-height: 2.2; font-family: 'Courier New', monospace; font-size: 14px;">
            {''.join(token_spans)}
        </div>
        <div style="margin-top: 10px; display: flex; align-items: center; font-size: 0.8em; color: #666;">
            <span style="margin-right: 10px;">Scale:</span>
            <span style="background: rgb(100, 149, 237); color: white; padding: 2px 6px; border-radius: 3px;">−{abs(vmin):.2f}</span>
            <span style="background: linear-gradient(to right, rgb(100, 149, 237), white, rgb(255, 99, 71)); width: 100px; height: 15px; display: inline-block; margin: 0 5px; border-radius: 3px;"></span>
            <span style="background: rgb(255, 99, 71); color: white; padding: 2px 6px; border-radius: 3px;">+{abs(vmax):.2f}</span>
        </div>
    </div>
    """
    return html_content


def create_comparison_html(
    tokens: List[str],
    values_neg: List[float],
    values_pos: List[float],
    title: str = "",
) -> str:
    """Create side-by-side comparison of ε=-1 and ε=+1."""
    
    # Use same scale for both
    all_values = values_neg + values_pos
    if all_values:
        abs_max = max(abs(min(all_values)), abs(max(all_values)), 0.01)
        vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax = -1, 1
    
    # Create token spans for both
    spans_neg = []
    spans_pos = []
    
    for i, token in enumerate(tokens[:max(len(values_neg), len(values_pos))]):
        if i < len(values_neg):
            spans_neg.append(create_token_html(token, values_neg[i], vmin, vmax))
        if i < len(values_pos):
            spans_pos.append(create_token_html(token, values_pos[i], vmin, vmax))
    
    html_content = f"""
    <div style="margin: 15px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;">
        {f'<h3 style="margin: 0 0 15px 0; color: #333;">{html.escape(title)}</h3>' if title else ''}
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <h4 style="margin: 0 0 10px 0; color: #d63384;">ε = −1 (Decrease target probability)</h4>
                <div style="line-height: 2.2; font-family: 'Courier New', monospace; font-size: 13px; background: white; padding: 10px; border-radius: 5px;">
                    {''.join(spans_neg)}
                </div>
            </div>
            <div>
                <h4 style="margin: 0 0 10px 0; color: #198754;">ε = +1 (Increase target probability)</h4>
                <div style="line-height: 2.2; font-family: 'Courier New', monospace; font-size: 13px; background: white; padding: 10px; border-radius: 5px;">
                    {''.join(spans_pos)}
                </div>
            </div>
        </div>
        
        <div style="margin-top: 15px; display: flex; align-items: center; justify-content: center; font-size: 0.85em; color: #666;">
            <span style="background: rgb(100, 149, 237); color: white; padding: 3px 8px; border-radius: 3px;">−{abs(vmax):.2f}</span>
            <span style="background: linear-gradient(to right, rgb(100, 149, 237), white, rgb(255, 99, 71)); width: 150px; height: 18px; display: inline-block; margin: 0 10px; border-radius: 3px;"></span>
            <span style="background: rgb(255, 99, 71); color: white; padding: 3px 8px; border-radius: 3px;">+{abs(vmax):.2f}</span>
            <span style="margin-left: 15px; font-style: italic;">Δ centered logit (steered − original)</span>
        </div>
    </div>
    """
    return html_content


def create_full_report_html(data: Dict) -> str:
    """Create a full HTML report for the steering visualization."""
    
    cloze_id = data.get("cloze_id", "Unknown")
    config = data.get("config", "")
    question = data.get("question", data.get("prefix", "")[:100])
    beta = data.get("beta", "")
    gamma = data.get("gamma", "")
    
    sections = []
    
    # Header
    header = f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px;">
        <h1 style="margin: 0 0 10px 0;">🎯 Steering Effect Visualization</h1>
        <p style="margin: 0; font-size: 1.1em; opacity: 0.9;">Token-wise logit changes from semantic steering</p>
        <div style="margin-top: 15px; background: rgba(255,255,255,0.1); padding: 10px 15px; border-radius: 5px;">
            <strong>Cloze:</strong> {html.escape(cloze_id)} | 
            <strong>Config:</strong> β={beta}, γ={gamma} |
            <strong>Question:</strong> {html.escape(question) if question else 'N/A'}
        </div>
    </div>
    """
    sections.append(header)
    
    # Process each cluster
    clusters = data.get("clusters", {})
    
    for cluster_id in sorted(clusters.keys()):
        cluster_data = clusters[cluster_id]
        branches = cluster_data.get("branches", [])
        
        cluster_section = f"""
        <div style="margin: 25px 0;">
            <h2 style="color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px;">
                📦 Cluster {cluster_id}
                <span style="font-size: 0.7em; color: #666; font-weight: normal;">({len(branches)} branches)</span>
            </h2>
        """
        
        for branch in branches[:5]:  # Limit to 5 branches per cluster
            tokens = branch.get("continuation_tokens", [])
            branch_id = branch.get("branch_id", "?")
            
            # Get epsilon data
            eps_neg = branch.get("epsilon_results", {}).get("-1.0", {})
            eps_pos = branch.get("epsilon_results", {}).get("1.0", {})
            
            diff_neg = eps_neg.get("diff", {}).get("centered_logits", [])
            diff_pos = eps_pos.get("diff", {}).get("centered_logits", [])
            
            total_diff_neg = eps_neg.get("diff", {}).get("total_log_prob", 0)
            total_diff_pos = eps_pos.get("diff", {}).get("total_log_prob", 0)
            
            cont_text = ' '.join([clean_token(t) for t in tokens[:10]])
            title = f"Branch {branch_id}: {cont_text}..."
            
            cluster_section += create_comparison_html(tokens, diff_neg, diff_pos, title)
            
            # Add summary stats
            cluster_section += f"""
            <div style="margin: 5px 0 20px 0; padding: 10px; background: #e9ecef; border-radius: 5px; font-size: 0.85em;">
                <strong>Summary:</strong>
                ε=−1: Total Δlog_prob = <span style="color: {'#d63384' if total_diff_neg < 0 else '#198754'};">{total_diff_neg:.3f}</span> |
                ε=+1: Total Δlog_prob = <span style="color: {'#d63384' if total_diff_pos < 0 else '#198754'};">{total_diff_pos:.3f}</span>
            </div>
            """
        
        cluster_section += "</div>"
        sections.append(cluster_section)
    
    # Footer
    footer = """
    <div style="margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px; text-align: center; color: #666;">
        <p style="margin: 0;">Generated by <strong>visualize_text_attribution.py</strong></p>
        <p style="margin: 5px 0 0 0; font-size: 0.9em;">
            🔵 Blue = decreased logit | ⚪ White = no change | 🔴 Red = increased logit
        </p>
    </div>
    """
    sections.append(footer)
    
    # Combine into full HTML
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Steering Effect Visualization - {cloze_id}</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                max-width: 1400px;
                margin: 0 auto;
                padding: 20px;
                background: #f5f5f5;
            }}
        </style>
    </head>
    <body>
        {''.join(sections)}
    </body>
    </html>
    """
    
    return full_html


def main():
    parser = argparse.ArgumentParser(description="Create text attribution visualization")
    parser.add_argument("--input", type=Path, required=True, help="Input JSON file")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML file")
    args = parser.parse_args()
    
    # Load data
    with open(args.input) as f:
        data = json.load(f)
    
    # Generate HTML
    html_content = create_full_report_html(data)
    
    # Determine output path
    if args.output is None:
        args.output = args.input.with_suffix('.html')
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Visualization saved to: {args.output}")


if __name__ == "__main__":
    main()

