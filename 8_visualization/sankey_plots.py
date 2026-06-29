#!/usr/bin/env -S uv run python
"""Sankey diagram visualization for cluster-to-token flow."""

from pathlib import Path
from typing import Dict

import numpy as np


def get_top_continuations_per_cluster(
    branches_data: Dict,
    assignments: np.ndarray,
    token_scores: Dict[int, Dict[int, float]]
) -> Dict[int, Dict[int, str]]:
    """
    Token-based Sankey diagrams require token-grouping metadata.
    This legacy path is disabled when that metadata is removed.

    Args:
        branches_data: Branches JSON data
        assignments: Cluster assignment per sample
        token_scores: π_{s,c} values (not used directly, but kept for API consistency)

    Returns:
        {token_id: {cluster_id: "top continuation text"}}
    """
    return {}


def plot_sankey_cluster_to_token(
    token_scores: Dict[int, Dict[int, float]],
    token_probs: Dict[int, float],
    P_bar: Dict[int, float],
    top_continuations: Dict[int, Dict[int, str]],
    output_path: Path,
    prefix_id: str,
    prefix_text: str = "",
    token_id_to_text: Dict[int, str] = None,
    top_k_tokens: int = 10,
    min_flow_weight: float = 0.01,
    figsize: tuple = (16, 12),
    logger=None
) -> Path:
    """
    Create Sankey diagram: Clusters -> Tokens with top continuation labels.

    Left nodes: Clusters (sized by P_bar_c)
    Right nodes: Tokens (sized by P(s))
    Flows: pi_{s,c} (cluster-token affinity)
    Labels: Top continuation text per (s, c) flow

    Args:
        token_scores: {token_id: {cluster_id: pi_{s,c}}}
        token_probs: {token_id: P(s)}
        P_bar: {cluster_id: P_bar_c}
        top_continuations: {token_id: {cluster_id: "argmax continuation"}}
        output_path: Where to save
        prefix_id: For title
        token_id_to_text: Token text lookup
        top_k_tokens: Number of tokens to show
        min_flow_weight: Minimum flow to display
        figsize: Figure size (width, height in inches)
        logger: Optional logger

    Returns:
        Path to saved plot
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        if logger:
            logger.error("plotly not installed. Run: uv add plotly kaleido")
        return None

    if logger:
        logger.info(f"    Building Sankey diagram...")

    # Select top tokens by probability
    sorted_tokens = sorted(token_probs.items(), key=lambda x: -x[1])[:top_k_tokens]
    token_ids = [t[0] for t in sorted_tokens]

    # Get valid cluster IDs
    cluster_ids = sorted([c for c in P_bar.keys() if c > 0])

    if not cluster_ids:
        if logger:
            logger.warning("    No valid clusters for Sankey diagram")
        return None

    # Build Sankey data
    source = []  # Cluster indices
    target = []  # Token indices (offset by n_clusters)
    value = []   # Flow weights
    labels_flow = []  # Continuation labels for hover

    n_clusters = len(cluster_ids)
    cluster_to_idx = {c: i for i, c in enumerate(cluster_ids)}
    token_to_idx = {t: i + n_clusters for i, t in enumerate(token_ids)}

    for token_id in token_ids:
        if token_id not in token_scores:
            continue
        for cluster_id in cluster_ids:
            pi_sc = token_scores[token_id].get(cluster_id, 0.0)
            if pi_sc >= min_flow_weight:
                source.append(cluster_to_idx[cluster_id])
                target.append(token_to_idx[token_id])
                value.append(pi_sc)

                # Get top continuation for this (token, cluster) pair
                cont_text = top_continuations.get(token_id, {}).get(cluster_id, "")
                if len(cont_text) > 40:
                    cont_text = cont_text[:37] + "..."
                labels_flow.append(cont_text)

    if not source:
        if logger:
            logger.warning("    No flows above threshold for Sankey diagram")
        return None

    # Node labels
    cluster_labels = [f"C{c} ({P_bar[c]:.2f})" for c in cluster_ids]
    token_labels = []
    for t in token_ids:
        if token_id_to_text and t in token_id_to_text:
            text = token_id_to_text[t].strip()
            if not text:
                text = repr(token_id_to_text.get(t, ""))
        else:
            text = str(t)
        if len(text) > 12:
            text = text[:9] + "..."
        token_labels.append(f"{text} ({token_probs[t]:.3f})")

    all_labels = cluster_labels + token_labels

    # Colors - clusters get distinct colors, tokens are gray
    cluster_colors = []
    for i in range(n_clusters):
        r = (i * 47 + 100) % 255
        g = (i * 83 + 80) % 255
        b = (i * 131 + 60) % 255
        cluster_colors.append(f"rgba({r}, {g}, {b}, 0.8)")

    token_colors = ["rgba(100, 100, 100, 0.6)"] * len(token_ids)

    # Flow colors - match source cluster color but semi-transparent
    link_colors = []
    for s in source:
        r = (s * 47 + 100) % 255
        g = (s * 83 + 80) % 255
        b = (s * 131 + 60) % 255
        link_colors.append(f"rgba({r}, {g}, {b}, 0.4)")

    # Create figure
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=all_labels,
            color=cluster_colors + token_colors,
            x=[0.01] * n_clusters + [0.4] * len(token_ids),  # Position nodes
            y=[(i + 0.5) / n_clusters for i in range(n_clusters)] +
              [(i + 0.5) / len(token_ids) for i in range(len(token_ids))]
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            label=labels_flow,
            color=link_colors,
            hovertemplate='%{label}<br>Flow: %{value:.3f}<extra></extra>'
        )
    )])

    # Build continuation annotation text (right side panel)
    # Group by token for cleaner display
    # Truncate prefix text if too long
    prefix_display = prefix_text[:60] + "..." if len(prefix_text) > 60 else prefix_text
    cont_annotations = [
        f"<b>═══ {prefix_id} ═══</b>",
        f"<i>\"{prefix_display}\"</i>" if prefix_text else "",
        "",
        "<b>Top Continuations by Token</b>",
        ""
    ]
    for i, token_id in enumerate(token_ids):
        token_text = token_id_to_text.get(token_id, str(token_id)) if token_id_to_text else str(token_id)
        token_text = token_text.strip()[:10]

        token_conts = []
        for cluster_id in cluster_ids:
            cont = top_continuations.get(token_id, {}).get(cluster_id, "")
            if cont:
                # Truncate continuation
                cont_short = cont[:35] + "..." if len(cont) > 35 else cont
                token_conts.append(f"  C{cluster_id}: {cont_short}")

        if token_conts:
            cont_annotations.append(f"<b>{token_text}</b>")
            cont_annotations.extend(token_conts[:3])  # Max 3 clusters per token
            cont_annotations.append("")  # Empty line between tokens

    # Add annotation panel on the right
    annotation_text = "<br>".join(cont_annotations[:40])  # Limit lines

    fig.add_annotation(
        x=1.02,  # Push right, outside the diagram area
        y=0.5,
        xref="paper",
        yref="paper",
        text=annotation_text,
        showarrow=False,
        font=dict(size=13, family="Courier New, monospace"),
        align="left",
        valign="middle",
        bgcolor="rgba(255,255,255,0.95)",
        bordercolor="black",
        borderwidth=2,
        borderpad=12
    )

    fig.update_layout(
        title_text=f"{prefix_id}: Cluster -> Token Flow with Top Continuations",
        font_size=13,
        width=2000,  # Wider to accommodate right annotation
        height=900,  # Fixed height
        margin=dict(l=50, r=700, t=50, b=50)  # Larger right margin for annotation
    )

    # Save as PNG
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fig.write_image(str(output_path))
        if logger:
            logger.info(f"    Saved: {output_path}")
    except Exception as e:
        if logger:
            logger.warning(f"    Could not save PNG (kaleido issue?): {e}")

    # Also save as HTML for interactivity
    html_path = output_path.with_suffix('.html')
    fig.write_html(str(html_path))
    if logger:
        logger.info(f"    Saved interactive: {html_path}")

    return output_path


def plot_sankey_with_annotations(
    token_scores: Dict[int, Dict[int, float]],
    token_probs: Dict[int, float],
    P_bar: Dict[int, float],
    top_continuations: Dict[int, Dict[int, str]],
    output_path: Path,
    prefix_id: str,
    token_id_to_text: Dict[int, str] = None,
    top_k_tokens: int = 8,
    min_flow_weight: float = 0.02,
    logger=None
) -> Path:
    """
    Create Sankey diagram with continuation text annotations on the side.

    Similar to plot_sankey_cluster_to_token but adds a text panel showing
    top continuations for major flows.

    Args:
        (same as plot_sankey_cluster_to_token)

    Returns:
        Path to saved plot
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        if logger:
            logger.error("plotly not installed")
        return None

    # Build main Sankey data (same logic as above)
    sorted_tokens = sorted(token_probs.items(), key=lambda x: -x[1])[:top_k_tokens]
    token_ids = [t[0] for t in sorted_tokens]
    cluster_ids = sorted([c for c in P_bar.keys() if c > 0])

    if not cluster_ids:
        return None

    # Collect flows and annotations
    flows = []
    for token_id in token_ids:
        if token_id not in token_scores:
            continue
        token_text = token_id_to_text.get(token_id, str(token_id)) if token_id_to_text else str(token_id)
        for cluster_id in cluster_ids:
            pi_sc = token_scores[token_id].get(cluster_id, 0.0)
            if pi_sc >= min_flow_weight:
                cont_text = top_continuations.get(token_id, {}).get(cluster_id, "")
                flows.append({
                    "cluster": cluster_id,
                    "token": token_id,
                    "token_text": token_text,
                    "pi": pi_sc,
                    "continuation": cont_text
                })

    if not flows:
        return None

    # Sort flows by pi for annotation ordering
    flows.sort(key=lambda x: -x["pi"])

    # Build annotation text with prefix header
    annotation_lines = [
        f"<b>═══ {prefix_id} ═══</b>",
        "<b>Top Flows (Cluster -> Token)</b>",
        ""
    ]
    for i, flow in enumerate(flows[:15]):  # Top 15 flows
        cont = flow["continuation"][:50] + "..." if len(flow["continuation"]) > 50 else flow["continuation"]
        line = f"C{flow['cluster']} -> {flow['token_text'][:10]} ({flow['pi']:.2f}): {cont}"
        annotation_lines.append(line)

    annotation_text = "<br>".join(annotation_lines)

    # Create figure with Sankey
    n_clusters = len(cluster_ids)
    cluster_to_idx = {c: i for i, c in enumerate(cluster_ids)}
    token_to_idx = {t: i + n_clusters for i, t in enumerate(token_ids)}

    source = [cluster_to_idx[f["cluster"]] for f in flows]
    target = [token_to_idx[f["token"]] for f in flows]
    value = [f["pi"] for f in flows]

    cluster_labels = [f"C{c}" for c in cluster_ids]
    token_labels = [token_id_to_text.get(t, str(t))[:10] if token_id_to_text else str(t)
                    for t in token_ids]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15, thickness=20,
            label=cluster_labels + token_labels
        ),
        link=dict(
            source=source, target=target, value=value
        )
    )])

    # Add annotation
    fig.add_annotation(
        xref="paper", yref="paper",
        x=1.05, y=0.5,
        text=annotation_text,
        showarrow=False,
        font=dict(size=12, family="monospace"),
        align="left",
        bordercolor="black",
        borderwidth=1,
        borderpad=8,
        bgcolor="white"
    )

    fig.update_layout(
        title_text=f"{prefix_id}: Cluster -> Token with Continuations",
        font_size=12,
        width=1600, height=800,
        margin=dict(l=50, r=550, t=50, b=50)  # Larger right margin for annotation
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path.with_suffix('.html')))

    if logger:
        logger.info(f"    Saved annotated Sankey: {output_path.with_suffix('.html')}")

    return output_path
