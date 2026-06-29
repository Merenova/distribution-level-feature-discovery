#!/usr/bin/env -S uv run python
"""Visualize RD curves and compare against KMeans baselines.

This script reads RD sweep results from results/5_clustering and
plots Rate (H) vs combined distortion:
    D = gamma * D_e + (1 - gamma) * D_a

It also computes two KMeans baselines per prefix:
  - semantic-only: KMeans on embeddings
  - attribution-only: KMeans on attributions

Each baseline uses K from the RD sweep config for each (beta, gamma).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
GAUSSIAN_DIR = REPO_ROOT / "5_gaussian_clustering"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GAUSSIAN_DIR))


def _import_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_cluster = _import_module("rd_cluster", GAUSSIAN_DIR / "cluster.py")
_em_loop = _import_module("rd_em_loop", GAUSSIAN_DIR / "em_loop.py")
_rd_objective = _import_module("rd_objective", GAUSSIAN_DIR / "rd_objective.py")

load_prefix_data = _cluster.load_prefix_data
weighted_median = _em_loop.weighted_median
compute_full_rd_statistics = _rd_objective.compute_full_rd_statistics


@dataclass
class GridEntry:
    beta: float
    gamma: float
    H: float
    D_e: float
    D_a: float
    K: int


def _try_import_ijson():
    try:
        import ijson  # type: ignore
    except Exception:
        return None
    return ijson


def _load_sweep_config_fast(path: Path) -> Dict[str, object]:
    ijson = _try_import_ijson()
    if ijson is None:
        data = json.loads(path.read_text())
        return data.get("sweep_config", {}) or {}

    sweep_config: Dict[str, object] = {}
    beta_values: List[float] = []
    gamma_values: List[float] = []
    with path.open("rb") as f:
        for prefix, event, value in ijson.parse(f):
            if prefix == "sweep_config.metric_a" and event in ("string", "number"):
                sweep_config["metric_a"] = value
            elif prefix == "sweep_config.normalize_dims" and event == "boolean":
                sweep_config["normalize_dims"] = value
            elif prefix == "sweep_config.K_max" and event == "number":
                sweep_config["K_max"] = value
            elif prefix == "sweep_config.K_clamp" and event == "number":
                sweep_config["K_clamp"] = value
            elif prefix == "sweep_config.d_e" and event == "number":
                sweep_config["d_e"] = value
            elif prefix == "sweep_config.d_a" and event == "number":
                sweep_config["d_a"] = value
            elif prefix == "sweep_config.beta_values.item" and event == "number":
                beta_values.append(float(value))
            elif prefix == "sweep_config.gamma_values.item" and event == "number":
                gamma_values.append(float(value))
            if prefix == "grid" and event == "start_array":
                break
    if beta_values:
        sweep_config["beta_values"] = beta_values
    if gamma_values:
        sweep_config["gamma_values"] = gamma_values
    return sweep_config


def _iter_grid_entries(path: Path) -> Iterable[GridEntry]:
    ijson = _try_import_ijson()
    if ijson is None:
        data = json.loads(path.read_text())
        for entry in data.get("grid", []) or []:
            try:
                yield GridEntry(
                    beta=float(entry.get("beta")),
                    gamma=float(entry.get("gamma")),
                    H=float(entry.get("H")),
                    D_e=float(entry.get("D_e")),
                    D_a=float(entry.get("D_a")),
                    K=int(entry.get("K", len(entry.get("components", {})) or 0)),
                )
            except Exception:
                continue
        return

    with path.open("rb") as f:
        for entry in ijson.items(f, "grid.item"):
            try:
                yield GridEntry(
                    beta=float(entry.get("beta")),
                    gamma=float(entry.get("gamma")),
                    H=float(entry.get("H")),
                    D_e=float(entry.get("D_e")),
                    D_a=float(entry.get("D_a")),
                    K=int(entry.get("K", len(entry.get("components", {})) or 0)),
                )
            except Exception:
                continue


def _weighted_mean(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = float(weights.sum())
    if total <= 0:
        return np.zeros(data.shape[1], dtype=np.float32)
    return np.sum(weights[:, None] * data, axis=0) / total


def _compute_cluster_centers(
    data: np.ndarray,
    assignments: np.ndarray,
    path_probs: np.ndarray,
    metric_a: str,
) -> Dict[int, np.ndarray]:
    centers: Dict[int, np.ndarray] = {}
    for c in sorted(set(int(x) for x in assignments)):
        mask = assignments == c
        if not np.any(mask):
            centers[c] = np.zeros(data.shape[1], dtype=np.float32)
            continue
        if metric_a == "l1":
            centers[c] = weighted_median(data[mask], path_probs[mask])
        else:
            centers[c] = _weighted_mean(data[mask], path_probs[mask])
    return centers


def _build_components(
    assignments: np.ndarray,
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    path_probs: np.ndarray,
    metric_a: str,
    mu_e_override: Optional[np.ndarray] = None,
    mu_a_override: Optional[np.ndarray] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    components: Dict[int, Dict[str, np.ndarray]] = {}
    for c in sorted(set(int(x) for x in assignments)):
        mask = assignments == c
        if mu_e_override is not None:
            mu_e = mu_e_override[c]
        else:
            mu_e = _weighted_mean(embeddings_e[mask], path_probs[mask]) if np.any(mask) else np.zeros(embeddings_e.shape[1], dtype=np.float32)
        if mu_a_override is not None:
            mu_a = mu_a_override[c]
        else:
            if np.any(mask):
                if metric_a == "l1":
                    mu_a = weighted_median(attributions_a[mask], path_probs[mask])
                else:
                    mu_a = _weighted_mean(attributions_a[mask], path_probs[mask])
            else:
                mu_a = np.zeros(attributions_a.shape[1], dtype=np.float32)
        components[c] = {"mu_e": mu_e, "mu_a": mu_a}
    return components


def _compute_rd_stats_for_assignments(
    assignments: np.ndarray,
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    path_probs: np.ndarray,
    components: Dict[int, Dict[str, np.ndarray]],
    metric_a: str,
) -> Dict[str, float]:
    stats = compute_full_rd_statistics(
        embeddings_e=embeddings_e,
        attributions_a=attributions_a,
        assignments=assignments.tolist(),
        path_probs=path_probs,
        components=components,
        beta_e=1.0,
        beta_a=1.0,
        metric_a=metric_a,
    )
    return {
        "H": float(stats["H"]),
        "D_e": float(stats["D_e"]),
        "D_a": float(stats["D_a"]),
    }


def _kmeans_assignments(data: np.ndarray, K: int, seed: int, n_init: int, max_iter: int) -> Tuple[np.ndarray, np.ndarray]:
    kmeans = KMeans(
        n_clusters=K,
        init="k-means++",
        n_init=int(n_init),
        max_iter=int(max_iter),
        random_state=int(seed),
    )
    assignments = kmeans.fit_predict(data)
    return assignments, kmeans.cluster_centers_


def _aggregate_add(
    agg: Dict[Tuple[float, float], Dict[str, float]],
    beta: float,
    gamma: float,
    H: float,
    D_e: float,
    D_a: float,
):
    key = (beta, gamma)
    if key not in agg:
        agg[key] = {"H": 0.0, "D_e": 0.0, "D_a": 0.0, "n": 0.0}
    agg[key]["H"] += H
    agg[key]["D_e"] += D_e
    agg[key]["D_a"] += D_a
    agg[key]["n"] += 1.0


def _aggregate_finalize(agg: Dict[Tuple[float, float], Dict[str, float]]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for (beta, gamma), vals in agg.items():
        n = max(vals["n"], 1.0)
        out.append(
            {
                "beta": float(beta),
                "gamma": float(gamma),
                "H": float(vals["H"] / n),
                "D_e": float(vals["D_e"] / n),
                "D_a": float(vals["D_a"] / n),
                "n": float(n),
            }
        )
    out.sort(key=lambda x: (x["beta"], x["gamma"]))
    return out


def _plot_rd_curves(
    aggregated: Dict[str, List[Dict[str, float]]],
    output_path: Path,
    title: str,
    gamma_values: List[float],
    per_gamma_panels: bool,
    gamma_wise: bool,
    per_gamma_files: bool,
):
    methods = [
        ("rd", "RD (semantic+attr)", "YlOrBr", "goldenrod", "-", "*"),
        ("kmeans_semantic", "K-Means semantic-only", "Blues", "tab:blue", "-", "s"),
        ("kmeans_attribution", "K-Means attribution-only", "Greens", "tab:green", "-", "^"),
    ]

    method_gamma_colors: Dict[str, Dict[float, Tuple[float, float, float, float]]] = {}
    # Darker for low gamma, brighter for high gamma.
    shade_levels = np.linspace(0.0, 1.0, num=max(len(gamma_values), 1)) ** 3
    for method_key, _, cmap_name, method_color, _, _ in methods:
        method_gamma_colors[method_key] = {}
        if gamma_wise:
            base_color = np.array(matplotlib.colors.to_rgba(method_color))
            if gamma_values:
                sorted_g = sorted(gamma_values)
                mid_g = sorted_g[len(sorted_g) // 2]
                max_dist = max(abs(g - mid_g) for g in sorted_g) or 1.0
            else:
                mid_g = 0.0
                max_dist = 1.0
            for g in gamma_values:
                # Middle gamma is the standard; lower gamma darker, higher gamma lighter.
                if max_dist > 0:
                    norm = (g - mid_g) / max_dist  # [-1, 1]
                else:
                    norm = 0.0
                if norm < 0:
                    matte = 1.0 + 0.4 * norm  # darker for low gamma
                else:
                    matte = 1.0 + 0.2 * norm  # lighter for high gamma
                matte = float(np.clip(matte, 0.5, 1.3))
                base = base_color.copy()
                base[:3] = np.clip(base[:3] * matte, 0.0, 1.0)
                method_gamma_colors[method_key][g] = tuple(base)
        else:
            cmap = plt.colormaps.get_cmap(cmap_name)
            for i, g in enumerate(gamma_values):
                base = np.array(cmap(shade_levels[i]))
                # Overlay transparent black for low gamma to deepen dark colors.
                black_alpha = 0.9 * (1.0 - shade_levels[i])
                base[:3] = (1.0 - black_alpha) * base[:3]
                method_gamma_colors[method_key][g] = tuple(base)

    linestyle_cycle = ["-", "--", ":", "-."]
    gamma_to_ls = {g: linestyle_cycle[i % len(linestyle_cycle)] for i, g in enumerate(gamma_values)}

    # Single plot: keep solid lines and use alpha for gamma distance to middle.
    is_single_plot = not per_gamma_panels and not per_gamma_files
    use_gamma_linestyle = gamma_wise and not per_gamma_files and not is_single_plot
    alpha_by_gamma: Dict[float, float] = {}
    mid_gamma = None
    if gamma_values:
        sorted_g = sorted(gamma_values)
        mid_gamma = sorted_g[len(sorted_g) // 2]
    if is_single_plot and gamma_values:
        sorted_g = sorted(gamma_values)
        mid = sorted_g[len(sorted_g) // 2]
        max_dist = max(abs(g - mid) for g in sorted_g) or 1.0
        for g in gamma_values:
            dist = abs(g - mid)
            alpha_by_gamma[g] = 0.3 + 0.7 * (1.0 - dist / max_dist)

    def plot_panel(ax, gamma: float, show_panel_labels: bool):
        method_points: Dict[str, Dict[float, Tuple[float, float]]] = {}
        for method_key, method_label, _, _, linestyle, marker in methods:
            entries = aggregated.get(method_key, [])
            if not entries:
                continue
            points = [e for e in entries if abs(e["gamma"] - gamma) < 1e-9]
            if not points:
                continue
            points.sort(key=lambda x: x["beta"])
            D_vals = [gamma * p["D_e"] + (1.0 - gamma) * p["D_a"] for p in points]
            H_vals = [p["H"] for p in points]
            x_vals = D_vals
            y_vals = H_vals
            ax.plot(
                x_vals,
                y_vals,
                linestyle=(gamma_to_ls[gamma] if use_gamma_linestyle else linestyle),
                marker=marker,
                color=method_gamma_colors[method_key][gamma],
                label=method_label,
                linewidth=2,
                markersize=11,
                alpha=alpha_by_gamma.get(gamma, 1.0),
            )
            # Annotate beta values for the main gamma next to K-Means semantic-only (single plot only).
            if is_single_plot and mid_gamma is not None and abs(gamma - mid_gamma) < 1e-9 and method_key == "kmeans_semantic":
                for d, h, p in zip(D_vals, H_vals, points):
                    beta_val = p.get("beta")
                    if beta_val is None:
                        continue
                    ax.annotate(
                        f"β={beta_val:g}",
                        (d, h),
                        textcoords="offset points",
                        xytext=(6, 0),
                        fontsize=13,
                        alpha=0.8,
                    )
            method_points[method_key] = {
                float(p["beta"]): (float(d), float(h)) for p, d, h in zip(points, D_vals, H_vals)
            }

        # Dotted "inter-method" connectors for the same beta across methods.
        if method_points and (mid_gamma is None or abs(gamma - mid_gamma) < 1e-9):
            common_betas = set.intersection(*[set(m.keys()) for m in method_points.values()])
            for beta in sorted(common_betas):
                coords = []
                for method_key, _, _, _, _, _ in methods:
                    if method_key in method_points:
                        coords.append(method_points[method_key][beta])
                if len(coords) >= 2:
                    D_line = [c[0] for c in coords]
                    H_line = [c[1] for c in coords]
                    ax.plot(D_line, H_line, linestyle=":", color="gray", linewidth=1, alpha=0.6)
        if show_panel_labels:
            ax.set_title(f"gamma={gamma:.2f}", fontsize=11)
            ax.set_xlabel("Combined Distortion")
            ax.set_ylabel("Rate H(C)")
        ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Precompute global axis limits for consistent scaling across per-gamma files.
    global_xlim = None
    global_ylim = None
    xticks = None
    yticks = None
    if per_gamma_files:
        all_D = []
        all_H = []
        for gamma in gamma_values:
            for method_key, _, _, _, _, _ in methods:
                entries = aggregated.get(method_key, [])
                if not entries:
                    continue
                points = [e for e in entries if abs(e["gamma"] - gamma) < 1e-9]
                if not points:
                    continue
                for p in points:
                    all_D.append(gamma * p["D_e"] + (1.0 - gamma) * p["D_a"])
                    all_H.append(p["H"])
        if all_D and all_H:
            global_xlim = (min(all_D), max(all_D))
            global_ylim = (min(all_H), max(all_H))
            # Fixed tick spacing for consistency across per-gamma files.
            x_start = np.floor(global_xlim[0]) + 0.5
            x_end = np.ceil(global_xlim[1]) + 0.5
            xticks = np.arange(x_start, x_end + 1e-9, 1.0)
            y_start = np.floor(global_ylim[0])
            y_end = np.ceil(global_ylim[1])
            yticks = np.arange(y_start, y_end + 1e-9, 0.5)

    if per_gamma_files:
        for gamma in gamma_values:
            fig, ax = plt.subplots(figsize=(10, 7))
            plot_panel(ax, gamma, show_panel_labels=True)
            if global_xlim and global_ylim:
                ax.set_xlim(global_xlim)
                ax.set_ylim(global_ylim)
            if xticks is not None and yticks is not None:
                ax.set_xticks(xticks)
                ax.set_yticks(yticks)
            ax.set_title(f"{title} (gamma={gamma:.2f})")
            from matplotlib.lines import Line2D

            method_handles = [
                Line2D([0], [0], color=method_color if gamma_wise else "black", marker=marker, linestyle="-", label=label)
                for _, label, _, method_color, _, marker in methods
            ]
            ax.legend(handles=method_handles, title="Method", loc="upper right", fontsize=9)
            plt.tight_layout()
            gamma_tag = str(gamma).replace(".", "p")
            out_path = output_path.parent / f"{output_path.stem}_gamma{gamma_tag}{output_path.suffix}"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
        return

    if per_gamma_panels:
        n = len(gamma_values)
        ncols = min(4, max(1, n))
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)
        for idx, gamma in enumerate(gamma_values):
            r = idx // ncols
            c = idx % ncols
            plot_panel(axes[r][c], gamma, show_panel_labels=True)
        # Hide unused axes
        for idx in range(n, nrows * ncols):
            r = idx // ncols
            c = idx % ncols
            axes[r][c].axis("off")
        fig.suptitle(title, fontsize=14)
        handles, labels = axes[0][0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    for gamma in gamma_values:
        plot_panel(ax, gamma, show_panel_labels=False)
    if not is_single_plot:
        ax.set_title(title)
    # Method legend (gamma colors are per-method colorbars)
    from matplotlib.lines import Line2D

    method_handles = [
        Line2D([0], [0], color=method_color if gamma_wise else "black", marker=marker, linestyle="-", label=label)
        for _, label, _, method_color, _, marker in methods
    ]
    if is_single_plot:
        ax.set_xlabel("Combined Distortion ($\\gamma D^{(e)} + (1 - \\gamma) D^{(a)}$)", fontsize=20)
        ax.set_ylabel("Rate $H(C)$", fontsize=20)
        ax.tick_params(labelsize=16)
        ax.legend(handles=method_handles, title="Method", loc="upper right", fontsize=15, title_fontsize=15)
        # Mean L_RD difference (middle gamma): show for RD vs K-Means baselines.
        if mid_gamma is not None:
            def _mean_lrd(method_key: str) -> Optional[float]:
                entries = aggregated.get(method_key, [])
                if not entries:
                    return None
                vals = []
                for e in entries:
                    if abs(e["gamma"] - mid_gamma) > 1e-9:
                        continue
                    beta = e.get("beta")
                    if beta is None:
                        continue
                    beta_e = float(beta) * float(mid_gamma)
                    beta_a = float(beta) * (1.0 - float(mid_gamma))
                    vals.append(float(e["H"]) + beta_e * float(e["D_e"]) + beta_a * float(e["D_a"]))
                if not vals:
                    return None
                return float(np.mean(vals))

            lrd_rd = _mean_lrd("rd")
            lrd_km_sem = _mean_lrd("kmeans_semantic")
            lrd_km_attr = _mean_lrd("kmeans_attribution")
            lines = []
            if lrd_rd is not None and lrd_km_sem is not None:
                lines.append(f"ΔL_RD (K-Means semantic − RD) @ γ={mid_gamma:.2f}: {lrd_km_sem - lrd_rd:.3f}")
            if lrd_rd is not None and lrd_km_attr is not None:
                lines.append(f"ΔL_RD (K-Means attribution − RD) @ γ={mid_gamma:.2f}: {lrd_km_attr - lrd_rd:.3f}")
            if lines:
                print("\n".join(lines))
    else:
        ax.legend(handles=method_handles, title="Method", loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_distortion_tradeoff(
    aggregated: Dict[str, List[Dict[str, float]]],
    output_path: Path,
    gamma_values: List[float],
    gamma_wise: bool,
    beta_filter: Optional[float],
):
    methods = [
        ("rd", "RD (semantic+attr)", "YlOrBr", "goldenrod", "-", "*"),
        ("kmeans_semantic", "K-Means semantic-only", "Blues", "tab:blue", "-", "s"),
        ("kmeans_attribution", "K-Means attribution-only", "Greens", "tab:green", "-", "^"),
    ]

    # Colors per gamma (same logic as RD plot)
    shade_levels = np.linspace(0.0, 1.0, num=max(len(gamma_values), 1)) ** 3
    method_gamma_colors: Dict[str, Dict[float, Tuple[float, float, float, float]]] = {}
    for method_key, _, cmap_name, method_color, _, _ in methods:
        method_gamma_colors[method_key] = {}
        if gamma_wise:
            base_color = np.array(matplotlib.colors.to_rgba(method_color))
            sorted_g = sorted(gamma_values)
            mid_g = sorted_g[len(sorted_g) // 2] if sorted_g else 0.0
            max_dist = max(abs(g - mid_g) for g in sorted_g) or 1.0
            for g in gamma_values:
                norm = (g - mid_g) / max_dist if max_dist > 0 else 0.0
                if norm < 0:
                    matte = 1.0 + 0.4 * norm
                else:
                    matte = 1.0 + 0.2 * norm
                matte = float(np.clip(matte, 0.5, 1.3))
                base = base_color.copy()
                base[:3] = np.clip(base[:3] * matte, 0.0, 1.0)
                method_gamma_colors[method_key][g] = tuple(base)
        else:
            cmap = plt.colormaps.get_cmap(cmap_name)
            for i, g in enumerate(gamma_values):
                base = np.array(cmap(shade_levels[i]))
                black_alpha = 0.9 * (1.0 - shade_levels[i])
                base[:3] = (1.0 - black_alpha) * base[:3]
                method_gamma_colors[method_key][g] = tuple(base)

    fig, ax = plt.subplots(figsize=(10, 7))
    
    # When beta_filter is set, sweep gamma at fixed beta (single line per method)
    # Otherwise, sweep beta at fixed gamma (one line per gamma per method)
    if beta_filter is not None:
        # Single beta, sweep gamma - one line per method
        for method_key, method_label, _, method_color, linestyle, marker in methods:
            entries = aggregated.get(method_key, [])
            if not entries:
                continue
            # Filter to specified beta and gamma values
            points = [e for e in entries if abs(e["beta"] - beta_filter) < 1e-9]
            points = [e for e in points if any(abs(e["gamma"] - g) < 1e-9 for g in gamma_values)]
            if not points:
                continue
            # Sort by gamma to connect points in gamma order
            points.sort(key=lambda x: x["gamma"])
            D_e_vals = [p["D_e"] for p in points]
            D_a_vals = [p["D_a"] for p in points]
            gammas = [p["gamma"] for p in points]
            
            # Plot the line
            ax.plot(
                D_e_vals,
                D_a_vals,
                linestyle=linestyle,
                marker=marker,
                color=method_color,
                linewidth=2,
                markersize=11,
                alpha=1.0,
                label=method_label,
            )
            # Annotate gamma values on points
            for i, (de, da, g) in enumerate(zip(D_e_vals, D_a_vals, gammas)):
                if i == 0 or i == len(gammas) - 1 or i == len(gammas) // 2:
                    ax.annotate(f"γ={g:.1f}", (de, da), textcoords="offset points", 
                               xytext=(5, 5), fontsize=9, alpha=0.7)
    else:
        # No beta filter - original behavior: sweep beta at fixed gamma
        for gamma in gamma_values:
            for method_key, method_label, _, _, linestyle, marker in methods:
                entries = aggregated.get(method_key, [])
                if not entries:
                    continue
                points = [e for e in entries if abs(e["gamma"] - gamma) < 1e-9]
                if not points:
                    continue
                points.sort(key=lambda x: x["beta"])
                D_e_vals = [p["D_e"] for p in points]
                D_a_vals = [p["D_a"] for p in points]
                ax.plot(
                    D_e_vals,
                    D_a_vals,
                    linestyle=linestyle,
                    marker=marker,
                    color=method_gamma_colors[method_key][gamma],
                    linewidth=2,
                    markersize=11,
                    alpha=1.0,
                    label=method_label,
                )

    ax.set_xlabel("Semantic Distortion $D^{(e)}$", fontsize=20)
    ax.set_ylabel("Attribution Distortion $D^{(a)}$", fontsize=20)
    if beta_filter is not None:
        ax.set_title(f"D_e vs D_a Tradeoff (β={beta_filter}, sweeping γ)", fontsize=16)
    ax.tick_params(labelsize=16)
    from matplotlib.lines import Line2D

    method_handles = [
        Line2D([0], [0], color=method_color, marker=marker, linestyle="-", label=label)
        for _, label, _, method_color, _, marker in methods
    ]
    ax.legend(handles=method_handles, title="Method", loc="upper right", fontsize=15, title_fontsize=15)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, required=True, help="Path to results directory (contains 2,3,4,5)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-prefixes", type=int, default=None, help="Limit number of prefixes (debug)")
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "sum"])
    ap.add_argument("--kmeans-seed", type=int, default=42)
    ap.add_argument("--kmeans-n-init", type=int, default=10)
    ap.add_argument("--kmeans-max-iter", type=int, default=300)
    ap.add_argument("--skip-baselines", action="store_true", help="Plot RD only (skip KMeans baselines)")
    ap.add_argument("--cache-json", type=Path, default=None, help="Optional output JSON cache")
    ap.add_argument("--baseline-cache", type=Path, default=None, help="Cache baseline results (per-prefix)")
    ap.add_argument("--gamma-values", type=str, default=None, help="Comma-separated gamma values to plot")
    ap.add_argument("--beta-values", type=str, default=None, help="Comma-separated beta values to filter (e.g. '0.5,1.0,2.0')")
    ap.add_argument("--max-gammas", type=int, default=6, help="Max gamma values to plot if not specified")
    ap.add_argument("--per-gamma-panels", action="store_true", help="Plot separate panels per gamma")
    ap.add_argument("--gamma-wise", action="store_true", help="Color by method; use linestyle to encode gamma")
    ap.add_argument("--per-gamma-files", action="store_true", help="Save separate PNG per gamma")
    ap.add_argument("--tradeoff-plot", action="store_true", help="Plot D_e vs D_a tradeoff with gamma lines")
    ap.add_argument("--tradeoff-only", action="store_true", help="Only output the D_e vs D_a tradeoff plot")
    ap.add_argument("--beta-filter", type=float, default=None, help="Filter to a single beta value for tradeoff plot")
    ap.add_argument("--n-workers", type=int, default=1, help="Number of parallel workers for K-means computation")
    args = ap.parse_args()

    results_dir = args.results_dir
    sweep_dir = results_dir / "5_clustering"
    embeddings_dir = results_dir / "4_feature_extraction" / "embeddings"
    attribution_graphs_dir = results_dir / "3_attribution_graphs"
    samples_dir = results_dir / "2_branch_sampling"

    sweep_files = sorted(sweep_dir.glob("*_sweep_results.json"))
    if args.max_prefixes is not None:
        sweep_files = sweep_files[: int(args.max_prefixes)]
    if not sweep_files:
        raise SystemExit(f"No sweep results found in {sweep_dir}")

    # Parse beta and gamma filter values
    beta_filter_set: Optional[set] = None
    if args.beta_values:
        beta_filter_set = {float(x.strip()) for x in args.beta_values.split(",") if x.strip()}
        print(f"Filtering to beta values: {sorted(beta_filter_set)}")
    
    gamma_filter_set: Optional[set] = None
    if args.gamma_values:
        gamma_filter_set = {float(x.strip()) for x in args.gamma_values.split(",") if x.strip()}
        print(f"Filtering to gamma values: {sorted(gamma_filter_set)}")

    def _matches_filter(beta: float, gamma: float) -> bool:
        """Check if (beta, gamma) passes the filter."""
        if beta_filter_set is not None:
            if not any(abs(beta - b) < 0.01 for b in beta_filter_set):
                return False
        if gamma_filter_set is not None:
            if not any(abs(gamma - g) < 0.01 for g in gamma_filter_set):
                return False
        return True

    rd_agg: Dict[Tuple[float, float], Dict[str, float]] = {}
    kmeans_sem_agg: Dict[Tuple[float, float], Dict[str, float]] = {}
    kmeans_attr_agg: Dict[Tuple[float, float], Dict[str, float]] = {}

    baseline_cache = {"meta": {}, "prefixes": {}}
    if args.baseline_cache is not None and args.baseline_cache.exists():
        try:
            baseline_cache = json.loads(args.baseline_cache.read_text())
        except Exception:
            baseline_cache = {"meta": {}, "prefixes": {}}

    logger = logging.getLogger("rd_curve_compare")
    logger.setLevel(logging.WARNING)

    def _process_prefix(sweep_file: Path) -> Tuple[str, Optional[Dict], List[Dict], List[Dict]]:
        """Process a single prefix and return results."""
        sweep_config = _load_sweep_config_fast(sweep_file)
        metric_a = str(sweep_config.get("metric_a", "l2"))
        prefix_id = sweep_file.name.replace("_sweep_results.json", "")

        entries = list(_iter_grid_entries(sweep_file))
        if not entries:
            return prefix_id, None, [], []

        # Filter entries by beta/gamma
        filtered_entries = [e for e in entries if _matches_filter(e.beta, e.gamma)]
        if not filtered_entries:
            return prefix_id, None, [], []

        # Check cache
        cache_key = f"{prefix_id}|metric_a={metric_a}|pooling={args.pooling}|seed={args.kmeans_seed}|n_init={args.kmeans_n_init}|max_iter={args.kmeans_max_iter}"
        cached = baseline_cache.get("prefixes", {}).get(cache_key)
        if cached:
            # Filter cached results too
            sem_filtered = [e for e in cached.get("kmeans_semantic", []) if _matches_filter(e["beta"], e["gamma"])]
            attr_filtered = [e for e in cached.get("kmeans_attribution", []) if _matches_filter(e["beta"], e["gamma"])]
            return prefix_id, {"entries": filtered_entries, "cached": True}, sem_filtered, attr_filtered

        if args.skip_baselines:
            return prefix_id, {"entries": filtered_entries, "cached": False, "skip": True}, [], []

        # Load data
        data = load_prefix_data(
            prefix_id=prefix_id,
            embeddings_dir=embeddings_dir,
            attribution_graphs_dir=attribution_graphs_dir,
            samples_dir=samples_dir,
            logger=logger,
            pooling=args.pooling,
            metric_a=metric_a,
        )
        embeddings_e = data["embeddings_e"]
        attributions_a = data["attributions_a"]
        path_probs = data["path_probs"]

        # Only compute K-means for K values we need
        k_values = sorted(set(int(e.K) for e in filtered_entries if e.K > 0))
        sem_cache_local: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        attr_cache_local: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        for K in k_values:
            sem_assign, sem_centers = _kmeans_assignments(
                embeddings_e, K, args.kmeans_seed, args.kmeans_n_init, args.kmeans_max_iter
            )
            sem_cache_local[K] = (sem_assign, sem_centers)

            attr_assign, attr_centers = _kmeans_assignments(
                attributions_a, K, args.kmeans_seed, args.kmeans_n_init, args.kmeans_max_iter
            )
            attr_cache_local[K] = (attr_assign, attr_centers)

        per_prefix_sem: List[Dict[str, float]] = []
        per_prefix_attr: List[Dict[str, float]] = []
        for e in filtered_entries:
            if e.K <= 0 or e.K not in sem_cache_local:
                continue

            sem_assign, sem_centers = sem_cache_local[e.K]
            sem_mu_a = _compute_cluster_centers(attributions_a, sem_assign, path_probs, metric_a)
            sem_components = _build_components(
                sem_assign, embeddings_e, attributions_a, path_probs, metric_a, mu_e_override=sem_centers, mu_a_override=sem_mu_a
            )
            sem_stats = _compute_rd_stats_for_assignments(
                sem_assign, embeddings_e, attributions_a, path_probs, sem_components, metric_a
            )
            per_prefix_sem.append(
                {"beta": e.beta, "gamma": e.gamma, "H": sem_stats["H"], "D_e": sem_stats["D_e"], "D_a": sem_stats["D_a"]}
            )

            attr_assign, attr_centers = attr_cache_local[e.K]
            attr_mu_a = _compute_cluster_centers(attributions_a, attr_assign, path_probs, metric_a)
            attr_components = _build_components(
                attr_assign, embeddings_e, attributions_a, path_probs, metric_a, mu_e_override=None, mu_a_override=attr_mu_a
            )
            attr_stats = _compute_rd_stats_for_assignments(
                attr_assign, embeddings_e, attributions_a, path_probs, attr_components, metric_a
            )
            per_prefix_attr.append(
                {"beta": e.beta, "gamma": e.gamma, "H": attr_stats["H"], "D_e": attr_stats["D_e"], "D_a": attr_stats["D_a"]}
            )

        return prefix_id, {"entries": filtered_entries, "cached": False, "cache_key": cache_key}, per_prefix_sem, per_prefix_attr

    # Process prefixes (parallel or sequential)
    if args.n_workers > 1 and not args.skip_baselines:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"Processing {len(sweep_files)} prefixes with {args.n_workers} workers...")
        
        results_list = []
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {executor.submit(_process_prefix, sf): sf for sf in sweep_files}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Prefixes"):
                try:
                    results_list.append(future.result())
                except Exception as exc:
                    print(f"Error processing prefix: {exc}")
    else:
        results_list = []
        for sweep_file in tqdm(sweep_files, desc="Prefixes"):
            results_list.append(_process_prefix(sweep_file))

    # Aggregate results
    for prefix_id, meta, per_prefix_sem, per_prefix_attr in results_list:
        if meta is None:
            continue
        
        # Aggregate RD entries
        for e in meta.get("entries", []):
            if hasattr(e, "beta"):  # GridEntry object
                _aggregate_add(rd_agg, e.beta, e.gamma, e.H, e.D_e, e.D_a)
        
        # Aggregate K-means results
        for entry in per_prefix_sem:
            _aggregate_add(kmeans_sem_agg, entry["beta"], entry["gamma"], entry["H"], entry["D_e"], entry["D_a"])
        for entry in per_prefix_attr:
            _aggregate_add(kmeans_attr_agg, entry["beta"], entry["gamma"], entry["H"], entry["D_e"], entry["D_a"])
        
        # Update cache if needed
        if args.baseline_cache is not None and not meta.get("cached", False) and not meta.get("skip", False):
            cache_key = meta.get("cache_key")
            if cache_key:
                baseline_cache.setdefault("prefixes", {})[cache_key] = {
                    "kmeans_semantic": per_prefix_sem,
                    "kmeans_attribution": per_prefix_attr,
                }

    aggregated = {
        "rd": _aggregate_finalize(rd_agg),
        "kmeans_semantic": _aggregate_finalize(kmeans_sem_agg),
        "kmeans_attribution": _aggregate_finalize(kmeans_attr_agg),
    }

    if args.cache_json is not None:
        args.cache_json.parent.mkdir(parents=True, exist_ok=True)
        args.cache_json.write_text(json.dumps(aggregated, indent=2))
    if args.baseline_cache is not None:
        args.baseline_cache.parent.mkdir(parents=True, exist_ok=True)
        args.baseline_cache.write_text(json.dumps(baseline_cache, indent=2))

    all_gammas = sorted({e["gamma"] for e in aggregated["rd"]})
    all_betas = sorted({e["beta"] for e in aggregated["rd"]})
    if all_betas:
        print(f"Beta range: {min(all_betas):.4g} to {max(all_betas):.4g} (n={len(all_betas)})")
    if all_gammas:
        print(f"Gamma range: {min(all_gammas):.4g} to {max(all_gammas):.4g} (n={len(all_gammas)})")
    
    # Use filtered gamma values for plotting
    if gamma_filter_set:
        gamma_values = sorted([g for g in gamma_filter_set if g in all_gammas or any(abs(g - ag) < 0.01 for ag in all_gammas)])
    else:
        if len(all_gammas) > args.max_gammas:
            gamma_values = list(np.linspace(min(all_gammas), max(all_gammas), num=args.max_gammas))
            gamma_values = [min(all_gammas, key=lambda x: abs(x - g)) for g in gamma_values]
            gamma_values = sorted(list(dict.fromkeys(gamma_values)))
        else:
            gamma_values = all_gammas

    if not args.tradeoff_only:
        title = "Rate-Distortion curves (RD vs KMeans baselines)"
        output_path = args.output_dir / "rd_curve_compare_kmeans.png"
        _plot_rd_curves(
            aggregated,
            output_path,
            title,
            gamma_values,
            args.per_gamma_panels,
            args.gamma_wise,
            args.per_gamma_files,
        )
        print(f"Saved plot: {output_path}")

    if args.tradeoff_plot or args.tradeoff_only:
        tradeoff_path = args.output_dir / "rd_distortion_tradeoff.png"
        _plot_distortion_tradeoff(aggregated, tradeoff_path, gamma_values, args.gamma_wise, args.beta_filter)
        print(f"Saved plot: {tradeoff_path}")


if __name__ == "__main__":
    main()
