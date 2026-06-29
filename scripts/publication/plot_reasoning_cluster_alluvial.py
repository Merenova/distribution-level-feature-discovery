#!/usr/bin/env python3
"""Plot alluvial local-cluster reasoning dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle

from plot_reasoning_fixed_cluster_effects import (
    DEFAULT_CASE_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RUN_KEY,
    DEFAULT_SELECTED_CASES,
    DEFAULT_STAGE7_ROOT,
    DEFAULT_SWEEP_KEY,
    finite_float,
    latex_escape,
    safe_name,
    select_cases,
)
from reasoning_dynamics_data import build_fixed_cluster_case


def _partition_masses(partition: dict[str, Any]) -> dict[str, float]:
    masses: dict[str, float] = {}
    members = partition.get("members") or {}
    if not isinstance(members, dict):
        return masses
    for cluster_id, cluster_members in members.items():
        if not isinstance(cluster_members, dict):
            continue
        mass = sum(max(0.0, finite_float(value) or 0.0) for value in cluster_members.values())
        if mass > 0.0:
            masses[str(cluster_id)] = mass
    return masses


def layout_nodes(
    partitions: list[dict[str, Any]],
    gap: float = 0.018,
) -> dict[int, dict[str, dict[str, float]]]:
    layouts: dict[int, dict[str, dict[str, float]]] = {}
    for column_index, partition in enumerate(partitions):
        masses = _partition_masses(partition)
        total_mass = sum(masses.values())
        if total_mass <= 0.0:
            continue
        ordered = sorted(masses.items(), key=lambda item: (-item[1], item[0]))
        cursor = 0.0
        column: dict[str, dict[str, float]] = {}
        column_gap = gap if len(ordered) > 1 else 0.0
        usable_height = max(0.0, 1.0 - column_gap * (len(ordered) - 1))
        for cluster_id, mass in ordered:
            height = usable_height * mass / total_mass
            column[str(cluster_id)] = {
                "y0": cursor,
                "y1": cursor + height,
                "height": height,
                "mass": mass,
                "mass_frac": mass / total_mass,
            }
            cursor += height + column_gap
        layouts[column_index] = column
    return layouts


def _node_center(layout: dict[str, float] | None) -> float:
    if layout is None:
        return 0.0
    return (layout["y0"] + layout["y1"]) / 2.0


def _scaled_flow_height(node: dict[str, float], mass: float) -> float:
    if node["mass"] <= 0.0:
        return 0.0
    return node["height"] * mass / node["mass"]


def _advance_segment_cursor(
    node: dict[str, float],
    cursor: float,
    height: float,
    epsilon: float,
) -> tuple[float, float, float]:
    end = cursor + height
    overflow = max(0.0, end - node["y1"])
    if overflow > epsilon:
        raise ValueError(
            "ribbon flow exceeds visual node height: "
            f"cluster y1={node['y1']:.12g} end={end:.12g} overflow={overflow:.12g}"
        )
    if overflow > 0.0:
        end = node["y1"]
    return cursor, end, overflow


def compute_ribbon_segments(
    flows_by_pair: list[dict[str, Any]],
    layouts: dict[int, dict[str, dict[str, float]]],
    source_steps: list[int],
    epsilon: float = 1e-9,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    step_to_index = {step: index for index, step in enumerate(source_steps)}
    segments: list[dict[str, Any]] = []
    skipped_flows = 0
    max_source_overflow = 0.0
    max_target_overflow = 0.0

    for pair in flows_by_pair:
        source_index = step_to_index.get(int(pair.get("source_step") or -1))
        target_index = step_to_index.get(int(pair.get("target_step") or -1))
        if source_index is None or target_index is None:
            skipped_flows += len(pair.get("flows") or [])
            continue
        source_column = layouts.get(source_index, {})
        target_column = layouts.get(target_index, {})
        pair_flows: list[dict[str, Any]] = []
        for flow_index, flow in enumerate(pair.get("flows") or []):
            source_node = source_column.get(str(flow.get("source_cluster_id")))
            target_node = target_column.get(str(flow.get("target_cluster_id")))
            mass = max(0.0, finite_float(flow.get("mass")) or 0.0)
            if source_node is None or target_node is None or mass <= 0.0:
                skipped_flows += 1
                continue
            pair_flows.append(
                {
                    "flow_index": flow_index,
                    "source_column": source_index,
                    "target_column": target_index,
                    "source_cluster_id": str(flow.get("source_cluster_id")),
                    "target_cluster_id": str(flow.get("target_cluster_id")),
                    "mass": mass,
                    "source_node": source_node,
                    "target_node": target_node,
                }
            )

        by_source: dict[str, list[dict[str, Any]]] = {}
        by_target: dict[str, list[dict[str, Any]]] = {}
        for flow in pair_flows:
            by_source.setdefault(str(flow["source_cluster_id"]), []).append(flow)
            by_target.setdefault(str(flow["target_cluster_id"]), []).append(flow)

        segment_by_flow_index: dict[int, dict[str, Any]] = {}
        for source_cluster_id, source_flows in by_source.items():
            source_flows.sort(
                key=lambda flow: (
                    _node_center(flow["target_node"]),
                    str(flow["target_cluster_id"]),
                    int(flow["flow_index"]),
                )
            )
            source_node = source_column[source_cluster_id]
            cursor = source_node["y0"]
            for flow in source_flows:
                height = _scaled_flow_height(source_node, float(flow["mass"]))
                y0, y1, overflow = _advance_segment_cursor(source_node, cursor, height, epsilon)
                max_source_overflow = max(max_source_overflow, overflow)
                cursor = y1
                segment_by_flow_index.setdefault(int(flow["flow_index"]), dict(flow)).update(
                    {
                        "source_y0": y0,
                        "source_y1": y1,
                        "source_visual_height": y1 - y0,
                    }
                )

        for target_cluster_id, target_flows in by_target.items():
            target_flows.sort(
                key=lambda flow: (
                    _node_center(flow["source_node"]),
                    str(flow["source_cluster_id"]),
                    int(flow["flow_index"]),
                )
            )
            target_node = target_column[target_cluster_id]
            cursor = target_node["y0"]
            for flow in target_flows:
                height = _scaled_flow_height(target_node, float(flow["mass"]))
                y0, y1, overflow = _advance_segment_cursor(target_node, cursor, height, epsilon)
                max_target_overflow = max(max_target_overflow, overflow)
                cursor = y1
                segment_by_flow_index.setdefault(int(flow["flow_index"]), dict(flow)).update(
                    {
                        "target_y0": y0,
                        "target_y1": y1,
                        "target_visual_height": y1 - y0,
                    }
                )

        for flow_index in sorted(segment_by_flow_index):
            segment = segment_by_flow_index[flow_index]
            if {"source_y0", "source_y1", "target_y0", "target_y1"} <= segment.keys():
                segment.pop("source_node", None)
                segment.pop("target_node", None)
                segments.append(segment)

    diagnostics = {
        "n_segments": len(segments),
        "n_skipped_flows": skipped_flows,
        "max_source_overflow": max_source_overflow,
        "max_target_overflow": max_target_overflow,
        "epsilon": epsilon,
    }
    return segments, diagnostics


def _rho_s_for_cluster(case_metadata: dict[str, Any], source_step: int, cluster_id: str) -> float | None:
    for source in case_metadata.get("per_source_effects", []):
        if int(source.get("source_step") or -1) != source_step:
            continue
        stats = source.get("stage7_cluster_stats") or {}
        if not isinstance(stats, dict):
            return None
        cluster_stats = stats.get(str(cluster_id)) or {}
        if not isinstance(cluster_stats, dict):
            return None
        return finite_float(cluster_stats.get("centered_logit_spearman"))
    return None


def _draw_ribbon(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y0_low: float,
    y0_high: float,
    y1_low: float,
    y1_high: float,
) -> None:
    curve = (x1 - x0) * 0.48
    verts = [
        (x0, y0_low),
        (x0 + curve, y0_low),
        (x1 - curve, y1_low),
        (x1, y1_low),
        (x1, y1_high),
        (x1 - curve, y1_high),
        (x0 + curve, y0_high),
        (x0, y0_high),
        (x0, y0_low),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(
        PathPatch(
            MplPath(verts, codes),
            facecolor="#bdbdbd",
            edgecolor="none",
            alpha=0.35,
            zorder=1,
        )
    )


def plot_case(case_metadata: dict[str, Any], output_base: Path) -> tuple[dict[str, str], dict[str, Any]]:
    partitions = case_metadata.get("local_cluster_partitions") or []
    flows_by_pair = case_metadata.get("adjacent_cluster_flows") or []
    layouts = layout_nodes(partitions)
    source_steps = [int(partition.get("source_step") or 0) for partition in partitions]
    ribbon_segments, diagnostics = compute_ribbon_segments(flows_by_pair, layouts, source_steps)

    n_columns = max(1, len(partitions))
    width = max(5.8, 1.2 * n_columns + 1.8)
    fig, ax = plt.subplots(figsize=(width, 4.2), constrained_layout=True)
    cmap = plt.cm.RdBu_r
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    node_width = 0.16

    for segment in ribbon_segments:
        _draw_ribbon(
            ax,
            float(segment["source_column"]) + node_width / 2.0,
            float(segment["target_column"]) - node_width / 2.0,
            float(segment["source_y0"]),
            float(segment["source_y1"]),
            float(segment["target_y0"]),
            float(segment["target_y1"]),
        )

    for column_index, partition in enumerate(partitions):
        source_step = int(partition.get("source_step") or 0)
        for cluster_id, layout in layouts.get(column_index, {}).items():
            rho_s = _rho_s_for_cluster(case_metadata, source_step, cluster_id)
            facecolor = "#d9d9d9" if rho_s is None else cmap(norm(rho_s))
            ax.add_patch(
                Rectangle(
                    (column_index - node_width / 2.0, layout["y0"]),
                    node_width,
                    layout["height"],
                    facecolor=facecolor,
                    edgecolor="white",
                    linewidth=1.0,
                    zorder=3,
                )
            )
            if layout["height"] >= 0.075:
                ax.text(
                    column_index,
                    (layout["y0"] + layout["y1"]) / 2.0,
                    str(cluster_id),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                    zorder=4,
                )

    ax.set_xlim(-0.45, max(0.45, n_columns - 0.55))
    ax.set_ylim(-0.02, 1.02)
    ax.invert_yaxis()
    ax.set_xticks(range(len(source_steps)))
    ax.set_xticklabels([f"i={step}" for step in source_steps])
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, labelsize=9)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.set_title(
        f"{case_metadata['dataset']} {case_metadata['example_id']} | "
        f"target j={case_metadata['target_step']}",
        fontsize=10,
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.055, pad=0.035)
    cbar.set_label("local cluster rho_s")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}, diagnostics


def write_tex_snippet(output_root: Path, outputs: list[dict[str, Any]]) -> None:
    lines = ["% Auto-generated by scripts/publication/plot_reasoning_cluster_alluvial.py"]
    for item in outputs:
        case = item["case"]
        label = safe_name(
            f"alluvial_cluster_dynamics_{case['dataset']}_{case['example_id']}_j{case['target_step']}"
        ).lower()
        label = label.replace("_", "-").replace(".", "-")
        lines.extend(
            [
                r"\begin{figure}[t]",
                r"\centering",
                r"\includegraphics[width=0.72\linewidth]{" + item["figure"]["pdf"] + r"}",
                r"\caption{Alluvial local-cluster dynamics for "
                + latex_escape(f"{case['dataset']} {case['example_id']}")
                + r", target step $j="
                + str(case["target_step"])
                + r"$. Columns are prior source steps $i<j$, node heights are local cluster probability mass, ribbons are adjacent continuation-overlap mass, and node color is local $\rho_s$.}",
                r"\label{fig:" + label + r"}",
                r"\end{figure}",
                "",
            ]
        )
    (output_root / "reasoning_alluvial_cluster_figures.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def render_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    selected_cases = select_cases(
        args.selected_cases,
        args.case_index,
        args.example_id,
        args.target_step,
        args.max_cases,
    )
    if not selected_cases:
        raise SystemExit("No selected cases matched the requested filters.")

    render_params = {
        "script": Path(__file__).name,
        "run_key": args.run_key,
        "sweep_key": args.sweep_key,
        "selected_cases": str(args.selected_cases),
        "case_root": str(args.case_root),
        "stage7_root": str(args.stage7_root),
        "output_root": str(args.output_root),
        "max_cases": args.max_cases,
        "case_index": args.case_index,
        "example_id": args.example_id,
        "target_step": args.target_step,
    }
    outputs: list[dict[str, Any]] = []
    for case in selected_cases:
        case_metadata = build_fixed_cluster_case(
            case,
            case_root=args.case_root,
            stage7_root=args.stage7_root,
            run_key=args.run_key,
            sweep_key=args.sweep_key,
        )
        output_name = (
            f"alluvial_cluster_dynamics_{safe_name(str(case_metadata['dataset']).lower())}_"
            f"{safe_name(str(case_metadata['example_id']))}_j"
            f"{int(case_metadata['target_step']):02d}"
        )
        figure_paths, diagnostics = plot_case(case_metadata, args.output_root / output_name)
        metadata_path = (args.output_root / output_name).with_suffix(".metadata.json")
        payload = {
            "figure": figure_paths,
            "case": case_metadata,
            "render": {
                **render_params,
                "ribbon_diagnostics": diagnostics,
            },
        }
        metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        outputs.append(
            {
                "figure": figure_paths,
                "metadata_json": str(metadata_path),
                "case": case_metadata,
                "render": {
                    **render_params,
                    "ribbon_diagnostics": diagnostics,
                },
            }
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-cases", type=Path, default=Path(DEFAULT_SELECTED_CASES))
    parser.add_argument("--case-root", type=Path, default=Path(DEFAULT_CASE_ROOT))
    parser.add_argument("--stage7-root", type=Path, default=Path(DEFAULT_STAGE7_ROOT))
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--max-cases", type=int, default=2)
    parser.add_argument("--case-index", type=int, default=None, help="1-based index in selected_cases.json.")
    parser.add_argument("--example-id", default=None, help="Optional selected-case example id filter.")
    parser.add_argument("--target-step", type=int, default=None, help="Optional selected-case target step filter.")
    parser.add_argument("--run-key", default=DEFAULT_RUN_KEY)
    parser.add_argument("--sweep-key", default=DEFAULT_SWEEP_KEY)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    outputs = render_cases(args)
    (args.output_root / "reasoning_alluvial_cluster_figures.json").write_text(
        json.dumps({"figures": outputs}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_tex_snippet(args.output_root, outputs)

    print(f"Rendered alluvial cluster figures: {len(outputs)}")
    print(f"Wrote outputs under: {args.output_root}")


if __name__ == "__main__":
    main()
