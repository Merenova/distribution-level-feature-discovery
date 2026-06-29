#!/usr/bin/env python3
"""Plot fixed target-cluster reasoning effects."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from reasoning_dynamics_data import build_fixed_cluster_case, load_json


DEFAULT_SELECTED_CASES = (
    "experiments/reasoning_runs/analysis/qwen3_0_6b/qualitative/"
    "selected_cases.json"
)
DEFAULT_CASE_ROOT = "experiments/reasoning_runs/analysis/qwen3_0_6b/qualitative"
DEFAULT_STAGE7_ROOT = "experiments/reasoning_runs"
DEFAULT_OUTPUT_ROOT = "experiments/reasoning_runs/analysis/qwen3_0_6b/figures"
DEFAULT_RUN_KEY = "beta0.75_gamma0.7"
DEFAULT_SWEEP_KEY = "sign_full_B5"


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_").replace(".json", "_json")


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def select_cases(
    selected_cases_path: Path,
    case_index: int | None,
    example_id: str | None,
    target_step: int | None,
    max_cases: int,
) -> list[dict[str, Any]]:
    if max_cases < 1:
        raise SystemExit("--max-cases must be >= 1")

    selected = load_json(selected_cases_path).get("selected_cases", [])
    if not isinstance(selected, list):
        raise ValueError(f"selected_cases must be a list in {selected_cases_path}")

    cases = [case for case in selected if isinstance(case, dict)]
    if example_id is not None:
        cases = [case for case in cases if str(case.get("example_id")) == example_id]
    if target_step is not None:
        cases = [
            case
            for case in cases
            if int(case.get("target_step") or -1) == target_step
        ]

    if case_index is not None:
        if case_index < 1 or case_index > len(cases):
            raise SystemExit(f"--case-index must be between 1 and {len(cases)}")
        return [cases[case_index - 1]]

    return cases[:max_cases]


def _cluster_edges(fixed_clusters: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    masses = [max(0.0, finite_float(cluster.get("mass_frac")) or 0.0) for cluster in fixed_clusters]
    total_mass = sum(masses)
    if total_mass <= 0.0:
        masses = [1.0 / len(fixed_clusters) for _ in fixed_clusters] if fixed_clusters else []
    else:
        masses = [mass / total_mass for mass in masses]

    left_edges: list[float] = []
    widths: list[float] = []
    cursor = 0.0
    for mass in masses:
        left_edges.append(cursor)
        widths.append(mass)
        cursor += mass
    return left_edges, widths


def plot_case(case_metadata: dict[str, Any], output_base: Path) -> dict[str, str]:
    fixed_clusters = case_metadata["fixed_clusters"]
    source_rows = case_metadata["per_source_effects"]
    left_edges, widths = _cluster_edges(fixed_clusters)

    height = max(2.7, 0.48 * len(source_rows) + 1.55)
    width = max(5.2, 0.55 * len(fixed_clusters) + 2.7)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)

    cmap = plt.cm.RdBu_r
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    cell_height = 0.82

    for y_index, source in enumerate(source_rows):
        effects = source.get("fixed_cluster_effects") or {}
        for cluster, x_left, cell_width in zip(fixed_clusters, left_edges, widths):
            fixed_id = str(cluster["fixed_cluster_id"])
            effect = effects.get(fixed_id, {}) if isinstance(effects, dict) else {}
            rho_s = finite_float(effect.get("rho_s")) if isinstance(effect, dict) else None
            facecolor = "#e0e0e0" if rho_s is None else cmap(norm(rho_s))
            ax.add_patch(
                Rectangle(
                    (x_left, y_index - cell_height / 2.0),
                    cell_width,
                    cell_height,
                    facecolor=facecolor,
                    edgecolor="white",
                    linewidth=1.1,
                )
            )
            if cell_width >= 0.075:
                ax.text(
                    x_left + cell_width / 2.0,
                    y_index,
                    fixed_id,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    centers = [left + width / 2.0 for left, width in zip(left_edges, widths)]
    labels = [str(cluster["fixed_cluster_id"]) for cluster in fixed_clusters]
    ax.set_xlim(0.0, 1.0 if widths else 1.0)
    ax.set_ylim(-0.6, max(0.4, len(source_rows) - 0.4))
    ax.invert_yaxis()
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.set_yticks(range(len(source_rows)))
    ax.set_yticklabels([f"i={source['source_step']}" for source in source_rows])
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title(
        f"{case_metadata['dataset']} {case_metadata['example_id']} | "
        f"target j={case_metadata['target_step']}",
        fontsize=10,
    )
    ax.tick_params(axis="both", labelsize=8)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.055, pad=0.035)
    cbar.set_label("projected rho_s")

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def write_tex_snippet(output_root: Path, outputs: list[dict[str, Any]]) -> None:
    lines = ["% Auto-generated by scripts/publication/plot_reasoning_fixed_cluster_effects.py"]
    for item in outputs:
        case = item["case"]
        label = safe_name(
            f"fixed_cluster_effects_{case['dataset']}_{case['example_id']}_j{case['target_step']}"
        ).lower()
        label = label.replace("_", "-").replace(".", "-")
        lines.extend(
            [
                r"\begin{figure}[t]",
                r"\centering",
                r"\includegraphics[width=0.62\linewidth]{" + item["figure"]["pdf"] + r"}",
                r"\caption{Fixed target-cluster effects for "
                + latex_escape(f"{case['dataset']} {case['example_id']}")
                + r", target step $j="
                + str(case["target_step"])
                + r"$. Rows are prior source steps $i<j$, columns are fixed target clusters, column widths are target-cluster mass, and color is projected $\rho_s$.}",
                r"\label{fig:" + label + r"}",
                r"\end{figure}",
                "",
            ]
        )
    (output_root / "reasoning_fixed_cluster_effect_figures.tex").write_text(
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
            f"fixed_cluster_effects_{safe_name(str(case_metadata['dataset']).lower())}_"
            f"{safe_name(str(case_metadata['example_id']))}_j"
            f"{int(case_metadata['target_step']):02d}"
        )
        figure_paths = plot_case(case_metadata, args.output_root / output_name)
        metadata_path = (args.output_root / output_name).with_suffix(".metadata.json")
        payload = {
            "figure": figure_paths,
            "case": case_metadata,
        }
        metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        outputs.append(
            {
                "figure": figure_paths,
                "metadata_json": str(metadata_path),
                "case": case_metadata,
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
    (args.output_root / "reasoning_fixed_cluster_effect_figures.json").write_text(
        json.dumps({"figures": outputs}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_tex_snippet(args.output_root, outputs)

    print(f"Rendered fixed-cluster effect figures: {len(outputs)}")
    print(f"Wrote outputs under: {args.output_root}")


if __name__ == "__main__":
    main()
