#!/usr/bin/env python3
"""Plot reasoning step-pair heatmaps for Stage 7 results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


DEFAULT_PAIR_CSV = (
    "experiments/reasoning_runs/analysis/qwen3_0_6b/quantitative/"
    "reasoning_stage7_pair_metrics.csv"
)
DEFAULT_SELECTED_CASES = (
    "experiments/reasoning_runs/analysis/qwen3_0_6b/qualitative/"
    "selected_cases.json"
)
DEFAULT_CASE_ROOT = "experiments/reasoning_runs/analysis/qwen3_0_6b/qualitative"
DEFAULT_OUTPUT_ROOT = "experiments/reasoning_runs/analysis/qwen3_0_6b/figures"


@dataclass(frozen=True)
class PairMetric:
    dataset: str
    prefix_id: str
    example_id: str
    source_step: int
    target_step: int
    rho_s: float
    abs_rho_s: float
    rho: float


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def load_pair_metrics(path: Path) -> list[PairMetric]:
    rows: list[PairMetric] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            rho_s = finite_float(raw.get("centered_logit_spearman_mean"))
            abs_rho_s = finite_float(raw.get("centered_logit_spearman_abs_mean"))
            rho = finite_float(raw.get("centered_logit_corr_mean"))
            if rho_s is None or abs_rho_s is None or rho is None:
                continue
            rows.append(
                PairMetric(
                    dataset=str(raw["dataset"]).upper(),
                    prefix_id=str(raw["prefix_id"]),
                    example_id=str(raw["example_id"]),
                    source_step=int(raw["source_step"]),
                    target_step=int(raw["target_step"]),
                    rho_s=rho_s,
                    abs_rho_s=abs_rho_s,
                    rho=rho,
                )
            )
    return rows


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def normalized_example_id(dataset: str, example_id: str) -> str:
    out = example_id.strip()
    if dataset.upper() == "MATH500" and out.startswith("math500_"):
        out = out[len("math500_") :]
    if dataset.upper() == "GSM8K" and out.startswith("gsm8k_"):
        out = out[len("gsm8k_") :]
        out = "gsm8k_" + out
    return out


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_").replace(".json", "_json")


def metric_value(row: PairMetric, metric: str) -> float:
    if metric == "rho_s":
        return row.rho_s
    if metric == "abs_rho_s":
        return row.abs_rho_s
    if metric == "rho":
        return row.rho
    raise ValueError(f"unsupported metric: {metric}")


def aggregate_pairs(rows: list[PairMetric], metric: str) -> dict[tuple[int, int], float]:
    grouped: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        grouped.setdefault((row.source_step, row.target_step), []).append(
            metric_value(row, metric)
        )
    return {
        pair: float(np.mean(values))
        for pair, values in grouped.items()
        if values
    }


def build_matrix(pair_values: dict[tuple[int, int], float]) -> np.ndarray:
    if not pair_values:
        return np.empty((0, 0))
    max_step = max(max(i, j) for i, j in pair_values)
    matrix = np.full((max_step, max_step), np.nan, dtype=float)
    for (source_step, target_step), value in pair_values.items():
        if source_step < target_step:
            matrix[target_step - 1, source_step - 1] = value
    return matrix


def find_case_file(case_root: Path, lane: str, suffix: str) -> Path | None:
    search_roots = [
        case_root / "selected_case_inputs" / lane,
        case_root / "selected_case_inputs",
        Path("experiments/reasoning_runs") / lane,
    ]
    for root in search_roots:
        if not root.exists():
            continue
        matches = sorted(root.rglob(f"*{suffix}"))
        if matches:
            return matches[0]
    return None


def extract_problem_statement(prompt_text: str) -> str:
    parts = [part.strip() for part in re.split(r"\n\s*\n", prompt_text) if part.strip()]
    if not parts:
        return prompt_text.strip()
    candidates = [
        part
        for part in parts
        if not part.startswith(("Solve the following", "Remember to put your answer"))
        and "last line of your response" not in part
    ]
    return max(candidates or parts, key=len).strip()


def case_question(case: dict[str, Any], case_root: Path) -> str:
    prefixes = case.get("prefixes") or []
    if not prefixes:
        return ""
    first_prefix = prefixes[0].get("prefix_id")
    lane = str(case.get("lane", ""))
    branch_file = find_case_file(case_root, lane, f"{first_prefix}_branches.json")
    if branch_file is None:
        return ""
    data = load_json(branch_file)
    prompt_text = (data.get("reasoning_metadata") or {}).get("prompt_text")
    if isinstance(prompt_text, str):
        return extract_problem_statement(prompt_text)
    prefix = str(data.get("prefix", ""))
    match = re.search(r"<\|im_start\|>user\n(?P<user>.*?)<\|im_end\|>", prefix, re.S)
    return extract_problem_statement(match.group("user") if match else prefix)


def draw_pair_heatmap(
    pair_values: dict[tuple[int, int], float],
    signed_values: dict[tuple[int, int], float],
    output_base: Path,
    title: str,
    subtitle: str = "",
    highlighted_pairs: list[tuple[int, int]] | None = None,
    annotate_top: int = 0,
    vmax: float = 1.0,
) -> dict[str, str] | None:
    matrix = build_matrix(pair_values)
    if matrix.size == 0:
        return None

    n_steps = matrix.shape[0]
    fig_width = max(6.5, min(10.5, 0.55 * n_steps + 3.0))
    fig_height = max(5.8, min(9.0, 0.55 * n_steps + 2.7))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    cmap = plt.cm.Reds.copy()
    cmap.set_bad("white")
    image = ax.imshow(matrix, origin="upper", cmap=cmap, vmin=0.0, vmax=vmax)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"mean $|\rho_s|$")

    for idx in range(n_steps):
        ax.add_patch(
            Rectangle(
                (idx - 0.5, idx - 0.5),
                1.0,
                1.0,
                facecolor="#5f1111",
                edgecolor="#5f1111",
                linewidth=0.0,
                zorder=4,
            )
        )

    ax.set_xticks(range(n_steps))
    ax.set_yticks(range(n_steps))
    ax.set_xticklabels([str(i) for i in range(1, n_steps + 1)])
    ax.set_yticklabels([str(i) for i in range(1, n_steps + 1)])
    ax.set_xlabel(r"Source reasoning step $i$")
    ax.set_ylabel(r"Target reasoning step $j$")
    ax.set_xlim(-0.5, n_steps - 0.5)
    ax.set_ylim(n_steps - 0.5, -0.5)

    ax.set_xticks(np.arange(-0.5, n_steps, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_steps, 1), minor=True)
    ax.grid(which="minor", color="#f0dada", linestyle="-", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)

    all_pairs = [
        (pair, value)
        for pair, value in pair_values.items()
        if pair[0] < pair[1] and math.isfinite(value)
    ]
    all_pairs.sort(key=lambda item: item[1], reverse=True)
    if highlighted_pairs is None:
        highlighted_pairs = [pair for pair, _ in all_pairs[:annotate_top]]
    else:
        highlighted_pairs = list(dict.fromkeys(highlighted_pairs))
        if annotate_top:
            for pair, _ in all_pairs:
                if len(highlighted_pairs) >= annotate_top:
                    break
                if pair not in highlighted_pairs:
                    highlighted_pairs.append(pair)

    colors = ["#2ca25f", "#3182bd", "#756bb1", "#e6550d", "#31a354", "#dd3497", "#636363"]
    stacked_target = None
    if len(highlighted_pairs) > 2:
        target_steps = {target_step for _, target_step in highlighted_pairs}
        if len(target_steps) == 1:
            stacked_target = next(iter(target_steps))
    for rank, (source_step, target_step) in enumerate(highlighted_pairs):
        if (source_step, target_step) not in pair_values:
            continue
        x = source_step - 1
        y = target_step - 1
        color = colors[rank % len(colors)]
        ax.plot([x, x], [x, y], color="black", linestyle="--", linewidth=1.0, alpha=0.75)
        ax.plot([x, y], [y, y], color="black", linestyle="--", linewidth=1.0, alpha=0.75)
        ax.scatter([x], [y], marker="s", s=70, color=color, edgecolor="white", linewidth=0.8, zorder=5)
        signed = signed_values.get((source_step, target_step))
        label = f"({source_step}, {target_step})"
        if signed is not None:
            label += rf" $\rho_s={signed:+.2f}$"
        if stacked_target is not None:
            x_text = min(n_steps - 1.1, stacked_target + 1.0)
            y_start = max(0.35, stacked_target - 3.4)
            y_text = min(n_steps - 0.8, y_start + 0.55 * rank)
        else:
            x_text = x + (0.35 if source_step <= n_steps * 0.55 else -2.2)
            y_text = y + (-0.8 if target_step > 2 else 0.9)
        ax.annotate(
            label,
            xy=(x, y),
            xytext=(x_text, y_text),
            color=color,
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.1},
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85},
            zorder=6,
        )

    wrapped_title = "\n".join(textwrap.wrap(title, width=76))
    if subtitle:
        wrapped_title += "\n" + subtitle
    ax.set_title(wrapped_title, fontsize=12, fontstyle="italic", pad=10)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    pdf_path = output_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path), "title": title, "subtitle": subtitle}


def plot_overall(rows: list[PairMetric], output_root: Path, metric: str) -> dict[str, str] | None:
    values = aggregate_pairs(rows, metric)
    signed = aggregate_pairs(rows, "rho_s")
    return draw_pair_heatmap(
        values,
        signed,
        output_root / f"overall_step_pair_{metric}",
        title="Overall reasoning step-pair validation",
        subtitle="Qwen3-0.6B, GSM8K + MATH500",
        annotate_top=0,
    )


def plot_cases(
    rows: list[PairMetric],
    selected_cases_path: Path,
    case_root: Path,
    output_root: Path,
    metric: str,
    max_cases: int,
) -> list[dict[str, str]]:
    selected = load_json(selected_cases_path).get("selected_cases", [])[:max_cases]
    outputs: list[dict[str, str]] = []
    for case in selected:
        dataset = str(case["dataset"]).upper()
        example_id = normalized_example_id(dataset, str(case["example_id"]))
        case_rows = [
            row
            for row in rows
            if row.dataset == dataset
            and normalized_example_id(dataset, row.example_id) == example_id
        ]
        if not case_rows:
            continue
        values = aggregate_pairs(case_rows, metric)
        signed = aggregate_pairs(case_rows, "rho_s")
        target_step = int(case["target_step"])
        highlighted_pairs = [
            (int(prefix["source_step"]), target_step)
            for prefix in case.get("prefixes", [])
        ]
        question = case_question(case, case_root)
        title = question or f"{dataset} {example_id}"
        output_name = (
            f"qual_{dataset.lower()}_{safe_name(example_id)}_j{target_step:02d}_"
            f"step_pair_blocks"
        )
        rendered = draw_pair_heatmap(
            values,
            signed,
            output_root / output_name,
            title=title,
            subtitle=(
                f"{dataset} {example_id}: highlighted target step j={target_step}"
            ),
            highlighted_pairs=highlighted_pairs,
            annotate_top=0,
        )
        if rendered is not None:
            rendered["dataset"] = dataset
            rendered["example_id"] = example_id
            rendered["target_step"] = str(target_step)
            outputs.append(rendered)
    return outputs


def write_tex_snippet(output_root: Path, overall: dict[str, str] | None, cases: list[dict[str, str]]) -> None:
    lines = [
        "% Auto-generated by scripts/publication/plot_reasoning_pairwise_blocks.py",
    ]
    if overall is not None:
        lines.extend(
            [
                r"\begin{figure}[t]",
                r"\centering",
                r"\includegraphics[width=0.62\linewidth]{" + overall["pdf"] + r"}",
                r"\caption{Overall reasoning step-pair validation heatmap. Rows are target steps $j$, columns are prior source steps $i$, and cells show mean $|\rho_s|$ for explicit pairs $(i,j)$.}",
                r"\label{fig:reasoning-overall-step-pair-heatmap}",
                r"\end{figure}",
                "",
            ]
        )
    for index, case in enumerate(cases, start=1):
        label_id = safe_name(f"{case['dataset']}_{case['example_id']}_j{case['target_step']}").lower()
        label_id = label_id.replace("_", "-").replace(".", "-")
        lines.extend(
            [
                r"\begin{figure}[t]",
                r"\centering",
                r"\includegraphics[width=0.72\linewidth]{" + case["pdf"] + r"}",
                r"\caption{Reasoning step-pair block visualization for "
                + case["dataset"]
                + " "
                + case["example_id"].replace("_", r"\_")
                + r". Highlighted cells fix the target step $j="
                + case["target_step"]
                + r"$ and compare prior source steps $i<j$.}",
                r"\label{fig:reasoning-step-pair-" + label_id + r"}",
                r"\end{figure}",
                "",
            ]
        )
    (output_root / "reasoning_pairwise_figures.tex").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    (output_root / "reasoning_pairwise_figures.json").write_text(
        json.dumps({"overall": overall, "cases": cases}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-metrics-csv", type=Path, default=Path(DEFAULT_PAIR_CSV))
    parser.add_argument("--selected-cases", type=Path, default=Path(DEFAULT_SELECTED_CASES))
    parser.add_argument("--case-root", type=Path, default=Path(DEFAULT_CASE_ROOT))
    parser.add_argument("--output-root", type=Path, default=Path(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--metric", choices=["abs_rho_s", "rho_s", "rho"], default="abs_rho_s")
    parser.add_argument("--max-cases", type=int, default=2)
    args = parser.parse_args()

    rows = load_pair_metrics(args.pair_metrics_csv)
    if not rows:
        raise SystemExit(f"no pair metrics found in {args.pair_metrics_csv}")

    overall = plot_overall(rows, args.output_root, args.metric)
    cases = plot_cases(
        rows,
        args.selected_cases,
        args.case_root,
        args.output_root,
        args.metric,
        args.max_cases,
    )
    write_tex_snippet(args.output_root, overall, cases)

    print(f"Loaded pair rows: {len(rows)}")
    print(f"Wrote figures under: {args.output_root}")


if __name__ == "__main__":
    main()
