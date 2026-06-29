#!/usr/bin/env python3
"""Render qualitative reasoning cases from selected Stage 7 candidates."""

from __future__ import annotations

import argparse
import html
import json
import re
import textwrap
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_file(case_root: Path, lane: str, suffix: str) -> Path | None:
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


def truncate(text: str, limit: int = 260) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def extract_problem_statement(prompt_text: str) -> str:
    parts = [part.strip() for part in re.split(r"\n\s*\n", prompt_text) if part.strip()]
    if not parts:
        return prompt_text.strip()

    instruction_starts = (
        "Solve the following",
        "Remember to put your answer",
    )
    candidates = [
        part
        for part in parts
        if not part.startswith(instruction_starts)
        and "last line of your response" not in part
    ]
    return max(candidates or parts, key=len).strip()


def extract_user_question(prefix: str, prompt_text: str | None = None) -> str:
    if prompt_text:
        return extract_problem_statement(prompt_text)

    match = re.search(r"<\|im_start\|>user\n(?P<user>.*?)<\|im_end\|>", prefix, re.S)
    user_text = match.group("user").strip() if match else prefix.strip()
    return extract_problem_statement(user_text)


def extract_assistant_context(prefix: str) -> str:
    match = re.search(r"<\|im_start\|>assistant\n(?P<assistant>.*)$", prefix, re.S)
    if not match:
        return ""
    text = match.group("assistant")
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def cluster_summaries(
    branches: dict[str, Any],
    clustering: dict[str, Any],
    stage7_prefix: dict[str, Any],
    max_clusters: int = 4,
) -> list[dict[str, Any]]:
    continuations = branches.get("continuations", [])
    run = next(iter(stage7_prefix.get("clustering_runs", {}).values()))
    stage7_result = run.get("results", {}).get("sign_full_B5", {})
    per_cluster = stage7_result.get("per_cluster_logit", {})

    grid = clustering.get("grid", [])
    entry = grid[0] if grid else {}
    assignments = entry.get("assignments", [])
    cluster_to_branch_ids: dict[str, list[int]] = {}
    for branch_idx, cluster_id in enumerate(assignments):
        cluster_to_branch_ids.setdefault(str(cluster_id), []).append(branch_idx)

    rows = []
    total_mass = sum(float(c.get("probability", 0.0)) for c in continuations) or 1.0
    for cluster_id, branch_ids in cluster_to_branch_ids.items():
        branch_conts = [
            continuations[idx]
            for idx in branch_ids
            if 0 <= idx < len(continuations)
        ]
        mass = sum(float(c.get("probability", 0.0)) for c in branch_conts)
        representative = max(
            branch_conts,
            key=lambda c: float(c.get("probability", 0.0)),
            default={},
        )
        stats = per_cluster.get(cluster_id, {})
        rows.append(
            {
                "cluster_id": cluster_id,
                "size": len(branch_conts),
                "mass": mass,
                "mass_frac": mass / total_mass,
                "representative": representative.get("text", ""),
                "rho_s": stats.get("centered_logit_spearman"),
                "rho": stats.get("centered_logit_corr"),
                "n_samples": stats.get("n_samples"),
            }
        )

    return sorted(rows, key=lambda r: (float(r["mass"]), int(r["size"])), reverse=True)[:max_clusters]


def load_case_prefix_artifacts(case_root: Path, lane: str, prefix_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    branches_file = find_file(case_root, lane, f"{prefix_id}_branches.json")
    clustering_file = find_file(case_root, lane, f"{prefix_id}_sweep_results.json")
    if branches_file is None or clustering_file is None:
        return None
    return load_json(branches_file), load_json(clustering_file)


def render_markdown_case(case: dict[str, Any], case_root: Path, stage7_root: Path) -> str:
    lane = case["lane"]
    dataset = case["dataset"].upper()
    lines = [
        f"## {dataset}: {case['example_id']} target step j={case['target_step']}",
        "",
        f"Score: {float(case['score']):.3f}. Source steps available: {case['source_steps_available']}.",
        "",
    ]

    first_artifacts = None
    for prefix in case["prefixes"]:
        first_artifacts = load_case_prefix_artifacts(case_root, lane, prefix["prefix_id"])
        if first_artifacts is not None:
            break
    if first_artifacts is None:
        lines.append("_Missing pulled Stage 2/5 inputs for this case._")
        return "\n".join(lines)

    branches, _ = first_artifacts
    question = extract_user_question(
        branches.get("prefix", ""),
        branches.get("reasoning_metadata", {}).get("prompt_text"),
    )
    context = extract_assistant_context(branches.get("prefix", ""))
    lines.extend(
        [
            f"**Question.** {truncate(question, 520)}",
            "",
            f"**Committed reasoning before step j.** {truncate(context, 700)}",
            "",
            "| source i | K | top target-step clusters for j | steering signal |",
            "|---:|---:|---|---|",
        ]
    )

    for prefix in sorted(case["prefixes"], key=lambda p: int(p["source_step"])):
        prefix_id = prefix["prefix_id"]
        artifacts = load_case_prefix_artifacts(case_root, lane, prefix_id)
        if artifacts is None:
            lines.append(f"| {prefix['source_step']} | - | missing pulled inputs for `{prefix_id}` | - |")
            continue
        branches, clustering = artifacts
        stage7_file = (
            stage7_root
            / lane
            / "7_validation/7c_combined_medoid/H4a_combined_medoid"
            / f"{prefix_id}_sweep_results.json"
        )
        stage7_prefix = load_json(stage7_file) if stage7_file.exists() else {"clustering_runs": {}}
        clusters = cluster_summaries(branches, clustering, stage7_prefix)
        cluster_bits = []
        for cluster in clusters:
            cluster_bits.append(
                "C{cid}: mass={mass:.2f}, n={n}, rep=\"{rep}\"".format(
                    cid=cluster["cluster_id"],
                    mass=cluster["mass_frac"],
                    n=cluster["size"],
                    rep=truncate(cluster["representative"], 120),
                )
            )
        signal = f"mean rho_s={float(prefix['mean_rho_s']):+.2f}, max |rho_s|={float(prefix['max_abs_rho_s']):.2f}"
        lines.append(
            f"| {prefix['source_step']} | {prefix['K']} | {'<br>'.join(cluster_bits)} | {signal} |"
        )

    lines.append("")
    lines.append(
        "Interpretation: different earlier source steps i induce different partitions and steering "
        "responses for the same target step j, exposing which prior reasoning contexts shape the next-step distribution."
    )
    lines.append("")
    return "\n".join(lines)


def render_latex_case(case: dict[str, Any], md_text: str) -> str:
    lines = [
        r"\paragraph{" + latex_escape(f"{case['dataset'].upper()} {case['example_id']} (j={case['target_step']})") + "}",
        r"\begin{quote}\small",
    ]
    for raw in md_text.splitlines():
        if not raw or raw.startswith("|") or raw.startswith("##"):
            continue
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", raw)
        cleaned = cleaned.replace("<br>", "; ")
        lines.append(latex_escape(truncate(cleaned, 900)) + r"\\")
    lines.append(r"\end{quote}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True)
    parser.add_argument("--stage7-root", type=Path, default=Path("experiments/reasoning_runs"))
    parser.add_argument("--max-cases", type=int, default=2)
    args = parser.parse_args()

    selected_path = args.case_root / "selected_cases.json"
    selected = load_json(selected_path).get("selected_cases", [])
    cases = selected[: args.max_cases]

    md_chunks = ["# Reasoning Qualitative Cases", ""]
    tex_chunks = [
        "% Auto-generated by scripts/publication/render_reasoning_qual_cases.py",
        r"\section*{Reasoning Qualitative Cases}",
        "",
    ]
    rendered_cases = []
    for case in cases:
        md = render_markdown_case(case, args.case_root, args.stage7_root)
        md_chunks.append(md)
        tex_chunks.append(render_latex_case(case, md))
        rendered_cases.append({"case": case, "markdown": md})

    args.case_root.mkdir(parents=True, exist_ok=True)
    (args.case_root / "reasoning_qualitative_cases.md").write_text(
        "\n".join(md_chunks),
        encoding="utf-8",
    )
    (args.case_root / "reasoning_qualitative_cases.tex").write_text(
        "\n\n".join(tex_chunks) + "\n",
        encoding="utf-8",
    )
    (args.case_root / "rendered_cases.json").write_text(
        json.dumps({"rendered_cases": rendered_cases}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Rendered cases: {len(cases)}")
    print(f"Wrote: {args.case_root / 'reasoning_qualitative_cases.md'}")
    print(f"Wrote: {args.case_root / 'reasoning_qualitative_cases.tex'}")
    print(f"Wrote: {args.case_root / 'rendered_cases.json'}")


if __name__ == "__main__":
    main()
