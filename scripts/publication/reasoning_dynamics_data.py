"""Data helpers for fixed-cluster reasoning dynamics figures."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def continuation_key(row: dict[str, Any]) -> str:
    token_ids = row.get("token_ids")
    if isinstance(token_ids, list) and token_ids:
        return "tok:" + ",".join(str(x) for x in token_ids)
    text = re.sub(r"\s+", " ", str(row.get("text", ""))).strip()
    return "txt:" + text


def _continuation_mass(row: dict[str, Any]) -> float:
    return finite_float(row.get("probability")) or 0.0


def _continuation_occurrence_rows(continuations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    identity_counts: dict[str, int] = {}
    for row in continuations:
        key = continuation_key(row)
        identity_counts[key] = identity_counts.get(key, 0) + 1

    seen: dict[str, int] = {}
    occurrences: list[dict[str, Any]] = []
    for index, row in enumerate(continuations):
        identity_key = continuation_key(row)
        occurrence_index = seen.get(identity_key, 0)
        seen[identity_key] = occurrence_index + 1
        occurrence_key = (
            identity_key
            if identity_counts[identity_key] == 1
            else f"{identity_key}#occ:{occurrence_index}"
        )
        occurrences.append(
            {
                "index": index,
                "identity_key": identity_key,
                "occurrence_key": occurrence_key,
                "mass": _continuation_mass(row),
                "row": row,
            }
        )
    return occurrences


def _identity_mass(occurrences: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for occurrence in occurrences:
        key = str(occurrence["identity_key"])
        out[key] = out.get(key, 0.0) + float(occurrence["mass"])
    return out


def _cluster_sort_value(item: tuple[str, list[tuple[int, dict[str, Any]]]]) -> tuple[float, int, str]:
    cluster_id, indexed_rows = item
    mass = sum(
        (finite_float(row.get("mass")) if "mass" in row else _continuation_mass(row)) or 0.0
        for _, row in indexed_rows
    )
    first_index = min((index for index, _ in indexed_rows), default=10**12)
    return (-mass, first_index, cluster_id)


def build_fixed_clusters_from_reference(
    continuations: list[dict[str, Any]],
    assignments: list[Any],
) -> list[dict[str, Any]]:
    total_mass = sum(_continuation_mass(row) for row in continuations) or 1.0
    occurrence_rows = _continuation_occurrence_rows(continuations)
    by_cluster: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, cluster_id in enumerate(assignments):
        if 0 <= index < len(occurrence_rows):
            by_cluster.setdefault(str(cluster_id), []).append((index, occurrence_rows[index]))

    fixed: list[dict[str, Any]] = []
    for fixed_index, (cluster_id, indexed_rows) in enumerate(
        sorted(by_cluster.items(), key=_cluster_sort_value),
        start=1,
    ):
        occurrence_rows_for_cluster = [row for _, row in indexed_rows]
        rows = [occurrence["row"] for occurrence in occurrence_rows_for_cluster]
        mass = sum(float(occurrence["mass"]) for occurrence in occurrence_rows_for_cluster)
        rows_sorted = sorted(rows, key=_continuation_mass, reverse=True)
        identity_mass = _identity_mass(occurrence_rows_for_cluster)
        fixed.append(
            {
                "fixed_cluster_id": f"F{fixed_index}",
                "reference_cluster_id": cluster_id,
                "mass": mass,
                "mass_frac": mass / total_mass,
                "continuation_keys": list(identity_mass),
                "continuation_key_mass": identity_mass,
                "continuation_occurrence_keys": [
                    str(occurrence["occurrence_key"])
                    for occurrence in occurrence_rows_for_cluster
                ],
                "representative_continuation": rows_sorted[0].get("text", "") if rows_sorted else "",
                "assigned_continuations": rows_sorted,
            }
        )
    return fixed


def local_cluster_members(
    continuations: list[dict[str, Any]],
    assignments: list[Any],
) -> dict[str, dict[str, float]]:
    members: dict[str, dict[str, float]] = {}
    occurrence_rows = _continuation_occurrence_rows(continuations)
    for index, cluster_id in enumerate(assignments):
        if 0 <= index < len(occurrence_rows):
            occurrence = occurrence_rows[index]
            key = str(occurrence["occurrence_key"])
            mass = float(occurrence["mass"])
            cluster_members = members.setdefault(str(cluster_id), {})
            cluster_members[key] = cluster_members.get(key, 0.0) + mass
    return members


def _local_cluster_identity_members(
    continuations: list[dict[str, Any]],
    assignments: list[Any],
) -> dict[str, dict[str, float]]:
    members: dict[str, dict[str, float]] = {}
    occurrence_rows = _continuation_occurrence_rows(continuations)
    for index, cluster_id in enumerate(assignments):
        if 0 <= index < len(occurrence_rows):
            occurrence = occurrence_rows[index]
            key = str(occurrence["identity_key"])
            mass = float(occurrence["mass"])
            cluster_members = members.setdefault(str(cluster_id), {})
            cluster_members[key] = cluster_members.get(key, 0.0) + mass
    return members


def project_local_effects_to_fixed_clusters(
    fixed_clusters: list[dict[str, Any]],
    continuations: list[dict[str, Any]],
    local_assignments: list[Any],
    local_cluster_stats: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    total_mass = sum(_continuation_mass(row) for row in continuations) or 1.0
    members = local_cluster_members(continuations, local_assignments)
    key_to_local_clusters: dict[str, list[tuple[str, float]]] = {}
    for cluster_id, cluster_members in members.items():
        for key, mass in cluster_members.items():
            key_to_local_clusters.setdefault(key, []).append((cluster_id, mass))

    projected: dict[str, dict[str, Any]] = {}
    for fixed in fixed_clusters:
        weighted_sum = 0.0
        overlap_mass = 0.0
        metric_overlap_mass = 0.0
        overlaps: dict[str, float] = {}
        metric_overlaps: dict[str, float] = {}
        fixed_keys = fixed.get("continuation_occurrence_keys") or fixed.get("continuation_keys", [])
        for key in fixed_keys:
            local_clusters = key_to_local_clusters.get(str(key), [])
            if not local_clusters:
                continue
            for local_cluster, mass in local_clusters:
                overlap_mass += mass
                overlaps[str(local_cluster)] = overlaps.get(str(local_cluster), 0.0) + mass
                stats = local_cluster_stats.get(str(local_cluster), {})
                rho_s = finite_float(stats.get("centered_logit_spearman"))
                if rho_s is None:
                    continue
                weighted_sum += mass * rho_s
                metric_overlap_mass += mass
                metric_overlaps[str(local_cluster)] = (
                    metric_overlaps.get(str(local_cluster), 0.0) + mass
                )
        projected[str(fixed["fixed_cluster_id"])] = {
            "rho_s": weighted_sum / metric_overlap_mass if metric_overlap_mass else None,
            "overlap_mass": overlap_mass,
            "overlap_mass_frac": overlap_mass / total_mass,
            "metric_overlap_mass": metric_overlap_mass,
            "metric_overlap_mass_frac": metric_overlap_mass / total_mass,
            "local_cluster_overlap_mass": overlaps,
            "metric_local_cluster_overlap_mass": metric_overlaps,
        }
    return projected


def compute_adjacent_cluster_flows(
    left_members: dict[str, dict[str, float]],
    right_members: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    key_to_right_clusters: dict[str, list[tuple[str, float]]] = {}
    for cluster_id, members in right_members.items():
        for key, mass in members.items():
            key_to_right_clusters.setdefault(key, []).append((cluster_id, mass))

    flows: dict[tuple[str, str], float] = {}
    for left_cluster, members in left_members.items():
        for key, left_mass in members.items():
            right_clusters = key_to_right_clusters.get(key, [])
            if not right_clusters:
                continue
            for right_cluster, right_mass in right_clusters:
                flow_key = (str(left_cluster), str(right_cluster))
                overlap_mass = min(left_mass, right_mass)
                flows[flow_key] = flows.get(flow_key, 0.0) + overlap_mass
    return [
        {"source_cluster_id": source, "target_cluster_id": target, "mass": mass}
        for (source, target), mass in sorted(flows.items())
    ]


def find_case_file(case_root: Path, lane: str, suffix: str) -> Path | None:
    for root in [
        case_root / "selected_case_inputs" / lane,
        case_root / "selected_case_inputs",
        Path("experiments/reasoning_runs") / lane,
    ]:
        if not root.exists():
            continue
        matches = sorted(root.rglob(f"*{suffix}"))
        if matches:
            return matches[0]
    return None


def stage7_file_path(stage7_root: Path, lane: str, prefix_id: str) -> Path:
    return (
        stage7_root
        / lane
        / "7_validation/7c_combined_medoid/H4a_combined_medoid"
        / f"{prefix_id}_sweep_results.json"
    )


def stage7_cluster_stats(
    stage7_prefix: dict[str, Any],
    run_key: str,
    sweep_key: str,
) -> dict[str, dict[str, Any]]:
    runs = stage7_prefix.get("clustering_runs", {})
    run = runs.get(run_key) if isinstance(runs, dict) else None
    if not isinstance(run, dict):
        return {}
    results = run.get("results")
    result = results.get(sweep_key) if isinstance(results, dict) else None
    if not isinstance(result, dict):
        return {}
    per_cluster = result.get("per_cluster_logit") or {}
    return {str(key): value for key, value in per_cluster.items() if isinstance(value, dict)}


def choose_reference_prefix(prefixes: list[dict[str, Any]]) -> dict[str, Any]:
    def score(prefix: dict[str, Any]) -> tuple[int, float, float, float, int, str]:
        k = int(prefix.get("K") or 0)
        readable = 1 if 2 <= k <= 6 else 0
        mean_abs = float(prefix.get("mean_abs_rho_s") or 0.0)
        coverage = float(
            prefix.get("continuation_coverage")
            or prefix.get("total_continuation_coverage")
            or prefix.get("coverage")
            or prefix.get("n_continuations")
            or 0.0
        )
        max_abs = float(prefix.get("max_abs_rho_s") or 0.0)
        source_step = -int(prefix.get("source_step") or 0)
        prefix_id = str(prefix.get("prefix_id") or "")
        return (readable, mean_abs, coverage, max_abs, source_step, prefix_id)

    if not prefixes:
        raise ValueError("selected case has no prefixes")
    return max(prefixes, key=score)


def _run_key_beta_gamma(run_key: str) -> tuple[float | None, float | None]:
    match = re.search(r"beta(?P<beta>[-+0-9.eE]+)_gamma(?P<gamma>[-+0-9.eE]+)", run_key)
    if match is None:
        return None, None
    return finite_float(match.group("beta")), finite_float(match.group("gamma"))


def _float_matches(left: Any, right: float | None) -> bool:
    if right is None:
        return False
    left_float = finite_float(left)
    return left_float is not None and math.isclose(left_float, right, rel_tol=1e-9, abs_tol=1e-9)


def _int_value(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _grid_entry(
    clustering: dict[str, Any],
    run_key: str | None = None,
    selected_k: Any = None,
) -> dict[str, Any]:
    grid = clustering.get("grid") or []
    entries = [entry for entry in grid if isinstance(entry, dict)]
    target_beta, target_gamma = _run_key_beta_gamma(run_key or "")
    target_k = _int_value(selected_k)
    context = f"run_key={run_key!r} beta={target_beta} gamma={target_gamma} K={target_k}"
    if target_beta is None or target_gamma is None:
        raise ValueError(f"could not parse Stage 5 beta/gamma from {context}")
    if not entries:
        raise ValueError(f"missing Stage 5 grid entries for {context}")

    matches: list[dict[str, Any]] = []
    for entry in entries:
        beta_match = _float_matches(entry.get("beta"), target_beta)
        gamma_match = _float_matches(entry.get("gamma"), target_gamma)
        entry_k = _int_value(entry.get("K") or entry.get("n_clusters"))
        k_match = target_k is None or entry_k == target_k
        if beta_match and gamma_match and k_match:
            matches.append(entry)
    if not matches:
        raise ValueError(f"no exact Stage 5 grid entry matches {context}")

    with_assignments = [entry for entry in matches if isinstance(entry.get("assignments"), list)]
    return with_assignments[0] if with_assignments else matches[0]


def _assignments_from_clustering(
    clustering: dict[str, Any],
    run_key: str | None = None,
    selected_k: Any = None,
    continuation_count: int | None = None,
) -> list[Any]:
    entry = _grid_entry(clustering, run_key=run_key, selected_k=selected_k)
    assignments = entry.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError(
            "matched Stage 5 grid entry has no assignments "
            f"for run_key={run_key!r} K={_int_value(selected_k)}"
        )
    if continuation_count is not None and len(assignments) != continuation_count:
        raise ValueError(
            "Stage 5 assignment length does not match continuation count: "
            f"assignments={len(assignments)} continuations={continuation_count} "
            f"run_key={run_key!r} K={_int_value(selected_k)}"
        )
    return assignments


def _extract_problem_statement(prompt_text: str) -> str:
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


def _source_step_text(branches: dict[str, Any], source_step: int) -> str:
    metadata = branches.get("reasoning_metadata") or {}
    steps = metadata.get("committed_previous_steps") or []
    if 1 <= source_step <= len(steps):
        return str(steps[source_step - 1]).strip()
    pair_metadata = branches.get("reasoning_pair_metadata") or {}
    target_metadata = pair_metadata.get("target_reasoning_metadata") or {}
    steps = target_metadata.get("committed_previous_steps") or []
    if 1 <= source_step <= len(steps):
        return str(steps[source_step - 1]).strip()
    return ""


def _file_or_raise(case_root: Path, lane: str, suffix: str, prefix_id: str, kind: str) -> Path:
    path = find_case_file(case_root, lane, suffix)
    if path is None:
        raise FileNotFoundError(f"missing {kind} file for prefix {prefix_id}: *{suffix}")
    return path


def build_fixed_cluster_case(
    case: dict[str, Any],
    case_root: Path,
    stage7_root: Path,
    run_key: str = "beta0.75_gamma0.7",
    sweep_key: str = "sign_full_B5",
) -> dict[str, Any]:
    lane = str(case["lane"])
    prefixes = sorted(case.get("prefixes", []), key=lambda item: int(item.get("source_step") or 0))
    if not prefixes:
        raise ValueError("selected case has no prefixes")

    loaded: list[dict[str, Any]] = []
    for prefix in prefixes:
        prefix_id = str(prefix["prefix_id"])
        branch_path = _file_or_raise(
            case_root, lane, f"{prefix_id}_branches.json", prefix_id, "branch"
        )
        clustering_path = _file_or_raise(
            case_root, lane, f"{prefix_id}_sweep_results.json", prefix_id, "clustering"
        )
        branches = load_json(branch_path)
        clustering = load_json(clustering_path)
        continuations = branches.get("continuations") or []
        selected_entry = _grid_entry(clustering, run_key=run_key, selected_k=prefix.get("K"))
        assignments = _assignments_from_clustering(
            clustering,
            run_key=run_key,
            selected_k=prefix.get("K"),
            continuation_count=len(continuations),
        )
        enriched_prefix = {
            **prefix,
            "K": prefix.get("K") or selected_entry.get("K"),
            "n_continuations": len(continuations),
            "continuation_coverage": sum(_continuation_mass(row) for row in continuations),
        }
        loaded.append(
            {
                "prefix": enriched_prefix,
                "branches": branches,
                "clustering": clustering,
                "selected_grid_entry": selected_entry,
                "continuations": continuations,
                "assignments": assignments,
                "branch_path": branch_path,
                "clustering_path": clustering_path,
            }
        )

    reference_prefix = choose_reference_prefix([item["prefix"] for item in loaded])
    reference_prefix_id = str(reference_prefix["prefix_id"])
    reference = next(item for item in loaded if str(item["prefix"]["prefix_id"]) == reference_prefix_id)
    fixed_clusters = build_fixed_clusters_from_reference(
        continuations=reference["continuations"],
        assignments=reference["assignments"],
    )

    per_source_effects: list[dict[str, Any]] = []
    local_cluster_partitions: list[dict[str, Any]] = []
    previous_members: dict[str, dict[str, float]] | None = None
    previous_step: int | None = None
    alluvial_flows: list[dict[str, Any]] = []

    for item in loaded:
        prefix = item["prefix"]
        prefix_id = str(prefix["prefix_id"])
        source_step = int(prefix.get("source_step") or 0)
        stage7_path = stage7_file_path(stage7_root, lane, prefix_id)
        if not stage7_path.exists():
            raise FileNotFoundError(f"missing stage7 validation file for prefix {prefix_id}: {stage7_path}")
        stage7_prefix = load_json(stage7_path)
        stats = stage7_cluster_stats(stage7_prefix, run_key, sweep_key)
        if not stats:
            raise ValueError(
                "missing Stage 7 cluster stats for selected prefix "
                f"{prefix_id}: run_key={run_key!r} sweep_key={sweep_key!r}"
            )
        effects = project_local_effects_to_fixed_clusters(
            fixed_clusters=fixed_clusters,
            continuations=item["continuations"],
            local_assignments=item["assignments"],
            local_cluster_stats=stats,
        )
        members = local_cluster_members(item["continuations"], item["assignments"])

        if previous_members is not None and previous_step is not None:
            alluvial_flows.append(
                {
                    "source_step": previous_step,
                    "target_step": source_step,
                    "flows": compute_adjacent_cluster_flows(previous_members, members),
                }
            )
        previous_members = members
        previous_step = source_step

        per_source_effects.append(
            {
                "source_step": source_step,
                "source_step_text": _source_step_text(item["branches"], source_step),
                "prefix_id": prefix_id,
                "fixed_cluster_effects": effects,
                "stage7_cluster_stats": stats,
                "stage5_grid_entry": {
                    "beta": item["selected_grid_entry"].get("beta"),
                    "gamma": item["selected_grid_entry"].get("gamma"),
                    "K": item["selected_grid_entry"].get("K"),
                },
                "paths": {
                    "stage2_branch_file": str(item["branch_path"]),
                    "stage5_clustering_file": str(item["clustering_path"]),
                    "stage7_prefix_file": str(stage7_path),
                },
            }
        )
        local_cluster_partitions.append(
            {
                "source_step": source_step,
                "prefix_id": prefix_id,
                "members": members,
                "identity_members": _local_cluster_identity_members(
                    item["continuations"],
                    item["assignments"],
                ),
                "assignments": [str(value) for value in item["assignments"]],
            }
        )

    first_branches = loaded[0]["branches"]
    metadata = first_branches.get("reasoning_metadata") or {}
    prompt_text = str(metadata.get("prompt_text") or first_branches.get("prefix") or "")
    committed_previous_steps = [
        str(step).strip()
        for step in (metadata.get("committed_previous_steps") or [])
    ]

    return {
        "dataset": str(case.get("dataset", "")).upper(),
        "lane": lane,
        "example_id": case.get("example_id"),
        "question": _extract_problem_statement(prompt_text) if prompt_text else "",
        "target_step": int(case.get("target_step") or reference_prefix.get("target_step") or 0),
        "committed_previous_steps": committed_previous_steps,
        "reference_prefix_id": reference_prefix_id,
        "fixed_clusters": fixed_clusters,
        "per_source_effects": per_source_effects,
        "local_cluster_partitions": local_cluster_partitions,
        "adjacent_cluster_flows": alluvial_flows,
        "selection_metrics": {
            "score": case.get("score"),
            "max_abs_rho_s": case.get("max_abs_rho_s"),
            "std_rho_s_across_sources": case.get("std_rho_s_across_sources"),
            "mean_n_clusters": case.get("mean_n_clusters"),
        },
        "paths": {
            "case_root": str(case_root),
            "stage7_root": str(stage7_root),
            "reference_branch_file": str(reference["branch_path"]),
            "reference_clustering_file": str(reference["clustering_path"]),
        },
    }
