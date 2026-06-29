#!/usr/bin/env -S uv run python
"""Compare minimal integrity artifacts between the original and paper-clean repos."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def normalize_method(raw: Any) -> str:
    method = str(raw or "rd")
    return {
        "combined_medoid": "rd",
        "rd": "rd",
        "kmeans": "km_sem",
        "kmeans_medoid": "km_sem",
        "km_sem": "km_sem",
        "single_continuation": "single",
        "single": "single",
    }.get(method, method)


def status_rank(status: str) -> int:
    return {"pass": 0, "pass_with_known_drift": 1, "fail": 2}[status]


def combine_statuses(statuses: Iterable[str]) -> str:
    highest = max(statuses, key=status_rank)
    return highest


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm == 0.0 or b_norm == 0.0:
        return 1.0 if a_norm == b_norm else 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def mean_rowwise_cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    values = [cosine_similarity(a_row, b_row) for a_row, b_row in zip(a, b)]
    return float(np.mean(values)) if values else 1.0


def relative_difference(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def tensor_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float().numpy()
    return np.asarray(value)


def maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def curve_value(curve: Dict[Any, Any], epsilon: Any) -> Optional[float]:
    if not isinstance(curve, dict):
        return None
    candidates = [epsilon]
    try:
        eps_float = float(epsilon)
        candidates.extend([eps_float, str(eps_float), str(epsilon)])
    except (TypeError, ValueError):
        candidates.append(str(epsilon))
    for candidate in candidates:
        if candidate in curve:
            value = curve[candidate]
            return maybe_float(value)
    return None


def shape_signature(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, np.ndarray):
        return list(value.shape)
    if isinstance(value, dict):
        return {str(key): shape_signature(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [shape_signature(item) for item in value]
    return type(value).__name__


def make_stage_result(stage: str, status: str, checks: Dict[str, Any], notes: List[str]) -> Dict[str, Any]:
    return {
        "stage": stage,
        "status": status,
        "checks": checks,
        "notes": notes,
    }


def compare_stage1(clean_prefixes_path: Path, clean_metadata_path: Path, original_stage1_path: Path) -> Dict[str, Any]:
    clean_prefixes = load_json(clean_prefixes_path)
    clean_metadata = load_json(clean_metadata_path)
    original = load_json(original_stage1_path)

    clean_entries = clean_metadata.get("entries", [])
    original_entries = original.get("clozes", [])

    def clean_record(prefix: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": prefix.get("prefix_id"),
            "prefix": prefix.get("prefix"),
            "original_id": meta.get("original_id"),
            "question": meta.get("question"),
            "role": meta.get("question_role"),
        }

    def original_record(entry: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": entry.get("cloze_id"),
            "prefix": entry.get("prefix"),
            "original_id": entry.get("original_id"),
            "question": entry.get("question"),
            "role": entry.get("cloze_type"),
        }

    clean_records = [clean_record(prefix, meta) for prefix, meta in zip(clean_prefixes, clean_entries)]
    original_records = [original_record(entry) for entry in original_entries]

    exact_match = clean_records == original_records

    clean_content = [
        {key: value for key, value in record.items() if key != "id"}
        for record in clean_records
    ]
    original_content = [
        {key: value for key, value in record.items() if key != "id"}
        for record in original_records
    ]
    sorted_clean_content = sorted(clean_content, key=lambda item: (str(item["original_id"]), str(item["role"]), str(item["question"])))
    sorted_original_content = sorted(original_content, key=lambda item: (str(item["original_id"]), str(item["role"]), str(item["question"])))
    content_match = sorted_clean_content == sorted_original_content

    prompt_match = [record["prefix"] for record in clean_records] == [record["prefix"] for record in original_records]
    notes: List[str] = []
    if exact_match:
        status = "pass"
    elif content_match:
        status = "pass_with_known_drift"
        notes.append("Stage 1 prompt content matches, but sequential IDs or ordering differ.")
    else:
        status = "fail"
        notes.append("Stage 1 formatted prompt content differs from the original selection output.")

    checks = {
        "n_clean_entries": len(clean_records),
        "n_original_entries": len(original_records),
        "exact_sequence_match": exact_match,
        "prompt_sequence_match": prompt_match,
        "order_insensitive_content_match": content_match,
    }
    return make_stage_result("stage1", status, checks, notes)


def load_branch_payload(branches_dir: Path, prefix_id: str) -> Dict[str, Any]:
    return load_json(branches_dir / f"{prefix_id}_branches.json")


def branch_probabilities(payload: Dict[str, Any]) -> List[float]:
    probabilities = []
    for entry in payload.get("continuations", []):
        probabilities.append(float(entry.get("probability", 0.0)))
    return probabilities


def branch_lengths(payload: Dict[str, Any]) -> List[int]:
    values = []
    for entry in payload.get("continuations", []):
        if entry.get("num_tokens") is not None:
            values.append(int(entry["num_tokens"]))
        elif isinstance(entry.get("token_ids"), list):
            values.append(len(entry["token_ids"]))
    return values


def compare_stage2(clean_branches_dir: Path, original_branches_dir: Path, prefix_id: str) -> Dict[str, Any]:
    clean_payload = load_branch_payload(clean_branches_dir, prefix_id)
    original_payload = load_branch_payload(original_branches_dir, prefix_id)

    clean_probs = sorted(branch_probabilities(clean_payload), reverse=True)
    original_probs = sorted(branch_probabilities(original_payload), reverse=True)
    clean_lengths = branch_lengths(clean_payload)
    original_lengths = branch_lengths(original_payload)

    clean_unique = len(clean_payload.get("continuations", []))
    original_unique = len(original_payload.get("continuations", []))
    unique_rel_diff = relative_difference(float(clean_unique), float(original_unique))

    clean_median_length = float(np.median(clean_lengths)) if clean_lengths else 0.0
    original_median_length = float(np.median(original_lengths)) if original_lengths else 0.0
    median_length_diff = abs(clean_median_length - original_median_length)

    clean_top10_mass = float(sum(clean_probs[:10]))
    original_top10_mass = float(sum(original_probs[:10]))
    top10_mass_diff = abs(clean_top10_mass - original_top10_mass)

    prefix_match = clean_payload.get("prefix") == original_payload.get("prefix")
    prefix_id_match = clean_payload.get("prefix_id") == original_payload.get("prefix_id") == prefix_id

    pass_checks = [
        prefix_match,
        prefix_id_match,
        unique_rel_diff <= 0.25,
        median_length_diff <= 5.0,
        top10_mass_diff <= 0.15,
    ]

    status = "pass" if all(pass_checks) else "fail"
    notes = []
    if not prefix_match:
        notes.append("Stage 2 prefix text differs between clean and original runs.")
    if not prefix_id_match:
        notes.append("Stage 2 prefix_id differs between clean and original runs.")
    if unique_rel_diff > 0.25:
        notes.append("Distinct continuation count drift exceeded the 25% threshold.")
    if median_length_diff > 5.0:
        notes.append("Median continuation length drift exceeded 5 tokens.")
    if top10_mass_diff > 0.15:
        notes.append("Top-10 probability mass drift exceeded 0.15.")

    checks = {
        "prefix_match": prefix_match,
        "prefix_id_match": prefix_id_match,
        "clean_unique_continuations": clean_unique,
        "original_unique_continuations": original_unique,
        "unique_relative_difference": unique_rel_diff,
        "clean_median_num_tokens": clean_median_length,
        "original_median_num_tokens": original_median_length,
        "median_num_tokens_difference": median_length_diff,
        "clean_top10_probability_mass": clean_top10_mass,
        "original_top10_probability_mass": original_top10_mass,
        "top10_probability_mass_difference": top10_mass_diff,
    }
    return make_stage_result("stage2", status, checks, notes)


def compare_stage3(clean_stage3_dir: Path, original_stage3_dir: Path, prefix_id: str) -> Dict[str, Any]:
    clean_ctx = torch.load(clean_stage3_dir / f"{prefix_id}_prefix_context.pt", map_location="cpu", weights_only=False)
    original_ctx = torch.load(original_stage3_dir / f"{prefix_id}_prefix_context.pt", map_location="cpu", weights_only=False)

    clean_attr = tensor_to_numpy(clean_ctx["aggregated_attributions"])
    original_attr = tensor_to_numpy(original_ctx["aggregated_attributions"])
    attr_cosine = cosine_similarity(clean_attr, original_attr)
    shape_match = list(clean_attr.shape) == list(original_attr.shape)

    count_keys = ["n_prefix_sources", "n_prefix_features", "n_prefix_errors", "n_prefix_tokens"]
    count_match = True
    counts: Dict[str, Any] = {}
    for key in count_keys:
        clean_value = int(clean_ctx[key])
        original_value = int(original_ctx[key])
        counts[f"clean_{key}"] = clean_value
        counts[f"original_{key}"] = original_value
        if clean_value != original_value:
            count_match = False

    pass_checks = [shape_match, attr_cosine >= 0.999, count_match]
    status = "pass" if all(pass_checks) else "fail"
    notes = []
    if not shape_match:
        notes.append("Stage 3 aggregated attribution tensor shapes differ.")
    if attr_cosine < 0.999:
        notes.append("Stage 3 flattened attribution cosine similarity fell below 0.999.")
    if not count_match:
        notes.append("Stage 3 prefix-context node counts differ.")

    checks = {
        "shape_match": shape_match,
        "clean_shape": list(clean_attr.shape),
        "original_shape": list(original_attr.shape),
        "flattened_cosine_similarity": attr_cosine,
        **counts,
    }
    return make_stage_result("stage3", status, checks, notes)


def compare_stage4(clean_stage4_dir: Path, original_stage4_dir: Path, prefix_id: str) -> Dict[str, Any]:
    clean_embeddings = np.load(clean_stage4_dir / f"{prefix_id}_embeddings.npy")
    original_embeddings = np.load(original_stage4_dir / f"{prefix_id}_embeddings.npy")
    clean_meta = load_json(clean_stage4_dir / f"{prefix_id}_embeddings_meta.json")
    original_meta = load_json(original_stage4_dir / f"{prefix_id}_embeddings_meta.json")

    shape_match = list(clean_embeddings.shape) == list(original_embeddings.shape)
    cosine_mean = mean_rowwise_cosine(clean_embeddings, original_embeddings)
    prefix_match = clean_meta.get("prefix") == original_meta.get("prefix")
    n_cont_match = clean_meta.get("n_continuations") == original_meta.get("n_continuations")

    pass_checks = [shape_match, cosine_mean >= 0.999, prefix_match, n_cont_match]
    status = "pass" if all(pass_checks) else "fail"
    notes = []
    if not shape_match:
        notes.append("Stage 4 embedding tensor shapes differ.")
    if cosine_mean < 0.999:
        notes.append("Stage 4 mean rowwise embedding cosine similarity fell below 0.999.")
    if not prefix_match:
        notes.append("Stage 4 metadata prefix text differs.")
    if not n_cont_match:
        notes.append("Stage 4 continuation count metadata differs.")

    checks = {
        "shape_match": shape_match,
        "clean_shape": list(clean_embeddings.shape),
        "original_shape": list(original_embeddings.shape),
        "mean_rowwise_cosine_similarity": cosine_mean,
        "prefix_match": prefix_match,
        "n_continuations_match": n_cont_match,
    }
    return make_stage_result("stage4", status, checks, notes)


def select_grid_entry(payload: Dict[str, Any], beta: float, gamma: float) -> Optional[Dict[str, Any]]:
    for entry in payload.get("grid", []):
        if round(float(entry.get("beta")), 6) == round(beta, 6) and round(float(entry.get("gamma")), 6) == round(gamma, 6):
            return entry
    return None


def compare_stage5(clean_stage5_dir: Path, original_stage5_dir: Path, prefix_id: str, beta: float, gamma: float) -> Dict[str, Any]:
    clean_payload = load_json(clean_stage5_dir / f"{prefix_id}_sweep_results.json")
    original_payload = load_json(original_stage5_dir / f"{prefix_id}_sweep_results.json")

    clean_entry = select_grid_entry(clean_payload, beta, gamma)
    original_entry = select_grid_entry(original_payload, beta, gamma)

    checks: Dict[str, Any] = {
        "beta": beta,
        "gamma": gamma,
        "clean_entry_found": clean_entry is not None,
        "original_entry_found": original_entry is not None,
    }
    notes: List[str] = []

    if clean_entry is None or original_entry is None:
        notes.append("Stage 5 target sweep entry was not found in one or both repos.")
        return make_stage_result("stage5", "fail", checks, notes)

    metrics = ["L_RD", "H", "D_e", "D_a"]
    metric_diffs: Dict[str, Any] = {}
    metrics_ok = True
    for metric in metrics:
        clean_value = float(clean_entry.get(metric, 0.0))
        original_value = float(original_entry.get(metric, 0.0))
        diff = relative_difference(clean_value, original_value)
        metric_diffs[f"clean_{metric}"] = clean_value
        metric_diffs[f"original_{metric}"] = original_value
        metric_diffs[f"{metric}_relative_difference"] = diff
        if diff > 2e-2:
            metrics_ok = False

    clean_assignments = clean_entry.get("assignments", [])
    original_assignments = original_entry.get("assignments", [])
    assignments_length_match = len(clean_assignments) == len(original_assignments)
    k_match = int(clean_entry.get("K", 0)) == int(original_entry.get("K", 0))
    h0_shape_match = list(np.asarray(clean_payload.get("H_0", [])).shape) == list(np.asarray(original_payload.get("H_0", [])).shape)

    status = "pass" if all([metrics_ok, assignments_length_match, k_match, h0_shape_match]) else "fail"
    if not metrics_ok:
        notes.append("Stage 5 R-D objective statistics drifted beyond tolerance.")
    if not assignments_length_match:
        notes.append("Stage 5 assignment vector lengths differ.")
    if not k_match:
        notes.append("Stage 5 cluster count K differs.")
    if not h0_shape_match:
        notes.append("Stage 5 H_0 shape differs.")

    checks.update(
        {
            "clean_K": int(clean_entry.get("K", 0)),
            "original_K": int(original_entry.get("K", 0)),
            "assignments_length_match": assignments_length_match,
            "clean_assignments_length": len(clean_assignments),
            "original_assignments_length": len(original_assignments),
            "H_0_shape_match": h0_shape_match,
            **metric_diffs,
        }
    )
    return make_stage_result("stage5", status, checks, notes)


def compare_stage6(clean_stage6_file: Path, original_stage6_file: Path) -> Dict[str, Any]:
    clean_payload = torch.load(clean_stage6_file, map_location="cpu", weights_only=False)
    original_payload = torch.load(original_stage6_file, map_location="cpu", weights_only=False)

    count_keys = ["n_features", "n_error_nodes", "n_token_nodes"]
    checks: Dict[str, Any] = {}
    counts_match = True
    for key in count_keys:
        clean_value = int(clean_payload.get(key, 0))
        original_value = int(original_payload.get(key, 0))
        checks[f"clean_{key}"] = clean_value
        checks[f"original_{key}"] = original_value
        if clean_value != original_value:
            counts_match = False

    clean_component_ids = [int(value) for value in clean_payload.get("component_ids", [])]
    original_component_ids = [int(value) for value in original_payload.get("component_ids", [])]
    component_ids_match = clean_component_ids == original_component_ids

    signature_keys = [
        "semantic_graphs",
        "semantic_graphs_centered",
        "soft_node_memberships",
        "active_features",
    ]
    signature_match = True
    for key in signature_keys:
        clean_sig = shape_signature(clean_payload.get(key))
        original_sig = shape_signature(original_payload.get(key))
        checks[f"clean_{key}_signature"] = clean_sig
        checks[f"original_{key}_signature"] = original_sig
        if clean_sig != original_sig:
            signature_match = False

    status = "pass" if all([counts_match, component_ids_match, signature_match]) else "fail"
    notes = []
    if not counts_match:
        notes.append("Stage 6 feature/error/token node counts differ.")
    if not component_ids_match:
        notes.append("Stage 6 component IDs differ.")
    if not signature_match:
        notes.append("Stage 6 graph tensor shapes differ.")

    checks["component_ids_match"] = component_ids_match
    checks["clean_component_ids"] = clean_component_ids
    checks["original_component_ids"] = original_component_ids
    return make_stage_result("stage6", status, checks, notes)


def select_clustering_run(payload: Dict[str, Any], beta: float, gamma: float) -> Optional[Tuple[str, Dict[str, Any]]]:
    for key, run in payload.get("clustering_runs", {}).items():
        if round(float(run.get("beta", 0.0)), 6) == round(beta, 6) and round(float(run.get("gamma", 0.0)), 6) == round(gamma, 6):
            return key, run
    return None


def normalize_stage7_result(payload: Dict[str, Any], beta: float, gamma: float, result_key: str) -> Optional[Dict[str, Any]]:
    selection = select_clustering_run(payload, beta, gamma)
    if selection is None:
        return None
    clustering_key, run = selection
    raw_result = run.get("results", {}).get(result_key)
    if raw_result is None:
        return {
            "method": normalize_method(payload.get("method") or payload.get("baseline_method")),
            "clustering_key": clustering_key,
            "selected_indices": {str(key): int(value) for key, value in (run.get("selected_indices") or {}).items()},
            "missing_result_key": True,
        }

    per_cluster_source = raw_result.get("per_cluster")
    if not isinstance(per_cluster_source, dict):
        per_cluster_source = raw_result.get("per_cluster_logit", {})
    per_cluster: Dict[str, Dict[str, Any]] = {}
    for cluster_id, stats in per_cluster_source.items():
        per_cluster[str(cluster_id)] = {
            "n_samples": int(stats.get("n_samples", 0)),
            "centered_logit_corr": maybe_float(stats.get("centered_logit_corr")),
            "centered_logit_spearman": maybe_float(stats.get("centered_logit_spearman")),
            "centered_logit_diff": stats.get("centered_logit_diff", {}),
            "sum_centered_logit_diff": stats.get("sum_centered_logit_diff", {}),
        }

    epsilon_values = raw_result.get("epsilon_values", [])
    corr_mean = raw_result.get("centered_logit_corr_mean")
    if corr_mean is None:
        corr_mean = raw_result.get("mean_logit_corr")
    spearman_mean = raw_result.get("centered_logit_spearman_mean")
    if spearman_mean is None:
        spearman_mean = raw_result.get("mean_logit_spearman")

    normalized = {
        "method": normalize_method(payload.get("method") or payload.get("baseline_method")),
        "clustering_key": clustering_key,
        "selected_indices": {str(key): int(value) for key, value in (run.get("selected_indices") or {}).items()},
        "epsilon_values": epsilon_values,
        "per_cluster": per_cluster,
        "centered_logit_corr_mean": maybe_float(corr_mean),
        "centered_logit_spearman_mean": maybe_float(spearman_mean),
    }

    for epsilon in epsilon_values:
        key = f"sum_diff_eps{epsilon}"
        if key in raw_result:
            normalized[key] = maybe_float(raw_result[key])
            continue
        total = 0.0
        saw_value = False
        for stats in per_cluster.values():
            value = curve_value(stats.get("sum_centered_logit_diff", {}), epsilon)
            if value is None:
                continue
            total += value
            saw_value = True
        if saw_value:
            normalized[key] = total
    return normalized


def compare_stage7(clean_stage7_file: Path, original_stage7_file: Path, beta: float, gamma: float, result_key: str) -> Dict[str, Any]:
    clean_payload = load_json(clean_stage7_file)
    original_payload = load_json(original_stage7_file)

    clean_run = normalize_stage7_result(clean_payload, beta, gamma, result_key)
    original_run = normalize_stage7_result(original_payload, beta, gamma, result_key)

    checks: Dict[str, Any] = {
        "beta": beta,
        "gamma": gamma,
        "result_key": result_key,
        "clean_run_found": clean_run is not None,
        "original_run_found": original_run is not None,
    }
    notes: List[str] = []

    if clean_run is None or original_run is None:
        notes.append("Stage 7 target clustering run was not found in one or both repos.")
        return make_stage_result("stage7_rd", "fail", checks, notes)

    if clean_run.get("missing_result_key") or original_run.get("missing_result_key"):
        notes.append("Stage 7 target sweep key is missing from one or both outputs.")
        return make_stage_result("stage7_rd", "fail", checks, notes)

    method_match = clean_run["method"] == original_run["method"] == "rd"
    selected_indices_match = clean_run.get("selected_indices") == original_run.get("selected_indices")
    epsilon_match = clean_run.get("epsilon_values") == original_run.get("epsilon_values")
    cluster_ids_match = sorted(clean_run.get("per_cluster", {}).keys()) == sorted(original_run.get("per_cluster", {}).keys())

    corr_diff = math.inf
    if clean_run.get("centered_logit_corr_mean") is not None and original_run.get("centered_logit_corr_mean") is not None:
        corr_diff = abs(float(clean_run["centered_logit_corr_mean"]) - float(original_run["centered_logit_corr_mean"]))

    spearman_diff = math.inf
    if clean_run.get("centered_logit_spearman_mean") is not None and original_run.get("centered_logit_spearman_mean") is not None:
        spearman_diff = abs(float(clean_run["centered_logit_spearman_mean"]) - float(original_run["centered_logit_spearman_mean"]))

    summary_keys = sorted(
        set(key for key in clean_run.keys() if str(key).startswith("sum_diff_eps"))
        & set(key for key in original_run.keys() if str(key).startswith("sum_diff_eps"))
    )
    summary_diffs: Dict[str, float] = {}
    summary_ok = True
    for key in summary_keys:
        diff = abs(float(clean_run[key]) - float(original_run[key]))
        summary_diffs[key] = diff
        if diff > 0.05:
            summary_ok = False

    pooled_ok = corr_diff <= 0.05 and spearman_diff <= 0.05
    status = "pass" if all([method_match, selected_indices_match, epsilon_match, cluster_ids_match, pooled_ok, summary_ok]) else "fail"

    if not method_match:
        notes.append("Stage 7 method normalization does not resolve to RD on both sides.")
    if not selected_indices_match:
        notes.append("Stage 7 selected medoid continuation indices differ.")
    if not epsilon_match:
        notes.append("Stage 7 epsilon grids differ.")
    if not cluster_ids_match:
        notes.append("Stage 7 cluster ID sets differ.")
    if corr_diff > 0.05:
        notes.append("Stage 7 centered-logit correlation mean drift exceeded 0.05.")
    if spearman_diff > 0.05:
        notes.append("Stage 7 centered-logit Spearman mean drift exceeded 0.05.")
    if not summary_ok:
        notes.append("Stage 7 summed centered-logit-difference summaries drifted beyond 0.05.")

    checks.update(
        {
            "method_match": method_match,
            "selected_indices_match": selected_indices_match,
            "clean_selected_indices": clean_run.get("selected_indices"),
            "original_selected_indices": original_run.get("selected_indices"),
            "epsilon_values_match": epsilon_match,
            "cluster_ids_match": cluster_ids_match,
            "clean_cluster_ids": sorted(clean_run.get("per_cluster", {}).keys()),
            "original_cluster_ids": sorted(original_run.get("per_cluster", {}).keys()),
            "clean_centered_logit_corr_mean": clean_run.get("centered_logit_corr_mean"),
            "original_centered_logit_corr_mean": original_run.get("centered_logit_corr_mean"),
            "centered_logit_corr_mean_difference": corr_diff,
            "clean_centered_logit_spearman_mean": clean_run.get("centered_logit_spearman_mean"),
            "original_centered_logit_spearman_mean": original_run.get("centered_logit_spearman_mean"),
            "centered_logit_spearman_mean_difference": spearman_diff,
            "sum_diff_differences": summary_diffs,
        }
    )
    return make_stage_result("stage7_rd", status, checks, notes)


def build_markdown_report(summary: Dict[str, Any]) -> str:
    lines = [
        "# Integrity Report",
        "",
        f"- Overall status: `{summary['overall_status']}`",
        f"- Prefix: `{summary['prefix_id']}`",
        f"- Beta/Gamma: `{summary['beta']}` / `{summary['gamma']}`",
        f"- Stage 7 sweep key: `{summary['stage7_result_key']}`",
        "",
    ]
    for stage in summary["stages"]:
        lines.append(f"## {stage['stage']}")
        lines.append(f"- Status: `{stage['status']}`")
        for key, value in stage["checks"].items():
            lines.append(f"- {key}: `{value}`")
        for note in stage["notes"]:
            lines.append(f"- Note: {note}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare integrity artifacts between the original and paper-clean repos")
    parser.add_argument("--clean-stage1-prefixes", type=Path, required=True)
    parser.add_argument("--clean-stage1-metadata", type=Path, required=True)
    parser.add_argument("--original-stage1", type=Path, required=True)
    parser.add_argument("--clean-stage2-dir", type=Path, required=True)
    parser.add_argument("--original-stage2-dir", type=Path, required=True)
    parser.add_argument("--clean-stage3-dir", type=Path, required=True)
    parser.add_argument("--original-stage3-dir", type=Path, required=True)
    parser.add_argument("--clean-stage4-dir", type=Path, required=True)
    parser.add_argument("--original-stage4-dir", type=Path, required=True)
    parser.add_argument("--clean-stage5-dir", type=Path, required=True)
    parser.add_argument("--original-stage5-dir", type=Path, required=True)
    parser.add_argument("--clean-stage6-file", type=Path, required=True)
    parser.add_argument("--original-stage6-file", type=Path, required=True)
    parser.add_argument("--clean-stage7-file", type=Path, required=True)
    parser.add_argument("--original-stage7-file", type=Path, required=True)
    parser.add_argument("--prefix-id", type=str, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--stage7-result-key", type=str, default="sign_full_B5")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    stages = [
        compare_stage1(args.clean_stage1_prefixes, args.clean_stage1_metadata, args.original_stage1),
        compare_stage2(args.clean_stage2_dir, args.original_stage2_dir, args.prefix_id),
        compare_stage3(args.clean_stage3_dir, args.original_stage3_dir, args.prefix_id),
        compare_stage4(args.clean_stage4_dir, args.original_stage4_dir, args.prefix_id),
        compare_stage5(args.clean_stage5_dir, args.original_stage5_dir, args.prefix_id, args.beta, args.gamma),
        compare_stage6(args.clean_stage6_file, args.original_stage6_file),
        compare_stage7(args.clean_stage7_file, args.original_stage7_file, args.beta, args.gamma, args.stage7_result_key),
    ]

    summary = {
        "overall_status": combine_statuses(stage["status"] for stage in stages),
        "prefix_id": args.prefix_id,
        "beta": args.beta,
        "gamma": args.gamma,
        "stage7_result_key": args.stage7_result_key,
        "stages": stages,
    }
    save_json(args.output_json, summary)
    save_text(args.output_md, build_markdown_report(summary))


if __name__ == "__main__":
    main()
