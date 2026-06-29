#!/usr/bin/env python3
"""Run a paired medoid-vs-single steering case study for the paper table."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIR = REPO_ROOT / "7_validation"
CIRCUIT_TRACER_PATH = REPO_ROOT / "circuit-tracer"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CIRCUIT_TRACER_PATH))

from circuit_tracer import ReplacementModel
from utils.attribution_pooling import load_pooled_attributions
from utils.data_utils import load_json, save_json
from utils.model_backend import get_model_device, resolve_backend, resolve_stage_backend


def _import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


graph = _import_module("case_study_7c_graph", VALIDATION_DIR / "7c_graph.py")
steering = _import_module("case_study_7c_steering", VALIDATION_DIR / "7c_steering.py")
utils7c = _import_module("case_study_7c_utils", VALIDATION_DIR / "7c_utils.py")


def parse_config(config: str) -> Tuple[float, float]:
    parts = config.split("_")
    beta = float(parts[0].replace("beta", ""))
    gamma = float(parts[1].replace("gamma", ""))
    return beta, gamma


def resolve_stage7c_backend(model_name: str, backend_arg: str) -> str:
    config = {"attribution": {"backend": backend_arg}, "stage_7c_steering": {}}
    return resolve_backend(model_name, resolve_stage_backend(config, "stage_7c_steering"))


def find_clustering_entry(clustering_file: Path, config: str) -> Dict[str, Any]:
    beta, gamma = parse_config(config)
    data = load_json(clustering_file)
    for entry in data.get("grid", []):
        if abs(float(entry.get("beta", -99)) - beta) < 1e-6 and abs(float(entry.get("gamma", -99)) - gamma) < 1e-6:
            return entry
    raise KeyError(f"Could not find {config} in {clustering_file}")


def aggregate_result_candidates(results_root: Path, prefix_id: str, source: str) -> List[Path]:
    if source == "RD-medoid":
        # For cloze_0394, the combined-medoid selected indices are in the
        # comparison sweep, not the standard 7c_baseline_combined_medoid run.
        return [
            results_root / "comparison/combined_medoid/7_validation/7c_steering/H4a_combined_medoid" / f"{prefix_id}_sweep_results.json",
            results_root / "7_validation/7c_baseline_combined_medoid/H4a_combined_medoid" / f"{prefix_id}_sweep_results.json",
            results_root / "7_validation/7c_steering/H4a_combined_medoid" / f"{prefix_id}_sweep_results.json",
        ]
    if source == "Single":
        return [
            results_root / "7_validation/7c_baseline_single/H4a_single" / f"{prefix_id}_sweep_results.json",
        ]
    raise ValueError(source)


def load_source_indices(results_root: Path, prefix_id: str, config: str, source: str) -> Dict[str, int]:
    candidates = aggregate_result_candidates(results_root, prefix_id, source)
    for path in candidates:
        if not path.exists():
            continue
        data = load_json(path)
        run = data.get("clustering_runs", {}).get(config)
        if run and "selected_indices" in run:
            return {str(k): int(v) for k, v in run["selected_indices"].items()}
    tried = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No selected_indices found for {source} {prefix_id} {config}. Tried:\n{tried}")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


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


def clean_display_text(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = text.replace("*", "")
    return text


def truncate_text(text: str, max_chars: int) -> str:
    text = clean_display_text(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def text_distance(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[A-Za-z0-9]+", left.lower()))
    right_tokens = set(re.findall(r"[A-Za-z0-9]+", right.lower()))
    if not left_tokens and not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return 1.0 - (len(left_tokens & right_tokens) / len(union))


def select_diverse_texts(texts: List[str], n: int = 3) -> List[str]:
    candidates = [normalize_text(t) for t in texts if normalize_text(t)]
    if len(candidates) <= n:
        return candidates

    selected = [candidates[0]]
    remaining = candidates[1:]
    while remaining and len(selected) < n:
        best_idx = max(
            range(len(remaining)),
            key=lambda idx: min(text_distance(remaining[idx], chosen) for chosen in selected),
        )
        selected.append(remaining.pop(best_idx))
    return selected


def extract_question(prefix_text: str) -> str:
    match = re.search(r"<\|im_start\|>user\s*(.*?)<\|im_end\|>", prefix_text, flags=re.S)
    if match:
        return normalize_text(match.group(1))
    return normalize_text(prefix_text)


def load_prefix_arrays(results_root: Path, prefix_id: str, pooling: str) -> Dict[str, Any]:
    branches_data = load_json(results_root / "2_branch_sampling" / f"{prefix_id}_branches.json")

    prefix_context_file = results_root / "3_attribution_graphs" / f"{prefix_id}_prefix_context.pt"
    pooled = load_pooled_attributions(
        prefix_context_file,
        pooling=pooling,
        meta_file=results_root / "3_attribution_graphs" / f"{prefix_id}_attribution.json",
    )

    embeddings = np.load(results_root / "4_feature_extraction/embeddings" / f"{prefix_id}_embeddings.npy")
    return {
        "branches_data": branches_data,
        "aggregated_attributions": pooled.values,
        "embeddings": embeddings,
    }


def build_decoder_and_encoder_cache(
    model,
    semantic_graphs: Dict[int, np.ndarray],
    active_features: torch.Tensor,
    selected_features: torch.Tensor,
    top_b: int,
    hc_selection: str,
    selection_mode: str,
) -> Tuple[Dict[int, Dict], Dict[int, Dict], Dict[int, List[Tuple[int, int, int, float]]]]:
    n_features = len(selected_features)
    max_top_b = max(top_b * 2, top_b)

    all_needed_indices: set[int] = set()
    for cluster_id, h_c in semantic_graphs.items():
        h_c_features = h_c[:n_features]
        rank_scores = graph.compute_feature_ranking_scores(
            cluster_id=cluster_id,
            semantic_graphs=semantic_graphs,
            n_features=n_features,
            selection_mode=selection_mode,
        )
        top_indices = np.argsort(rank_scores)[-max_top_b:]
        for idx in top_indices:
            if abs(h_c_features[idx]) >= utils7c.EPSILON_SMALL:
                all_needed_indices.add(int(idx))

    device = get_model_device(model)
    global_decoder_cache: Dict[int, torch.Tensor] = {}
    layer_to_indices: Dict[int, List[Tuple[int, int]]] = {}
    for h_c_idx in sorted(all_needed_indices):
        feat_idx = selected_features[h_c_idx].item()
        layer, _pos, feat_id = active_features[feat_idx].tolist()
        layer_to_indices.setdefault(int(layer), []).append((h_c_idx, int(feat_id)))

    for layer, idx_list in layer_to_indices.items():
        h_c_indices = [x[0] for x in idx_list]
        feat_ids = [x[1] for x in idx_list]
        feat_ids_t = torch.tensor(feat_ids, device=device, dtype=torch.long)
        dec_vecs = model.transcoders._get_decoder_vectors(layer, feat_ids_t)
        for i, h_c_idx in enumerate(h_c_indices):
            global_decoder_cache[h_c_idx] = dec_vecs[i]

    decoder_cache = graph.build_cluster_decoder_cache(
        semantic_graphs,
        global_decoder_cache,
        active_features,
        selected_features,
        max_features=max_top_b,
        selection_mode=selection_mode,
    )

    features_by_cluster: Dict[int, List[Tuple[int, int, int, float]]] = {}
    for cluster_id, cache_data in decoder_cache.items():
        h_c_vals, _dec_vecs, layers, positions, feat_ids = graph.select_features_with_hc_selection(
            cache_data, top_b, hc_selection
        )
        features_by_cluster[cluster_id] = [
            (layers[i], positions[i], feat_ids[i], h_c_vals[i]) for i in range(len(h_c_vals))
        ]

    encoder_cache = graph.precompute_cluster_encoder_weights(model, features_by_cluster, device)
    return decoder_cache, encoder_cache, features_by_cluster


def run_centered_logits(
    model,
    branches: List[Dict[str, Any]],
    features: List[Tuple[int, int, int, float]],
    encoder_cache: Dict[int, Dict],
    decoder_cache: Dict,
    epsilon: float,
    steering_method: str,
    max_seq_len: int,
    batch_size: int,
) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}
    for start in range(0, len(branches), batch_size):
        batch = branches[start : start + batch_size]
        batch_token_ids = [b["full_token_ids"] for b in batch]
        batch_cont_info = []
        for branch in batch:
            cont_ids = branch["continuation_token_ids"]
            cont_start = len(branch["full_token_ids"]) - len(cont_ids)
            batch_cont_info.append((cont_ids, cont_start))

        with torch.inference_mode():
            logits, _ = steering.run_batched_steered_pass_on_the_fly(
                model,
                batch_token_ids,
                features=features,
                encoder_cache=encoder_cache,
                decoder_cache=decoder_cache,
                steering_method=steering_method,
                epsilon=epsilon,
                max_seq_len=max_seq_len,
            )

        centered = steering.compute_per_token_centered_logits_batched(
            logits, batch_cont_info, return_per_token=True
        )
        for branch, (per_token, mean_centered) in zip(batch, centered):
            results[int(branch["branch_id"])] = {
                "mean_centered_logit": float(mean_centered),
                "per_token_centered_logits": [float(v) for v in per_token],
            }

        del logits
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def choose_eval_branch_ids(
    assignments: List[int],
    clusters: Iterable[str],
    source_indices: Dict[str, Dict[str, int]],
    max_eval_per_cluster: int,
    seed: int,
) -> Dict[str, List[int]]:
    rng = np.random.RandomState(seed)
    eval_ids: Dict[str, List[int]] = {}
    for cluster in clusters:
        excluded = {
            int(indices[cluster])
            for indices in source_indices.values()
            if cluster in indices
        }
        candidates = [
            idx
            for idx, assigned in enumerate(assignments)
            if str(assigned) == str(cluster) and idx not in excluded
        ]
        rng.shuffle(candidates)
        if max_eval_per_cluster > 0:
            candidates = candidates[:max_eval_per_cluster]
        eval_ids[str(cluster)] = sorted(int(x) for x in candidates)
    return eval_ids


def build_latex_table(rows: List[Dict[str, Any]]) -> str:
    def metric_cell(value: float, bold: bool = False) -> str:
        text = f"{value:+.3f}"
        return rf"\textbf{{{text}}}" if bold else text

    def pos_cell(value: float, bold: bool = False) -> str:
        text = f"{100.0 * value:.0f}\\%"
        return rf"\textbf{{{text}}}" if bold else text

    rows_by_cluster: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        rows_by_cluster.setdefault((row["prefix_label"], row["cluster"]), []).append(row)

    first_row = rows[0]
    question = latex_escape(first_row.get("question") or "Who pays for the renovations on Holmes Next Generation?")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tcolorbox}[",
        r"  enhanced,",
        r"  colback=gray!5,",
        r"  colframe=gray!40,",
        r"  boxrule=0.4pt,",
        r"  left=2pt, right=2pt, top=2pt, bottom=2pt,",
        r"  fontupper=\scriptsize",
        r"]",
        rf"\textbf{{Q:}} \textit{{{question}}}",
        r"\vspace{0.2em}\hrule\vspace{0.2em}",
    ]

    for cluster_idx, ((_prefix, _cluster), cluster_rows) in enumerate(rows_by_cluster.items(), start=1):
        cluster_rows = sorted(cluster_rows, key=lambda r: 0 if r["source"] == "RD-medoid" else 1)
        rd_row = next((r for r in cluster_rows if r["source"] == "RD-medoid"), cluster_rows[0])
        cluster_label = latex_escape(rd_row["cluster_label"])
        eval_texts = [latex_escape(truncate_text(t, 84)) for t in select_diverse_texts(rd_row["eval_texts"], n=3)]
        source_bits = []
        for row in cluster_rows:
            label = "RD" if row["source"] == "RD-medoid" else "Single"
            source_bits.append(
                rf"\textbf{{{label}:}} ``{latex_escape(truncate_text(row['source_text'], 74))}''"
            )

        if cluster_idx > 1:
            lines.append(r"\vspace{0.2em}\hrule\vspace{0.2em}")
        lines.append(rf"\textbf{{C$_{cluster_idx}$:}} {cluster_label}.")
        lines.append(r"\vspace{0.15em}")
        for ans_idx, text in enumerate(eval_texts, start=1):
            suffix = r"\\" if ans_idx < len(eval_texts) else ""
            lines.append(rf"\textbf{{A$_{ans_idx}$}}: ``{text}''{suffix}")
        lines.append(r"\vspace{0.2em}")
        lines.append(r"{\tiny\textcolor{gray}{")
        lines.append(r"Direction sources: " + r"; ".join(source_bits))
        lines.append(r"}}")
        lines.append(r"\vspace{-0.2em}")
        lines.append(r"\begin{center}")
        lines.append(r"\setlength{\tabcolsep}{3pt}")
        lines.append(r"\begin{tabular}{lccc}")
        lines.append(r"\toprule")
        lines.append(r"Source & $\Delta L_{+1}$ & Pos. & $\Delta L_{-1}$ \\")
        lines.append(r"\midrule")
        for row in cluster_rows:
            is_rd = row["source"] == "RD-medoid"
            source = r"\textbf{RD-medoid}" if is_rd else "Single"
            lines.append(
                " & ".join(
                    [
                        source,
                        metric_cell(row["mean_delta_pos"], bold=is_rd),
                        pos_cell(row["positive_rate"], bold=is_rd),
                        metric_cell(row["mean_delta_neg"], bold=False),
                    ]
                )
                + r" \\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{center}")

    lines.extend(
        [
            r"\end{tcolorbox}",
            r"\caption{\textbf{Cluster-level transfer case study on AmbigQA with Qwen3-8B.} Each box lists one held-out set per cluster and compares steering from the combined-distance medoid against a single randomly selected continuation from the same cluster. $\Delta L_{+1}$ and $\Delta L_{-1}$ are mean changes in centered target-token logits under amplification and ablation; Pos. is the fraction of held-out continuations with positive $\Delta L_{+1}$.}",
            r"\label{tab:case-study-cluster-causality}",
            r"\vspace{-0.15in}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "prefix_id",
        "config",
        "cluster",
        "cluster_label",
        "source",
        "source_idx",
        "n_eval",
        "mean_delta_pos",
        "positive_rate",
        "mean_delta_neg",
        "corr_three_point",
        "source_text",
        "eval_branch_ids",
        "eval_texts",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def pearson_three_point(delta_neg: float, delta_pos: float) -> float:
    x = np.array([-1.0, 0.0, 1.0])
    y = np.array([delta_neg, 0.0, delta_pos])
    if np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--prefix-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--clusters", nargs="+", required=True)
    parser.add_argument("--cluster-labels", nargs="*", default=None)
    parser.add_argument("--prefix-label", default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--transcoder", default="mwhanna/qwen3-8b-transcoders")
    parser.add_argument("--backend", choices=["auto", "transformerlens", "nnsight"], default="auto")
    parser.add_argument("--pooling", choices=["mean", "max", "sum"], default="mean")
    parser.add_argument("--top-B", type=int, default=10)
    parser.add_argument("--hc-selection", default="full", choices=["full", "positive", "negative"])
    parser.add_argument("--selection-mode", default="magnitude", choices=["magnitude", "distinct"])
    parser.add_argument("--steering-method", default="sign")
    parser.add_argument("--epsilons", type=float, nargs="+", default=[-1.0, 1.0])
    parser.add_argument("--max-eval-per-cluster", type=int, default=10)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=96)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--paper-table", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger("case_study")

    if set(args.epsilons) != {-1.0, 1.0}:
        raise ValueError("This table expects exactly --epsilons -1.0 1.0")

    prefix_data = load_prefix_arrays(args.results_root, args.prefix_id, args.pooling)
    branches_data = prefix_data["branches_data"]
    continuations = branches_data.get("continuations", [])
    clustering_entry = find_clustering_entry(
        args.results_root / "5_clustering" / f"{args.prefix_id}_sweep_results.json",
        args.config,
    )
    assignments = [int(x) for x in clustering_entry["assignments"]]
    branches = utils7c.build_branches_from_data(branches_data, assignments)

    cluster_labels = {str(c): f"cluster {c}" for c in args.clusters}
    if args.cluster_labels:
        for cluster, label in zip(args.clusters, args.cluster_labels):
            cluster_labels[str(cluster)] = label

    source_indices = {
        "RD-medoid": load_source_indices(args.results_root, args.prefix_id, args.config, "RD-medoid"),
        "Single": load_source_indices(args.results_root, args.prefix_id, args.config, "Single"),
    }
    eval_branch_ids = choose_eval_branch_ids(
        assignments,
        args.clusters,
        source_indices,
        args.max_eval_per_cluster,
        args.eval_seed,
    )

    selected_eval_branches: Dict[str, List[Dict[str, Any]]] = {
        cluster: [branches[idx] for idx in ids] for cluster, ids in eval_branch_ids.items()
    }
    all_eval_branches = [branch for cluster in args.clusters for branch in selected_eval_branches[str(cluster)]]
    logger.info("Evaluation branch IDs: %s", eval_branch_ids)

    backend = resolve_stage7c_backend(args.model, args.backend)
    logger.info("Loading model=%s transcoder=%s backend=%s", args.model, args.transcoder, backend)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReplacementModel.from_pretrained(
        args.model,
        args.transcoder,
        backend=backend,
        device=device,
        dtype=torch.bfloat16,
        lazy_encoder=True,
        lazy_decoder=True,
    )

    logger.info("Computing original centered logits for %d held-out continuations", len(all_eval_branches))
    original = run_centered_logits(
        model,
        all_eval_branches,
        features=[],
        encoder_cache={},
        decoder_cache={},
        epsilon=0.0,
        steering_method=args.steering_method,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
    )

    active_features, selected_features = graph.load_attribution_context(
        args.results_root / "3_attribution_graphs", args.prefix_id, use_continuation_attribution=True
    )

    rows: List[Dict[str, Any]] = []
    detailed_results: Dict[str, Any] = {
        "prefix_id": args.prefix_id,
        "config": args.config,
        "clusters": {},
        "source_indices": source_indices,
        "eval_branch_ids": eval_branch_ids,
    }

    for source_name, indices in source_indices.items():
        semantic_graphs = {
            int(cluster): prefix_data["aggregated_attributions"][indices[str(cluster)]].copy()
            for cluster in args.clusters
        }
        decoder_cache, encoder_cache, features_by_cluster = build_decoder_and_encoder_cache(
            model,
            semantic_graphs,
            active_features,
            selected_features,
            top_b=args.top_B,
            hc_selection=args.hc_selection,
            selection_mode=args.selection_mode,
        )

        for cluster in args.clusters:
            cluster_key = str(cluster)
            cluster_id = int(cluster)
            eval_branches = selected_eval_branches[cluster_key]
            features = features_by_cluster.get(cluster_id, [])
            if not features:
                logger.warning("No features for %s cluster %s", source_name, cluster)
                continue

            eps_results: Dict[float, Dict[int, Dict[str, Any]]] = {}
            for eps in args.epsilons:
                logger.info("Running %s cluster=%s epsilon=%s n=%d", source_name, cluster, eps, len(eval_branches))
                eps_results[eps] = run_centered_logits(
                    model,
                    eval_branches,
                    features=features,
                    encoder_cache=encoder_cache[cluster_id],
                    decoder_cache=decoder_cache[cluster_id],
                    epsilon=eps,
                    steering_method=args.steering_method,
                    max_seq_len=args.max_seq_len,
                    batch_size=args.batch_size,
                )

            deltas_by_eps: Dict[str, List[float]] = {}
            branch_details = []
            for eps in args.epsilons:
                deltas = []
                for branch in eval_branches:
                    bid = int(branch["branch_id"])
                    delta = eps_results[eps][bid]["mean_centered_logit"] - original[bid]["mean_centered_logit"]
                    deltas.append(float(delta))
                deltas_by_eps[str(eps)] = deltas

            for branch in eval_branches:
                bid = int(branch["branch_id"])
                branch_details.append(
                    {
                        "branch_id": bid,
                        "text": continuations[bid].get("text", ""),
                        "original": original[bid],
                        "delta_by_epsilon": {
                            str(eps): eps_results[eps][bid]["mean_centered_logit"] - original[bid]["mean_centered_logit"]
                            for eps in args.epsilons
                        },
                    }
                )

            delta_neg = float(np.mean(deltas_by_eps["-1.0"]))
            delta_pos = float(np.mean(deltas_by_eps["1.0"]))
            positive_rate = float(np.mean(np.array(deltas_by_eps["1.0"]) > 0))
            corr = pearson_three_point(delta_neg, delta_pos)
            source_idx = int(indices[cluster_key])
            row = {
                "prefix_id": args.prefix_id,
                "prefix_label": args.prefix_label or args.prefix_id,
                "question": extract_question(branches_data.get("prefix", "")),
                "config": args.config,
                "cluster": cluster_key,
                "cluster_label": cluster_labels[cluster_key],
                "source": source_name,
                "source_idx": source_idx,
                "source_text": continuations[source_idx].get("text", ""),
                "n_eval": len(eval_branches),
                "eval_branch_ids": [int(b["branch_id"]) for b in eval_branches],
                "eval_texts": [continuations[int(b["branch_id"])].get("text", "") for b in eval_branches],
                "mean_delta_pos": delta_pos,
                "positive_rate": positive_rate,
                "mean_delta_neg": delta_neg,
                "corr_three_point": corr,
            }
            rows.append(row)
            detailed_results["clusters"].setdefault(cluster_key, {})[source_name] = {
                **row,
                "branch_details": branch_details,
                "deltas_by_epsilon": deltas_by_eps,
            }

    rows.sort(key=lambda r: (r["prefix_id"], int(r["cluster"]), 0 if r["source"] == "RD-medoid" else 1))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "case_study_cluster_causality.json"
    csv_path = args.out_dir / "case_study_cluster_causality.csv"
    tex_path = args.out_dir / "case_study_cluster_causality.tex"
    save_json(detailed_results, json_path)
    write_csv(csv_path, rows)
    tex = build_latex_table(rows)
    tex_path.write_text(tex)
    if args.paper_table:
        args.paper_table.parent.mkdir(parents=True, exist_ok=True)
        args.paper_table.write_text(tex)

    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", csv_path)
    logger.info("Wrote %s", tex_path)
    if args.paper_table:
        logger.info("Wrote paper table %s", args.paper_table)


if __name__ == "__main__":
    main()
