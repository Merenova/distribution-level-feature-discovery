#!/usr/bin/env -S uv run python
"""
Compare RD-clustering H_c vs K-means-on-embeddings H_c for the first N continuations of a prefix.

For each of the first N continuations:
  1) get RD cluster assignment -> H_c(rd) -> pick top-1 feature by |H_c|
  2) run K-means on semantic embeddings with K taken from RD sweep -> assignment -> H_c(kmeans) -> pick top-1 feature
  3) steer with that single feature and compare vs baseline (eps=0) per-token:
     - Δcentered_logit (demeaned logit; centered by mean over vocab)
     - Δlogit (raw target-token logit)
     - Δlogprob (target-token log-prob)
     - Δprob (target-token prob)

Outputs a JSON report.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "circuit-tracer"))

from circuit_tracer import ReplacementModel  # noqa: E402


def _import_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mod_dir = REPO_ROOT / "7_validation"
graph = _import_module("7c_graph", _mod_dir / "7c_graph.py")
steering = _import_module("7c_steering", _mod_dir / "7c_steering.py")
baseline_kmeans = _import_module("7c_baseline_kmeans", _mod_dir / "7c_baseline_kmeans.py")


def _find_grid_entry(sweep: Dict[str, Any], beta: float, gamma: float) -> Dict[str, Any] | None:
    for e in (sweep.get("grid", []) or []):
        try:
            b = float(e.get("beta"))
            g = float(e.get("gamma"))
        except Exception:
            continue
        if abs(b - beta) < 1e-9 and abs(g - gamma) < 1e-9:
            return e
    return None


def _summ_stats(x: List[float]) -> Dict[str, Any]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.quantile(arr, 0.1)),
        "p50": float(np.quantile(arr, 0.5)),
        "p90": float(np.quantile(arr, 0.9)),
    }


def _gather_per_token_target_logit_and_logprob(
    logits: torch.Tensor,  # [B, S, V]
    cont_ids: List[int],
    cont_start: int,
    batch_idx: int,
) -> Tuple[List[float], List[float], List[float]]:
    """Return (logit, logprob, prob) sequences for the continuation target tokens."""
    if not cont_ids:
        return [], [], []

    seq_len = logits.shape[1]
    valid_len = min(len(cont_ids), seq_len - cont_start)
    if valid_len <= 0:
        return [], [], []

    positions = torch.arange(cont_start, cont_start + valid_len, device=logits.device)
    cont_logits = logits[batch_idx, positions, :].float()  # [T, V]
    token_ids = torch.tensor(cont_ids[:valid_len], device=logits.device, dtype=torch.long)  # [T]

    target_logits = cont_logits[torch.arange(valid_len, device=logits.device), token_ids]  # [T]
    log_probs = F.log_softmax(cont_logits, dim=-1)
    target_logprobs = log_probs[torch.arange(valid_len, device=logits.device), token_ids]  # [T]
    target_probs = torch.exp(target_logprobs)

    return target_logits.tolist(), target_logprobs.tolist(), target_probs.tolist()


def _make_single_feature_decoder_cache(
    model,
    layer: int,
    pos: int,
    feat_id: int,
    h_c_val: float,
) -> Dict[str, Any]:
    """Build the flattened decoder_cache format expected by heterogeneous steering."""
    feat_ids_t = torch.tensor([int(feat_id)], device=model.cfg.device, dtype=torch.long)
    dec_vecs = model.transcoders._get_decoder_vectors(int(layer), feat_ids_t)  # [1, d_model]
    return {
        "layers": [int(layer)],
        "positions": [int(pos)],
        "feat_ids": [int(feat_id)],
        "decoder_vecs": [dec_vecs[0]],
        "h_c_values": [float(h_c_val)],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", required=True, help="gpu*/results directory (contains 2,3,4,5)")
    ap.add_argument("--prefix-id", required=True)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--steering-method", default="sign", choices=["sign", "additive", "multiplicative", "scaling", "absolute"])
    ap.add_argument("--epsilon", type=float, default=0.5)
    ap.add_argument("--kmeans-random-state", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)

    cfg = json.loads(Path(args.config).read_text())
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("base_model", "Qwen/Qwen3-8B")
    transcoder_set = model_cfg.get("transcoder", "mwhanna/qwen3-8b-transcoders")
    global_cfg = cfg.get("global", {})
    max_seq_len = global_cfg.get("max_seq_len", None)

    # Load branches (continuations)
    branches_path = results_dir / "2_branch_sampling" / f"{args.prefix_id}_branches.json"
    branches = json.loads(branches_path.read_text())
    conts = branches.get("continuations", []) or []
    if len(conts) == 0:
        raise SystemExit("No continuations found")

    n = min(int(args.n), len(conts))
    cont_indices = list(range(n))

    # Load feature mapping (for H_c indices -> (layer,pos,feat_id))
    active_features, selected_features = graph.load_attribution_context(
        results_dir / "3_attribution_graphs", args.prefix_id, use_continuation_attribution=True
    )

    # Load RD sweep results for assignments + H_c (mu_a)
    rd_path = results_dir / "5_clustering" / f"{args.prefix_id}_sweep_results.json"
    rd = json.loads(rd_path.read_text())
    rd_entry = _find_grid_entry(rd, args.beta, args.gamma)
    if rd_entry is None:
        raise SystemExit(f"No RD grid entry for beta={args.beta}, gamma={args.gamma}")
    K = int(rd_entry.get("K", len(rd_entry.get("components", {})) or 0))
    rd_assignments = rd_entry.get("assignments", [])
    rd_components = rd_entry.get("components", {}) or {}
    if not rd_assignments or not rd_components:
        raise SystemExit("RD entry missing assignments/components")
    rd_graphs = graph.build_semantic_graphs_from_clustering(rd_entry)

    # Load attributions + probs for KMeans H_c construction (L1-consistent weighted median)
    ctx_path = results_dir / "3_attribution_graphs" / f"{args.prefix_id}_prefix_context.pt"
    ctx = torch.load(ctx_path, map_location="cpu", weights_only=False)
    if "aggregated_attributions" not in ctx:
        raise SystemExit("prefix_context.pt missing aggregated_attributions")
    attributions = ctx["aggregated_attributions"].float().numpy()  # [n_cont, d_attr]

    path_probs = np.asarray([float(c.get("probability", 0.0)) for c in conts], dtype=np.float64)
    H_0 = baseline_kmeans.compute_weighted_global_center(attributions, path_probs, use_median=True)

    # Load embeddings, run KMeans with the SAME K as RD config
    emb_path = results_dir / "4_feature_extraction" / "embeddings" / f"{args.prefix_id}_embeddings.npy"
    embeddings = np.load(emb_path)
    if embeddings.shape[0] != len(conts):
        raise SystemExit(f"Embeddings n={embeddings.shape[0]} != continuations n={len(conts)}")

    kmeans = KMeans(
        n_clusters=K,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=int(args.kmeans_random_state),
    )
    km_assignments = kmeans.fit_predict(embeddings)  # [n_cont]
    km_graphs = baseline_kmeans.build_semantic_graphs_from_kmeans(
        km_assignments, attributions, path_probs, H_0, use_median=True
    )

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReplacementModel.from_pretrained(
        model_name,
        transcoder_set,
        device=device,
        dtype=torch.bfloat16,
        lazy_encoder=True,
        lazy_decoder=False,
    )

    # Prepare baseline batch items (eps=0)
    batch_full_ids: List[List[int]] = []
    batch_cont_info: List[Tuple[List[int], int]] = []
    for i in cont_indices:
        full_ids = conts[i]["full_token_ids"]
        cont_ids = conts[i]["token_ids"]
        cont_start = len(full_ids) - len(cont_ids)
        batch_full_ids.append(full_ids)
        batch_cont_info.append((cont_ids, cont_start))

    baseline_items = [{"token_ids": toks, "features": [], "decoder_cache": {}, "cluster_id": None} for toks in batch_full_ids]
    logits_base, _ = steering.run_heterogeneous_steered_pass(
        model=model,
        batch_items=baseline_items,
        steering_method="additive",
        epsilon=0.0,
        max_seq_len=max_seq_len,
    )

    # Helper to build per-continuation item for a given feature
    def make_item(i: int, feature: Tuple[int, int, int, float], cluster_id: int):
        layer, pos, feat_id, h_c_val = feature
        return {
            "token_ids": batch_full_ids[i],
            "features": [feature],
            "decoder_cache": _make_single_feature_decoder_cache(model, layer, pos, feat_id, h_c_val),
            "cluster_id": int(cluster_id),
        }

    # Build RD and KMeans batch items (one feature per continuation, chosen by its cluster H_c)
    rd_items = []
    km_items = []
    rd_meta = []
    km_meta = []
    for local_i, cont_i in enumerate(cont_indices):
        c_rd = int(rd_assignments[cont_i])
        Hc_rd = rd_graphs.get(c_rd)
        if Hc_rd is None:
            raise SystemExit(f"Missing RD H_c for cluster {c_rd}")
        feat_rd = graph.select_top_features_by_magnitude(Hc_rd, active_features, selected_features, top_B=1)[0]

        c_km = int(km_assignments[cont_i])
        Hc_km = km_graphs.get(c_km)
        if Hc_km is None:
            raise SystemExit(f"Missing KMeans H_c for cluster {c_km}")
        feat_km = graph.select_top_features_by_magnitude(Hc_km, active_features, selected_features, top_B=1)[0]

        rd_items.append(make_item(local_i, feat_rd, c_rd))
        km_items.append(make_item(local_i, feat_km, c_km))

        rd_meta.append({"cluster_id": c_rd, "feature": {"layer": feat_rd[0], "pos": feat_rd[1], "feat_id": feat_rd[2], "h_c_val": feat_rd[3]}})
        km_meta.append({"cluster_id": c_km, "feature": {"layer": feat_km[0], "pos": feat_km[1], "feat_id": feat_km[2], "h_c_val": feat_km[3]}})

    logits_rd, _ = steering.run_heterogeneous_steered_pass(
        model=model,
        batch_items=rd_items,
        steering_method=args.steering_method,
        epsilon=float(args.epsilon),
        max_seq_len=max_seq_len,
    )
    logits_km, _ = steering.run_heterogeneous_steered_pass(
        model=model,
        batch_items=km_items,
        steering_method=args.steering_method,
        epsilon=float(args.epsilon),
        max_seq_len=max_seq_len,
    )

    # Centered logits (demeaned)
    base_centered = steering.compute_per_token_centered_logits_batched(logits_base, batch_cont_info)
    rd_centered = steering.compute_per_token_centered_logits_batched(logits_rd, batch_cont_info)
    km_centered = steering.compute_per_token_centered_logits_batched(logits_km, batch_cont_info)

    per_cont = []
    for local_i, cont_i in enumerate(cont_indices):
        cont_ids, cont_start = batch_cont_info[local_i]
        base_logit, base_logprob, base_prob = _gather_per_token_target_logit_and_logprob(logits_base, cont_ids, cont_start, local_i)
        rd_logit, rd_logprob, rd_prob = _gather_per_token_target_logit_and_logprob(logits_rd, cont_ids, cont_start, local_i)
        km_logit, km_logprob, km_prob = _gather_per_token_target_logit_and_logprob(logits_km, cont_ids, cont_start, local_i)

        base_cent = base_centered[local_i][0]
        rd_cent = rd_centered[local_i][0]
        km_cent = km_centered[local_i][0]

        T = min(len(base_logit), len(rd_logit), len(km_logit), len(base_cent), len(rd_cent), len(km_cent))
        if T == 0:
            continue

        d_rd_centered = [float(rd_cent[t] - base_cent[t]) for t in range(T)]
        d_km_centered = [float(km_cent[t] - base_cent[t]) for t in range(T)]
        d_rd_logit = [float(rd_logit[t] - base_logit[t]) for t in range(T)]
        d_km_logit = [float(km_logit[t] - base_logit[t]) for t in range(T)]
        d_rd_logprob = [float(rd_logprob[t] - base_logprob[t]) for t in range(T)]
        d_km_logprob = [float(km_logprob[t] - base_logprob[t]) for t in range(T)]
        d_rd_prob = [float(rd_prob[t] - base_prob[t]) for t in range(T)]
        d_km_prob = [float(km_prob[t] - base_prob[t]) for t in range(T)]

        per_cont.append(
            {
                "cont_idx": int(cont_i),
                "n_tokens": int(T),
                "rd": {
                    **rd_meta[local_i],
                    "delta_centered_logit": d_rd_centered,
                    "delta_logit": d_rd_logit,
                    "delta_logprob": d_rd_logprob,
                    "delta_prob": d_rd_prob,
                    "summary": {
                        "delta_centered_logit": _summ_stats(d_rd_centered),
                        "delta_logit": _summ_stats(d_rd_logit),
                        "delta_logprob": _summ_stats(d_rd_logprob),
                        "delta_prob": _summ_stats(d_rd_prob),
                    },
                },
                "kmeans": {
                    **km_meta[local_i],
                    "delta_centered_logit": d_km_centered,
                    "delta_logit": d_km_logit,
                    "delta_logprob": d_km_logprob,
                    "delta_prob": d_km_prob,
                    "summary": {
                        "delta_centered_logit": _summ_stats(d_km_centered),
                        "delta_logit": _summ_stats(d_km_logit),
                        "delta_logprob": _summ_stats(d_km_logprob),
                        "delta_prob": _summ_stats(d_km_prob),
                    },
                },
            }
        )

    report = {
        "prefix_id": args.prefix_id,
        "beta": float(args.beta),
        "gamma": float(args.gamma),
        "K": int(K),
        "steering_method": args.steering_method,
        "epsilon": float(args.epsilon),
        "n_continuations": int(len(per_cont)),
        "note": "Δ* values are (steered - baseline) at the continuation target-token positions; baseline is eps=0.",
        "continuations": per_cont,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(str(out_path))


if __name__ == "__main__":
    main()


