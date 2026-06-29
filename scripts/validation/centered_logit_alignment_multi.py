#!/usr/bin/env python3
"""
Multi-continuation alignment check: attribution vs Δ(centered logit) under steering.

We fix:
  - a prefix_id
  - a feature selection rule (by cluster H_c(mu_a) or by per-continuation attribution)
  - a steering method + epsilon

Then, across N continuations, compute per-token:
  - token_attribution for the chosen feature
  - delta_centered_logit = centered_logit_steered - centered_logit_baseline

and report distribution across continuations of:
  - Pearson corr(attr, Δcentered_logit)
  - sign agreement rate (sign(attr) == sign(Δcentered_logit))

Important: We CHUNK batches to avoid holding huge logits tensors.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

# Repo root and circuit tracer (same pattern as other scripts)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "circuit-tracer"))

from circuit_tracer import ReplacementModel  # noqa: E402

import importlib.util  # noqa: E402


def _import_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mod_dir = REPO_ROOT / "7_validation"
steering = _import_module("7c_steering", _mod_dir / "7c_steering.py")
utils7c = _import_module("7c_utils", _mod_dir / "7c_utils.py")
graph7c = _import_module("7c_graph", _mod_dir / "7c_graph.py")


def _find_grid_entry(clustering_sweep: Dict[str, Any], beta: float, gamma: float) -> Dict[str, Any] | None:
    for e in (clustering_sweep.get("grid", []) or []):
        try:
            b = float(e.get("beta"))
            g = float(e.get("gamma"))
        except Exception:
            continue
        if abs(b - beta) < 1e-9 and abs(g - gamma) < 1e-9:
            return e
    return None


def _summarize(arr: List[float]) -> Dict[str, Any]:
    a = np.asarray([x for x in arr if x is not None and math.isfinite(x)], dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.quantile(a, 0.1)),
        "p90": float(np.quantile(a, 0.9)),
    }


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results-dir", required=True, help="gpu*/results directory (contains 2_branch_sampling, 3_attribution_graphs, 5_clustering)")
    ap.add_argument("--prefix-id", required=True)
    ap.add_argument("--feature-source", choices=["hc", "attr"], default="hc")
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--cluster-id", type=int, default=-1, help="-1 picks largest cluster (hc only)")
    ap.add_argument("--steering-method", choices=["sign", "additive", "multiplicative", "scaling", "absolute"], default="sign")
    ap.add_argument("--epsilon", type=float, default=1e-4)
    ap.add_argument("--n", type=int, default=50, help="Number of continuations to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=0, help="0 => dynamic (None)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = json.loads(Path(args.config).read_text())
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("base_model", "Qwen/Qwen3-8B")
    transcoder_set = model_cfg.get("transcoder", "mwhanna/qwen3-8b-transcoders")

    results_dir = Path(args.results_dir)
    pt_path = results_dir / "3_attribution_graphs" / f"{args.prefix_id}_prefix_context.pt"
    branches_path = results_dir / "2_branch_sampling" / f"{args.prefix_id}_branches.json"
    if not pt_path.exists() or not branches_path.exists():
        raise SystemExit(f"Missing required files for {args.prefix_id} under {results_dir}")

    ctx = torch.load(pt_path, map_location="cpu", weights_only=False)
    token_attributions = ctx.get("token_attributions") or []
    n_cont_total = len(token_attributions)
    n_features = int(ctx.get("n_prefix_features", 0) or 0)
    if n_cont_total == 0 or n_features <= 0:
        raise SystemExit("prefix_context missing token_attributions or n_prefix_features")

    # Sample continuation indices
    all_indices = list(range(n_cont_total))
    random.shuffle(all_indices)
    cont_indices = sorted(all_indices[: min(args.n, n_cont_total)])

    branches = json.loads(branches_path.read_text())
    continuations = branches.get("continuations", [])
    if len(continuations) < n_cont_total:
        raise SystemExit("branches.json has fewer continuations than prefix_context.pt")

    # Feature mapping
    active_features, selected_features = graph7c.load_attribution_context(
        results_dir / "3_attribution_graphs", args.prefix_id, use_continuation_attribution=True
    )

    # Choose feature (either fixed via H_c, or per-continuation attr)
    mu_a_val = None
    cluster_id = None
    if args.feature_source == "hc":
        clus_path = results_dir / "5_clustering" / f"{args.prefix_id}_sweep_results.json"
        if not clus_path.exists():
            raise SystemExit(f"Missing clustering sweep: {clus_path}")
        clus = json.loads(clus_path.read_text())
        entry = _find_grid_entry(clus, args.beta, args.gamma)
        if entry is None:
            raise SystemExit(f"No clustering entry for beta={args.beta}, gamma={args.gamma}")
        assignments = entry.get("assignments", [])
        components = entry.get("components", {}) or {}
        if not assignments or not components:
            raise SystemExit("clustering entry missing assignments/components")
        if args.cluster_id != -1:
            cluster_id = int(args.cluster_id)
        else:
            counts = {}
            for a in assignments:
                try:
                    a = int(a)
                except Exception:
                    continue
                counts[a] = counts.get(a, 0) + 1
            cluster_id = max(counts.items(), key=lambda kv: kv[1])[0]
        comp = components.get(str(cluster_id))
        if comp is None or "mu_a" not in comp:
            raise SystemExit(f"cluster {cluster_id} missing mu_a")
        mu_a = np.asarray(comp["mu_a"], dtype=np.float64)
        if mu_a.size < n_features:
            raise SystemExit("mu_a shorter than n_features")
        top_f = int(np.argmax(np.abs(mu_a[:n_features])))
        mu_a_val = float(mu_a[top_f])
        h_c_val = float(mu_a_val)

        layer, pos, feat_id = [int(x) for x in active_features[selected_features[top_f]].tolist()]

        # minimal caches for this single feature
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model: {model_name} + {transcoder_set} on {device} ...")
        model = ReplacementModel.from_pretrained(
            model_name, transcoder_set, device=device, dtype=torch.bfloat16,
            lazy_encoder=True, lazy_decoder=False
        )

        feat_ids_t = torch.tensor([feat_id], device=device, dtype=torch.long)
        W_enc_subset, b_enc_subset = utils7c.get_encoder_weights(model, layer, feat_ids_t)
        encoder_cache = {
            layer: {"W_enc": W_enc_subset, "b_enc": b_enc_subset, "feat_ids": [feat_id], "feat_id_to_idx": {feat_id: 0}}
        }
        dec_vec = model.transcoders._get_decoder_vectors(layer, feat_ids_t)  # [1, d_model]
        decoder_cache = {layer: ([pos], [feat_id], dec_vec, torch.tensor([h_c_val], device=device, dtype=model.cfg.dtype))}
        features = [(layer, pos, feat_id, h_c_val)]
    else:
        # For per-continuation attr mode, we will pick top_f (and corresponding caches) per continuation.
        # This requires model loaded once, but caches will be rebuilt per continuation.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model: {model_name} + {transcoder_set} on {device} ...")
        model = ReplacementModel.from_pretrained(
            model_name, transcoder_set, device=device, dtype=torch.bfloat16,
            lazy_encoder=True, lazy_decoder=False
        )

    max_seq_len = None if int(args.max_seq_len) == 0 else int(args.max_seq_len)

    per_cont_rows = []
    corrs = []
    agrees = []

    def _process_chunk(batch_full_ids: List[List[int]], batch_cont_info: List[Tuple[List[int], int]]):
        # baseline
        logits_base, _ = steering.run_batched_steered_pass_on_the_fly(
            model, batch_full_ids,
            features=[], encoder_cache={}, decoder_cache={},
            steering_method="additive", epsilon=0.0, max_seq_len=max_seq_len
        )
        # steered (uses outer-scope features/caches)
        logits_steer, _ = steering.run_batched_steered_pass_on_the_fly(
            model, batch_full_ids,
            features=features, encoder_cache=encoder_cache, decoder_cache=decoder_cache,
            steering_method=args.steering_method, epsilon=float(args.epsilon), max_seq_len=max_seq_len
        )
        base_centered = steering.compute_per_token_centered_logits_batched(logits_base, batch_cont_info)
        steer_centered = steering.compute_per_token_centered_logits_batched(logits_steer, batch_cont_info)
        return base_centered, steer_centered

    if args.feature_source == "hc":
        # Prepare all items once, then chunk
        items = []
        for i in cont_indices:
            cont = continuations[i]
            full_ids = cont["full_token_ids"]
            cont_ids = cont["token_ids"]
            cont_start = len(full_ids) - len(cont_ids)
            items.append((i, full_ids, cont_ids, cont_start))

        for start in range(0, len(items), args.chunk_size):
            chunk = items[start : start + args.chunk_size]
            batch_full_ids = [x[1] for x in chunk]
            batch_cont_info = [(x[2], x[3]) for x in chunk]
            base_centered, steer_centered = _process_chunk(batch_full_ids, batch_cont_info)

            for local_idx, (i, _, _, _) in enumerate(chunk):
                base_seq, _ = base_centered[local_idx]
                steer_seq, _ = steer_centered[local_idx]
                L = min(len(base_seq), len(steer_seq), token_attributions[i].shape[0])
                if L <= 1:
                    continue
                dcl = np.asarray(steer_seq[:L], dtype=np.float64) - np.asarray(base_seq[:L], dtype=np.float64)
                a = token_attributions[i][:L, top_f].to(torch.float32).numpy().astype(np.float64)
                corr = _pearson_corr(a, dcl)
                agree = float(np.mean((np.sign(a) == np.sign(dcl)).astype(np.float64)))
                per_cont_rows.append({
                    "cont_idx": int(i),
                    "corr_attr_delta_centered_logit": corr,
                    "sign_agree": agree,
                    "std_attr": float(np.std(a)) if a.size else None,
                    "std_delta_centered_logit": float(np.std(dcl)) if dcl.size else None,
                    "mean_abs_delta_centered_logit": float(np.mean(np.abs(dcl))) if dcl.size else None,
                })
                if corr is not None:
                    corrs.append(float(corr))
                agrees.append(float(agree))

            # free logits
            del base_centered, steer_centered
            torch.cuda.empty_cache()
    else:
        # attr-per-continuation mode: run one-by-one (still fast enough for N=50), since features differ
        for i in cont_indices:
            attrs_i = token_attributions[i].to(torch.float32)  # [T, D]
            summed = attrs_i[:, :n_features].sum(dim=0)
            top_f = int(torch.argmax(torch.abs(summed)).item())
            a_total = float(summed[top_f].item())
            a_sign = 0.0 if a_total == 0 else float(np.sign(a_total))
            layer, pos, feat_id = [int(x) for x in active_features[selected_features[top_f]].tolist()]

            feat_ids_t = torch.tensor([feat_id], device=model.cfg.device, dtype=torch.long)
            W_enc_subset, b_enc_subset = utils7c.get_encoder_weights(model, layer, feat_ids_t)
            encoder_cache = {
                layer: {"W_enc": W_enc_subset, "b_enc": b_enc_subset, "feat_ids": [feat_id], "feat_id_to_idx": {feat_id: 0}}
            }
            dec_vec = model.transcoders._get_decoder_vectors(layer, feat_ids_t)
            decoder_cache = {layer: ([pos], [feat_id], dec_vec, torch.tensor([a_sign], device=model.cfg.device, dtype=model.cfg.dtype))}
            features = [(layer, pos, feat_id, a_sign)]

            cont = continuations[i]
            full_ids = cont["full_token_ids"]
            cont_ids = cont["token_ids"]
            cont_start = len(full_ids) - len(cont_ids)
            batch_full_ids = [full_ids]
            batch_cont_info = [(cont_ids, cont_start)]
            base_centered, steer_centered = _process_chunk(batch_full_ids, batch_cont_info)
            base_seq, _ = base_centered[0]
            steer_seq, _ = steer_centered[0]

            L = min(len(base_seq), len(steer_seq), attrs_i.shape[0])
            if L <= 1:
                continue
            dcl = np.asarray(steer_seq[:L], dtype=np.float64) - np.asarray(base_seq[:L], dtype=np.float64)
            a = attrs_i[:L, top_f].numpy().astype(np.float64)
            corr = _pearson_corr(a, dcl)
            agree = float(np.mean((np.sign(a) == np.sign(dcl)).astype(np.float64)))

            per_cont_rows.append({
                "cont_idx": int(i),
                "top_f": int(top_f),
                "layer": int(layer),
                "pos": int(pos),
                "feat_id": int(feat_id),
                "sum_attr": float(a_total),
                "corr_attr_delta_centered_logit": corr,
                "sign_agree": agree,
            })
            if corr is not None:
                corrs.append(float(corr))
            agrees.append(float(agree))
            torch.cuda.empty_cache()

    payload = {
        "prefix_id": args.prefix_id,
        "results_dir": str(results_dir),
        "feature_source": args.feature_source,
        "beta": args.beta,
        "gamma": args.gamma,
        "cluster_id": cluster_id,
        "steering_method": args.steering_method,
        "epsilon": float(args.epsilon),
        "n_requested": int(args.n),
        "n_used": int(len(per_cont_rows)),
        "summary": {
            "corr_attr_delta_centered_logit": _summarize(corrs),
            "sign_agree": _summarize(agrees),
        },
        "rows": per_cont_rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(str(out_path))


if __name__ == "__main__":
    main()


