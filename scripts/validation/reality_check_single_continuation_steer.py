#!/usr/bin/env python3
"""
Reality check: pick one continuation, pick top-1 feature by attribution, steer that feature,
and compare per-token Δlogp with attribution sign.

This uses the SAME steering hook path as Stage 7c:
  - ReplacementModel + transcoders
  - 7c_steering.run_batched_steered_pass_on_the_fly

Inputs:
  - prefix_id (e.g. cloze_0105)
  - continuation index (0..)
  - epsilon and steering_method (default: additive)

We can select the top-1 feature either:
  - by cluster semantic graph H_c (Stage-5 mu_a) (recommended for matching Stage 7),
  - or by per-continuation token attribution (legacy sanity check).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Make latent_planning importable (repo root assumed two parents up from this script)
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# circuit-tracer path (same as 7c_hypotheses.py)
CIRCUIT_TRACER_PATH = REPO_ROOT / "circuit-tracer"
sys.path.insert(0, str(CIRCUIT_TRACER_PATH))

from circuit_tracer import ReplacementModel  # noqa: E402

# Import 7c modules the same way Stage 7 does
import importlib.util  # noqa: E402


def _import_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mod_dir = (REPO_ROOT / "7_validation")
steering = _import_module("7c_steering", _mod_dir / "7c_steering.py")
utils7c = _import_module("7c_utils", _mod_dir / "7c_utils.py")
graph7c = _import_module("7c_graph", _mod_dir / "7c_graph.py")


def per_token_logprobs_for_continuation(
    logits: torch.Tensor,
    cont_ids: list[int],
    cont_start: int,
) -> list[float]:
    """Match compute_continuation_log_prob_batched indexing, but return per-token logp list."""
    # logits: [1, seq_len, vocab]
    seq_len = logits.shape[1]
    valid_len = min(len(cont_ids), seq_len - cont_start)
    if valid_len <= 0:
        return []

    positions = torch.arange(cont_start, cont_start + valid_len, device=logits.device)
    cont_logits = logits[0, positions, :].float()  # [valid_len, vocab]
    log_probs = F.log_softmax(cont_logits, dim=-1)  # [valid_len, vocab]
    token_ids = torch.tensor(cont_ids[:valid_len], device=logits.device, dtype=torch.long)
    selected = log_probs[torch.arange(valid_len, device=logits.device), token_ids]
    return [float(x) for x in selected.detach().cpu().tolist()]


def per_token_target_logits_for_continuation(
    logits: torch.Tensor,
    cont_ids: list[int],
    cont_start: int,
) -> list[float]:
    """Return the raw logit assigned to the realized continuation token at each step."""
    # logits: [1, seq_len, vocab]
    seq_len = logits.shape[1]
    valid_len = min(len(cont_ids), seq_len - cont_start)
    if valid_len <= 0:
        return []

    positions = torch.arange(cont_start, cont_start + valid_len, device=logits.device)
    token_ids = torch.tensor(cont_ids[:valid_len], device=logits.device, dtype=torch.long)
    # gather logits for each (pos, token_id)
    selected = logits[0, positions, token_ids].float()
    return [float(x) for x in selected.detach().cpu().tolist()]


def per_token_centered_logits_for_continuation(
    logits: torch.Tensor,
    cont_ids: list[int],
    cont_start: int,
) -> list[float]:
    """Centered logit = logit(target_token) - mean(logits over vocabulary) at each position."""
    # logits: [1, seq_len, vocab]
    seq_len = logits.shape[1]
    valid_len = min(len(cont_ids), seq_len - cont_start)
    if valid_len <= 0:
        return []

    positions = torch.arange(cont_start, cont_start + valid_len, device=logits.device)
    cont_logits = logits[0, positions, :].float()  # [valid_len, vocab]
    mean_logits = cont_logits.mean(dim=-1)  # [valid_len]
    token_ids = torch.tensor(cont_ids[:valid_len], device=logits.device, dtype=torch.long)
    target_logits = cont_logits[torch.arange(valid_len, device=logits.device), token_ids]
    centered = target_logits - mean_logits
    return [float(x) for x in centered.detach().cpu().tolist()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Main config JSON (for model + dtype).")
    ap.add_argument("--results-dir", required=True, help="gpu*/results directory containing 2_branch_sampling and 3_attribution_graphs")
    ap.add_argument("--prefix-id", required=True)
    ap.add_argument("--cont-idx", type=int, default=0)
    ap.add_argument("--feature-source", choices=["hc", "attr"], default="hc", help="How to pick the top-1 feature.")
    ap.add_argument("--beta", type=float, default=3.0, help="Beta for clustering entry (used when feature-source=hc).")
    ap.add_argument("--gamma", type=float, default=0.5, help="Gamma for clustering entry (used when feature-source=hc).")
    ap.add_argument("--cluster-id", type=int, default=-1, help="Cluster id for H_c (mu_a). -1 chooses largest cluster by assignments.")
    ap.add_argument("--epsilon", type=float, default=0.5)
    ap.add_argument("--steering-method", type=str, default="additive", choices=["additive", "multiplicative", "sign", "scaling", "absolute"])
    ap.add_argument("--max-seq-len", type=int, default=0, help="0 means dynamic padding (None)")
    ap.add_argument("--topk-print", type=int, default=20, help="How many continuation tokens to print")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("base_model", "Qwen/Qwen3-8B")
    transcoder_set = model_cfg.get("transcoder", "mwhanna/qwen3-8b-transcoders")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load prefix_context (for token_attributions + feature mapping)
    results_dir = Path(args.results_dir)
    pt_path = results_dir / "3_attribution_graphs" / f"{args.prefix_id}_prefix_context.pt"
    branches_path = results_dir / "2_branch_sampling" / f"{args.prefix_id}_branches.json"
    if not pt_path.exists():
        raise SystemExit(f"Missing: {pt_path}")
    if not branches_path.exists():
        raise SystemExit(f"Missing: {branches_path}")

    ctx = torch.load(pt_path, map_location="cpu", weights_only=False)
    token_attributions = ctx.get("token_attributions")
    if not token_attributions or args.cont_idx >= len(token_attributions):
        raise SystemExit(f"Invalid cont-idx={args.cont_idx}; token_attributions len={len(token_attributions) if token_attributions else 0}")
    attrs = token_attributions[args.cont_idx].to(torch.float32)  # [T, D]

    n_features = int(ctx.get("n_prefix_features", 0) or 0)
    if n_features <= 0:
        raise SystemExit("n_prefix_features missing/invalid in prefix_context.pt")

    # Choose top-1 feature
    chosen_by = args.feature_source
    mu_a_val = None
    cluster_id = None
    if args.feature_source == "attr":
        # Per-continuation: choose top-1 feature by |sum_t attr[t,f]|
        summed = attrs[:, :n_features].sum(dim=0)  # [n_features]
        top_f = int(torch.argmax(torch.abs(summed)).item())
        a_total = float(summed[top_f].item())
        a_sign = 0.0 if a_total == 0 else float(np.sign(a_total))
    else:
        # By cluster H_c: use Stage-5 mu_a for chosen (beta,gamma,cluster)
        clus_path = results_dir / "5_clustering" / f"{args.prefix_id}_sweep_results.json"
        if not clus_path.exists():
            raise SystemExit(f"Missing clustering sweep: {clus_path}")
        clus = json.loads(clus_path.read_text())
        entry = None
        for e in (clus.get("grid", []) or []):
            try:
                b = float(e.get("beta"))
                g = float(e.get("gamma"))
            except Exception:
                continue
            if abs(b - float(args.beta)) < 1e-9 and abs(g - float(args.gamma)) < 1e-9:
                entry = e
                break
        if entry is None:
            raise SystemExit(f"No grid entry for beta={args.beta}, gamma={args.gamma} in {clus_path}")
        assignments = entry.get("assignments", [])
        components = entry.get("components", {}) or {}
        if not assignments or not components:
            raise SystemExit("Clustering entry missing assignments/components.")

        # choose cluster id
        if args.cluster_id != -1:
            cluster_id = int(args.cluster_id)
        else:
            # largest cluster by assignments
            counts = {}
            for a in assignments:
                try:
                    a = int(a)
                except Exception:
                    continue
                counts[a] = counts.get(a, 0) + 1
            if not counts:
                raise SystemExit("Could not determine largest cluster (empty/invalid assignments).")
            cluster_id = max(counts.items(), key=lambda kv: kv[1])[0]

        comp = components.get(str(cluster_id))
        if comp is None or "mu_a" not in comp:
            raise SystemExit(f"Cluster {cluster_id} missing mu_a in clustering entry.")
        mu_a = np.asarray(comp["mu_a"], dtype=np.float64)
        if mu_a.ndim != 1:
            raise SystemExit("mu_a is not a 1D vector.")
        if mu_a.size < n_features:
            raise SystemExit(f"mu_a length {mu_a.size} < n_features {n_features}; cannot map H_c feature dims.")

        # top-1 feature index by |H_c|
        top_f = int(np.argmax(np.abs(mu_a[:n_features])))
        mu_a_val = float(mu_a[top_f])
        a_sign = 0.0 if mu_a_val == 0 else float(np.sign(mu_a_val))
        # still compute attribution sum for reporting
        a_total = float(attrs[:, top_f].sum().item())

    # map feature index -> (layer,pos,feat_id)
    # (this reconstructs active_features consistent with Stage 7)
    active_features, selected_features = graph7c.load_attribution_context(
        results_dir / "3_attribution_graphs", args.prefix_id, use_continuation_attribution=True
    )
    layer, pos, feat_id = [int(x) for x in active_features[selected_features[top_f]].tolist()]

    # Load branch tokens
    branches = json.loads(branches_path.read_text())
    cont = branches["continuations"][args.cont_idx]
    full_token_ids = cont["full_token_ids"]
    cont_ids = cont["token_ids"]
    cont_start = len(full_token_ids) - len(cont_ids)

    # Load model
    print(f"Loading model: {model_name} + {transcoder_set} on {device} ...")
    model = ReplacementModel.from_pretrained(
        model_name, transcoder_set, device=device, dtype=torch.bfloat16,
        lazy_encoder=True, lazy_decoder=False
    )

    # Build minimal caches for this single feature
    # - encoder weights for this feat_id in this layer
    feat_ids_t = torch.tensor([feat_id], device=device, dtype=torch.long)
    W_enc_subset, b_enc_subset = utils7c.get_encoder_weights(model, layer, feat_ids_t)
    encoder_cache = {
        layer: {
            "W_enc": W_enc_subset,
            "b_enc": b_enc_subset,
            "feat_ids": [feat_id],
            "feat_id_to_idx": {feat_id: 0},
        }
    }

    # - decoder vector for this feat_id
    dec_vec = model.transcoders._get_decoder_vectors(layer, feat_ids_t)  # [1, d_model]
    decoder_cache = {
        # h_c_vals tensor is not used directly in hook, but we keep it consistent
        layer: ([pos], [feat_id], dec_vec, torch.tensor([float(mu_a_val) if mu_a_val is not None else float(a_sign)], device=device, dtype=model.cfg.dtype))
    }

    # Steering features:
    # - if chosen by H_c: use h_c_val = mu_a[top_f] (Stage-7 semantics)
    # - if chosen by attr: use h_c_val = sign(sum_t attr) to probe sign alignment
    h_c_val = float(mu_a_val) if mu_a_val is not None else float(a_sign)
    features = [(layer, pos, feat_id, h_c_val)]
    max_seq_len = None if args.max_seq_len == 0 else int(args.max_seq_len)

    # Baseline (no steering) using same runner with empty features
    logits_base, _ = steering.run_batched_steered_pass_on_the_fly(
        model, [full_token_ids],
        features=[], encoder_cache={}, decoder_cache={},
        steering_method="additive", epsilon=0.0,
        max_seq_len=max_seq_len
    )
    logits_steer, _ = steering.run_batched_steered_pass_on_the_fly(
        model, [full_token_ids],
        features=features,
        encoder_cache=encoder_cache,
        decoder_cache=decoder_cache,
        steering_method=args.steering_method,
        epsilon=float(args.epsilon),
        max_seq_len=max_seq_len
    )

    base_lp = per_token_logprobs_for_continuation(logits_base, cont_ids, cont_start)
    steer_lp = per_token_logprobs_for_continuation(logits_steer, cont_ids, cont_start)
    base_logit = per_token_target_logits_for_continuation(logits_base, cont_ids, cont_start)
    steer_logit = per_token_target_logits_for_continuation(logits_steer, cont_ids, cont_start)
    base_clogit = per_token_centered_logits_for_continuation(logits_base, cont_ids, cont_start)
    steer_clogit = per_token_centered_logits_for_continuation(logits_steer, cont_ids, cont_start)
    L = min(len(base_lp), len(steer_lp), attrs.shape[0])
    if L == 0:
        raise SystemExit("No valid continuation token positions to compare.")

    dlp = np.asarray(steer_lp[:L], dtype=np.float64) - np.asarray(base_lp[:L], dtype=np.float64)
    dlogit = np.asarray(steer_logit[:L], dtype=np.float64) - np.asarray(base_logit[:L], dtype=np.float64)
    dclogit = np.asarray(steer_clogit[:L], dtype=np.float64) - np.asarray(base_clogit[:L], dtype=np.float64)
    a_tok = attrs[:L, top_f].detach().cpu().numpy().astype(np.float64)

    # Relationship metrics
    corr = float(np.corrcoef(a_tok, dlp)[0, 1]) if np.std(a_tok) > 0 and np.std(dlp) > 0 else float("nan")
    agree = float(np.mean((np.sign(a_tok) == np.sign(dlp)).astype(np.float64)))
    corr_logit = float(np.corrcoef(a_tok, dlogit)[0, 1]) if np.std(a_tok) > 0 and np.std(dlogit) > 0 else float("nan")
    agree_logit = float(np.mean((np.sign(a_tok) == np.sign(dlogit)).astype(np.float64)))
    corr_clogit = float(np.corrcoef(a_tok, dclogit)[0, 1]) if np.std(a_tok) > 0 and np.std(dclogit) > 0 else float("nan")
    agree_clogit = float(np.mean((np.sign(a_tok) == np.sign(dclogit)).astype(np.float64)))

    print("\n=== Selected example ===")
    print(f"prefix_id: {args.prefix_id}")
    print(f"cont_idx: {args.cont_idx}")
    print(f"top_feature_index (within first {n_features}): {top_f}")
    print(f"mapped (layer,pos,feat_id): ({layer},{pos},{feat_id})")
    if chosen_by == "hc":
        print(f"feature_source: H_c (mu_a)   beta={args.beta} gamma={args.gamma}   cluster_id={cluster_id}")
        print(f"H_c[top_f] = {mu_a_val:+.6g} => sign(H_c)={a_sign:+.0f}")
        print(f"sum_token_attr(feature) = {a_total:+.6g} (for reference)")
    else:
        print(f"feature_source: token_attribution")
        print(f"sum_token_attr(feature) = {a_total:+.6g} => sign={a_sign:+.0f}")
    print(f"steering_method: {args.steering_method}   epsilon: {args.epsilon}")
    print(f"per-token corr(attr, Δlogp): {corr:+.4f}")
    print(f"per-token sign agreement (attr vs Δlogp): {agree:.3f}")
    print(f"per-token corr(attr, Δlogit): {corr_logit:+.4f}")
    print(f"per-token sign agreement (attr vs Δlogit): {agree_logit:.3f}")
    print(f"per-token corr(attr, Δcentered_logit): {corr_clogit:+.4f}")
    print(f"per-token sign agreement (attr vs Δcentered_logit): {agree_clogit:.3f}")

    # Print per-token table
    print("\nidx  token_id   attr(feature)   base_cLogit  steer_cLogit   ΔcLogit   base_logit    steer_logit     Δlogit    base_logp     steer_logp      Δlogp   sign(attr) sign(ΔcLogit) sign(Δlogit) sign(Δlogp)")
    for j in range(min(args.topk_print, L)):
        sa = int(np.sign(a_tok[j]))
        sdc = int(np.sign(dclogit[j]))
        sdlogit = int(np.sign(dlogit[j]))
        sdlogp = int(np.sign(dlp[j]))
        print(
            f"{j:>3d}  {cont_ids[j]:>7d}  {a_tok[j]:>+12.5e}  "
            f"{base_clogit[j]:>+11.5f}  {steer_clogit[j]:>+11.5f}  {dclogit[j]:>+9.5f}  "
            f"{base_logit[j]:>+11.5f}  {steer_logit[j]:>+11.5f}  {dlogit[j]:>+10.5f}  "
            f"{base_lp[j]:>+11.5f}  {steer_lp[j]:>+11.5f}  {dlp[j]:>+10.5f}     "
            f"{sa:>+2d}        {sdc:>+2d}        {sdlogit:>+2d}        {sdlogp:>+2d}"
        )


if __name__ == "__main__":
    main()


