#!/usr/bin/env -S uv run python
"""
Unified script for:
1) Sampling continuations + token/word attribution magnitudes (L1).
2) Teacher-forced steering sweeps over epsilons (per-continuation H_c) and per-token/per-word Δlogit/Δprob.
3) Position-wise attribution magnitude statistics (including variance).

Inputs:
  - One or more *_prefix_context.pt files produced by Stage 3 (continuation attribution).

For each continuation i, token position t:
  token_attributions[i] is a tensor of shape [T, n_sources]
We reduce over sources to a scalar magnitude per token position:
  - l1 (default): sum(|attr|) over sources (token-wise magnitude only; source ids discarded)

Then we aggregate across all continuations (and optionally across many prefixes)
and report distribution statistics per token position.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _reduce_token_attr_l1(x: torch.Tensor) -> torch.Tensor:
    """Token-wise L1 magnitude over sources.

    Args:
        x: [T, n_sources] attribution tensor

    Returns:
        [T] tensor: sum(abs(attr)) over sources for each token position.
    """
    return x.abs().to(torch.float32).sum(dim=1)


def _quantiles(values: np.ndarray, qs: Tuple[float, ...]) -> Dict[str, float]:
    if values.size == 0:
        return {f"p{int(q*100):02d}": float("nan") for q in qs}
    out = np.quantile(values, qs, method="linear")
    return {f"p{int(q*100):02d}": float(v) for q, v in zip(qs, out)}


def _safe_prefix_id_from_context_path(path: str) -> str:
    stem = Path(path).name
    if stem.endswith("_prefix_context.pt"):
        return stem[: -len("_prefix_context.pt")]
    return Path(stem).stem


def _load_prefix_context(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _require_token_attributions(d: Dict[str, Any], path: str) -> List[torch.Tensor]:
    token_attrs = d.get("token_attributions")
    if not isinstance(token_attrs, list) or not token_attrs:
        raise ValueError(
            f"{path}: missing `token_attributions` (Stage 3 must be run with --store-all)"
        )
    return token_attrs


def _get_continuation_tokens(d: Dict[str, Any], path: str) -> List[List[int]]:
    cont_tokens = d.get("continuation_tokens")
    if isinstance(cont_tokens, list) and cont_tokens and isinstance(cont_tokens[0], list):
        return cont_tokens
    raise ValueError(
        f"{path}: missing `continuation_tokens` (Stage 3 must be run with --store-all)"
    )


def _summarize_positions_l1(
    pt_files: List[str],
    max_files: int | None = None,
) -> List[Dict[str, Any]]:
    # position -> list of L1 magnitudes
    by_pos: Dict[int, List[float]] = defaultdict(list)
    used_files = 0

    for path in pt_files:
        if max_files is not None and used_files >= max_files:
            break
        used_files += 1
        d = _load_prefix_context(path)
        token_attrs = _require_token_attributions(d, path)

        for ta in token_attrs:
            if not torch.is_tensor(ta) or ta.ndim != 2:
                continue
            mags = _reduce_token_attr_l1(ta)  # [T]
            mags_np = mags.numpy()
            for t, v in enumerate(mags_np.tolist()):
                by_pos[t].append(float(v))

        del d

    rows: List[Dict[str, Any]] = []
    qs = (0.1, 0.5, 0.9, 0.95, 0.99)
    for t in sorted(by_pos.keys()):
        vals = np.asarray(by_pos[t], dtype=np.float32)
        row: Dict[str, Any] = {
            "pos": int(t),
            "n": int(vals.size),
            "mean": float(vals.mean()) if vals.size else float("nan"),
            "std": float(vals.std()) if vals.size else float("nan"),
            "var": float(vals.var()) if vals.size else float("nan"),
        }
        row.update(_quantiles(vals, qs))
        rows.append(row)
    return rows


def _try_load_tokenizer(model_name: str):
    # Avoid hard dependency on transformers for position-only mode.
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Word-level outputs require `transformers` (AutoTokenizer). "
            "In this repo, run via `uv run python ...` so dependencies from pyproject.toml are available, "
            "or run without word-level outputs."
        ) from e
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def _group_tokens_to_words(token_strs: List[str]) -> List[List[int]]:
    """Group token indices into word groups.

    Uses tokenizer token markers when present (▁ or Ġ). Falls back to leading whitespace.
    """
    if not token_strs:
        return []

    has_sentencepiece = any(s.startswith("▁") for s in token_strs)
    has_gpt2_bpe = any(s.startswith("Ġ") for s in token_strs)

    groups: List[List[int]] = []
    cur: List[int] = []

    def _starts_new_word(s: str) -> bool:
        if has_sentencepiece:
            return s.startswith("▁")
        if has_gpt2_bpe:
            return s.startswith("Ġ")
        # fallback: whitespace prefix in decoded token piece
        return s[:1].isspace()

    for i, s in enumerate(token_strs):
        if i == 0:
            cur = [0]
            continue
        if _starts_new_word(s) and cur:
            groups.append(cur)
            cur = [i]
        else:
            cur.append(i)

    if cur:
        groups.append(cur)
    return groups


def _reduce_by_groups(values: List[float], groups: List[List[int]], reduction: str) -> List[float]:
    out: List[float] = []
    arr = np.asarray(values, dtype=np.float64)
    for g in groups:
        if not g:
            continue
        v = arr[np.asarray(g, dtype=np.int64)]
        if reduction == "sum":
            out.append(float(v.sum()))
        elif reduction == "mean":
            out.append(float(v.mean()))
        else:
            raise ValueError(f"Unknown word reduction: {reduction}")
    return out


def _compute_target_logit_prob_arrays(
    logits: torch.Tensor,  # [1, seq, vocab]
    cont_ids: List[int],
    cont_start: int,
) -> Tuple[List[float], List[float], List[float]]:
    """Teacher-forced arrays along the continuation:
    - target logits
    - target probs
    - demeaned target logits = target_logit - mean(logits over vocab) at each position
    """
    seq_len = logits.shape[1]
    valid_len = min(len(cont_ids), seq_len - cont_start)
    if valid_len <= 0:
        return [], [], []

    positions = torch.arange(cont_start, cont_start + valid_len, device=logits.device)
    cont_logits = logits[0, positions, :].float()  # [T, V]
    token_ids = torch.tensor(cont_ids[:valid_len], device=logits.device, dtype=torch.long)

    target_logits = cont_logits[torch.arange(valid_len, device=logits.device), token_ids]  # [T]
    mean_logits = cont_logits.mean(dim=-1)  # [T]
    demeaned_target_logits = target_logits - mean_logits

    log_probs = F.log_softmax(cont_logits, dim=-1)
    target_logprobs = log_probs[torch.arange(valid_len, device=logits.device), token_ids]  # [T]
    target_probs = torch.exp(target_logprobs)

    return target_logits.tolist(), target_probs.tolist(), demeaned_target_logits.tolist()


def _import_7c_module(name: str, file_path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_replacement_model(model_name: str, transcoder_name: str, dtype_str: str):
    # Match Stage 7 scripts: add repo root + circuit-tracer to sys.path.
    script_dir = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(script_dir))
    sys.path.insert(0, str(script_dir / "circuit-tracer"))
    from circuit_tracer import ReplacementModel  # type: ignore

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}")
    return ReplacementModel.from_pretrained(model_name, transcoder_name, dtype=dtype_map[dtype_str])


@dataclass
class SampleSpec:
    context_path: str
    prefix_id: str
    cont_idx: int


def _sample_continuations(
    pt_files: List[str],
    n_samples: int,
    seed: int,
    max_files: Optional[int] = None,
) -> List[SampleSpec]:
    rng = np.random.RandomState(seed)
    candidates: List[SampleSpec] = []
    used_files = 0
    for path in pt_files:
        if max_files is not None and used_files >= max_files:
            break
        used_files += 1
        d = _load_prefix_context(path)
        token_attrs = _require_token_attributions(d, path)
        prefix_id = _safe_prefix_id_from_context_path(path)
        for i in range(len(token_attrs)):
            candidates.append(SampleSpec(context_path=path, prefix_id=prefix_id, cont_idx=i))
        del d

    if not candidates:
        return []
    if n_samples >= len(candidates):
        rng.shuffle(candidates)
        return candidates
    idx = rng.choice(len(candidates), size=n_samples, replace=False)
    return [candidates[i] for i in idx.tolist()]


def _select_topB_features_from_Hc(
    H_c: np.ndarray,
    active_features: torch.Tensor,      # [N,3]
    selected_features: torch.Tensor,    # [N]
    top_B: int,
    hc_selection: str,
) -> List[Tuple[int, int, int, float]]:
    n_features = int(selected_features.numel())
    Hf = H_c[:n_features]
    if hc_selection == "positive":
        valid = np.where(Hf > 0)[0]
    elif hc_selection == "negative":
        valid = np.where(Hf < 0)[0]
    else:
        valid = np.arange(len(Hf))
    if valid.size == 0:
        return []
    top = valid[np.argsort(np.abs(Hf[valid]))[::-1]]
    if top_B > 0:
        top = top[:top_B]

    feats: List[Tuple[int, int, int, float]] = []
    for idx in top.tolist():
        feat_idx = int(selected_features[idx].item())
        layer, pos, feat_id = active_features[feat_idx].tolist()
        feats.append((int(layer), int(pos), int(feat_id), float(Hf[idx])))
    return feats


def _make_random_feature_baseline(
    *,
    active_features: torch.Tensor,  # [N,3]
    selected_features: torch.Tensor,  # [N]
    avoid: List[Tuple[int, int, int, float]],
    k: int,
    h_c_vals_source: List[float],
    seed: int,
) -> List[Tuple[int, int, int, float]]:
    """Pick random (layer,pos,feat_id) and assign H_c values by shuffling provided values.

    This preserves the H_c value distribution/signs while breaking alignment to actual features.
    """
    rng = np.random.RandomState(seed)
    avoid_set = {(int(l), int(p), int(f)) for (l, p, f, _) in avoid}

    # Candidate locations from active_features/selected_features
    cand: List[Tuple[int, int, int]] = []
    for idx in range(int(selected_features.numel())):
        feat_idx = int(selected_features[idx].item())
        l, p, f = active_features[feat_idx].tolist()
        tpl = (int(l), int(p), int(f))
        if tpl in avoid_set:
            continue
        cand.append(tpl)

    if not cand:
        # fallback: allow overlap if avoid removed everything
        for idx in range(int(selected_features.numel())):
            feat_idx = int(selected_features[idx].item())
            l, p, f = active_features[feat_idx].tolist()
            cand.append((int(l), int(p), int(f)))

    if not cand:
        return []

    k_eff = min(k, len(cand))
    picks = [cand[i] for i in rng.choice(len(cand), size=k_eff, replace=False).tolist()]

    hc = list(h_c_vals_source)[:k_eff]
    rng.shuffle(hc)

    return [(picks[i][0], picks[i][1], picks[i][2], float(hc[i])) for i in range(k_eff)]


def main():
    # CLI supports both:
    # - Legacy: `python script.py --pt-glob ...` (runs position-stats)
    # - New:    `python script.py sample --pt-glob ...` etc (subcommands)
    argv = sys.argv[1:]
    cmds = {"position-stats", "sample", "steer"}
    first_pos = next((a for a in argv if not a.startswith("-")), None)
    use_subcommands = first_pos in cmds

    if use_subcommands:
        ap = argparse.ArgumentParser(description="Attribution magnitude + steering analysis")
        sub = ap.add_subparsers(dest="cmd", required=True)

        common = argparse.ArgumentParser(add_help=False)
        common.add_argument(
            "--pt-glob",
            required=True,
            help="Glob for *_prefix_context.pt files (quote it).",
        )
        common.add_argument("--max-files", type=int, default=None)
        common.add_argument("--out", default=None, help="Output JSON path (or stdout if omitted).")

        sub.add_parser("position-stats", parents=[common], help="Position-wise L1 attribution stats (includes variance).")

        ap_sample = sub.add_parser("sample", parents=[common], help="Sample continuations and compute token/word L1 magnitudes.")
        ap_sample.add_argument("--n-samples", type=int, default=10)
        ap_sample.add_argument("--seed", type=int, default=42)
        ap_sample.add_argument("--tokenizer-model", type=str, required=True, help="HF model id for tokenizer.")
        ap_sample.add_argument("--word-reduction", choices=["sum", "mean"], default="sum")

        ap_steer = sub.add_parser("steer", parents=[common], help="Teacher-forced steering sweeps (per-continuation H_c).")
        ap_steer.add_argument("--n-samples", type=int, default=10)
        ap_steer.add_argument("--seed", type=int, default=42)
        ap_steer.add_argument("--model", type=str, required=True)
        ap_steer.add_argument("--transcoder", type=str, required=True)
        ap_steer.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
        ap_steer.add_argument("--epsilons", type=float, nargs="+", required=True)
        ap_steer.add_argument("--top-B", type=int, default=10)
        ap_steer.add_argument("--hc-selection", choices=["full", "positive", "negative"], default="full")
        ap_steer.add_argument("--steering-method", choices=["additive", "multiplicative", "absolute", "sign", "scaling"], default="multiplicative")
        ap_steer.add_argument("--word-reduction", choices=["sum", "mean"], default="mean")
        ap_steer.add_argument("--batch-size", type=int, default=16)
        ap_steer.add_argument("--max-seq-len", type=int, default=None)
        ap_steer.add_argument(
            "--random-baseline",
            action="store_true",
            help="Also run a baseline steering pass using random features (same count, shuffled H_c values).",
        )
        ap_steer.add_argument(
            "--random-baseline-seed",
            type=int,
            default=0,
            help="Seed for random feature baseline selection (per-sample seed = base + hash(sample_key)).",
        )

        args = ap.parse_args()
        cmd = args.cmd
        pt_glob = args.pt_glob
        max_files = args.max_files
        out_path = args.out
    else:
        ap = argparse.ArgumentParser(description="Attribution magnitude + steering analysis (legacy mode)")
        ap.add_argument(
            "--pt-glob",
            required=True,
            help="Glob for *_prefix_context.pt files (quote it).",
        )
        ap.add_argument("--max-files", type=int, default=None)
        ap.add_argument("--out", default=None, help="Output JSON path (or stdout if omitted).")
        args = ap.parse_args()
        cmd = "position-stats"
        pt_glob = args.pt_glob
        max_files = args.max_files
        out_path = args.out

    pt_files = sorted(glob.glob(pt_glob))
    if not pt_files:
        raise SystemExit(f"No files matched: {pt_glob}")

    payload: Dict[str, Any] = {"cmd": cmd, "pt_glob": pt_glob, "n_files_matched": len(pt_files), "max_files": max_files}

    if cmd == "position-stats":
        rows = _summarize_positions_l1(pt_files, max_files=max_files)
        payload["position_stats"] = rows

    elif cmd == "sample":
        # Also include global position-wise stats for convenience (requirement 3).
        payload["position_stats"] = _summarize_positions_l1(pt_files, max_files=max_files)

        tok = _try_load_tokenizer(args.tokenizer_model)
        specs = _sample_continuations(pt_files, n_samples=args.n_samples, seed=args.seed, max_files=max_files)
        samples: List[Dict[str, Any]] = []
        for spec in specs:
            d = _load_prefix_context(spec.context_path)
            token_attrs = _require_token_attributions(d, spec.context_path)
            cont_tokens = _get_continuation_tokens(d, spec.context_path)
            ta = token_attrs[spec.cont_idx]
            cont_ids = cont_tokens[spec.cont_idx]
            l1 = _reduce_token_attr_l1(ta).tolist()
            token_strs = tok.convert_ids_to_tokens(cont_ids)
            groups = _group_tokens_to_words(token_strs)
            word_l1 = _reduce_by_groups(l1, groups, reduction=args.word_reduction)
            samples.append(
                {
                    "prefix_id": spec.prefix_id,
                    "context_path": spec.context_path,
                    "cont_idx": spec.cont_idx,
                    "continuation_token_ids": cont_ids,
                    "continuation_token_strs": token_strs,
                    "token_attr_l1": l1,
                    "word_groups": groups,
                    "word_attr_l1": word_l1,
                }
            )
            del d
        payload["samples"] = samples
        payload["tokenizer_model"] = args.tokenizer_model
        payload["word_reduction"] = args.word_reduction
        payload["attr_magnitude"] = {"type": "l1", "definition": "sum(abs(token_attributions[t, :])) over sources"}

    elif cmd == "steer":
        # Also include global position-wise stats for convenience (requirement 3).
        payload["position_stats"] = _summarize_positions_l1(pt_files, max_files=max_files)

        # Load model + 7c modules
        model = _load_replacement_model(args.model, args.transcoder, args.dtype)
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("ReplacementModel has no tokenizer; cannot produce word-level arrays.")

        module_dir = Path(__file__).resolve().parents[1] / "7_validation"
        graph = _import_7c_module("7c_graph", module_dir / "7c_graph.py")
        steering = _import_7c_module("7c_steering", module_dir / "7c_steering.py")

        specs = _sample_continuations(pt_files, n_samples=args.n_samples, seed=args.seed, max_files=max_files)
        if not specs:
            payload["samples"] = []
        else:
            # Build items
            items: List[Dict[str, Any]] = []
            baselines: List[Dict[str, Any]] = []

            for spec in specs:
                d = _load_prefix_context(spec.context_path)
                token_attrs = _require_token_attributions(d, spec.context_path)
                cont_tokens = _get_continuation_tokens(d, spec.context_path)
                aggregated = d.get("aggregated_attributions")
                if not torch.is_tensor(aggregated):
                    raise ValueError(f"{spec.context_path}: missing `aggregated_attributions`")
                H_c = aggregated[spec.cont_idx].float().cpu().numpy()

                # Map feature indices -> (layer,pos,feat_id)
                ctx_dir = Path(spec.context_path).parent
                active_features, selected_features = graph.load_attribution_context(ctx_dir, spec.prefix_id, use_continuation_attribution=True)

                feats = _select_topB_features_from_Hc(
                    H_c,
                    active_features=active_features,
                    selected_features=selected_features,
                    top_B=args.top_B,
                    hc_selection=args.hc_selection,
                )

                # Optional random-feature baseline (same count as feats, shuffled H_c values).
                rand_feats: List[Tuple[int, int, int, float]] = []
                rand_decoder_cache: Dict[int, Tuple[List[int], List[int], torch.Tensor, torch.Tensor]] = {}
                if getattr(args, "random_baseline", False) and feats:
                    hc_vals_src = [float(v) for (_, _, _, v) in feats]
                    seed0 = int(getattr(args, "random_baseline_seed", 0))
                    # stable-ish per-sample seed
                    seed = seed0 + (abs(hash(f"{spec.prefix_id}:{spec.cont_idx}")) % 1_000_000_000)
                    rand_feats = _make_random_feature_baseline(
                        active_features=active_features,
                        selected_features=selected_features,
                        avoid=feats,
                        k=len(feats),
                        h_c_vals_source=hc_vals_src,
                        seed=seed,
                    )
                    if rand_feats:
                        rand_decoder_cache_by_cluster = graph.precompute_cluster_decoder_vectors(
                            model, {0: rand_feats}, device=model.cfg.device
                        )
                        rand_decoder_cache = rand_decoder_cache_by_cluster.get(0, {})

                # Stage 3 stores prefix tokens as a Python list in most runs.
                # Accept both list and tensor for robustness.
                full_token_ids = d.get("prefix_tokens")
                if isinstance(full_token_ids, torch.Tensor):
                    prefix_ids = full_token_ids.tolist()
                elif isinstance(full_token_ids, list) and full_token_ids and isinstance(full_token_ids[0], int):
                    prefix_ids = full_token_ids
                else:
                    # Fallback: load Stage-2 branches JSON next to results (if present).
                    results_dir = Path(spec.context_path).parent.parent  # .../results
                    branches_path = results_dir / "2_branch_sampling" / f"{spec.prefix_id}_branches.json"
                    if branches_path.exists():
                        import json as _json
                        with open(branches_path, "r") as f:
                            bd = _json.load(f)
                        prefix_ids = bd.get("prefix_tokens_with_bos")
                        if not (isinstance(prefix_ids, list) and prefix_ids and isinstance(prefix_ids[0], int)):
                            raise ValueError(f"{branches_path}: missing `prefix_tokens_with_bos`")
                    else:
                        raise ValueError(
                            f"{spec.context_path}: missing usable `prefix_tokens` for building full_token_ids. "
                            f"Expected list[int] in the .pt, or Stage-2 branches at {branches_path}."
                        )

                cont_ids = cont_tokens[spec.cont_idx]
                full_ids = prefix_ids + cont_ids
                cont_start = len(full_ids) - len(cont_ids)

                # Precompute caches in the same shapes as Stage 7 expects.
                encoder_cache = graph.preload_encoder_weights_for_cluster(model, feats, device=model.cfg.device)
                decoder_cache_by_cluster = graph.precompute_cluster_decoder_vectors(model, {0: feats}, device=model.cfg.device)
                decoder_cache = decoder_cache_by_cluster.get(0, {})

                token_strs = tokenizer.convert_ids_to_tokens(cont_ids)
                word_groups = _group_tokens_to_words(token_strs)

                items.append(
                    {
                        "sample_key": f"{spec.prefix_id}:{spec.cont_idx}",
                        "prefix_id": spec.prefix_id,
                        "context_path": spec.context_path,
                        "cont_idx": spec.cont_idx,
                        "token_ids": full_ids,
                        "cont_ids": cont_ids,
                        "cont_start": cont_start,
                        "features": feats,
                        "decoder_cache": decoder_cache,
                        "encoder_cache": encoder_cache,  # not used by heterogeneous pass, but keep for reference
                        "random_baseline_enabled": bool(getattr(args, "random_baseline", False)),
                        "random_features": rand_feats,
                        "random_decoder_cache": rand_decoder_cache,
                        "token_strs": token_strs,
                        "word_groups": word_groups,
                    }
                )

                # Baseline attribution magnitude arrays for this continuation
                l1 = _reduce_token_attr_l1(token_attrs[spec.cont_idx]).tolist()
                baselines.append(
                    {
                        "sample_key": f"{spec.prefix_id}:{spec.cont_idx}",
                        "token_attr_l1": l1,
                        "word_attr_l1": _reduce_by_groups(l1, word_groups, reduction="sum"),
                    }
                )
                del d

            # Baseline logits (no steering): heterogeneous pass with empty features.
            baseline_samples: List[Dict[str, Any]] = []
            for it, base_attr in zip(items, baselines):
                baseline_samples.append(
                    {
                        "token_ids": it["token_ids"],
                        "cont_ids": it["cont_ids"],
                        "cont_start": it["cont_start"],
                        "features": [],
                        "decoder_cache": {},
                    }
                )

            # Run baseline in mini-batches
            baseline_arrays: Dict[str, Dict[str, List[float]]] = {}
            for start in range(0, len(baseline_samples), args.batch_size):
                chunk = baseline_samples[start : start + args.batch_size]
                logits, _ = steering.run_heterogeneous_steered_pass(
                    model,
                    batch_items=chunk,
                    steering_method=args.steering_method,
                    epsilon=0.0,
                    max_seq_len=args.max_seq_len,
                )
                for j, it in enumerate(chunk):
                    key = items[start + j]["sample_key"]
                    base_logit, base_prob, base_dlogit = _compute_target_logit_prob_arrays(
                        logits[j : j + 1], it["cont_ids"], it["cont_start"]
                    )
                    baseline_arrays[key] = {
                        "target_logits": base_logit,
                        "target_probs": base_prob,
                        "demeaned_target_logits": base_dlogit,
                    }
                del logits

            # Run steered passes for each epsilon
            results: List[Dict[str, Any]] = []
            for it, base_attr in zip(items, baselines):
                key = it["sample_key"]
                base_logit = baseline_arrays[key]["target_logits"]
                base_prob = baseline_arrays[key]["target_probs"]
                base_dlogit = baseline_arrays[key]["demeaned_target_logits"]
                # Word-level baseline
                word_base_logit = _reduce_by_groups(base_logit, it["word_groups"], reduction=args.word_reduction)
                word_base_prob = _reduce_by_groups(base_prob, it["word_groups"], reduction=args.word_reduction)
                word_base_dlogit = _reduce_by_groups(base_dlogit, it["word_groups"], reduction=args.word_reduction)

                entry: Dict[str, Any] = {
                    "sample_key": key,
                    "prefix_id": it["prefix_id"],
                    "context_path": it["context_path"],
                    "cont_idx": it["cont_idx"],
                    "continuation_token_ids": it["cont_ids"],
                    "continuation_token_strs": it["token_strs"],
                    "word_groups": it["word_groups"],
                    "token_attr_l1": base_attr["token_attr_l1"],
                    "word_attr_l1": base_attr["word_attr_l1"],
                    "baseline": {
                        "token_target_logits": base_logit,
                        "token_target_probs": base_prob,
                        "token_demeaned_target_logits": base_dlogit,
                        "word_target_logits": word_base_logit,
                        "word_target_probs": word_base_prob,
                        "word_demeaned_target_logits": word_base_dlogit,
                    },
                    "steering": {
                        "steering_method": args.steering_method,
                        "top_B": args.top_B,
                        "hc_selection": args.hc_selection,
                        "epsilons": args.epsilons,
                        "per_epsilon": {},
                    },
                }
                if it.get("random_baseline_enabled", False):
                    entry["random_baseline"] = {
                        "type": "random_features_shuffled_hc",
                        "seed_base": int(getattr(args, "random_baseline_seed", 0)),
                        "k": len(it.get("random_features", [])),
                        "per_epsilon": {},
                    }

                # Evaluate epsilons one-by-one (keeps memory bounded)
                for eps in args.epsilons:
                    chunk_item = [{
                        "token_ids": it["token_ids"],
                        "cont_ids": it["cont_ids"],
                        "cont_start": it["cont_start"],
                        "features": it["features"],
                        "decoder_cache": it["decoder_cache"],
                        "cluster_id": key,  # unique to avoid cache collisions
                    }]
                    logits, _ = steering.run_heterogeneous_steered_pass(
                        model,
                        batch_items=chunk_item,
                        steering_method=args.steering_method,
                        epsilon=float(eps),
                        max_seq_len=args.max_seq_len,
                    )
                    st_logit, st_prob, st_dlogit = _compute_target_logit_prob_arrays(
                        logits[0:1], it["cont_ids"], it["cont_start"]
                    )
                    # Δ arrays
                    dlogit = [(st_logit[i] - base_logit[i]) for i in range(min(len(st_logit), len(base_logit)))]
                    dprob = [(st_prob[i] - base_prob[i]) for i in range(min(len(st_prob), len(base_prob)))]
                    ddlogit = [(st_dlogit[i] - base_dlogit[i]) for i in range(min(len(st_dlogit), len(base_dlogit)))]
                    w_st_logit = _reduce_by_groups(st_logit, it["word_groups"], reduction=args.word_reduction)
                    w_st_prob = _reduce_by_groups(st_prob, it["word_groups"], reduction=args.word_reduction)
                    w_dlogit = [(w_st_logit[i] - word_base_logit[i]) for i in range(min(len(w_st_logit), len(word_base_logit)))]
                    w_dprob = [(w_st_prob[i] - word_base_prob[i]) for i in range(min(len(w_st_prob), len(word_base_prob)))]
                    w_st_dlogit = _reduce_by_groups(st_dlogit, it["word_groups"], reduction=args.word_reduction)
                    w_ddlogit = [(w_st_dlogit[i] - word_base_dlogit[i]) for i in range(min(len(w_st_dlogit), len(word_base_dlogit)))]

                    entry["steering"]["per_epsilon"][str(eps)] = {
                        "token_target_logits": st_logit,
                        "token_target_probs": st_prob,
                        "token_delta_target_logits": dlogit,
                        "token_delta_target_probs": dprob,
                        "token_demeaned_target_logits": st_dlogit,
                        "token_delta_demeaned_target_logits": ddlogit,
                        "word_target_logits": w_st_logit,
                        "word_target_probs": w_st_prob,
                        "word_delta_target_logits": w_dlogit,
                        "word_delta_target_probs": w_dprob,
                        "word_demeaned_target_logits": w_st_dlogit,
                        "word_delta_demeaned_target_logits": w_ddlogit,
                    }
                    del logits

                    # Random baseline (if enabled)
                    if it.get("random_baseline_enabled", False) and it.get("random_features") and it.get("random_decoder_cache") is not None:
                        rand_item = [{
                            "token_ids": it["token_ids"],
                            "cont_ids": it["cont_ids"],
                            "cont_start": it["cont_start"],
                            "features": it["random_features"],
                            "decoder_cache": it["random_decoder_cache"],
                            "cluster_id": f"rand:{key}",
                        }]
                        logits_r, _ = steering.run_heterogeneous_steered_pass(
                            model,
                            batch_items=rand_item,
                            steering_method=args.steering_method,
                            epsilon=float(eps),
                            max_seq_len=args.max_seq_len,
                        )
                        r_logit, r_prob, r_dlogit = _compute_target_logit_prob_arrays(
                            logits_r[0:1], it["cont_ids"], it["cont_start"]
                        )
                        r_ddlogit = [(r_dlogit[i] - base_dlogit[i]) for i in range(min(len(r_dlogit), len(base_dlogit)))]
                        r_dprob = [(r_prob[i] - base_prob[i]) for i in range(min(len(r_prob), len(base_prob)))]
                        rw_dlogit = _reduce_by_groups(r_dlogit, it["word_groups"], reduction=args.word_reduction)
                        rw_ddlogit = [(rw_dlogit[i] - word_base_dlogit[i]) for i in range(min(len(rw_dlogit), len(word_base_dlogit)))]
                        rw_prob = _reduce_by_groups(r_prob, it["word_groups"], reduction=args.word_reduction)
                        rw_dprob = [(rw_prob[i] - word_base_prob[i]) for i in range(min(len(rw_prob), len(word_base_prob)))]

                        entry["random_baseline"]["per_epsilon"][str(eps)] = {
                            "token_delta_demeaned_target_logits": r_ddlogit,
                            "token_delta_target_probs": r_dprob,
                            "word_delta_demeaned_target_logits": rw_ddlogit,
                            "word_delta_target_probs": rw_dprob,
                        }
                        del logits_r

                results.append(entry)

            payload["samples"] = results
            payload["model"] = args.model
            payload["transcoder"] = args.transcoder
            payload["dtype"] = args.dtype
            payload["word_reduction"] = args.word_reduction
            payload["batch_size"] = args.batch_size
            payload["max_seq_len"] = args.max_seq_len
            payload["attr_magnitude"] = {"type": "l1", "definition": "sum(abs(token_attributions[t, :])) over sources"}

    else:
        raise SystemExit(f"Unknown cmd: {cmd}")

    out_text = json.dumps(payload, indent=2)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.write(out_text)
        print(out_path)
    else:
        print(out_text)


if __name__ == "__main__":
    main()


