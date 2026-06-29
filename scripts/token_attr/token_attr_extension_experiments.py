#!/usr/bin/env -S uv run python
"""Standalone continuation attribution extension experiments.

This script extends the current token-attribution / steering workflow with:
1. Token-wise attribution consistency across continuation positions
2. Dynamic prefix-vs-history attribution mass decomposition
3. Span-specific steering derived from token-level attribution vectors

Outputs are written as JSON files inside the requested output directory:
  - consistency.json
  - span_ratio.json
  - span_steer.json
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "circuit-tracer"))

from circuit_tracer import ReplacementModel  # type: ignore
from circuit_tracer.attribution import attribute_prefix_to_continuations  # type: ignore
from utils.data_utils import reconstruct_active_features, save_json


def _import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _stable_int_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _load_replacement_model(
    model_name: str,
    transcoder_name: str,
    dtype_str: str,
    device_str: Optional[str] = None,
):
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unknown dtype: {dtype_str}")
    device = None if device_str in (None, "auto") else torch.device(device_str)
    return ReplacementModel.from_pretrained(
        model_name,
        transcoder_name,
        device=device,
        dtype=dtype_map[dtype_str],
    )


@dataclass(frozen=True)
class SampleSpec:
    context_path: str
    prefix_id: str
    cont_idx: int

    @property
    def sample_key(self) -> str:
        return f"{self.prefix_id}:{self.cont_idx}"


@dataclass
class LoadedSample:
    spec: SampleSpec
    prefix_tokens: List[int]
    continuation_tokens: List[int]
    token_attributions: torch.Tensor  # [T, n_prefix_sources]
    aggregated_attribution: torch.Tensor  # [n_prefix_sources]
    active_features: torch.Tensor  # [n_features, 3]
    selected_features: torch.Tensor  # [n_features]
    n_prefix_features: int

    @property
    def sample_key(self) -> str:
        return self.spec.sample_key

    @property
    def continuation_length(self) -> int:
        return int(self.token_attributions.shape[0])

    @property
    def feature_cap(self) -> int:
        return self.n_prefix_features


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


def _get_prefix_tokens(d: Dict[str, Any], spec: SampleSpec) -> List[int]:
    full_token_ids = d.get("prefix_tokens")
    if isinstance(full_token_ids, torch.Tensor):
        return full_token_ids.tolist()
    if isinstance(full_token_ids, list) and full_token_ids and isinstance(full_token_ids[0], int):
        return full_token_ids

    results_dir = Path(spec.context_path).parent.parent
    branches_path = results_dir / "2_branch_sampling" / f"{spec.prefix_id}_branches.json"
    if branches_path.exists():
        import json as _json

        with open(branches_path, "r") as f:
            bd = _json.load(f)
        prefix_ids = bd.get("prefix_tokens_with_bos")
        if isinstance(prefix_ids, list) and prefix_ids and isinstance(prefix_ids[0], int):
            return prefix_ids

    raise ValueError(
        f"{spec.context_path}: missing usable `prefix_tokens`; expected list[int] in the .pt "
        f"or Stage-2 branches at {branches_path}"
    )


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


def _load_samples(specs: Sequence[SampleSpec]) -> List[LoadedSample]:
    out: List[LoadedSample] = []
    for spec in specs:
        d = _load_prefix_context(spec.context_path)
        token_attrs = _require_token_attributions(d, spec.context_path)
        cont_tokens = _get_continuation_tokens(d, spec.context_path)
        prefix_tokens = _get_prefix_tokens(d, spec)
        aggregated = d.get("aggregated_attributions")
        if not torch.is_tensor(aggregated):
            raise ValueError(f"{spec.context_path}: missing `aggregated_attributions`")
        active_features = reconstruct_active_features(
            d["decoder_locations"],
            d["selected_features"],
            activation_matrix=d.get("activation_matrix"),
            return_numpy=False,
        )
        selected_features = torch.arange(active_features.shape[0], dtype=torch.long)
        out.append(
            LoadedSample(
                spec=spec,
                prefix_tokens=prefix_tokens,
                continuation_tokens=cont_tokens[spec.cont_idx],
                token_attributions=token_attrs[spec.cont_idx].to(torch.float32).cpu(),
                aggregated_attribution=aggregated[spec.cont_idx].to(torch.float32).cpu(),
                active_features=active_features.cpu(),
                selected_features=selected_features,
                n_prefix_features=int(d["n_prefix_features"]),
            )
        )
        del d
    return out


def _reduce_token_attr_l1(x: torch.Tensor) -> torch.Tensor:
    return x.abs().to(torch.float32).sum(dim=1)


def _group_tokens_to_words(token_strs: List[str]) -> List[List[int]]:
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
    seq_len = logits.shape[1]
    valid_len = min(len(cont_ids), seq_len - cont_start)
    if valid_len <= 0:
        return [], [], []

    positions = torch.arange(cont_start, cont_start + valid_len, device=logits.device)
    cont_logits = logits[0, positions, :].float()
    token_ids = torch.tensor(cont_ids[:valid_len], device=logits.device, dtype=torch.long)

    target_logits = cont_logits[torch.arange(valid_len, device=logits.device), token_ids]
    mean_logits = cont_logits.mean(dim=-1)
    demeaned_target_logits = target_logits - mean_logits

    log_probs = F.log_softmax(cont_logits, dim=-1)
    target_logprobs = log_probs[torch.arange(valid_len, device=logits.device), token_ids]
    target_probs = torch.exp(target_logprobs)

    return (
        target_logits.tolist(),
        target_probs.tolist(),
        demeaned_target_logits.tolist(),
    )


def _select_topB_features_from_Hc(
    H_c: np.ndarray,
    active_features: torch.Tensor,
    selected_features: torch.Tensor,
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
    active_features: torch.Tensor,
    selected_features: torch.Tensor,
    avoid: List[Tuple[int, int, int, float]],
    k: int,
    h_c_vals_source: List[float],
    seed: int,
) -> List[Tuple[int, int, int, float]]:
    rng = np.random.RandomState(seed)
    avoid_set = {(int(l), int(p), int(f)) for (l, p, f, _) in avoid}

    cand: List[Tuple[int, int, int]] = []
    for idx in range(int(selected_features.numel())):
        feat_idx = int(selected_features[idx].item())
        l, p, f = active_features[feat_idx].tolist()
        tpl = (int(l), int(p), int(f))
        if tpl in avoid_set:
            continue
        cand.append(tpl)

    if not cand:
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


def _top_feature_set(vec: torch.Tensor, n_prefix_features: int, top_k: int) -> set[int]:
    if n_prefix_features <= 0:
        return set()
    feature_scores = vec[:n_prefix_features].abs()
    k = min(top_k, int(feature_scores.numel()))
    if k <= 0:
        return set()
    idx = torch.argsort(feature_scores, descending=True)[:k]
    return {int(i) for i in idx.tolist()}


def _jaccard_similarity(a: set[int], b: set[int]) -> Optional[float]:
    if not a and not b:
        return None
    union = a | b
    if not union:
        return None
    return float(len(a & b) / len(union))


def _float_or_none(x: Optional[float]) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else float(x)


def _mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    xs = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not xs:
        return None
    return float(np.mean(np.asarray(xs, dtype=np.float64)))


def _serialize_feature_interventions(
    features: List[Tuple[int, int, int, float]]
) -> List[Dict[str, Any]]:
    return [
        {
            "layer": int(layer),
            "position": int(pos),
            "feature_id": int(feat_id),
            "h_c_value": float(h_c_val),
        }
        for layer, pos, feat_id, h_c_val in features
    ]


def _draw_random_baseline_vector_same_context(
    context_token_attributions: Sequence[torch.Tensor],
    exclude_cont_idx: int,
    forbidden_positions: set[int],
    rng: np.random.RandomState,
    max_attempts: int = 512,
) -> Optional[Tuple[int, int, torch.Tensor]]:
    if len(context_token_attributions) <= 1:
        return None
    for _ in range(max_attempts):
        cont_idx = int(rng.randint(0, len(context_token_attributions)))
        if cont_idx == exclude_cont_idx:
            continue
        candidate = context_token_attributions[cont_idx].to(torch.float32).cpu()
        valid_positions = [p for p in range(int(candidate.shape[0])) if p not in forbidden_positions]
        if not valid_positions:
            continue
        pos = int(valid_positions[int(rng.randint(0, len(valid_positions)))])
        return cont_idx, pos, candidate[pos]
    return None


def run_consistency_experiment(
    samples: Sequence[LoadedSample],
    *,
    pt_glob: str,
    n_files_matched: int,
    max_files: Optional[int],
    n_samples: int,
    seed: int,
    top_k: int,
    output_path: Path,
) -> None:
    rng = np.random.RandomState(seed)
    pair_agg: Dict[Tuple[int, int], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    gap_agg: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    sample_rows: List[Dict[str, Any]] = []
    for sample_idx, sample in enumerate(samples):
        context_data = _load_prefix_context(sample.spec.context_path)
        context_token_attributions = _require_token_attributions(context_data, sample.spec.context_path)
        top_sets = [
            _top_feature_set(sample.token_attributions[pos], sample.n_prefix_features, top_k)
            for pos in range(sample.continuation_length)
        ]
        pair_rows: List[Dict[str, Any]] = []

        for i in range(sample.continuation_length):
            vec_i = sample.token_attributions[i]
            set_i = top_sets[i]
            for j in range(i + 1, sample.continuation_length):
                vec_j = sample.token_attributions[j]
                set_j = top_sets[j]

                real_l1 = float(torch.abs(vec_i - vec_j).sum().item())
                real_jaccard = _jaccard_similarity(set_i, set_j)

                rand_sample = _draw_random_baseline_vector_same_context(
                    context_token_attributions=context_token_attributions,
                    exclude_cont_idx=sample.spec.cont_idx,
                    forbidden_positions={i, j},
                    rng=rng,
                )
                rand_row: Dict[str, Any] = {
                    "random_sample_key": None,
                    "random_position": None,
                    "random_l1_anchor_i": None,
                    "random_l1_anchor_j": None,
                    "random_l1_mean": None,
                    "random_jaccard_anchor_i": None,
                    "random_jaccard_anchor_j": None,
                    "random_jaccard_mean": None,
                }
                rand_l1_mean = None
                rand_jaccard_mean = None
                if rand_sample is not None:
                    rand_cont_idx, rand_pos, rand_vec = rand_sample
                    rand_set = _top_feature_set(
                        rand_vec,
                        sample.n_prefix_features,
                        top_k,
                    )
                    rand_l1_i = float(torch.abs(vec_i - rand_vec).sum().item())
                    rand_l1_j = float(torch.abs(vec_j - rand_vec).sum().item())
                    rand_j_i = _jaccard_similarity(set_i, rand_set)
                    rand_j_j = _jaccard_similarity(set_j, rand_set)
                    rand_l1_mean = _mean_or_none([rand_l1_i, rand_l1_j])
                    rand_jaccard_mean = _mean_or_none([rand_j_i, rand_j_j])
                    rand_row = {
                        "random_sample_key": f"{sample.spec.prefix_id}:{rand_cont_idx}",
                        "random_position": rand_pos,
                        "random_l1_anchor_i": rand_l1_i,
                        "random_l1_anchor_j": rand_l1_j,
                        "random_l1_mean": rand_l1_mean,
                        "random_jaccard_anchor_i": rand_j_i,
                        "random_jaccard_anchor_j": rand_j_j,
                        "random_jaccard_mean": rand_jaccard_mean,
                    }

                pair_rows.append(
                    {
                        "i": i,
                        "j": j,
                        "gap": j - i,
                        "real_l1": real_l1,
                        "real_jaccard_topk": real_jaccard,
                        **rand_row,
                    }
                )

                pair_bucket = pair_agg[(i, j)]
                pair_bucket["real_l1"].append(real_l1)
                if real_jaccard is not None:
                    pair_bucket["real_jaccard"].append(real_jaccard)
                if rand_l1_mean is not None:
                    pair_bucket["random_l1"].append(rand_l1_mean)
                    pair_bucket["real_minus_random_l1"].append(real_l1 - rand_l1_mean)
                if rand_jaccard_mean is not None and real_jaccard is not None:
                    pair_bucket["random_jaccard"].append(rand_jaccard_mean)
                    pair_bucket["real_minus_random_jaccard"].append(real_jaccard - rand_jaccard_mean)

                gap_bucket = gap_agg[j - i]
                gap_bucket["real_l1"].append(real_l1)
                if real_jaccard is not None:
                    gap_bucket["real_jaccard"].append(real_jaccard)
                if rand_l1_mean is not None:
                    gap_bucket["random_l1"].append(rand_l1_mean)
                    gap_bucket["real_minus_random_l1"].append(real_l1 - rand_l1_mean)
                if rand_jaccard_mean is not None and real_jaccard is not None:
                    gap_bucket["random_jaccard"].append(rand_jaccard_mean)
                    gap_bucket["real_minus_random_jaccard"].append(real_jaccard - rand_jaccard_mean)

        sample_rows.append(
            {
                "sample_key": sample.sample_key,
                "prefix_id": sample.spec.prefix_id,
                "context_path": sample.spec.context_path,
                "cont_idx": sample.spec.cont_idx,
                "continuation_length": sample.continuation_length,
                "pair_metrics": pair_rows,
            }
        )
        del context_data

    pair_summary = []
    for (i, j), bucket in sorted(pair_agg.items()):
        pair_summary.append(
            {
                "i": i,
                "j": j,
                "gap": j - i,
                "n": max(len(v) for v in bucket.values()) if bucket else 0,
                "real_l1_mean": _mean_or_none(bucket.get("real_l1", [])),
                "random_l1_mean": _mean_or_none(bucket.get("random_l1", [])),
                "real_minus_random_l1_mean": _mean_or_none(bucket.get("real_minus_random_l1", [])),
                "real_jaccard_mean": _mean_or_none(bucket.get("real_jaccard", [])),
                "random_jaccard_mean": _mean_or_none(bucket.get("random_jaccard", [])),
                "real_minus_random_jaccard_mean": _mean_or_none(bucket.get("real_minus_random_jaccard", [])),
            }
        )

    gap_summary = []
    for gap, bucket in sorted(gap_agg.items()):
        gap_summary.append(
            {
                "gap": gap,
                "n": max(len(v) for v in bucket.values()) if bucket else 0,
                "real_l1_mean": _mean_or_none(bucket.get("real_l1", [])),
                "random_l1_mean": _mean_or_none(bucket.get("random_l1", [])),
                "real_minus_random_l1_mean": _mean_or_none(bucket.get("real_minus_random_l1", [])),
                "real_jaccard_mean": _mean_or_none(bucket.get("real_jaccard", [])),
                "random_jaccard_mean": _mean_or_none(bucket.get("random_jaccard", [])),
                "real_minus_random_jaccard_mean": _mean_or_none(bucket.get("real_minus_random_jaccard", [])),
            }
        )

    payload = {
        "cmd": "consistency",
        "pt_glob": pt_glob,
        "n_files_matched": n_files_matched,
        "max_files": max_files,
        "n_samples": n_samples,
        "seed": seed,
        "top_k_features": top_k,
        "jaccard_basis": "feature_nodes",
        "baseline": "external_random_vector_both_anchors",
        "samples": sample_rows,
        "pair_summary": pair_summary,
        "gap_summary": gap_summary,
    }
    save_json(payload, output_path)


def _partition_dynamic_source_masses(
    prefix_ctx: Any,
    source_attr: torch.Tensor,
    original_prefix_length: int,
) -> Tuple[float, float]:
    source_len = prefix_ctx.prefix_length
    n_features = prefix_ctx.n_prefix_features
    n_layers = prefix_ctx.n_layers

    feature_prefix_mass = 0.0
    feature_history_mass = 0.0
    if n_features > 0:
        active_features = reconstruct_active_features(
            prefix_ctx.decoder_locations,
            prefix_ctx.selected_features if prefix_ctx.selected_features is not None else torch.arange(n_features),
            activation_matrix=prefix_ctx.activation_matrix,
            return_numpy=False,
        )
        feature_positions = active_features[:, 1].to(torch.long)
        feature_vals = source_attr[:n_features].abs()
        prefix_mask = feature_positions < original_prefix_length
        feature_prefix_mass = float(feature_vals[prefix_mask].sum().item())
        feature_history_mass = float(feature_vals[~prefix_mask].sum().item())

    error_start = n_features
    error_end = error_start + n_layers * source_len
    error_vals = source_attr[error_start:error_end].abs().view(n_layers, source_len)
    error_prefix_mass = float(error_vals[:, :original_prefix_length].sum().item())
    error_history_mass = float(error_vals[:, original_prefix_length:].sum().item())

    token_vals = source_attr[error_end:].abs()
    token_prefix_mass = float(token_vals[:original_prefix_length].sum().item())
    token_history_mass = float(token_vals[original_prefix_length:].sum().item())

    prefix_mass = feature_prefix_mass + error_prefix_mass + token_prefix_mass
    history_mass = feature_history_mass + error_history_mass + token_history_mass
    return prefix_mass, history_mass


def run_span_ratio_experiment(
    samples: Sequence[LoadedSample],
    *,
    pt_glob: str,
    n_files_matched: int,
    max_files: Optional[int],
    n_samples: int,
    seed: int,
    model: ReplacementModel,
    model_name: str,
    transcoder_name: str,
    dtype_str: str,
    device_str: str,
    max_feature_nodes: Optional[int],
    max_target_positions: Optional[int],
    output_path: Path,
) -> None:
    by_pos: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    sample_rows: List[Dict[str, Any]] = []

    for sample in samples:
        cached_prefix_l1 = _reduce_token_attr_l1(sample.token_attributions).tolist()
        prefix_len = len(sample.prefix_tokens)
        per_pos_rows: List[Dict[str, Any]] = []

        target_tokens = sample.continuation_tokens
        if max_target_positions is not None:
            target_tokens = target_tokens[:max_target_positions]

        for target_pos, target_token in enumerate(target_tokens):
            dynamic_prefix = sample.prefix_tokens + sample.continuation_tokens[:target_pos]
            feature_cap = sample.feature_cap
            if max_feature_nodes is not None:
                feature_cap = min(feature_cap, max_feature_nodes)
            try:
                result = attribute_prefix_to_continuations(
                    prefix=dynamic_prefix,
                    continuations=[[target_token]],
                    model=model,
                    batch_size=1,
                    add_bos=False,
                    max_feature_nodes=feature_cap,
                    verbose=False,
                )
            except torch.OutOfMemoryError as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "span-ratio attribution ran out of memory at "
                    f"{sample.sample_key} target_pos={target_pos}. "
                    "Retry with a smaller `--span-ratio-max-feature-nodes`, fewer target positions via "
                    "`--span-ratio-max-target-positions`, or `--device cpu`."
                ) from exc
            attr_vec = result.continuation_attributions[0][0].source_attribution.to(torch.float32).cpu()
            dynamic_ctx = result.prefix_context

            prefix_mass, history_mass = _partition_dynamic_source_masses(
                dynamic_ctx,
                attr_vec,
                original_prefix_length=prefix_len,
            )
            total_mass = float(attr_vec.abs().sum().item())
            prefix_fraction = float(prefix_mass / total_mass) if total_mass > 0 else None
            history_fraction = float(history_mass / total_mass) if total_mass > 0 else None
            prefix_to_history_ratio = float(prefix_mass / history_mass) if history_mass > 0 else None

            row = {
                "target_pos": target_pos,
                "history_token_count": target_pos,
                "source_length": len(dynamic_prefix),
                "prefix_only_cached_l1": float(cached_prefix_l1[target_pos]),
                "dynamic_prefix_l1": prefix_mass,
                "dynamic_history_l1": history_mass,
                "dynamic_total_l1": total_mass,
                "prefix_fraction": _float_or_none(prefix_fraction),
                "history_fraction": _float_or_none(history_fraction),
                "prefix_to_history_ratio": _float_or_none(prefix_to_history_ratio),
            }
            per_pos_rows.append(row)

            by_pos[target_pos]["prefix_only_cached_l1"].append(float(cached_prefix_l1[target_pos]))
            by_pos[target_pos]["dynamic_prefix_l1"].append(prefix_mass)
            by_pos[target_pos]["dynamic_history_l1"].append(history_mass)
            by_pos[target_pos]["dynamic_total_l1"].append(total_mass)
            if prefix_fraction is not None:
                by_pos[target_pos]["prefix_fraction"].append(prefix_fraction)
            if history_fraction is not None:
                by_pos[target_pos]["history_fraction"].append(history_fraction)
            if prefix_to_history_ratio is not None:
                by_pos[target_pos]["prefix_to_history_ratio"].append(prefix_to_history_ratio)

            del result

        sample_rows.append(
            {
                "sample_key": sample.sample_key,
                "prefix_id": sample.spec.prefix_id,
                "context_path": sample.spec.context_path,
                "cont_idx": sample.spec.cont_idx,
                "continuation_length": sample.continuation_length,
                "positions": per_pos_rows,
            }
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    position_summary = []
    for pos, bucket in sorted(by_pos.items()):
        position_summary.append(
            {
                "target_pos": pos,
                "n": max(len(v) for v in bucket.values()) if bucket else 0,
                "prefix_only_cached_l1_mean": _mean_or_none(bucket.get("prefix_only_cached_l1", [])),
                "dynamic_prefix_l1_mean": _mean_or_none(bucket.get("dynamic_prefix_l1", [])),
                "dynamic_history_l1_mean": _mean_or_none(bucket.get("dynamic_history_l1", [])),
                "dynamic_total_l1_mean": _mean_or_none(bucket.get("dynamic_total_l1", [])),
                "prefix_fraction_mean": _mean_or_none(bucket.get("prefix_fraction", [])),
                "history_fraction_mean": _mean_or_none(bucket.get("history_fraction", [])),
                "prefix_to_history_ratio_mean": _mean_or_none(bucket.get("prefix_to_history_ratio", [])),
            }
        )

    payload = {
        "cmd": "span-ratio",
        "pt_glob": pt_glob,
        "n_files_matched": n_files_matched,
        "max_files": max_files,
        "n_samples": n_samples,
        "seed": seed,
        "model": model_name,
        "transcoder": transcoder_name,
        "dtype": dtype_str,
        "device": device_str,
        "dynamic_source_span": "original_prefix_plus_previous_continuation_tokens",
        "max_feature_nodes": max_feature_nodes,
        "max_target_positions": max_target_positions,
        "partition_basis": "all_source_nodes_by_position",
        "samples": sample_rows,
        "position_summary": position_summary,
    }
    save_json(payload, output_path)


def _build_span_vector(sample: LoadedSample, mode: str) -> torch.Tensor:
    if mode == "full_sum":
        return sample.aggregated_attribution
    if mode == "first_token":
        return sample.token_attributions[0]
    if mode == "last_token":
        return sample.token_attributions[-1]
    if mode == "first5_sum":
        return sample.token_attributions[: min(5, sample.continuation_length)].sum(dim=0)
    raise ValueError(f"Unknown span mode: {mode}")


def _mode_positions(sample: LoadedSample, mode: str) -> List[int]:
    if mode == "full_sum":
        return list(range(sample.continuation_length))
    if mode == "first_token":
        return [0]
    if mode == "last_token":
        return [sample.continuation_length - 1]
    if mode == "first5_sum":
        return list(range(min(5, sample.continuation_length)))
    raise ValueError(f"Unknown span mode: {mode}")


def run_span_steer_experiment(
    samples: Sequence[LoadedSample],
    *,
    pt_glob: str,
    n_files_matched: int,
    max_files: Optional[int],
    n_samples: int,
    seed: int,
    model: ReplacementModel,
    model_name: str,
    transcoder_name: str,
    dtype_str: str,
    device_str: str,
    top_B: int,
    hc_selection: str,
    steering_method: str,
    epsilons: Sequence[float],
    word_reduction: str,
    batch_size: int,
    max_seq_len: Optional[int],
    random_baseline_seed: int,
    output_path: Path,
) -> None:
    graph = _import_module("lp_7c_graph", REPO_ROOT / "7_validation" / "7c_graph.py")
    steering = _import_module("lp_7c_steering", REPO_ROOT / "7_validation" / "7c_steering.py")

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("ReplacementModel has no tokenizer")

    span_modes = ["full_sum", "first_token", "last_token", "first5_sum"]
    items_by_mode: Dict[str, List[Dict[str, Any]]] = {mode: [] for mode in span_modes}
    baseline_items: List[Dict[str, Any]] = []
    entries_by_key: Dict[str, Dict[str, Any]] = {}

    for sample in samples:
        full_ids = sample.prefix_tokens + sample.continuation_tokens
        cont_start = len(sample.prefix_tokens)
        token_strs = tokenizer.convert_ids_to_tokens(sample.continuation_tokens)
        word_groups = _group_tokens_to_words(token_strs)
        token_attr_l1 = _reduce_token_attr_l1(sample.token_attributions).tolist()
        word_attr_l1 = _reduce_by_groups(token_attr_l1, word_groups, reduction="sum")

        entries_by_key[sample.sample_key] = {
            "sample_key": sample.sample_key,
            "prefix_id": sample.spec.prefix_id,
            "context_path": sample.spec.context_path,
            "cont_idx": sample.spec.cont_idx,
            "continuation_token_ids": sample.continuation_tokens,
            "continuation_token_strs": token_strs,
            "word_groups": word_groups,
            "token_attr_l1": token_attr_l1,
            "word_attr_l1": word_attr_l1,
            "baseline": {},
            "span_modes": {},
        }

        baseline_items.append(
            {
                "sample_key": sample.sample_key,
                "token_ids": full_ids,
                "cont_ids": sample.continuation_tokens,
                "cont_start": cont_start,
                "features": [],
                "decoder_cache": {},
                "cluster_id": f"baseline:{sample.sample_key}",
            }
        )

        for mode in span_modes:
            H_c = _build_span_vector(sample, mode).numpy()
            feats = _select_topB_features_from_Hc(
                H_c,
                active_features=sample.active_features,
                selected_features=sample.selected_features,
                top_B=top_B,
                hc_selection=hc_selection,
            )
            decoder_cache = graph.precompute_cluster_decoder_vectors(
                model,
                {0: feats},
                device=model.cfg.device,
            ).get(0, {})

            hc_vals_src = [float(v) for (_, _, _, v) in feats]
            rand_feats: List[Tuple[int, int, int, float]] = []
            rand_decoder_cache: Dict[int, Tuple[List[int], List[int], torch.Tensor, torch.Tensor]] = {}
            if feats:
                rand_seed = random_baseline_seed + (
                    _stable_int_hash(f"{sample.sample_key}:{mode}") % 1_000_000_000
                )
                rand_feats = _make_random_feature_baseline(
                    active_features=sample.active_features,
                    selected_features=sample.selected_features,
                    avoid=feats,
                    k=len(feats),
                    h_c_vals_source=hc_vals_src,
                    seed=rand_seed,
                )
                if rand_feats:
                    rand_decoder_cache = graph.precompute_cluster_decoder_vectors(
                        model,
                        {0: rand_feats},
                        device=model.cfg.device,
                    ).get(0, {})

            entries_by_key[sample.sample_key]["span_modes"][mode] = {
                "span_mode": mode,
                "positions": _mode_positions(sample, mode),
                "selected_features": _serialize_feature_interventions(feats),
                "steering": {
                    "steering_method": steering_method,
                    "top_B": top_B,
                    "hc_selection": hc_selection,
                    "epsilons": [float(e) for e in epsilons],
                    "per_epsilon": {},
                },
                "random_baseline": {
                    "type": "random_features_shuffled_hc",
                    "seed_base": random_baseline_seed,
                    "k": len(rand_feats),
                    "per_epsilon": {},
                },
            }

            items_by_mode[mode].append(
                {
                    "sample_key": sample.sample_key,
                    "token_ids": full_ids,
                    "cont_ids": sample.continuation_tokens,
                    "cont_start": cont_start,
                    "features": feats,
                    "decoder_cache": decoder_cache,
                    "cluster_id": f"{sample.sample_key}:{mode}",
                    "random_features": rand_feats,
                    "random_decoder_cache": rand_decoder_cache,
                    "word_groups": word_groups,
                }
            )

    baseline_arrays: Dict[str, Dict[str, List[float]]] = {}
    for start in range(0, len(baseline_items), batch_size):
        chunk = baseline_items[start : start + batch_size]
        logits, _ = steering.run_heterogeneous_steered_pass(
            model,
            batch_items=chunk,
            steering_method=steering_method,
            epsilon=0.0,
            max_seq_len=max_seq_len,
        )
        for j, item in enumerate(chunk):
            base_logit, base_prob, base_dlogit = _compute_target_logit_prob_arrays(
                logits[j : j + 1],
                item["cont_ids"],
                item["cont_start"],
            )
            key = item["sample_key"]
            baseline_arrays[key] = {
                "target_logits": base_logit,
                "target_probs": base_prob,
                "demeaned_target_logits": base_dlogit,
            }
        del logits

    for sample in samples:
        key = sample.sample_key
        base_logit = baseline_arrays[key]["target_logits"]
        base_prob = baseline_arrays[key]["target_probs"]
        base_dlogit = baseline_arrays[key]["demeaned_target_logits"]
        word_groups = entries_by_key[key]["word_groups"]
        entries_by_key[key]["baseline"] = {
            "token_target_logits": base_logit,
            "token_target_probs": base_prob,
            "token_demeaned_target_logits": base_dlogit,
            "word_target_logits": _reduce_by_groups(base_logit, word_groups, reduction=word_reduction),
            "word_target_probs": _reduce_by_groups(base_prob, word_groups, reduction=word_reduction),
            "word_demeaned_target_logits": _reduce_by_groups(base_dlogit, word_groups, reduction=word_reduction),
        }

    for mode in span_modes:
        mode_items = items_by_mode[mode]
        for eps in epsilons:
            for start in range(0, len(mode_items), batch_size):
                chunk = mode_items[start : start + batch_size]
                logits, _ = steering.run_heterogeneous_steered_pass(
                    model,
                    batch_items=chunk,
                    steering_method=steering_method,
                    epsilon=float(eps),
                    max_seq_len=max_seq_len,
                )
                for j, item in enumerate(chunk):
                    key = item["sample_key"]
                    base = entries_by_key[key]["baseline"]
                    st_logit, st_prob, st_dlogit = _compute_target_logit_prob_arrays(
                        logits[j : j + 1],
                        item["cont_ids"],
                        item["cont_start"],
                    )
                    base_logit = base["token_target_logits"]
                    base_prob = base["token_target_probs"]
                    base_dlogit = base["token_demeaned_target_logits"]

                    dlogit = [(st_logit[i] - base_logit[i]) for i in range(min(len(st_logit), len(base_logit)))]
                    dprob = [(st_prob[i] - base_prob[i]) for i in range(min(len(st_prob), len(base_prob)))]
                    ddlogit = [(st_dlogit[i] - base_dlogit[i]) for i in range(min(len(st_dlogit), len(base_dlogit)))]

                    word_groups = item["word_groups"]
                    word_base_logit = base["word_target_logits"]
                    word_base_prob = base["word_target_probs"]
                    word_base_dlogit = base["word_demeaned_target_logits"]
                    w_st_logit = _reduce_by_groups(st_logit, word_groups, reduction=word_reduction)
                    w_st_prob = _reduce_by_groups(st_prob, word_groups, reduction=word_reduction)
                    w_st_dlogit = _reduce_by_groups(st_dlogit, word_groups, reduction=word_reduction)
                    w_dlogit = [(w_st_logit[i] - word_base_logit[i]) for i in range(min(len(w_st_logit), len(word_base_logit)))]
                    w_dprob = [(w_st_prob[i] - word_base_prob[i]) for i in range(min(len(w_st_prob), len(word_base_prob)))]
                    w_ddlogit = [(w_st_dlogit[i] - word_base_dlogit[i]) for i in range(min(len(w_st_dlogit), len(word_base_dlogit)))]

                    entries_by_key[key]["span_modes"][mode]["steering"]["per_epsilon"][str(float(eps))] = {
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

            random_items = [
                {
                    "sample_key": item["sample_key"],
                    "token_ids": item["token_ids"],
                    "cont_ids": item["cont_ids"],
                    "cont_start": item["cont_start"],
                    "features": item["random_features"],
                    "decoder_cache": item["random_decoder_cache"],
                    "cluster_id": f"random:{item['sample_key']}:{mode}",
                    "word_groups": item["word_groups"],
                }
                for item in mode_items
                if item["random_features"]
            ]
            for start in range(0, len(random_items), batch_size):
                chunk = random_items[start : start + batch_size]
                logits_r, _ = steering.run_heterogeneous_steered_pass(
                    model,
                    batch_items=chunk,
                    steering_method=steering_method,
                    epsilon=float(eps),
                    max_seq_len=max_seq_len,
                )
                for j, item in enumerate(chunk):
                    key = item["sample_key"]
                    base = entries_by_key[key]["baseline"]
                    r_logit, r_prob, r_dlogit = _compute_target_logit_prob_arrays(
                        logits_r[j : j + 1],
                        item["cont_ids"],
                        item["cont_start"],
                    )
                    base_prob = base["token_target_probs"]
                    base_dlogit = base["token_demeaned_target_logits"]
                    r_ddlogit = [(r_dlogit[i] - base_dlogit[i]) for i in range(min(len(r_dlogit), len(base_dlogit)))]
                    r_dprob = [(r_prob[i] - base_prob[i]) for i in range(min(len(r_prob), len(base_prob)))]

                    word_groups = item["word_groups"]
                    word_base_prob = base["word_target_probs"]
                    word_base_dlogit = base["word_demeaned_target_logits"]
                    rw_prob = _reduce_by_groups(r_prob, word_groups, reduction=word_reduction)
                    rw_dprob = [(rw_prob[i] - word_base_prob[i]) for i in range(min(len(rw_prob), len(word_base_prob)))]
                    rw_dlogit = _reduce_by_groups(r_dlogit, word_groups, reduction=word_reduction)
                    rw_ddlogit = [(rw_dlogit[i] - word_base_dlogit[i]) for i in range(min(len(rw_dlogit), len(word_base_dlogit)))]

                    entries_by_key[key]["span_modes"][mode]["random_baseline"]["per_epsilon"][str(float(eps))] = {
                        "token_delta_demeaned_target_logits": r_ddlogit,
                        "token_delta_target_probs": r_dprob,
                        "word_delta_demeaned_target_logits": rw_ddlogit,
                        "word_delta_target_probs": rw_dprob,
                    }
                del logits_r

    payload = {
        "cmd": "span-steer",
        "pt_glob": pt_glob,
        "n_files_matched": n_files_matched,
        "max_files": max_files,
        "n_samples": n_samples,
        "seed": seed,
        "model": model_name,
        "transcoder": transcoder_name,
        "dtype": dtype_str,
        "device": device_str,
        "word_reduction": word_reduction,
        "batch_size": batch_size,
        "max_seq_len": max_seq_len,
        "span_modes": span_modes,
        "samples": [entries_by_key[sample.sample_key] for sample in samples],
    }
    save_json(payload, output_path)


def _normalize_experiments(experiments: Sequence[str]) -> List[str]:
    if "all" in experiments:
        return ["consistency", "span-ratio", "span-steer"]
    return list(dict.fromkeys(experiments))


def main() -> None:
    ap = argparse.ArgumentParser(description="Standalone token-attribution extension experiments")
    ap.add_argument("--pt-glob", required=True, help="Glob for *_prefix_context.pt files")
    ap.add_argument("--out-dir", type=Path, required=True, help="Directory for output JSON files")
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        choices=["all", "consistency", "span-ratio", "span-steer"],
        help="Which experiments to run",
    )
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--consistency-top-k", type=int, default=100)

    ap.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--transcoder", type=str, default="mwhanna/qwen3-8b-transcoders")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    ap.add_argument("--device", type=str, default="auto", help="Device for ReplacementModel (default: auto)")

    ap.add_argument("--top-B", type=int, default=10)
    ap.add_argument("--hc-selection", choices=["full", "positive", "negative"], default="full")
    ap.add_argument("--steering-method", choices=["additive", "multiplicative", "absolute", "sign", "scaling"], default="sign")
    ap.add_argument("--epsilons", type=float, nargs="+", default=[-1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0])
    ap.add_argument("--word-reduction", choices=["sum", "mean"], default="mean")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--random-baseline-seed", type=int, default=123)
    ap.add_argument(
        "--span-ratio-max-feature-nodes",
        type=int,
        default=None,
        help="Optional cap for dynamic span-ratio attribution feature nodes; defaults to the Stage 3 feature cap",
    )
    ap.add_argument(
        "--span-ratio-max-target-positions",
        type=int,
        default=None,
        help="Optional limit on evaluated continuation target positions for span-ratio",
    )
    args = ap.parse_args()

    pt_files = sorted(glob.glob(args.pt_glob))
    if not pt_files:
        raise SystemExit(f"No files matched: {args.pt_glob}")

    experiments = _normalize_experiments(args.experiments)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    specs = _sample_continuations(
        pt_files,
        n_samples=args.n_samples,
        seed=args.seed,
        max_files=args.max_files,
    )
    if not specs:
        raise SystemExit("No sampled continuations available")
    samples = _load_samples(specs)

    model = None
    if any(exp in {"span-ratio", "span-steer"} for exp in experiments):
        model = _load_replacement_model(args.model, args.transcoder, args.dtype, args.device)

    if "consistency" in experiments:
        run_consistency_experiment(
            samples,
            pt_glob=args.pt_glob,
            n_files_matched=len(pt_files),
            max_files=args.max_files,
            n_samples=len(samples),
            seed=args.seed,
            top_k=args.consistency_top_k,
            output_path=args.out_dir / "consistency.json",
        )
        print(str(args.out_dir / "consistency.json"))

    if "span-ratio" in experiments:
        assert model is not None
        run_span_ratio_experiment(
            samples,
            pt_glob=args.pt_glob,
            n_files_matched=len(pt_files),
            max_files=args.max_files,
            n_samples=len(samples),
            seed=args.seed,
            model=model,
            model_name=args.model,
            transcoder_name=args.transcoder,
            dtype_str=args.dtype,
            device_str=args.device,
            max_feature_nodes=args.span_ratio_max_feature_nodes,
            max_target_positions=args.span_ratio_max_target_positions,
            output_path=args.out_dir / "span_ratio.json",
        )
        print(str(args.out_dir / "span_ratio.json"))

    if "span-steer" in experiments:
        assert model is not None
        run_span_steer_experiment(
            samples,
            pt_glob=args.pt_glob,
            n_files_matched=len(pt_files),
            max_files=args.max_files,
            n_samples=len(samples),
            seed=args.seed,
            model=model,
            model_name=args.model,
            transcoder_name=args.transcoder,
            dtype_str=args.dtype,
            device_str=args.device,
            top_B=args.top_B,
            hc_selection=args.hc_selection,
            steering_method=args.steering_method,
            epsilons=args.epsilons,
            word_reduction=args.word_reduction,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            random_baseline_seed=args.random_baseline_seed,
            output_path=args.out_dir / "span_steer.json",
        )
        print(str(args.out_dir / "span_steer.json"))


if __name__ == "__main__":
    main()
