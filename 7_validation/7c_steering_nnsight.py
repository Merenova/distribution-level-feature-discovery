#!/usr/bin/env python3
"""nnsight Stage 7c steering implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from circuit_tracer.replacement_model.replacement_model_nnsight import FeatureSteeringSpec

_module_dir = Path(__file__).parent


def _import_7c_utils():
    spec = importlib.util.spec_from_file_location("lp_7c_utils_nnsight", _module_dir / "7c_utils.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


utils = _import_7c_utils()


def _device(model):
    return getattr(model, "device", getattr(model.cfg, "device", torch.device("cpu")))


def _dtype(model):
    return getattr(model, "dtype", getattr(model.cfg, "dtype", torch.float32))


def _to_1d_tokens(token_ids: list[int], model) -> torch.Tensor:
    return torch.as_tensor(token_ids, dtype=torch.long, device=_device(model))


def _materialize(value):
    return getattr(value, "value", value)


def _run_logits(model, token_ids: list[int]) -> torch.Tensor:
    from nnsight import save

    tokens = _to_1d_tokens(token_ids, model)
    with torch.inference_mode(), model.trace(tokens):
        logits = save(model.output.logits)
    logits = _materialize(logits)
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    return logits


def _get_activation_cache(model, token_ids: list[int]) -> torch.Tensor:
    tokens = _to_1d_tokens(token_ids, model)
    with torch.inference_mode():
        _, activation_cache = model.get_activations(tokens, sparse=False)
    return _materialize(activation_cache)


def build_absolute_interventions(
    model,
    activation_cache: torch.Tensor,
    features: list[tuple[int, int, int, float]],
    steering_method: str,
    epsilon: float,
) -> list[tuple[int, int, int, float]]:
    h_c_values = [h_c_val for _, _, _, h_c_val in features]
    if h_c_values:
        h_c_tensor = torch.as_tensor(h_c_values, device=_device(model), dtype=_dtype(model))
        h_c_norm = float(torch.linalg.norm(h_c_tensor).item()) or 1.0
    else:
        h_c_norm = 1.0

    interventions: list[tuple[int, int, int, float]] = []
    for layer, pos, feat_id, h_c_val in features:
        if pos <= 0:
            continue
        if layer >= activation_cache.shape[0] or pos >= activation_cache.shape[1]:
            continue
        current_val = activation_cache[layer, pos, feat_id]
        delta = utils.compute_steering_delta(
            current_val,
            float(h_c_val),
            h_c_norm,
            steering_method,
            epsilon,
        )
        target = current_val + delta
        if isinstance(target, torch.Tensor):
            target_value = float(target.item())
        else:
            target_value = float(target)
        interventions.append((int(layer), int(pos), int(feat_id), target_value))
    return interventions


def pad_logits(logits_list: list[torch.Tensor], pad_to: int | None = None):
    max_len = pad_to if pad_to is not None else max(logits.shape[1] for logits in logits_list)
    vocab = logits_list[0].shape[-1]
    out = torch.zeros(
        len(logits_list),
        max_len,
        vocab,
        device=logits_list[0].device,
        dtype=logits_list[0].dtype,
    )
    mask = torch.zeros(len(logits_list), max_len, device=logits_list[0].device, dtype=torch.long)
    for idx, logits in enumerate(logits_list):
        seq_len = min(logits.shape[1], max_len)
        out[idx, :seq_len] = logits[0, :seq_len]
        mask[idx, :seq_len] = 1
    return out, mask


def run_batched_steered_pass_on_the_fly(
    model,
    batch_token_ids: list[list[int]],
    features: list[tuple[int, int, int, float]],
    encoder_cache: dict,
    decoder_cache: dict,
    steering_method: str,
    epsilon: float,
    max_seq_len: int | None = None,
):
    logits_list = []
    h_c_values = [h_c_val for _, _, _, h_c_val in features]
    if h_c_values:
        h_c_tensor = torch.as_tensor(h_c_values, device=_device(model), dtype=_dtype(model))
        h_c_norm = float(torch.linalg.norm(h_c_tensor).item()) or 1.0
    else:
        h_c_norm = 1.0

    for token_ids in batch_token_ids:
        if features:
            seq_len = len(token_ids)
            interventions = [
                (
                    int(layer),
                    int(pos),
                    int(feat_id),
                    FeatureSteeringSpec(
                        h_c_value=float(h_c_val),
                        h_c_norm=h_c_norm,
                        steering_method=steering_method,
                        epsilon=float(epsilon),
                    ),
                )
                for layer, pos, feat_id, h_c_val in features
                if 0 < int(pos) < seq_len
            ]
            tokens = _to_1d_tokens(token_ids, model)
            if interventions:
                logits, _ = model.feature_intervention(
                    tokens,
                    interventions,
                    freeze_attention=False,
                    return_activations=False,
                )
                logits = _materialize(logits)
                if logits.ndim == 2:
                    logits = logits.unsqueeze(0)
            else:
                logits = _run_logits(model, token_ids)
        else:
            logits = _run_logits(model, token_ids)
        logits_list.append(logits)
    return pad_logits(logits_list, max_seq_len)


def run_steered_pass_on_the_fly(
    model,
    full_token_ids: list[int],
    features: list[tuple[int, int, int, float]],
    encoder_cache: dict,
    decoder_cache: dict,
    steering_method: str,
    epsilon: float,
    max_seq_len: int | None = None,
):
    logits, _ = run_batched_steered_pass_on_the_fly(
        model,
        [full_token_ids],
        features,
        encoder_cache,
        decoder_cache,
        steering_method,
        epsilon,
        max_seq_len,
    )
    return logits


def prepare_heterogeneous_layer_metadata(
    model,
    batch_items: list[dict],
    max_seq_len: int | None = None,
):
    return None


def run_heterogeneous_steered_pass(
    model,
    batch_items: list[dict],
    steering_method: str,
    epsilon: float,
    max_seq_len: int | None = None,
    precomputed_metadata=None,
):
    logits_list = []
    for item in batch_items:
        logits, _ = run_batched_steered_pass_on_the_fly(
            model,
            [item["token_ids"]],
            item.get("features", []),
            {},
            item.get("decoder_cache", {}),
            steering_method,
            epsilon,
            max_seq_len,
        )
        logits_list.append(logits[:1])
    return pad_logits(logits_list, max_seq_len)


def compute_per_token_centered_logits_batched(
    logits: torch.Tensor,
    batch_cont_info: list[tuple[list[int], int]],
    return_per_token: bool = True,
) -> list[tuple[list[float], float]]:
    results = []
    for batch_idx, (cont_ids, cont_start) in enumerate(batch_cont_info):
        valid_len = min(len(cont_ids), logits.shape[1] - cont_start)
        if valid_len <= 0:
            results.append(([], 0.0))
            continue
        positions = torch.arange(cont_start, cont_start + valid_len, device=logits.device)
        token_ids = torch.as_tensor(cont_ids[:valid_len], device=logits.device, dtype=torch.long)
        cont_logits = logits[batch_idx, positions].float()
        target_logits = cont_logits[torch.arange(valid_len, device=logits.device), token_ids]
        centered = target_logits - cont_logits.mean(dim=-1)
        results.append(
            (centered.tolist() if return_per_token else [], float(centered.mean().item()))
        )
    return results


def compute_branch_log_probs_batch(
    model,
    branches: list[dict],
    logger,
    batch_size: int = 32,
    max_seq_len: int | None = None,
    desc: str | None = None,
    progress=None,
    store_per_token: bool = True,
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    for start in range(0, len(branches), batch_size):
        batch = branches[start : start + batch_size]
        logits, _ = run_batched_steered_pass_on_the_fly(
            model,
            [item["full_token_ids"] for item in batch],
            [],
            {},
            {},
            "additive",
            0.0,
            max_seq_len,
        )
        batch_cont_info = [
            (
                item["continuation_token_ids"],
                len(item["full_token_ids"]) - len(item["continuation_token_ids"]),
            )
            for item in batch
        ]
        centered = compute_per_token_centered_logits_batched(
            logits, batch_cont_info, store_per_token
        )
        for idx, item in enumerate(batch):
            cont_ids, cont_start = batch_cont_info[idx]
            valid_len = min(len(cont_ids), logits.shape[1] - cont_start)
            entry: dict[str, Any] = {
                "log_P_original": 0.0,
                "mean_centered_logit_original": centered[idx][1],
                "mean_target_logit_original": 0.0,
                "mean_target_prob_original": 0.0,
            }
            if valid_len > 0:
                positions = torch.arange(cont_start, cont_start + valid_len, device=logits.device)
                token_ids = torch.as_tensor(
                    cont_ids[:valid_len], device=logits.device, dtype=torch.long
                )
                cont_logits = logits[idx, positions].float()
                log_probs = F.log_softmax(cont_logits, dim=-1)
                selected = log_probs[torch.arange(valid_len, device=logits.device), token_ids]
                target_logits = cont_logits[
                    torch.arange(valid_len, device=logits.device), token_ids
                ]
                entry["log_P_original"] = float(selected.sum().item())
                entry["mean_target_logit_original"] = float(target_logits.mean().item())
                entry["mean_target_prob_original"] = float(torch.exp(selected).mean().item())
                if store_per_token:
                    entry["per_token_log_probs_original"] = selected.tolist()
                    entry["per_token_centered_logits_original"] = centered[idx][0]
                    entry["per_token_target_logits_original"] = target_logits.tolist()
                    entry["per_token_target_probs_original"] = torch.exp(selected).tolist()
            results[item["branch_id"]] = entry
        if progress is not None:
            progress.update(1)
    return results


def compute_baseline_metadata(branches: list[dict], branch_log_probs: dict[int, dict[str, Any]]):
    metadata = {}
    for branch in branches:
        branch_id = branch["branch_id"]
        if branch_id in branch_log_probs and "error" not in branch_log_probs[branch_id]:
            metadata[branch_id] = branch_log_probs[branch_id]
    return metadata


def generate_steered_sequences(
    model,
    prefix_token_ids: list[int],
    features: list[tuple[int, int, int, float]],
    encoder_cache: dict,
    decoder_cache: dict,
    steering_method: str,
    epsilon: float,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    num_samples: int = 1,
    max_seq_len: int | None = None,
) -> list[dict[str, Any]]:
    results = []
    for sample_idx in range(num_samples):
        current_tokens = list(prefix_token_ids)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            logits = run_steered_pass_on_the_fly(
                model,
                current_tokens,
                features,
                encoder_cache,
                decoder_cache,
                steering_method,
                epsilon,
                max_seq_len,
            )
            next_token_logits = logits[0, len(current_tokens) - 1].float() / temperature
            if top_k > 0:
                cutoff = torch.topk(next_token_logits, top_k).values[-1]
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < cutoff, -float("inf")
                )
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cumulative > top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                next_token_logits[sorted_indices[remove]] = -float("inf")
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())
            generated.append(next_token)
            current_tokens.append(next_token)
            tokenizer = getattr(model, "tokenizer", None)
            if getattr(tokenizer, "eos_token_id", None) == next_token:
                break
            if max_seq_len is not None and len(current_tokens) >= max_seq_len:
                break
        generated_text = None
        if getattr(model, "tokenizer", None) is not None:
            generated_text = model.tokenizer.decode(generated, skip_special_tokens=True)
        results.append(
            {
                "sample_idx": sample_idx,
                "generated_tokens": generated,
                "generated_text": generated_text,
                "full_token_ids": current_tokens,
            }
        )
    return results
