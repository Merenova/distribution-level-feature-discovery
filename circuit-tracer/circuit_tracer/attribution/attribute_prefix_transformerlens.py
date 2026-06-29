"""TransformerLens prefix-to-continuation attribution."""

from __future__ import annotations

import logging

import torch

from circuit_tracer.attribution.prefix_context import (
    ContinuationAttributionContext,
    ContinuationTokenAttribution,
    PrefixAttributionContext,
    PrefixContinuationResult,
)
from circuit_tracer.replacement_model.replacement_model_transformerlens import (
    TransformerLensReplacementModel,
)


@torch.no_grad()
def setup_prefix_context(
    prefix_ids: torch.Tensor,
    model: TransformerLensReplacementModel,
) -> PrefixAttributionContext:
    """Extract and cache prefix components for reuse across continuations."""

    prefix_ids = prefix_ids.to(model.cfg.device)

    mlp_in_cache, mlp_in_caching_hooks, _ = model.get_caching_hooks(
        lambda name: model.feature_input_hook in name
    )
    mlp_out_cache, mlp_out_caching_hooks, _ = model.get_caching_hooks(
        lambda name: model.feature_output_hook in name
    )

    model.run_with_hooks(prefix_ids, fwd_hooks=mlp_in_caching_hooks + mlp_out_caching_hooks)

    mlp_in_cache = torch.cat(list(mlp_in_cache.values()), dim=0)
    mlp_out_cache = torch.cat(list(mlp_out_cache.values()), dim=0)

    attribution_data = model.transcoders.compute_attribution_components(mlp_in_cache)

    error_vectors = mlp_out_cache - attribution_data["reconstruction"]
    error_vectors[:, 0] = 0

    token_vectors = model.W_E[prefix_ids].detach()

    return PrefixAttributionContext(
        prefix_tokens=prefix_ids,
        activation_matrix=attribution_data["activation_matrix"],
        error_vectors=error_vectors,
        token_vectors=token_vectors,
        decoder_vecs=attribution_data["decoder_vecs"],
        encoder_vecs=attribution_data["encoder_vecs"],
        encoder_to_decoder_map=attribution_data["encoder_to_decoder_map"],
        decoder_locations=attribution_data["decoder_locations"],
        n_layers=model.cfg.n_layers,
    )


def attribute_prefix_to_continuations(
    prefix: torch.Tensor | list[int],
    continuations: list[torch.Tensor | list[int]],
    model: TransformerLensReplacementModel,
    *,
    batch_size: int = 512,
    add_bos: bool = True,
    max_feature_nodes: int | None = None,
    verbose: bool = False,
) -> PrefixContinuationResult:
    """Compute attribution from prefix components to continuation-token logits."""

    logger = logging.getLogger("attribution")

    if isinstance(prefix, list):
        prefix_ids = torch.tensor(prefix, dtype=torch.long, device=model.cfg.device)
    else:
        prefix_ids = prefix.to(model.cfg.device)

    if add_bos:
        bos_token_id = model.tokenizer.bos_token_id
        if bos_token_id is not None:
            prefix_ids = torch.cat(
                [
                    torch.tensor([bos_token_id], dtype=torch.long, device=model.cfg.device),
                    prefix_ids,
                ]
            )
        else:
            logger.warning("No BOS token found, No changes made to prefix")

    if verbose:
        logger.info("Setting up prefix context...")
    prefix_ctx = setup_prefix_context(prefix_ids, model)

    total_active = prefix_ctx.activation_matrix._nnz()
    if max_feature_nodes is not None and max_feature_nodes < total_active:
        if verbose:
            logger.info(
                "Selecting top %s of %s features by activation magnitude",
                max_feature_nodes,
                total_active,
            )

        activation_values = prefix_ctx.activation_matrix.values()
        top_indices = torch.argsort(activation_values.abs(), descending=True)[:max_feature_nodes]
        top_indices = top_indices.sort().values

        prefix_ctx = PrefixAttributionContext(
            prefix_tokens=prefix_ctx.prefix_tokens,
            activation_matrix=prefix_ctx.activation_matrix,
            error_vectors=prefix_ctx.error_vectors,
            token_vectors=prefix_ctx.token_vectors,
            decoder_vecs=prefix_ctx.decoder_vecs[top_indices],
            encoder_vecs=prefix_ctx.encoder_vecs[top_indices],
            encoder_to_decoder_map=torch.arange(len(top_indices), device=model.cfg.device),
            decoder_locations=prefix_ctx.decoder_locations[:, top_indices],
            n_layers=prefix_ctx.n_layers,
            selected_features=top_indices,
            total_active_features=total_active,
        )

    n_prefix = prefix_ctx.prefix_length
    n_layers = model.cfg.n_layers

    all_continuation_attributions: list[list[ContinuationTokenAttribution]] = []

    for cont_idx, continuation in enumerate(continuations):
        if isinstance(continuation, list):
            cont_ids = torch.tensor(continuation, dtype=torch.long, device=model.cfg.device)
        else:
            cont_ids = continuation.to(model.cfg.device)

        n_continuation = len(cont_ids)
        full_tokens = torch.cat([prefix_ids, cont_ids])
        full_length = len(full_tokens)

        if verbose:
            logger.info("Processing continuation %s: %s tokens", cont_idx, n_continuation)

        cont_ctx = ContinuationAttributionContext(
            prefix_ctx=prefix_ctx,
            full_sequence_length=full_length,
        )

        with cont_ctx.install_hooks(model):
            residual = model.forward(
                full_tokens.expand(min(batch_size, n_continuation), -1),
                stop_at_layer=n_layers,
            )
            cont_ctx._resid_activations[-1] = model.ln_final(residual)

        token_attributions: list[ContinuationTokenAttribution] = []

        for batch_start in range(0, n_continuation, batch_size):
            batch_end = min(batch_start + batch_size, n_continuation)
            batch_size_actual = batch_end - batch_start

            batch_positions = torch.arange(
                n_prefix + batch_start,
                n_prefix + batch_end,
                device=model.cfg.device,
            )
            batch_token_ids = cont_ids[batch_start:batch_end]

            unembed_cols = model.unembed.W_U[:, batch_token_ids]
            demeaned = unembed_cols - model.unembed.W_U.mean(dim=-1, keepdim=True)
            inject_values = demeaned.T

            rows = cont_ctx.compute_batch(
                layers=torch.full((batch_size_actual,), n_layers, device=model.cfg.device),
                positions=batch_positions,
                inject_values=inject_values,
                retain_graph=batch_end < n_continuation,
            )

            for i in range(batch_size_actual):
                token_attributions.append(
                    ContinuationTokenAttribution(
                        token_id=batch_token_ids[i].item(),
                        position=n_prefix + batch_start + i,
                        source_attribution=rows[i].cpu(),
                    )
                )

        all_continuation_attributions.append(token_attributions)

    return PrefixContinuationResult(
        prefix_tokens=prefix_ids,
        prefix_context=prefix_ctx,
        continuation_attributions=all_continuation_attributions,
    )
