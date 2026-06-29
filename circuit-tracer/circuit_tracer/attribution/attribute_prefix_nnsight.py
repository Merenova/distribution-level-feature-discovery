"""nnsight prefix-to-continuation attribution."""

from __future__ import annotations

import logging

import numpy as np
import torch

from circuit_tracer.attribution.context_nnsight import (
    AttributionContext as NNSightAttributionContext,
)
from circuit_tracer.attribution.prefix_context import (
    ContinuationTokenAttribution,
    PrefixAttributionContext,
    PrefixContinuationResult,
)

logger = logging.getLogger("attribution.nnsight")


def _compute_attribution_components(transcoders, mlp_in_tensor, zero_positions):
    """Call TranscoderSet.compute_attribution_components across supported signatures."""

    compute = transcoders.compute_attribution_components

    def _is_signature_error(err: TypeError) -> bool:
        msg = str(err)
        return (
            "unexpected keyword argument" in msg
            or "positional argument" in msg
            or "positional arguments" in msg
        )

    try:
        return compute(mlp_in_tensor, zero_positions=zero_positions)
    except TypeError as err:
        if not _is_signature_error(err):
            raise
    try:
        return compute(mlp_in_tensor, zero_positions)
    except TypeError as err:
        if not _is_signature_error(err):
            raise
    return compute(mlp_in_tensor)


@torch.no_grad()
def setup_prefix_context(prefix_ids: torch.Tensor, model) -> PrefixAttributionContext:
    """Extract and cache prefix components using nnsight tracing."""

    from nnsight import save as nnsight_save  # type: ignore

    device = getattr(model, "device", None) or model.cfg.device
    prefix_ids = prefix_ids.to(device)
    assert prefix_ids.ndim == 1, "prefix_ids must be 1D"

    with model.trace(prefix_ids):
        mlp_in_list = []
        mlp_out_list = []
        for feature_input_loc, feature_output_loc in zip(
            model.feature_input_locs, model.feature_output_locs
        ):
            x = feature_input_loc.output
            if x.ndim == 2:
                x = x.unsqueeze(0)
            mlp_in_list.append(x)

            y = feature_output_loc.output
            if y.ndim == 2:
                y = y.unsqueeze(0)
            mlp_out_list.append(y)

        mlp_in_cache = nnsight_save(torch.cat(mlp_in_list, dim=0))
        mlp_out_cache = nnsight_save(torch.cat(mlp_out_list, dim=0))

    mlp_in_tensor = getattr(mlp_in_cache, "value", mlp_in_cache)
    mlp_out_tensor = getattr(mlp_out_cache, "value", mlp_out_cache)

    zero_positions = getattr(model, "zero_positions", slice(0, 1))
    attribution_data = _compute_attribution_components(
        model.transcoders,
        mlp_in_tensor,
        zero_positions,
    )

    error_vectors = mlp_out_tensor - attribution_data["reconstruction"]
    error_vectors[:, zero_positions] = 0

    embed_weight = getattr(model, "W_E", None)
    if embed_weight is None:
        embed_weight = model.embed_weight
    token_vectors = embed_weight[prefix_ids].detach()

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


def make_prefix_only_attribution_context(prefix_ctx: PrefixAttributionContext):
    """Build an nnsight AttributionContext restricted to prefix-source rows."""

    n_prefix = prefix_ctx.prefix_length
    feature_count = prefix_ctx.n_prefix_features

    class PrefixOnlyNNSightAttributionContext(NNSightAttributionContext):  # type: ignore[misc]
        def compute_error_attributions(self, layer, grads):  # type: ignore[override]
            def error_offset(lyr: int) -> int:
                return feature_count + lyr * n_prefix

            self.compute_score(
                grads,
                self.error_vectors[layer],
                write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
                read_index=np.s_[:, :n_prefix],
            )

        def compute_token_attributions(self, grads):  # type: ignore[override]
            tok_start = feature_count + self.n_layers * n_prefix
            self.compute_score(
                grads,
                self.token_vectors,
                write_index=np.s_[tok_start : tok_start + n_prefix],
                read_index=np.s_[:, :n_prefix],
            )

    cont_ctx = PrefixOnlyNNSightAttributionContext(
        activation_matrix=prefix_ctx.activation_matrix,
        error_vectors=prefix_ctx.error_vectors,
        token_vectors=prefix_ctx.token_vectors,
        decoder_vecs=prefix_ctx.decoder_vecs,
        encoder_vecs=prefix_ctx.encoder_vecs,
        encoder_to_decoder_map=prefix_ctx.encoder_to_decoder_map,
        decoder_locations=prefix_ctx.decoder_locations,
        logits=None,
    )
    cont_ctx.n_layers = prefix_ctx.n_layers
    cont_ctx._row_size = prefix_ctx.n_prefix_sources
    return cont_ctx


def attribute_prefix_to_continuations(
    prefix,
    continuations,
    model,
    *,
    batch_size: int = 512,
    add_bos: bool = True,
    max_feature_nodes: int | None = None,
    verbose: bool = False,
) -> PrefixContinuationResult:
    """Compute attribution from prefix components to continuation-token logits."""

    log = logger if verbose else logging.getLogger("attribution.nnsight")
    device = getattr(model, "device", None) or model.cfg.device

    if isinstance(prefix, list):
        prefix_ids = torch.tensor(prefix, dtype=torch.long, device=device)
    else:
        prefix_ids = prefix.to(device=device, dtype=torch.long)

    if add_bos:
        bos_token_id = getattr(model.tokenizer, "bos_token_id", None)
        if bos_token_id is not None:
            prefix_ids = torch.cat(
                [
                    torch.tensor([bos_token_id], dtype=torch.long, device=device),
                    prefix_ids,
                ]
            )
        else:
            log.warning("No BOS token found, No changes made to prefix")

    if verbose:
        log.info("Setting up prefix context (nnsight backend)...")
    prefix_ctx = setup_prefix_context(prefix_ids, model)

    total_active = prefix_ctx.activation_matrix._nnz()
    if max_feature_nodes is not None and max_feature_nodes < total_active:
        if verbose:
            log.info(
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
            encoder_to_decoder_map=torch.arange(len(top_indices), device=device),
            decoder_locations=prefix_ctx.decoder_locations[:, top_indices],
            n_layers=prefix_ctx.n_layers,
            selected_features=top_indices,
            total_active_features=total_active,
        )

    n_prefix = prefix_ctx.prefix_length
    n_layers = model.cfg.n_layers

    unembed_weight = model.unembed_weight
    d_model = model.cfg.d_model
    if unembed_weight.shape[0] != d_model:
        unembed_weight = unembed_weight.T
    unembed_mean = unembed_weight.mean(dim=-1, keepdim=True)

    all_continuation_attributions: list[list[ContinuationTokenAttribution]] = []

    for cont_idx, continuation in enumerate(continuations):
        if isinstance(continuation, list):
            cont_ids = torch.tensor(continuation, dtype=torch.long, device=device)
        else:
            cont_ids = continuation.to(device=device, dtype=torch.long)

        n_continuation = len(cont_ids)
        full_tokens = torch.cat([prefix_ids, cont_ids])
        full_length = len(full_tokens)

        if verbose:
            log.info(
                "Processing continuation %s: %s tokens (full_length=%s)",
                cont_idx,
                n_continuation,
                full_length,
            )

        token_attributions: list[ContinuationTokenAttribution] = []

        for batch_start in range(0, n_continuation, batch_size):
            batch_end = min(batch_start + batch_size, n_continuation)
            chunk = batch_end - batch_start

            batch_positions = torch.arange(
                n_prefix + batch_start,
                n_prefix + batch_end,
                device=device,
            )
            batch_token_ids = cont_ids[batch_start:batch_end]
            inject_values = (unembed_weight[:, batch_token_ids] - unembed_mean).T

            cont_ctx = make_prefix_only_attribution_context(prefix_ctx)

            with model.trace() as tracer:
                with tracer.invoke(full_tokens.expand(chunk, -1)):
                    pass
                detach_barrier = tracer.barrier(2)
                model.configure_gradient_flow(tracer)
                model.configure_skip_connection(tracer, barrier=detach_barrier)
                cont_ctx.cache_residual(model, tracer, barrier=detach_barrier)

            rows = cont_ctx.compute_batch(
                layers=torch.full((chunk,), n_layers, device=device),
                positions=batch_positions,
                inject_values=inject_values,
                retain_graph=False,
            )
            row_prefix = rows.detach().cpu()[:, : prefix_ctx.n_prefix_sources]

            for i in range(chunk):
                token_attributions.append(
                    ContinuationTokenAttribution(
                        token_id=int(batch_token_ids[i].item()),
                        position=n_prefix + batch_start + i,
                        source_attribution=row_prefix[i],
                    )
                )

            del cont_ctx, rows

        all_continuation_attributions.append(token_attributions)

    return PrefixContinuationResult(
        prefix_tokens=prefix_ids,
        prefix_context=prefix_ctx,
        continuation_attributions=all_continuation_attributions,
    )
