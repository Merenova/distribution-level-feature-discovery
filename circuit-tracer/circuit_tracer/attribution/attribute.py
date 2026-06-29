"""
Build an **attribution graph** that captures the *direct*, *linear* effects
between features and next-token logits for a *prompt-specific*
**local replacement model**.

High-level algorithm (matches the 2025 ``Attribution Graphs`` paper):
https://transformer-circuits.pub/2025/attribution-graphs/methods.html

1. **Local replacement model** - we configure gradients to flow only through
   linear components of the network, effectively bypassing attention mechanisms,
   MLP non-linearities, and layer normalization scales.
2. **Forward pass** - record residual-stream activations and mark every active
   feature.
3. **Backward passes** - for each source node (feature or logit), inject a
   *custom* gradient that selects its encoder/decoder direction.  Because the
   model is linear in the residual stream under our freezes, this contraction
   equals the *direct effect* A_{s->t}.
4. **Assemble graph** - store edge weights in a dense matrix and package a
   ``Graph`` object.  Downstream utilities can *prune* the graph to the subset
   needed for interpretation.
"""

import logging
import time
from typing import Literal

import torch
from tqdm import tqdm

from circuit_tracer.graph import Graph
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.utils import get_default_device
from circuit_tracer.utils.disk_offload import offload_modules


@torch.no_grad()
def compute_salient_logits(
    logits: torch.Tensor,
    unembed_proj: torch.Tensor,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick the smallest logit set whose cumulative prob >= *desired_logit_prob*.

    Args:
        logits: ``(d_vocab,)`` vector (single position).
        unembed_proj: ``(d_model, d_vocab)`` unembedding matrix.
        max_n_logits: Hard cap *k*.
        desired_logit_prob: Cumulative probability threshold *p*.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            * logit_indices - ``(k,)`` vocabulary ids.
            * logit_probs   - ``(k,)`` softmax probabilities.
            * demeaned_vecs - ``(k, d_model)`` unembedding columns, demeaned.
    """

    probs = torch.softmax(logits, dim=-1)
    top_p, top_idx = torch.topk(probs, max_n_logits)
    cutoff = int(torch.searchsorted(torch.cumsum(top_p, 0), desired_logit_prob)) + 1
    top_p, top_idx = top_p[:cutoff], top_idx[:cutoff]

    cols = unembed_proj[:, top_idx]
    demeaned = cols - unembed_proj.mean(dim=-1, keepdim=True)
    return top_idx, top_p, demeaned.T


def compute_partial_influences(edge_matrix, logit_p, row_to_node_index, max_iter=128, device=None):
    """Compute partial influences using power iteration method."""
    device = device or get_default_device()

    normalized_matrix = torch.empty_like(edge_matrix, device=device).copy_(edge_matrix)
    normalized_matrix = normalized_matrix.abs_()
    normalized_matrix /= normalized_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)

    influences = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod = torch.zeros(edge_matrix.shape[1], device=normalized_matrix.device)
    prod[-len(logit_p) :] = logit_p

    for _ in range(max_iter):
        prod = prod[row_to_node_index] @ normalized_matrix
        if not prod.any():
            break
        influences += prod
    else:
        raise RuntimeError("Failed to converge")

    return influences


def attribute(
    prompt: str | torch.Tensor | list[int],
    model: ReplacementModel,
    *,
    max_n_logits: int = 10,
    desired_logit_prob: float = 0.95,
    batch_size: int = 512,
    max_feature_nodes: int | None = None,
    offload: Literal["cpu", "disk", None] = None,
    verbose: bool = False,
    update_interval: int = 4,
) -> Graph:
    """Compute an attribution graph for *prompt*.

    Args:
        prompt: Text, token ids, or tensor - will be tokenized if str.
        model: Frozen ``ReplacementModel``
        max_n_logits: Max number of logit nodes.
        desired_logit_prob: Keep logits until cumulative prob >= this value.
        batch_size: How many source nodes to process per backward pass.
        max_feature_nodes: Max number of feature nodes to include in the graph.
        offload: Method for offloading model parameters to save memory.
                 Options are "cpu" (move to CPU), "disk" (save to disk),
                 or None (no offloading).
        verbose: Whether to show progress information.
        update_interval: Number of batches to process before updating the feature ranking.

    Returns:
        Graph: Fully dense adjacency (unpruned).
    """

    logger = logging.getLogger("attribution")
    logger.propagate = False
    handler = None
    if verbose and not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    offload_handles = []
    try:
        return _run_attribution(
            model=model,
            prompt=prompt,
            max_n_logits=max_n_logits,
            desired_logit_prob=desired_logit_prob,
            batch_size=batch_size,
            max_feature_nodes=max_feature_nodes,
            offload=offload,
            verbose=verbose,
            offload_handles=offload_handles,
            update_interval=update_interval,
            logger=logger,
        )
    finally:
        for reload_handle in offload_handles:
            reload_handle()

        if handler:
            logger.removeHandler(handler)


def _run_attribution(
    model,
    prompt,
    max_n_logits,
    desired_logit_prob,
    batch_size,
    max_feature_nodes,
    offload,
    verbose,
    offload_handles,
    logger,
    update_interval=4,
):
    start_time = time.time()
    # Phase 0: precompute
    logger.info("Phase 0: Precomputing activations and vectors")
    phase_start = time.time()
    input_ids = model.ensure_tokenized(prompt)

    ctx = model.setup_attribution(input_ids)
    activation_matrix = ctx.activation_matrix

    logger.info(f"Precomputation completed in {time.time() - phase_start:.2f}s")
    logger.info(f"Found {ctx.activation_matrix._nnz()} active features")

    if offload:
        offload_handles += offload_modules(model.transcoders, offload)

    # Phase 1: forward pass
    logger.info("Phase 1: Running forward pass")
    phase_start = time.time()
    with ctx.install_hooks(model):
        residual = model.forward(input_ids.expand(batch_size, -1), stop_at_layer=model.cfg.n_layers)
        ctx._resid_activations[-1] = model.ln_final(residual)
    logger.info(f"Forward pass completed in {time.time() - phase_start:.2f}s")

    if offload:
        offload_handles += offload_modules([block.mlp for block in model.blocks], offload)

    # Phase 2: build input vector list
    logger.info("Phase 2: Building input vectors")
    phase_start = time.time()
    feat_layers, feat_pos, _ = activation_matrix.indices()
    n_layers, n_pos, _ = activation_matrix.shape
    total_active_feats = activation_matrix._nnz()

    logit_idx, logit_p, logit_vecs = compute_salient_logits(
        ctx.logits[0, -1],
        model.unembed.W_U,
        max_n_logits=max_n_logits,
        desired_logit_prob=desired_logit_prob,
    )
    logger.info(
        f"Selected {len(logit_idx)} logits with cumulative probability {logit_p.sum().item():.4f}"
    )

    if offload:
        offload_handles += offload_modules([model.unembed, model.embed], offload)

    logit_offset = len(feat_layers) + (n_layers + 1) * n_pos
    n_logits = len(logit_idx)
    total_nodes = logit_offset + n_logits

    max_feature_nodes = min(max_feature_nodes or total_active_feats, total_active_feats)
    logger.info(f"Will include {max_feature_nodes} of {total_active_feats} feature nodes")

    edge_matrix = torch.zeros(max_feature_nodes + n_logits, total_nodes)
    # Maps row indices in edge_matrix to original feature/node indices
    # First populated with logit node IDs, then feature IDs in attribution order
    row_to_node_index = torch.zeros(max_feature_nodes + n_logits, dtype=torch.int32)
    logger.info(f"Input vectors built in {time.time() - phase_start:.2f}s")

    # Phase 3: logit attribution
    logger.info("Phase 3: Computing logit attributions")
    phase_start = time.time()
    for i in range(0, len(logit_idx), batch_size):
        batch = logit_vecs[i : i + batch_size]
        rows = ctx.compute_batch(
            layers=torch.full((batch.shape[0],), n_layers),
            positions=torch.full((batch.shape[0],), n_pos - 1),
            inject_values=batch,
        )
        edge_matrix[i : i + batch.shape[0], :logit_offset] = rows.cpu()
        row_to_node_index[i : i + batch.shape[0]] = (
            torch.arange(i, i + batch.shape[0]) + logit_offset
        )
    logger.info(f"Logit attributions completed in {time.time() - phase_start:.2f}s")

    # Phase 4: feature attribution
    logger.info("Phase 4: Computing feature attributions")
    phase_start = time.time()
    st = n_logits
    visited = torch.zeros(total_active_feats, dtype=torch.bool)
    n_visited = 0

    pbar = tqdm(total=max_feature_nodes, desc="Feature influence computation", disable=not verbose)

    while n_visited < max_feature_nodes:
        if max_feature_nodes == total_active_feats:
            pending = torch.arange(total_active_feats)
        else:
            influences = compute_partial_influences(
                edge_matrix[:st], logit_p, row_to_node_index[:st]
            )
            feature_rank = torch.argsort(influences[:total_active_feats], descending=True).cpu()
            queue_size = min(update_interval * batch_size, max_feature_nodes - n_visited)
            pending = feature_rank[~visited[feature_rank]][:queue_size]

        queue = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]

        for idx_batch in queue:
            n_visited += len(idx_batch)

            rows = ctx.compute_batch(
                layers=feat_layers[idx_batch],
                positions=feat_pos[idx_batch],
                inject_values=ctx.encoder_vecs[idx_batch],
                retain_graph=n_visited < max_feature_nodes,
            )

            end = min(st + batch_size, st + rows.shape[0])
            edge_matrix[st:end, :logit_offset] = rows.cpu()
            row_to_node_index[st:end] = idx_batch
            visited[idx_batch] = True
            st = end
            pbar.update(len(idx_batch))

    pbar.close()
    logger.info(f"Feature attributions completed in {time.time() - phase_start:.2f}s")

    # Phase 5: packaging graph
    selected_features = torch.where(visited)[0]
    if max_feature_nodes < total_active_feats:
        non_feature_nodes = torch.arange(total_active_feats, total_nodes)
        col_read = torch.cat([selected_features, non_feature_nodes])
        edge_matrix = edge_matrix[:, col_read]

    # sort rows such that features are in order
    edge_matrix = edge_matrix[row_to_node_index.argsort()]
    final_node_count = edge_matrix.shape[1]
    full_edge_matrix = torch.zeros(final_node_count, final_node_count)
    full_edge_matrix[:max_feature_nodes] = edge_matrix[:max_feature_nodes]
    full_edge_matrix[-n_logits:] = edge_matrix[max_feature_nodes:]

    graph = Graph(
        input_string=model.tokenizer.decode(input_ids),
        input_tokens=input_ids,
        logit_tokens=logit_idx,
        logit_probabilities=logit_p,
        active_features=activation_matrix.indices().T,
        activation_values=activation_matrix.values(),
        selected_features=selected_features,
        adjacency_matrix=full_edge_matrix,
        cfg=model.cfg,
        scan=model.scan,
    )

    total_time = time.time() - start_time
    logger.info(f"Attribution completed in {total_time:.2f}s")

    return graph


# ============================================================================
# Prefix-to-Continuation Attribution
# ============================================================================

from dataclasses import dataclass
from circuit_tracer.attribution.context import PrefixAttributionContext, ContinuationAttributionContext


@dataclass
class ContinuationTokenAttribution:
    """Attribution scores for a single continuation token.

    Attributes:
        token_id: The token ID being predicted
        position: Position in the full sequence (prefix + continuation)
        source_attribution: (n_prefix_sources,) attribution scores to prefix components
    """
    token_id: int
    position: int
    source_attribution: torch.Tensor


@dataclass
class PrefixContinuationResult:
    """Result of prefix-to-continuation attribution.

    Attributes:
        prefix_tokens: (n_prefix,) token IDs of the prefix
        prefix_context: Cached prefix computations for potential reuse
        continuation_attributions: List of lists - [continuation_idx][token_idx]
    """
    prefix_tokens: torch.Tensor
    prefix_context: PrefixAttributionContext
    continuation_attributions: list[list[ContinuationTokenAttribution]]


@torch.no_grad()
def setup_prefix_context(
    prefix_ids: torch.Tensor,
    model: ReplacementModel,
) -> PrefixAttributionContext:
    """Extract and cache prefix components for reuse across continuations.

    Args:
        prefix_ids: (n_prefix,) token IDs with BOS already prepended if needed
        model: Frozen ReplacementModel

    Returns:
        PrefixAttributionContext with cached prefix components
    """
    prefix_ids = prefix_ids.to(model.cfg.device)

    # Get MLP in/out caches for prefix
    mlp_in_cache, mlp_in_caching_hooks, _ = model.get_caching_hooks(
        lambda name: model.feature_input_hook in name
    )
    mlp_out_cache, mlp_out_caching_hooks, _ = model.get_caching_hooks(
        lambda name: model.feature_output_hook in name
    )

    _ = model.run_with_hooks(prefix_ids, fwd_hooks=mlp_in_caching_hooks + mlp_out_caching_hooks)

    mlp_in_cache = torch.cat(list(mlp_in_cache.values()), dim=0)
    mlp_out_cache = torch.cat(list(mlp_out_cache.values()), dim=0)

    # Compute attribution components
    attribution_data = model.transcoders.compute_attribution_components(mlp_in_cache)

    # Compute error vectors
    error_vectors = mlp_out_cache - attribution_data["reconstruction"]
    error_vectors[:, 0] = 0  # Zero first position (BOS artifact)

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
    model: ReplacementModel,
    *,
    batch_size: int = 512,
    add_bos: bool = True,
    max_feature_nodes: int | None = None,
    verbose: bool = False,
) -> PrefixContinuationResult:
    """Compute attribution graphs from prefix components to continuation tokens.

    For each continuation token, computes attribution scores from all prefix
    components (features, errors, tokens) to that token's logit prediction.

    Args:
        prefix: Token IDs of the prefix (without BOS unless add_bos=False)
        continuations: List of continuation token ID sequences
        model: Frozen ReplacementModel
        batch_size: How many continuation tokens to process per backward pass
        add_bos: Whether to prepend BOS token to prefix
        max_feature_nodes: Max number of prefix feature nodes to include (by activation magnitude).
            If None, includes all active features.
        verbose: Whether to show progress information

    Returns:
        PrefixContinuationResult containing:
        - prefix_tokens: Token IDs (with BOS if add_bos=True)
        - prefix_context: Cached prefix computations
        - continuation_attributions: [continuation_idx][token_idx] attribution scores
    """
    logger = logging.getLogger("attribution")

    # Convert prefix to tensor
    if isinstance(prefix, list):
        prefix_ids = torch.tensor(prefix, dtype=torch.long, device=model.cfg.device)
    else:
        prefix_ids = prefix.to(model.cfg.device)

    # Optionally prepend BOS
    if add_bos:
        bos_token_id = model.tokenizer.bos_token_id 
        if bos_token_id is not None:
            prefix_ids = torch.cat([
                torch.tensor([bos_token_id], dtype=torch.long, device=model.cfg.device),
                prefix_ids
            ])
        else:
            logger.warning("No BOS token found, No changes made to prefix")

    # Phase 1: Setup prefix context (shared across all continuations)
    if verbose:
        logger.info("Setting up prefix context...")
    prefix_ctx = setup_prefix_context(prefix_ids, model)

    # Apply max_feature_nodes selection if specified
    total_active = prefix_ctx.activation_matrix._nnz()
    if max_feature_nodes is not None and max_feature_nodes < total_active:
        if verbose:
            logger.info(f"Selecting top {max_feature_nodes} of {total_active} features by activation magnitude")

        # Select top features by activation magnitude
        activation_values = prefix_ctx.activation_matrix.values()
        top_indices = torch.argsort(activation_values.abs(), descending=True)[:max_feature_nodes]
        top_indices = top_indices.sort().values  # Keep original order

        # Filter prefix_ctx to only include selected features
        prefix_ctx = PrefixAttributionContext(
            prefix_tokens=prefix_ctx.prefix_tokens,
            activation_matrix=prefix_ctx.activation_matrix,  # Keep full for reference
            error_vectors=prefix_ctx.error_vectors,
            token_vectors=prefix_ctx.token_vectors,
            decoder_vecs=prefix_ctx.decoder_vecs[top_indices],
            encoder_vecs=prefix_ctx.encoder_vecs[top_indices],
            encoder_to_decoder_map=torch.arange(len(top_indices), device=model.cfg.device),  # Remap to 0..n-1
            decoder_locations=prefix_ctx.decoder_locations[:, top_indices],
            n_layers=prefix_ctx.n_layers,
            selected_features=top_indices,
            total_active_features=total_active,
        )

    n_prefix = prefix_ctx.prefix_length
    n_layers = model.cfg.n_layers

    all_continuation_attributions: list[list[ContinuationTokenAttribution]] = []

    # Phase 2: Process each continuation
    for cont_idx, continuation in enumerate(continuations):
        # Convert continuation to tensor
        if isinstance(continuation, list):
            cont_ids = torch.tensor(continuation, dtype=torch.long, device=model.cfg.device)
        else:
            cont_ids = continuation.to(model.cfg.device)

        n_continuation = len(cont_ids)
        full_tokens = torch.cat([prefix_ids, cont_ids])
        full_length = len(full_tokens)

        if verbose:
            logger.info(f"Processing continuation {cont_idx}: {n_continuation} tokens")

        # Create continuation context
        cont_ctx = ContinuationAttributionContext(
            prefix_ctx=prefix_ctx,
            full_sequence_length=full_length,
        )

        # Forward pass with caching hooks for full sequence
        with cont_ctx.install_hooks(model):
            residual = model.forward(
                full_tokens.expand(min(batch_size, n_continuation), -1),
                stop_at_layer=n_layers,
            )
            cont_ctx._resid_activations[-1] = model.ln_final(residual)

        # Compute attribution for each continuation token
        token_attributions: list[ContinuationTokenAttribution] = []

        for batch_start in range(0, n_continuation, batch_size):
            batch_end = min(batch_start + batch_size, n_continuation)
            batch_size_actual = batch_end - batch_start

            # Positions in full sequence
            batch_positions = torch.arange(
                n_prefix + batch_start,
                n_prefix + batch_end,
                device=model.cfg.device,
            )
            batch_token_ids = cont_ids[batch_start:batch_end]

            # Get demeaned unembedding vectors for teacher-forced prediction
            unembed_cols = model.unembed.W_U[:, batch_token_ids]  # (d_model, batch)
            demeaned = unembed_cols - model.unembed.W_U.mean(dim=-1, keepdim=True)
            inject_values = demeaned.T  # (batch, d_model)

            # Compute attribution
            rows = cont_ctx.compute_batch(
                layers=torch.full((batch_size_actual,), n_layers, device=model.cfg.device),
                positions=batch_positions,
                inject_values=inject_values,
                retain_graph=(batch_end < n_continuation),
            )

            # Store results
            for i in range(batch_size_actual):
                token_attributions.append(ContinuationTokenAttribution(
                    token_id=batch_token_ids[i].item(),
                    position=n_prefix + batch_start + i,
                    source_attribution=rows[i].cpu(),
                ))

        all_continuation_attributions.append(token_attributions)

    return PrefixContinuationResult(
        prefix_tokens=prefix_ids,
        prefix_context=prefix_ctx,
        continuation_attributions=all_continuation_attributions,
    )
