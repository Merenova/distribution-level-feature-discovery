"""
Attribution context for managing hooks during attribution computation.
"""

import contextlib
import weakref
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING
from collections.abc import Callable

import numpy as np
import torch
from einops import einsum
from transformer_lens.hook_points import HookPoint

if TYPE_CHECKING:
    from circuit_tracer.replacement_model import ReplacementModel


@dataclass
class PrefixAttributionContext:
    """Cached prefix computations shared across all continuations.

    This stores all the components needed for attribution that are specific
    to the prefix and can be reused when computing attribution to different
    continuations.

    Attributes:
        prefix_tokens: (n_prefix,) token IDs of the prefix
        activation_matrix: sparse (n_layers, n_prefix, d_transcoder) feature activations
        error_vectors: (n_layers, n_prefix, d_model) reconstruction errors
        token_vectors: (n_prefix, d_model) token embeddings
        decoder_vecs: (n_selected_features, d_model) scaled decoder vectors for selected features
        encoder_vecs: (n_selected_features, d_model) encoder vectors for selected features
        encoder_to_decoder_map: mapping from encoder to decoder indices
        decoder_locations: (2, n_selected_features) layer and position indices for selected features
        n_layers: number of model layers
        selected_features: indices of selected features (None = all features)
        total_active_features: total number of active features before selection
    """
    prefix_tokens: torch.Tensor
    activation_matrix: torch.Tensor  # sparse
    error_vectors: torch.Tensor
    token_vectors: torch.Tensor
    decoder_vecs: torch.Tensor
    encoder_vecs: torch.Tensor
    encoder_to_decoder_map: torch.Tensor
    decoder_locations: torch.Tensor
    n_layers: int
    selected_features: torch.Tensor | None = None  # indices of selected features
    total_active_features: int | None = None  # total before selection

    @property
    def prefix_length(self) -> int:
        return len(self.prefix_tokens)

    @property
    def n_prefix_features(self) -> int:
        """Number of selected features (or all if no selection)."""
        if self.selected_features is not None:
            return len(self.selected_features)
        return self.activation_matrix._nnz()

    @property
    def n_prefix_errors(self) -> int:
        return self.n_layers * self.prefix_length

    @property
    def n_prefix_tokens(self) -> int:
        return self.prefix_length

    @property
    def n_prefix_sources(self) -> int:
        """Total number of source nodes in the prefix."""
        return self.n_prefix_features + self.n_prefix_errors + self.n_prefix_tokens

    def get_feature_info(self, attr_idx: int) -> tuple[int, int, int, float] | None:
        """Get (layer, position, feature_idx, activation_value) for an attribution index.

        Args:
            attr_idx: Index in the source_attribution vector (0 to n_prefix_sources-1)

        Returns:
            Tuple of (layer, position, feature_idx, activation_value) if this is a feature node,
            None if this is an error or token node.
        """
        if attr_idx >= self.n_prefix_features:
            return None  # Error or token node

        # Get the full activation matrix indices
        all_layers, all_positions, all_feature_idxs = self.activation_matrix.indices()
        all_values = self.activation_matrix.values()

        # Map to original index if we have selected features
        if self.selected_features is not None:
            original_idx = self.selected_features[attr_idx].item()
        else:
            original_idx = attr_idx

        layer = all_layers[original_idx].item()
        position = all_positions[original_idx].item()
        feature_idx = all_feature_idxs[original_idx].item()
        value = all_values[original_idx].item()

        return (layer, position, feature_idx, value)

    def attribution_to_interventions(
        self,
        attr_indices: list[int] | torch.Tensor,
        values: list[float] | torch.Tensor | None = None,
        scale: float = 1.0,
    ) -> list[tuple[int, int, int, float]]:
        """Convert attribution indices to intervention format for feature_intervention().

        Args:
            attr_indices: Indices into source_attribution vector (feature nodes only, < n_prefix_features)
            values: Override activation values. If None, uses original activations scaled by `scale`.
            scale: Multiplier for original activation values (only used if values is None)

        Returns:
            List of (layer, position, feature_idx, value) tuples for feature_intervention()
        """
        if isinstance(attr_indices, torch.Tensor):
            attr_indices = attr_indices.tolist()
        if isinstance(values, torch.Tensor):
            values = values.tolist()

        interventions = []
        for i, attr_idx in enumerate(attr_indices):
            info = self.get_feature_info(attr_idx)
            if info is None:
                continue  # Skip non-feature nodes

            layer, position, feature_idx, orig_value = info
            if values is not None:
                value = values[i]
            else:
                value = orig_value * scale

            interventions.append((layer, position, feature_idx, value))

        return interventions

    def get_top_features(
        self,
        source_attribution: torch.Tensor,
        k: int = 10,
        by_abs: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get top-k feature indices by attribution score.

        Args:
            source_attribution: (n_prefix_sources,) attribution scores
            k: Number of top features to return
            by_abs: If True, sort by absolute value; otherwise by raw value

        Returns:
            Tuple of (top_indices, top_scores) where top_indices are into source_attribution
        """
        # Only consider feature nodes (first n_prefix_features indices)
        feature_scores = source_attribution[:self.n_prefix_features]

        if by_abs:
            sorted_indices = torch.argsort(feature_scores.abs(), descending=True)
        else:
            sorted_indices = torch.argsort(feature_scores, descending=True)

        top_k = min(k, len(sorted_indices))
        top_indices = sorted_indices[:top_k]
        top_scores = feature_scores[top_indices]

        return top_indices, top_scores


class AttributionContext:
    """Manage hooks for computing attribution rows.

    This helper caches residual-stream activations **(forward pass)** and then
    registers backward hooks that populate a write-only buffer with
    *direct-effect rows* **(backward pass)**.

    The buffer layout concatenates rows for **feature nodes**, **error nodes**,
    **token-embedding nodes**

    Args:
        activation_matrix (torch.sparse.Tensor):
            Sparse `(n_layers, n_pos, n_features)` tensor indicating **which**
            features fired at each layer/position.
        error_vectors (torch.Tensor):
            `(n_layers, n_pos, d_model)` - *residual* the CLT / PLT failed to
            reconstruct ("error nodes").
        token_vectors (torch.Tensor):
            `(n_pos, d_model)` - embeddings of the prompt tokens.
        decoder_vectors (torch.Tensor):
            `(total_active_features, d_model)` - decoder rows **only for active
            features**, already multiplied by feature activations so they
            represent a_s * W^dec.
    """

    def __init__(
        self,
        activation_matrix: torch.Tensor,
        error_vectors: torch.Tensor,
        token_vectors: torch.Tensor,
        decoder_vecs: torch.Tensor,
        encoder_vecs: torch.Tensor,
        encoder_to_decoder_map: torch.Tensor,
        decoder_locations: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        n_layers, n_pos, _ = activation_matrix.shape

        # Forward-pass cache
        self._resid_activations: list[torch.Tensor | None] = [None] * (n_layers + 1)
        self._batch_buffer: torch.Tensor | None = None
        self.n_layers: int = n_layers

        self.logits = logits
        self.activation_matrix = activation_matrix
        self.error_vectors = error_vectors
        self.token_vectors = token_vectors
        self.decoder_vecs = decoder_vecs
        self.encoder_vecs = encoder_vecs

        self.encoder_to_decoder_map = encoder_to_decoder_map
        self.decoder_locations = decoder_locations

        total_active_feats = activation_matrix._nnz()
        self._row_size: int = total_active_feats + (n_layers + 1) * n_pos  # + logits later

    def _caching_hooks(self, feature_input_hook: str) -> list[tuple[str, Callable]]:
        """Return hooks that store residual activations layer-by-layer."""

        proxy = weakref.proxy(self)

        def _cache(acts: torch.Tensor, hook: HookPoint, *, layer: int) -> torch.Tensor:
            proxy._resid_activations[layer] = acts
            return acts

        hooks = [
            (f"blocks.{layer}.{feature_input_hook}", partial(_cache, layer=layer))
            for layer in range(self.n_layers)
        ]
        hooks.append(("unembed.hook_pre", partial(_cache, layer=self.n_layers)))
        return hooks

    def _compute_score_hook(
        self,
        hook_name: str,
        output_vecs: torch.Tensor,
        write_index: slice,
        read_index: slice | np.ndarray = np.s_[:],
    ) -> tuple[str, Callable]:
        """
        Factory that contracts *gradients* with an **output vector set**.
        The hook computes A_{s->t} and writes the result into an in-place buffer row.
        """

        proxy = weakref.proxy(self)

        def _hook_fn(grads: torch.Tensor, hook: HookPoint) -> None:
            proxy._batch_buffer[write_index] += einsum(
                grads.to(output_vecs.dtype)[read_index],
                output_vecs,
                "batch position d_model, position d_model -> position batch",
            )

        return hook_name, _hook_fn

    def _make_attribution_hooks(self, feature_output_hook: str) -> list[tuple[str, Callable]]:
        """Create the complete backward-hook for computing attribution scores."""

        n_layers, n_pos, _ = self.activation_matrix.shape
        nnz_layers, nnz_positions = self.decoder_locations

        # Feature nodes
        feature_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{feature_output_hook}",
                self.decoder_vecs[layer_mask],
                write_index=self.encoder_to_decoder_map[layer_mask],  # type: ignore
                read_index=np.s_[:, nnz_positions[layer_mask]],  # type: ignore
            )
            for layer in range(n_layers)
            if (layer_mask := nnz_layers == layer).any()
        ]

        # Error nodes
        def error_offset(layer: int) -> int:  # starting row for this layer
            return self.activation_matrix._nnz() + layer * n_pos

        error_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{feature_output_hook}",
                self.error_vectors[layer],
                write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
            )
            for layer in range(n_layers)
        ]

        # Token-embedding nodes
        tok_start = error_offset(n_layers)
        token_hook = [
            self._compute_score_hook(
                "hook_embed",
                self.token_vectors,
                write_index=np.s_[tok_start : tok_start + n_pos],
            )
        ]

        return feature_hooks + error_hooks + token_hook

    @contextlib.contextmanager
    def install_hooks(self, model: "ReplacementModel"):
        """Context manager instruments the hooks for the forward and backward passes."""
        with model.hooks(
            fwd_hooks=self._caching_hooks(model.feature_input_hook),  # type: ignore
            bwd_hooks=self._make_attribution_hooks(model.feature_output_hook),  # type: ignore
        ):
            yield

    def compute_batch(
        self,
        layers: torch.Tensor,
        positions: torch.Tensor,
        inject_values: torch.Tensor,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        """Return attribution rows for a batch of (layer, pos) nodes.

        The routine overrides gradients at **exact** residual-stream locations
        triggers one backward pass, and copies the rows from the internal buffer.

        Args:
            layers: 1-D tensor of layer indices *l* for the source nodes.
            positions: 1-D tensor of token positions *c* for the source nodes.
            inject_values: `(batch, d_model)` tensor with outer product
                a_s * W^(enc/dec) to inject as custom gradient.

        Returns:
            torch.Tensor: ``(batch, row_size)`` matrix - one row per node.
        """

        assert self._resid_activations[0] is not None, "Residual activations are not cached"
        batch_size = self._resid_activations[0].shape[0]
        self._batch_buffer = torch.zeros(
            self._row_size,
            batch_size,
            dtype=inject_values.dtype,
            device=inject_values.device,
        )

        # Custom gradient injection (per-layer registration)
        batch_idx = torch.arange(len(layers), device=layers.device)

        def _inject(grads, *, batch_indices, pos_indices, values):
            grads_out = grads.clone().to(values.dtype)
            grads_out.index_put_((batch_indices, pos_indices), values)
            return grads_out.to(grads.dtype)

        handles = []
        layers_in_batch = layers.unique().tolist()

        for layer in layers_in_batch:
            mask = layers == layer
            if not mask.any():
                continue
            fn = partial(
                _inject,
                batch_indices=batch_idx[mask],
                pos_indices=positions[mask],
                values=inject_values[mask],
            )
            resid_activations = self._resid_activations[int(layer)]
            assert resid_activations is not None, "Residual activations are not cached"
            handles.append(resid_activations.register_hook(fn))

        try:
            last_layer = max(layers_in_batch)
            self._resid_activations[last_layer].backward(
                gradient=torch.zeros_like(self._resid_activations[last_layer]),
                retain_graph=retain_graph,
            )
        finally:
            for h in handles:
                h.remove()

        buf, self._batch_buffer = self._batch_buffer, None
        return buf.T[: len(layers)]


class ContinuationAttributionContext:
    """Attribution context for computing attribution from prefix to continuation tokens.

    This context handles the case where we want to attribute continuation token
    logits to prefix components only. It caches residual activations for the full
    sequence (prefix + continuation) but only accumulates attribution scores from
    prefix positions.

    Args:
        prefix_ctx: Cached prefix computations (PrefixAttributionContext)
        full_sequence_length: Total length of prefix + continuation
    """

    def __init__(
        self,
        prefix_ctx: PrefixAttributionContext,
        full_sequence_length: int,
    ) -> None:
        self.prefix_ctx = prefix_ctx
        self.full_length = full_sequence_length
        self.n_layers = prefix_ctx.n_layers

        # Forward-pass cache for full sequence (prefix + continuation)
        self._resid_activations: list[torch.Tensor | None] = [None] * (self.n_layers + 1)
        self._batch_buffer: torch.Tensor | None = None

        # Row size only includes prefix sources
        self._row_size = prefix_ctx.n_prefix_sources

    def _caching_hooks(self, feature_input_hook: str) -> list[tuple[str, Callable]]:
        """Return hooks that store residual activations layer-by-layer for full sequence."""

        proxy = weakref.proxy(self)

        def _cache(acts: torch.Tensor, hook: HookPoint, *, layer: int) -> torch.Tensor:
            proxy._resid_activations[layer] = acts
            return acts

        hooks = [
            (f"blocks.{layer}.{feature_input_hook}", partial(_cache, layer=layer))
            for layer in range(self.n_layers)
        ]
        hooks.append(("unembed.hook_pre", partial(_cache, layer=self.n_layers)))
        return hooks

    def _compute_score_hook(
        self,
        hook_name: str,
        output_vecs: torch.Tensor,
        write_index: slice,
        read_index: slice | np.ndarray = np.s_[:],
    ) -> tuple[str, Callable]:
        """Factory that contracts gradients with output vectors, writing to buffer."""

        proxy = weakref.proxy(self)

        def _hook_fn(grads: torch.Tensor, hook: HookPoint) -> None:
            proxy._batch_buffer[write_index] += einsum(
                grads.to(output_vecs.dtype)[read_index],
                output_vecs,
                "batch position d_model, position d_model -> position batch",
            )

        return hook_name, _hook_fn

    def _make_prefix_only_attribution_hooks(
        self, feature_output_hook: str
    ) -> list[tuple[str, Callable]]:
        """Create backward hooks that only accumulate from prefix positions.

        This is the key difference from AttributionContext._make_attribution_hooks:
        the read_index is restricted to prefix positions only.
        """
        n_layers = self.n_layers
        n_prefix = self.prefix_ctx.prefix_length
        nnz_layers, nnz_positions = self.prefix_ctx.decoder_locations

        # Feature hooks - only include features at prefix positions
        # (prefix_ctx already only has prefix features)
        feature_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{feature_output_hook}",
                self.prefix_ctx.decoder_vecs[layer_mask],
                write_index=self.prefix_ctx.encoder_to_decoder_map[layer_mask],  # type: ignore
                read_index=np.s_[:, nnz_positions[layer_mask]],  # type: ignore
            )
            for layer in range(n_layers)
            if (layer_mask := nnz_layers == layer).any()
        ]

        # Error hooks - only prefix positions (0:n_prefix)
        def error_offset(layer: int) -> int:
            return self.prefix_ctx.n_prefix_features + layer * n_prefix

        error_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{feature_output_hook}",
                self.prefix_ctx.error_vectors[layer],  # (n_prefix, d_model)
                write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
                read_index=np.s_[:, :n_prefix],  # Only read prefix gradients
            )
            for layer in range(n_layers)
        ]

        # Token-embedding hooks - only prefix positions
        tok_start = error_offset(n_layers)
        token_hook = [
            self._compute_score_hook(
                "hook_embed",
                self.prefix_ctx.token_vectors,  # (n_prefix, d_model)
                write_index=np.s_[tok_start : tok_start + n_prefix],
                read_index=np.s_[:, :n_prefix],  # Only read prefix gradients
            )
        ]

        return feature_hooks + error_hooks + token_hook

    @contextlib.contextmanager
    def install_hooks(self, model: "ReplacementModel"):
        """Context manager instruments hooks for forward and backward passes."""
        with model.hooks(
            fwd_hooks=self._caching_hooks(model.feature_input_hook),  # type: ignore
            bwd_hooks=self._make_prefix_only_attribution_hooks(model.feature_output_hook),  # type: ignore
        ):
            yield

    def compute_batch(
        self,
        layers: torch.Tensor,
        positions: torch.Tensor,
        inject_values: torch.Tensor,
        retain_graph: bool = True,
    ) -> torch.Tensor:
        """Return attribution rows for a batch of continuation token positions.

        Injects gradients at the specified continuation positions and computes
        attribution to prefix components only.

        Args:
            layers: 1-D tensor of layer indices (typically all n_layers for logit attribution)
            positions: 1-D tensor of token positions in the full sequence
            inject_values: (batch, d_model) tensor of unembedding vectors to inject

        Returns:
            torch.Tensor: (batch, n_prefix_sources) matrix of attribution scores
        """
        assert self._resid_activations[0] is not None, "Residual activations are not cached"
        batch_size = self._resid_activations[0].shape[0]

        self._batch_buffer = torch.zeros(
            self._row_size,
            batch_size,
            dtype=inject_values.dtype,
            device=inject_values.device,
        )

        batch_idx = torch.arange(len(layers), device=layers.device)

        def _inject(grads, *, batch_indices, pos_indices, values):
            grads_out = grads.clone().to(values.dtype)
            grads_out.index_put_((batch_indices, pos_indices), values)
            return grads_out.to(grads.dtype)

        handles = []
        layers_in_batch = layers.unique().tolist()

        for layer in layers_in_batch:
            mask = layers == layer
            if not mask.any():
                continue
            fn = partial(
                _inject,
                batch_indices=batch_idx[mask],
                pos_indices=positions[mask],
                values=inject_values[mask],
            )
            resid_activations = self._resid_activations[int(layer)]
            assert resid_activations is not None, "Residual activations are not cached"
            handles.append(resid_activations.register_hook(fn))

        try:
            last_layer = max(layers_in_batch)
            self._resid_activations[last_layer].backward(
                gradient=torch.zeros_like(self._resid_activations[last_layer]),
                retain_graph=retain_graph,
            )
        finally:
            for h in handles:
                h.remove()

        buf, self._batch_buffer = self._batch_buffer, None
        return buf.T[: len(layers)]
