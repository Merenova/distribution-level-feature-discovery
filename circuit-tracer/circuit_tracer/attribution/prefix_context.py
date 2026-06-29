"""Shared prefix-to-continuation attribution data structures and TL hooks."""

import contextlib
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch
from einops import einsum
from transformer_lens.hook_points import HookPoint

if TYPE_CHECKING:
    from circuit_tracer.replacement_model.replacement_model_transformerlens import (
        TransformerLensReplacementModel,
    )


@dataclass
class PrefixAttributionContext:
    """Cached prefix components reused across continuation attributions."""

    prefix_tokens: torch.Tensor
    activation_matrix: torch.Tensor
    error_vectors: torch.Tensor
    token_vectors: torch.Tensor
    decoder_vecs: torch.Tensor
    encoder_vecs: torch.Tensor
    encoder_to_decoder_map: torch.Tensor
    decoder_locations: torch.Tensor
    n_layers: int
    selected_features: torch.Tensor | None = None
    total_active_features: int | None = None

    @property
    def prefix_length(self) -> int:
        return len(self.prefix_tokens)

    @property
    def n_prefix_features(self) -> int:
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
        return self.n_prefix_features + self.n_prefix_errors + self.n_prefix_tokens

    def get_feature_info(self, attr_idx: int) -> tuple[int, int, int, float] | None:
        """Return feature metadata for a feature-source attribution index."""

        if attr_idx >= self.n_prefix_features:
            return None

        all_layers, all_positions, all_feature_idxs = self.activation_matrix.indices()
        all_values = self.activation_matrix.values()

        if self.selected_features is not None:
            original_idx = self.selected_features[attr_idx].item()
        else:
            original_idx = attr_idx

        return (
            all_layers[original_idx].item(),
            all_positions[original_idx].item(),
            all_feature_idxs[original_idx].item(),
            all_values[original_idx].item(),
        )

    def attribution_to_interventions(
        self,
        attr_indices: list[int] | torch.Tensor,
        values: list[float] | torch.Tensor | None = None,
        scale: float = 1.0,
    ) -> list[tuple[int, int, int, float]]:
        """Convert feature-source indices to feature intervention tuples."""

        if isinstance(attr_indices, torch.Tensor):
            attr_indices = attr_indices.tolist()
        if isinstance(values, torch.Tensor):
            values = values.tolist()

        interventions = []
        for i, attr_idx in enumerate(attr_indices):
            info = self.get_feature_info(attr_idx)
            if info is None:
                continue

            layer, position, feature_idx, orig_value = info
            value = values[i] if values is not None else orig_value * scale
            interventions.append((layer, position, feature_idx, value))

        return interventions

    def get_top_features(
        self,
        source_attribution: torch.Tensor,
        k: int = 10,
        by_abs: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return top-k feature-source indices and scores."""

        feature_scores = source_attribution[: self.n_prefix_features]
        if by_abs:
            sorted_indices = torch.argsort(feature_scores.abs(), descending=True)
        else:
            sorted_indices = torch.argsort(feature_scores, descending=True)

        top_indices = sorted_indices[: min(k, len(sorted_indices))]
        return top_indices, feature_scores[top_indices]


@dataclass
class ContinuationTokenAttribution:
    """Attribution scores for one continuation token."""

    token_id: int
    position: int
    source_attribution: torch.Tensor


@dataclass
class PrefixContinuationResult:
    """Prefix-to-continuation attribution result."""

    prefix_tokens: torch.Tensor
    prefix_context: PrefixAttributionContext
    continuation_attributions: list[list[ContinuationTokenAttribution]]


class ContinuationAttributionContext:
    """TransformerLens context that attributes continuation logits to prefix sources."""

    def __init__(
        self,
        prefix_ctx: PrefixAttributionContext,
        full_sequence_length: int,
    ) -> None:
        self.prefix_ctx = prefix_ctx
        self.full_length = full_sequence_length
        self.n_layers = prefix_ctx.n_layers
        self._resid_activations: list[torch.Tensor | None] = [None] * (self.n_layers + 1)
        self._batch_buffer: torch.Tensor | None = None
        self._row_size = prefix_ctx.n_prefix_sources

    def _caching_hooks(self, feature_input_hook: str) -> list[tuple[str, Callable]]:
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
        n_prefix = self.prefix_ctx.prefix_length
        nnz_layers, nnz_positions = self.prefix_ctx.decoder_locations

        feature_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{feature_output_hook}",
                self.prefix_ctx.decoder_vecs[layer_mask],
                write_index=self.prefix_ctx.encoder_to_decoder_map[layer_mask],  # type: ignore
                read_index=np.s_[:, nnz_positions[layer_mask]],  # type: ignore
            )
            for layer in range(self.n_layers)
            if (layer_mask := nnz_layers == layer).any()
        ]

        def error_offset(layer: int) -> int:
            return self.prefix_ctx.n_prefix_features + layer * n_prefix

        error_hooks = [
            self._compute_score_hook(
                f"blocks.{layer}.{feature_output_hook}",
                self.prefix_ctx.error_vectors[layer],
                write_index=np.s_[error_offset(layer) : error_offset(layer + 1)],
                read_index=np.s_[:, :n_prefix],
            )
            for layer in range(self.n_layers)
        ]

        tok_start = error_offset(self.n_layers)
        token_hook = [
            self._compute_score_hook(
                "hook_embed",
                self.prefix_ctx.token_vectors,
                write_index=np.s_[tok_start : tok_start + n_prefix],
                read_index=np.s_[:, :n_prefix],
            )
        ]

        return feature_hooks + error_hooks + token_hook

    @contextlib.contextmanager
    def install_hooks(self, model: "TransformerLensReplacementModel"):
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
            resid_activations = self._resid_activations[int(layer)]
            assert resid_activations is not None, "Residual activations are not cached"
            handles.append(
                resid_activations.register_hook(
                    partial(
                        _inject,
                        batch_indices=batch_idx[mask],
                        pos_indices=positions[mask],
                        values=inject_values[mask],
                    )
                )
            )

        try:
            last_layer = max(layers_in_batch)
            last_acts = self._resid_activations[last_layer]
            assert last_acts is not None, "Residual activations are not cached"
            last_acts.backward(
                gradient=torch.zeros_like(last_acts),
                retain_graph=retain_graph,
            )
        finally:
            for handle in handles:
                handle.remove()

        buf, self._batch_buffer = self._batch_buffer, None
        return buf.T[: len(layers)]
