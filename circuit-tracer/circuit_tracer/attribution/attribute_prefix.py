"""Unified prefix-to-continuation attribution dispatcher."""

from __future__ import annotations

import torch

from circuit_tracer.attribution.prefix_context import (
    PrefixAttributionContext,
    PrefixContinuationResult,
)


def setup_prefix_context(prefix_ids: torch.Tensor, model) -> PrefixAttributionContext:
    """Route prefix context setup to the model's backend."""

    if getattr(model, "backend", "transformerlens") == "nnsight":
        from circuit_tracer.attribution.attribute_prefix_nnsight import (
            setup_prefix_context as setup_prefix_context_nnsight,
        )

        return setup_prefix_context_nnsight(prefix_ids, model)

    from circuit_tracer.attribution.attribute_prefix_transformerlens import (
        setup_prefix_context as setup_prefix_context_transformerlens,
    )

    return setup_prefix_context_transformerlens(prefix_ids, model)


def attribute_prefix_to_continuations(
    prefix: torch.Tensor | list[int],
    continuations: list[torch.Tensor | list[int]],
    model,
    *,
    batch_size: int = 512,
    add_bos: bool = True,
    max_feature_nodes: int | None = None,
    verbose: bool = False,
) -> PrefixContinuationResult:
    """Route prefix-to-continuation attribution to the model's backend."""

    if getattr(model, "backend", "transformerlens") == "nnsight":
        from circuit_tracer.attribution.attribute_prefix_nnsight import (
            attribute_prefix_to_continuations as attribute_prefix_to_continuations_nnsight,
        )

        return attribute_prefix_to_continuations_nnsight(
            prefix=prefix,
            continuations=continuations,
            model=model,
            batch_size=batch_size,
            add_bos=add_bos,
            max_feature_nodes=max_feature_nodes,
            verbose=verbose,
        )

    from circuit_tracer.attribution.attribute_prefix_transformerlens import (
        attribute_prefix_to_continuations as attribute_prefix_to_continuations_transformerlens,
    )

    return attribute_prefix_to_continuations_transformerlens(
        prefix=prefix,
        continuations=continuations,
        model=model,
        batch_size=batch_size,
        add_bos=add_bos,
        max_feature_nodes=max_feature_nodes,
        verbose=verbose,
    )
