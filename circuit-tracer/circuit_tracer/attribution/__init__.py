"""Attribution graph and prefix-to-continuation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from circuit_tracer.attribution.attribute import attribute
    from circuit_tracer.attribution.attribute_prefix import (
        attribute_prefix_to_continuations,
        setup_prefix_context,
    )
    from circuit_tracer.attribution.prefix_context import (
        ContinuationTokenAttribution,
        PrefixAttributionContext,
        PrefixContinuationResult,
    )

__all__ = [
    "ContinuationTokenAttribution",
    "PrefixAttributionContext",
    "PrefixContinuationResult",
    "attribute",
    "attribute_prefix_to_continuations",
    "setup_prefix_context",
]


def __getattr__(name):
    lazy_imports = {
        "attribute": ("circuit_tracer.attribution.attribute", "attribute"),
        "attribute_prefix_to_continuations": (
            "circuit_tracer.attribution.attribute_prefix",
            "attribute_prefix_to_continuations",
        ),
        "setup_prefix_context": (
            "circuit_tracer.attribution.attribute_prefix",
            "setup_prefix_context",
        ),
        "ContinuationTokenAttribution": (
            "circuit_tracer.attribution.prefix_context",
            "ContinuationTokenAttribution",
        ),
        "PrefixAttributionContext": (
            "circuit_tracer.attribution.prefix_context",
            "PrefixAttributionContext",
        ),
        "PrefixContinuationResult": (
            "circuit_tracer.attribution.prefix_context",
            "PrefixContinuationResult",
        ),
    }

    if name in lazy_imports:
        module_name, attr_name = lazy_imports[name]
        module = __import__(module_name, fromlist=[attr_name])
        return getattr(module, attr_name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
