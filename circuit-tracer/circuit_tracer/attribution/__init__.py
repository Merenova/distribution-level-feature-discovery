"""Attribution graph computation for circuit tracing."""

# Import context classes directly (no circular import issue)
from circuit_tracer.attribution.context import (
    PrefixAttributionContext,
    ContinuationAttributionContext,
)

# Lazy imports for attribute.py to avoid circular import with replacement_model.py
_lazy_imports = {
    "attribute": "circuit_tracer.attribution.attribute",
    "compute_salient_logits": "circuit_tracer.attribution.attribute",
    "attribute_prefix_to_continuations": "circuit_tracer.attribution.attribute",
    "setup_prefix_context": "circuit_tracer.attribution.attribute",
    "ContinuationTokenAttribution": "circuit_tracer.attribution.attribute",
    "PrefixContinuationResult": "circuit_tracer.attribution.attribute",
}


def __getattr__(name):
    if name in _lazy_imports:
        module_name = _lazy_imports[name]
        import importlib
        module = importlib.import_module(module_name)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "attribute",
    "compute_salient_logits",
    "attribute_prefix_to_continuations",
    "setup_prefix_context",
    "ContinuationTokenAttribution",
    "PrefixContinuationResult",
    "PrefixAttributionContext",
    "ContinuationAttributionContext",
]
