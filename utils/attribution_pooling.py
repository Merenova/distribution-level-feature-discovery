from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from utils.data_utils import load_json


VALID_POOLINGS = {"mean", "sum", "max"}


@dataclass(frozen=True)
class PooledAttributions:
    values: np.ndarray
    requested_pooling: str
    effective_pooling: str
    source: str
    context_file: Path | None = None
    meta_file: Path | None = None


def pool_context_attributions(
    context_data: Mapping[str, Any],
    attr_meta: Mapping[str, Any] | None = None,
    *,
    pooling: str = "mean",
    context_file: Path | None = None,
    meta_file: Path | None = None,
) -> PooledAttributions:
    requested_pooling = _validate_pooling(pooling)
    span_info = _get_metadata_field(context_data, attr_meta, "span_info")

    if context_data.get("store_all", False) and context_data.get("token_attributions") is not None:
        values = _pool_token_attributions(
            context_data["token_attributions"],
            span_info,
            requested_pooling,
        )
        return PooledAttributions(
            values=values,
            requested_pooling=requested_pooling,
            effective_pooling=requested_pooling,
            source="token_attributions",
            context_file=_optional_path(context_file),
            meta_file=_optional_path(meta_file),
        )

    aggregated_pooling = _aggregated_pooling(context_data, attr_meta)

    if requested_pooling == "max":
        raise ValueError("Pooling 'max' requires token_attributions")
    if requested_pooling == "sum" and aggregated_pooling == "mean":
        raise ValueError(
            "Cannot reconstruct sum from mean aggregated_attributions; "
            "pooling 'sum' requires token_attributions"
        )
    if aggregated_pooling not in {"mean", "sum"}:
        raise ValueError(
            "Unsupported aggregated_attributions_pooling "
            f"{aggregated_pooling!r}; expected 'mean' or 'sum'"
        )

    aggregated = _aggregated_attributions(context_data)
    if requested_pooling == "sum":
        return PooledAttributions(
            values=aggregated.astype(np.float32, copy=False),
            requested_pooling=requested_pooling,
            effective_pooling="sum",
            source="aggregated_attributions_sum",
            context_file=_optional_path(context_file),
            meta_file=_optional_path(meta_file),
        )

    if aggregated_pooling == "mean":
        values = aggregated
        source = "aggregated_attributions_mean"
    elif aggregated_pooling == "sum":
        lengths = _span_lengths(span_info, aggregated.shape[0])
        values = aggregated / lengths[:, np.newaxis]
        source = "aggregated_attributions_mean_from_sum"

    return PooledAttributions(
        values=values.astype(np.float32, copy=False),
        requested_pooling=requested_pooling,
        effective_pooling="mean",
        source=source,
        context_file=_optional_path(context_file),
        meta_file=_optional_path(meta_file),
    )


def load_pooled_attributions(
    context_file: Path,
    *,
    pooling: str = "mean",
    meta_file: Path | None = None,
) -> PooledAttributions:
    context_path = Path(context_file)
    metadata_path = Path(meta_file) if meta_file is not None else _infer_meta_file(context_path)

    context_data = torch.load(context_path, map_location="cpu", weights_only=False)
    attr_meta = None
    if meta_file is not None or metadata_path.exists():
        attr_meta = load_json(metadata_path)

    return pool_context_attributions(
        context_data,
        attr_meta,
        pooling=pooling,
        context_file=context_path,
        meta_file=metadata_path if attr_meta is not None else None,
    )


def _validate_pooling(pooling: str) -> str:
    if not isinstance(pooling, str):
        raise ValueError(f"Unknown attribution pooling {pooling!r}; expected one of {sorted(VALID_POOLINGS)}")

    normalized = pooling.lower()
    if normalized not in VALID_POOLINGS:
        raise ValueError(f"Unknown attribution pooling {pooling!r}; expected one of {sorted(VALID_POOLINGS)}")
    return normalized


def _get_metadata_field(
    context_data: Mapping[str, Any],
    attr_meta: Mapping[str, Any] | None,
    field_name: str,
) -> Any:
    if field_name in context_data:
        return context_data[field_name]
    if attr_meta is None:
        return None
    return attr_meta.get(field_name)


def _aggregated_pooling(
    context_data: Mapping[str, Any],
    attr_meta: Mapping[str, Any] | None,
) -> str:
    pooling = context_data.get("aggregated_attributions_pooling")
    if pooling is None and attr_meta is not None:
        pooling = attr_meta.get("aggregated_attributions_pooling")
    if pooling is None:
        return "sum"
    if not isinstance(pooling, str):
        raise ValueError("aggregated_attributions_pooling must be 'mean' or 'sum'")
    return pooling.lower()


def _aggregated_attributions(context_data: Mapping[str, Any]) -> np.ndarray:
    if "aggregated_attributions" not in context_data:
        raise ValueError("context_data is missing aggregated_attributions")

    values = _to_float_numpy(context_data["aggregated_attributions"])
    if values.ndim != 2:
        raise ValueError("aggregated_attributions must be a 2D tensor or array")
    return values


def _pool_token_attributions(
    token_attributions: Sequence[Any],
    span_info: Sequence[Mapping[str, Any]] | None,
    pooling: str,
) -> np.ndarray:
    if len(token_attributions) == 0:
        raise ValueError("token_attributions is empty")
    _validate_token_span_info(span_info, len(token_attributions))

    pooled = []
    for index, token_attr in enumerate(token_attributions):
        tensor = _to_float_tensor(token_attr)
        if tensor.ndim != 2:
            raise ValueError("token_attributions must contain 2D tensors")

        span = span_info[index]
        start, end = _span_bounds(span, tensor.shape[0], continuation_index=index)
        span_tensor = tensor[start:end]

        if span_tensor.shape[0] == 0:
            pooled_attr = torch.zeros(tensor.shape[1], dtype=torch.float32)
        elif pooling == "mean":
            pooled_attr = span_tensor.mean(dim=0)
        elif pooling == "sum":
            pooled_attr = span_tensor.sum(dim=0)
        elif pooling == "max":
            pooled_attr = span_tensor.max(dim=0).values
        else:
            raise ValueError(f"Unknown attribution pooling {pooling!r}")

        pooled.append(pooled_attr)

    return torch.stack(pooled).detach().cpu().numpy().astype(np.float32, copy=False)


def _validate_token_span_info(
    span_info: Sequence[Mapping[str, Any]] | None,
    expected_rows: int,
) -> None:
    if span_info is None:
        raise ValueError("span_info is required when pooling token_attributions")
    if len(span_info) < expected_rows:
        raise ValueError(
            "span_info must contain one entry for each token_attributions row"
        )


def _span_bounds(
    span: Mapping[str, Any] | None,
    default_length: int,
    *,
    continuation_index: int | None = None,
) -> tuple[int, int]:
    if span is None:
        raise ValueError(_span_error("span_info entry is missing", continuation_index))

    start = int(span.get("start", 0))
    if "end" in span:
        end = int(span["end"])
    elif "span_length" in span:
        end = start + int(span["span_length"])
    elif "continuation_length" in span:
        end = start + int(span["continuation_length"])
    else:
        raise ValueError(
            _span_error(
                "span_info entries must include end, span_length, or continuation_length",
                continuation_index,
            )
        )

    if start < 0 or end < 0 or end < start or start > default_length:
        raise ValueError(
            _span_error(
                f"malformed span_info bounds start={start}, end={end}, "
                f"token_length={default_length}",
                continuation_index,
            )
        )

    return start, min(end, default_length)


def _span_error(message: str, continuation_index: int | None) -> str:
    if continuation_index is None:
        return message
    return f"{message} for continuation {continuation_index}"


def _span_lengths(span_info: Any, expected_rows: int) -> np.ndarray:
    if expected_rows == 0:
        return np.asarray([], dtype=np.float32)
    if not span_info:
        raise ValueError("span_info is required to convert summed aggregated_attributions to mean")
    if len(span_info) < expected_rows:
        raise ValueError(
            "span_info must contain one entry for each aggregated_attributions row"
        )

    lengths = []
    for index in range(expected_rows):
        span = span_info[index]
        if "span_length" in span:
            length = int(span["span_length"])
        elif "end" in span and "start" in span:
            length = int(span["end"]) - int(span["start"])
        elif "continuation_length" in span:
            length = int(span["continuation_length"])
        else:
            raise ValueError(
                "span_info entries must include span_length, start/end, or continuation_length"
            )

        if length <= 0:
            raise ValueError("span_info entries must have positive span lengths")
        lengths.append(length)

    return np.asarray(lengths, dtype=np.float32)


def _to_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def _to_float_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float32).numpy()
    return np.asarray(value, dtype=np.float32)


def _infer_meta_file(context_file: Path) -> Path:
    suffix = "_prefix_context.pt"
    if not context_file.name.endswith(suffix):
        raise ValueError(
            f"Cannot infer attribution metadata path from {context_file}; "
            f"expected a filename ending in {suffix!r}"
        )
    return context_file.with_name(
        f"{context_file.name[:-len(suffix)]}_attribution.json"
    )


def _optional_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    return Path(path)
