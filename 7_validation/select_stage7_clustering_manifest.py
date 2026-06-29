#!/usr/bin/env -S uv run python
"""Select top-k Stage 5 clustering configs for Stage 7 validation."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple


PREFIX_KEYS = ("prefix_id", "cloze_id", "sample_id")
NESTED_SCORE_CONTAINERS = ("scores", "metrics", "validation", "summary")
LIST_CONTAINERS = (
    "grid",
    "results",
    "entries",
    "manifest",
    "selected",
    "selections",
    "prefixes",
    "configs",
    "rows",
)


def _coerce_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _coerce_int(value: Any) -> Optional[int]:
    numeric = _coerce_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _path_get(mapping: Dict[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _score_value(entry: Dict[str, Any], score_key: str) -> Optional[float]:
    value = _path_get(entry, score_key) if "." in score_key else entry.get(score_key)
    score = _coerce_float(value)
    if score is not None:
        return score

    for container_key in NESTED_SCORE_CONTAINERS:
        container = entry.get(container_key)
        if not isinstance(container, dict):
            continue
        value = _path_get(container, score_key) if "." in score_key else container.get(score_key)
        score = _coerce_float(value)
        if score is not None:
            return score
    return None


def _prefix_from(mapping: Dict[str, Any], fallback: Optional[str] = None) -> Optional[str]:
    for key in PREFIX_KEYS:
        value = mapping.get(key)
        if value is not None:
            return str(value)
    return fallback


def _prefix_from_path(path: Path) -> Optional[str]:
    stem = path.stem
    if stem.endswith("_sweep_results"):
        return stem[: -len("_sweep_results")]
    return None


def _looks_like_candidate(mapping: Dict[str, Any]) -> bool:
    return "beta" in mapping and "gamma" in mapping


def _infer_k(entry: Dict[str, Any]) -> Optional[int]:
    for key in ("K", "k", "n_components"):
        if key in entry:
            return _coerce_int(entry[key])

    components = entry.get("components")
    if isinstance(components, dict):
        return len(components)
    if isinstance(components, list):
        return len(components)

    assignments = entry.get("assignments")
    if isinstance(assignments, list):
        return len(set(assignments))

    return None


def _iter_candidates(
    payload: Any,
    source_path: Path,
    prefix_hint: Optional[str] = None,
    source_ref: str = "$",
) -> Iterator[Tuple[Dict[str, Any], str, str]]:
    if isinstance(payload, list):
        for index, item in enumerate(payload):
            yield from _iter_candidates(
                item,
                source_path,
                prefix_hint=prefix_hint,
                source_ref=f"{source_ref}[{index}]",
            )
        return

    if not isinstance(payload, dict):
        return

    prefix_id = _prefix_from(payload, prefix_hint)
    if _looks_like_candidate(payload):
        if prefix_id is not None:
            yield payload, prefix_id, source_ref
        return

    for container_key in LIST_CONTAINERS:
        container = payload.get(container_key)
        if isinstance(container, list):
            for index, item in enumerate(container):
                item_prefix = prefix_id
                if isinstance(item, dict):
                    item_prefix = _prefix_from(item, item_prefix)
                yield from _iter_candidates(
                    item,
                    source_path,
                    prefix_hint=item_prefix,
                    source_ref=f"{source_ref}.{container_key}[{index}]",
                )
        elif isinstance(container, dict):
            for key, item in sorted(container.items(), key=lambda pair: str(pair[0])):
                item_prefix = str(key) if prefix_id is None else prefix_id
                if isinstance(item, dict):
                    item_prefix = _prefix_from(item, item_prefix)
                yield from _iter_candidates(
                    item,
                    source_path,
                    prefix_hint=item_prefix,
                    source_ref=f"{source_ref}.{container_key}.{key}",
                )

    for key, item in sorted(payload.items(), key=lambda pair: str(pair[0])):
        if key in LIST_CONTAINERS or key in NESTED_SCORE_CONTAINERS:
            continue
        if not isinstance(item, (dict, list)):
            continue
        if not (str(key).startswith("cloze_") or str(key).startswith("prefix_")):
            continue
        item_prefix = str(key)
        if isinstance(item, dict):
            item_prefix = _prefix_from(item, item_prefix)
        yield from _iter_candidates(
            item,
            source_path,
            prefix_hint=item_prefix,
            source_ref=f"{source_ref}.{key}",
        )


def _json_paths(stage5_dir: Path) -> List[Path]:
    if stage5_dir.is_file():
        return [stage5_dir] if stage5_dir.suffix == ".json" else []
    return sorted(path for path in stage5_dir.glob("*.json") if path.is_file())


def _load_json(path: Path) -> Any:
    with path.open("r") as handle:
        return json.load(handle)


def _normalise_candidate(
    entry: Dict[str, Any],
    prefix_id: str,
    source_path: Path,
    source_ref: str,
    score_key: str,
    min_k: int,
    max_k: Optional[int],
) -> Optional[Dict[str, Any]]:
    if entry.get("error"):
        return None

    beta = _coerce_float(entry.get("beta"))
    gamma = _coerce_float(entry.get("gamma"))
    score = _score_value(entry, score_key)
    K = _infer_k(entry)

    if beta is None or gamma is None or score is None or K is None:
        return None
    if K < min_k:
        return None
    if max_k is not None and K > max_k:
        return None

    cloze_id = str(entry.get("cloze_id", prefix_id))
    return {
        "prefix_id": str(prefix_id),
        "cloze_id": cloze_id,
        "beta": beta,
        "gamma": gamma,
        "K": K,
        "score": score,
        "score_key": score_key,
        "source_path": str(source_path),
        "source_ref": source_ref,
    }


def _candidate_sort_key(candidate: Dict[str, Any], score_order: str) -> Tuple[Any, ...]:
    score = candidate["score"]
    score_key = -score if score_order == "desc" else score
    return (
        score_key,
        candidate["beta"],
        candidate["gamma"],
        candidate["K"],
        candidate["source_path"],
        candidate["prefix_id"],
        candidate["source_ref"],
    )


def _select_for_prefix(
    candidates: Iterable[Dict[str, Any]],
    top_k: int,
    score_order: str,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen_configs = set()

    for candidate in sorted(candidates, key=lambda item: _candidate_sort_key(item, score_order)):
        config_key = (candidate["beta"], candidate["gamma"], candidate["K"])
        if config_key in seen_configs:
            continue
        seen_configs.add(config_key)
        selected.append(candidate)
        if len(selected) >= top_k:
            break

    return selected


def build_manifest(
    stage5_dir: Path,
    top_k: int = 1,
    score_key: str = "harmonic",
    score_order: str = "desc",
    min_k: int = 2,
    max_k: Optional[int] = None,
    max_prefixes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build a compact Stage 7 manifest from Stage 5 clustering sweep JSONs."""
    stage5_dir = Path(stage5_dir)
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if score_order not in {"asc", "desc"}:
        raise ValueError("score_order must be 'asc' or 'desc'")
    if max_k is not None and max_k < min_k:
        raise ValueError("max_k must be >= min_k")
    if max_prefixes is not None and max_prefixes < 1:
        raise ValueError("max_prefixes must be >= 1")

    by_prefix: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    json_paths = _json_paths(stage5_dir)
    if max_prefixes is not None:
        json_paths = json_paths[:max_prefixes]

    for source_path in json_paths:
        try:
            payload = _load_json(source_path)
        except (OSError, json.JSONDecodeError):
            continue

        prefix_hint = _prefix_from_path(source_path)
        for entry, prefix_id, source_ref in _iter_candidates(
            payload,
            source_path,
            prefix_hint=prefix_hint,
        ):
            candidate = _normalise_candidate(
                entry,
                prefix_id,
                source_path,
                source_ref,
                score_key,
                min_k,
                max_k,
            )
            if candidate is not None:
                by_prefix[candidate["prefix_id"]].append(candidate)

    manifest: List[Dict[str, Any]] = []
    for prefix_id in sorted(by_prefix):
        manifest.extend(_select_for_prefix(by_prefix[prefix_id], top_k, score_order))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select top-k Stage 5 clustering configs for Stage 7 validation"
    )
    parser.add_argument("--stage5-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--score-key", default="harmonic")
    parser.add_argument("--score-order", choices=["asc", "desc"], default="desc")
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=None)
    parser.add_argument(
        "--max-prefixes",
        type=int,
        default=None,
        help="Only scan the first N sorted Stage 5 per-prefix JSON files",
    )
    args = parser.parse_args()

    manifest = build_manifest(
        args.stage5_dir,
        top_k=args.top_k,
        score_key=args.score_key,
        score_order=args.score_order,
        min_k=args.min_k,
        max_k=args.max_k,
        max_prefixes=args.max_prefixes,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    counts_by_prefix: Dict[str, int] = defaultdict(int)
    for entry in manifest:
        counts_by_prefix[entry["prefix_id"]] += 1

    print(
        json.dumps(
            {
                "output": str(args.output),
                "n_prefixes": len(counts_by_prefix),
                "n_entries": len(manifest),
                "top_k": args.top_k,
                "score_key": args.score_key,
                "score_order": args.score_order,
                "min_k": args.min_k,
                "max_k": args.max_k,
                "max_prefixes": args.max_prefixes,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
