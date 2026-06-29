"""Gemma-3 transcoder config synthesis for GemmaScope2-style repos."""

from __future__ import annotations

import re
from typing import Protocol

from huggingface_hub import list_repo_files

_LAYER_RE = re.compile(r"layer_(\d+)\.safetensors$")


class HfUriLike(Protocol):
    repo_id: str
    file_path: str | None
    revision: str | None


def build_gemma3_transcoder_config(uri: HfUriLike) -> dict | None:
    """Build a circuit-tracer transcoder config for repo/subfolder refs without config.yaml."""

    if "gemma-scope-2" not in uri.repo_id.lower():
        return None
    if not uri.file_path:
        return None

    prefix = uri.file_path.strip("/")
    files = list_repo_files(uri.repo_id, revision=uri.revision)

    indexed_paths: list[tuple[int, str]] = []
    for file_path in files:
        if not file_path.startswith(prefix + "/"):
            continue
        match = _LAYER_RE.search(file_path)
        if match is None:
            continue
        indexed_paths.append((int(match.group(1)), file_path))

    if not indexed_paths:
        return None

    indexed_paths.sort(key=lambda item: item[0])
    expected_layers = list(range(indexed_paths[-1][0] + 1))
    actual_layers = [layer for layer, _ in indexed_paths]
    if actual_layers != expected_layers:
        raise ValueError(
            f"GemmaScope2 transcoder layers are not contiguous for {uri.repo_id}/{prefix}: "
            f"expected {expected_layers}, got {actual_layers}"
        )

    revision_query = f"?revision={uri.revision}" if uri.revision else ""
    transcoders = [
        f"hf://{uri.repo_id}/{file_path}{revision_query}" for _, file_path in indexed_paths
    ]
    scan = f"{uri.repo_id}/{prefix}"
    if uri.revision:
        scan = f"{scan}@{uri.revision}"

    return {
        "repo_id": uri.repo_id,
        "revision": uri.revision,
        "subfolder": prefix,
        "scan": scan,
        "model_kind": "transcoder_set",
        "transcoders": transcoders,
        "feature_input_hook": "ln2.hook_normalized",
        "feature_output_hook": "hook_mlp_out",
    }
