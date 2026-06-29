"""Configuration helpers for the paper-aligned pipeline."""

import argparse
import copy
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ConfigDict = dict[str, Any]


@dataclass
class ModelConfig:
    """Configuration for model and transcoder."""

    base_model: str = "Qwen/Qwen3-8B"
    transcoder: str = "mwhanna/qwen3-8b-transcoders"
    dtype: str = "bfloat16"
    device: str = "cuda"


@dataclass
class SamplingConfig:
    """Configuration for branch sampling."""

    nucleus_p: float = 0.9
    temperature: float = 1.0
    max_tokens: int = 20
    stop_tokens: list[str] = field(default_factory=lambda: [".", "?", "!"])
    batch_size: int = 32


@dataclass
class EmbeddingConfig:
    """Configuration for contextual continuation embeddings."""

    model_name: str = "google/embeddinggemma-300m"
    batch_size: int = 32
    device: str = "cuda"


@dataclass
class PathConfig:
    """Project-relative paths for the clean duplicate."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    circuit_tracer: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "circuit-tracer")
    results: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "results")
    logs: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "logs")

    def __post_init__(self) -> None:
        self.branch_sampling = self.project_root / "2_branch_sampling"
        self.attribution_graphs = self.project_root / "3_attribution_graphs"
        self.feature_extraction = self.project_root / "4_feature_extraction"
        self.gaussian_clustering = self.project_root / "5_gaussian_clustering"
        self.semantic_graphs = self.project_root / "6_semantic_graphs"
        self.validation = self.project_root / "7_validation"

        self.results_branch_sampling = self.results / "2_branch_sampling"
        self.results_attribution_graphs = self.results / "3_attribution_graphs"
        self.results_feature_extraction = self.results / "4_feature_extraction"
        self.results_clustering = self.results / "5_clustering"
        self.results_semantic_graphs = self.results / "6_semantic_graphs"
        self.results_validation = self.results / "7_validation"

    def ensure_dirs(self) -> None:
        for path in [
            self.results,
            self.logs,
            self.results_branch_sampling,
            self.results_attribution_graphs,
            self.results_feature_extraction,
            self.results_clustering,
            self.results_semantic_graphs,
            self.results_validation,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def deep_merge(base: ConfigDict, overlay: ConfigDict) -> ConfigDict:
    """Recursively merge overlay into base without mutating either input."""

    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path, _seen: set[Path] | None = None) -> ConfigDict:
    """Load a JSON config, recursively resolving relative ``extends`` entries."""

    config_path = Path(path).expanduser().resolve()
    seen = set() if _seen is None else set(_seen)
    if config_path in seen:
        cycle = " -> ".join(str(p) for p in [*seen, config_path])
        raise ValueError(f"Config extends cycle detected: {cycle}")
    seen.add(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        raw_config = json.load(file)
    if not isinstance(raw_config, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")

    extends = raw_config.get("extends", [])
    if isinstance(extends, str):
        extend_paths = [extends]
    elif isinstance(extends, list):
        extend_paths = extends
    else:
        raise ValueError(f"Config extends must be a string or list: {config_path}")

    resolved: ConfigDict = {}
    for extend_path in extend_paths:
        if not isinstance(extend_path, str):
            raise ValueError(f"Config extends entries must be strings: {config_path}")
        parent_config = load_config(config_path.parent / extend_path, seen)
        resolved = deep_merge(resolved, parent_config)

    overlay = {key: value for key, value in raw_config.items() if key != "extends"}
    return deep_merge(resolved, overlay)


def write_resolved_config(path: str | Path, output_path: str | Path) -> ConfigDict:
    """Resolve a config and write the fully expanded JSON representation."""

    resolved = load_config(path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def _get_nested(config: ConfigDict, *keys: str) -> Any:
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _shell_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def config_to_shell_exports(config: ConfigDict) -> dict[str, str]:
    """Convert resolved config values into runner environment variables."""

    mappings: dict[str, tuple[str, ...]] = {
        "EXPERIMENT_NAME": ("experiment_name",),
        "RANDOM_SEED": ("random_seed",),
        "DATASET_NAME": ("data", "dataset"),
        "DATA_ADAPTER": ("data", "adapter"),
        "MODEL_NAME": ("model", "base_model"),
        "TRANSCODER_NAME": ("model", "transcoder"),
        "MODEL_DTYPE": ("model", "dtype"),
        "GPU_MEMORY": ("vllm", "gpu_memory_utilization"),
        "TENSOR_PARALLEL": ("vllm", "tensor_parallel_size"),
        "MAX_MODEL_LEN": ("vllm", "max_model_len"),
        "TRUST_REMOTE_CODE": ("vllm", "trust_remote_code"),
        "MAX_TOTAL_CONTINUATIONS": ("sampling", "max_total_continuations"),
        "NUCLEUS_P": ("sampling", "nucleus_p"),
        "TEMPERATURE": ("sampling", "temperature"),
        "SAMPLING_MAX_TOKENS": ("sampling", "max_tokens"),
        "SAMPLING_BATCH_SIZE": ("sampling", "batch_size"),
        "SAMPLING_MAX_BATCHES": ("sampling", "max_batches"),
        "ATTR_MAX_FEATURES": ("attribution", "max_feature_nodes"),
        "ATTR_BATCH_SIZE": ("attribution", "batch_size"),
        "EMBEDDING_MODEL": ("embedding", "model_name"),
        "EMBEDDING_BATCH_SIZE": ("embedding", "batch_size"),
        "DATA_SPLIT_DEFAULT": ("data", "split"),
        "STAGE1_N_GROUPS": ("stage_1_data_prep", "n_groups"),
    }

    exports: dict[str, str] = {}
    for export_name, keys in mappings.items():
        value = _get_nested(config, *keys)
        if value is not None:
            exports[export_name] = _shell_value(value)
    return exports


def print_shell_exports(config: ConfigDict) -> None:
    """Print shell export statements for the resolved config."""

    for key, value in config_to_shell_exports(config).items():
        print(f"export {key}={shlex.quote(value)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resolve composable JSON configs.")
    parser.add_argument("config", type=Path, help="Config JSON file to resolve")
    parser.add_argument("--shell-env", action="store_true", help="Print shell exports")
    parser.add_argument("--json", action="store_true", help="Print resolved JSON")
    parser.add_argument("--write", type=Path, help="Write resolved JSON to this path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)

    if args.write:
        write_resolved_config(args.config, args.write)
    if args.shell_env:
        print_shell_exports(config)
    if args.json or not args.shell_env:
        print(json.dumps(config, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
