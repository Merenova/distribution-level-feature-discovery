"""Configuration management for Gaussian optimization experiments."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


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
    n_samples: int = 1024  # N_x: number of continuations per prefix
    nucleus_p: float = 0.95  # Top-p for nucleus sampling
    temperature: float = 1.0  # Temperature for sampling
    max_tokens: int = 50  # Maximum tokens per continuation
    stop_tokens: list[str] = field(default_factory=lambda: [".", "?", "!"])  # Punctuation stopping
    batch_size: int = 32  # Batch size for vLLM sampling
    format_prefix: bool = True  # Whether to apply chat template to prefix


@dataclass
class EmbeddingConfig:
    """Configuration for semantic embeddings."""
    model_name: str = "google/embeddinggemma-300m"
    embed_full_sequence: bool = True  # Embed prefix + continuation
    batch_size: int = 32
    device: str = "cuda"


@dataclass
class AttributionConfig:
    """Configuration for attribution graph extraction."""
    use_circuit_tracer: bool = True
    batch_size: int = 16
    device: str = "cuda"


@dataclass
class ClusteringConfig:
    """Configuration for legacy (threshold-based) Gaussian clustering algorithm."""
    K_0: int = 3  # Initial number of components
    max_iterations: int = 50  # Maximum EM iterations
    convergence_threshold: float = 1e-4  # Convergence criterion

    # Thresholds for dynamic operations
    tau_clone: float = 2.0  # Reconstruction error threshold for cloning
    tau_split_a: float = 10.0  # Attribution variance threshold for splitting

    # Size constraints
    n_min: int = 8  # Minimum component size
    T_max_split: int = 2  # Maximum split attempts per component
    K_clone: int = 5  # Maximum clones per iteration

    # Probability weighting
    prob_temperature: float = 1.0  # Temperature for probability weights (β in P_n^β)


@dataclass
class RateDistortionConfig:
    """Configuration for rate-distortion based clustering.

    Uses principled information-theoretic criteria instead of hand-tuned thresholds.
    Objective: L_RD = H(C) + β_e D^(e) + β_a D^(a)

    Note: No tunable thresholds - all operations use exact R-D criteria.
    """
    # Clustering method
    method: str = "rate_distortion"

    # Rate-distortion tradeoff (equal weights by default)
    beta_e: float = 1.0  # Semantic distortion weight
    beta_a: float = 1.0  # Attribution distortion weight

    # Capacity constraints
    K_max: int = 20  # Maximum number of components

    # Convergence
    max_iterations: int = 50  # Maximum EM iterations
    convergence_threshold: float = 1e-6  # Relative change in L_RD


@dataclass
class AnnealingConfig:
    """Configuration for β-annealing.

    β-annealing runs clustering across a range of β values, using warm-start
    from previous solution. This traces the rate-distortion curve.

    Per tex Section 7: Run for β ∈ [β_min, β_max] (log-spaced, ascending),
    using previous solution as warm start.
    """
    enabled: bool = False  # Whether to run annealing
    beta_min: float = 0.1  # Starting β (low = favor compression, fewer clusters)
    beta_max: float = 10.0  # Ending β (high = favor reconstruction, more clusters)
    n_steps: int = 20  # Number of β values (log-spaced)
    gamma: float = 0.5  # View ratio: β_e = γβ, β_a = (1-γ)β


@dataclass
class PathConfig:
    """Path configuration for project directories."""
    # Derive project root from this file's location
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    # Parent projects (optional, only used for cross-project references)
    knowledge_attribution: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3] / "knowledge_attribution")
    qwen_pilot: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3] / "qwen_pilot")

    # Circuit-tracer (relative to project root)
    circuit_tracer: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1] / "circuit-tracer")

    # Source code directories (numbered stages)
    data_preparation: Optional[Path] = None
    attribution_graphs: Optional[Path] = None
    branch_sampling: Optional[Path] = None
    feature_extraction: Optional[Path] = None
    gaussian_clustering: Optional[Path] = None
    semantic_graphs: Optional[Path] = None

    # Results directory structure
    results: Optional[Path] = None
    results_attribution_graphs: Optional[Path] = None
    results_branch_sampling: Optional[Path] = None
    results_feature_extraction: Optional[Path] = None
    results_clustering: Optional[Path] = None
    results_semantic_graphs: Optional[Path] = None
    results_validation: Optional[Path] = None
    results_visualization: Optional[Path] = None

    # Validation subdirectories
    validation_7a: Optional[Path] = None
    validation_7b: Optional[Path] = None
    validation_7c: Optional[Path] = None

    # Support directories
    logs: Optional[Path] = None
    configs: Optional[Path] = None
    manifests: Optional[Path] = None

    # Legacy (deprecated)
    test_results: Optional[Path] = None

    def __post_init__(self):
        """Initialize relative paths."""
        # Source code directories
        self.data_preparation = self.project_root / "1_data_preparation"
        self.branch_sampling = self.project_root / "2_branch_sampling"
        self.attribution_graphs = self.project_root / "3_attribution_graphs"
        self.feature_extraction = self.project_root / "4_feature_extraction"
        self.gaussian_clustering = self.project_root / "5_gaussian_clustering"
        self.semantic_graphs = self.project_root / "6_semantic_graphs"

        # Results directory structure (numbered)
        self.results = self.project_root / "results"
        self.results_branch_sampling = self.results / "2_branch_sampling"
        self.results_attribution_graphs = self.results / "3_attribution_graphs"
        self.results_feature_extraction = self.results / "4_feature_extraction"
        self.results_clustering = self.results / "5_clustering"
        self.results_semantic_graphs = self.results / "6_semantic_graphs"
        self.results_validation = self.results / "7_validation"
        self.results_visualization = self.results / "8_visualization"

        # Validation subdirectories
        self.validation_7a = self.results_validation / "7a_graph_validation"
        self.validation_7b = self.results_validation / "7b_clustering_sweep"
        self.validation_7c = self.results_validation / "7c_steering"

        # Support directories
        self.logs = self.project_root / "logs"
        self.configs = self.results / "configs"
        self.manifests = self.results / "manifests"

        # Legacy (deprecated)
        self.test_results = self.project_root / "test_results"

    def ensure_dirs(self):
        """Create all directories if they don't exist."""
        for path in [
            self.results,
            self.results_attribution_graphs,
            self.results_branch_sampling,
            self.results_feature_extraction,
            self.results_clustering,
            self.results_semantic_graphs,
            self.results_validation,
            self.results_visualization,
            self.validation_7a,
            self.validation_7b,
            self.validation_7c,
            self.logs,
            self.configs,
            self.manifests,
        ]:
            if path:
                path.mkdir(parents=True, exist_ok=True)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    attribution: AttributionConfig = field(default_factory=AttributionConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    # Experiment metadata
    experiment_name: str = "gaussian_test"
    num_test_prefixes: int = 10
    random_seed: int = 42


def load_clustering_config_from_json(config_path: Path) -> ClusteringConfig:
    """Load legacy clustering configuration from JSON file.

    Args:
        config_path: Path to JSON config file

    Returns:
        ClusteringConfig with values from JSON
    """
    with open(config_path) as f:
        config = json.load(f)

    clustering_dict = config.get("clustering", {})

    return ClusteringConfig(
        K_0=clustering_dict.get("K_0", 3),
        max_iterations=clustering_dict.get("max_iterations", 50),
        convergence_threshold=clustering_dict.get("convergence_threshold", 1e-4),
        tau_clone=clustering_dict.get("tau_clone", 2.0),
        tau_split_a=clustering_dict.get("tau_split_a", 100.0),
        n_min=clustering_dict.get("n_min", 8),
        T_max_split=clustering_dict.get("T_max_split", 2),
        K_clone=clustering_dict.get("K_clone", 5),
        prob_temperature=clustering_dict.get("prob_temperature", 1.0),
    )


def load_rd_config_from_json(config_path: Path) -> RateDistortionConfig:
    """Load rate-distortion clustering configuration from JSON file.

    Args:
        config_path: Path to JSON config file

    Returns:
        RateDistortionConfig with values from JSON
    """
    with open(config_path) as f:
        config = json.load(f)

    clustering_dict = config.get("clustering", {})

    return RateDistortionConfig(
        method=clustering_dict.get("method", "rate_distortion"),
        beta_e=clustering_dict.get("beta_e", 1.0),
        beta_a=clustering_dict.get("beta_a", 1.0),
        K_max=clustering_dict.get("K_max", 20),
        max_iterations=clustering_dict.get("max_iterations", 50),
        convergence_threshold=clustering_dict.get("convergence_threshold", 1e-6),
    )


def load_annealing_config_from_json(config_path: Path) -> AnnealingConfig:
    """Load β-annealing configuration from JSON file.

    Args:
        config_path: Path to JSON config file

    Returns:
        AnnealingConfig with values from JSON
    """
    with open(config_path) as f:
        config = json.load(f)

    annealing_dict = config.get("annealing", {})

    return AnnealingConfig(
        enabled=annealing_dict.get("enabled", False),
        beta_min=annealing_dict.get("beta_min", 0.1),
        beta_max=annealing_dict.get("beta_max", 10.0),
        n_steps=annealing_dict.get("n_steps", 20),
        gamma=annealing_dict.get("gamma", 0.5),
    )


def get_clustering_method(config_path: Path) -> str:
    """Determine clustering method from config file.

    Args:
        config_path: Path to JSON config file

    Returns:
        Method name: "rate_distortion" or "legacy"
    """
    with open(config_path) as f:
        config = json.load(f)

    clustering_dict = config.get("clustering", {})
    return clustering_dict.get("method", "rate_distortion")
