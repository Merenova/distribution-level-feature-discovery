import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE7 = ROOT / "7_validation"
ARCHIVE = ROOT / "archive" / "7_validation_stage7_legacy"


def read_stage7(name: str) -> str:
    return (STAGE7 / name).read_text()


def parse_stage7(name: str) -> ast.Module:
    return ast.parse(read_stage7(name), filename=name)


def function_names(name: str) -> set[str]:
    tree = parse_stage7(name)
    return {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def imported_names(name: str) -> set[str]:
    tree = parse_stage7(name)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def referenced_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def test_dead_metrics_helpers_removed_from_core_stage7():
    functions = function_names("7c_metrics.py")
    forbidden = [
        "compute_mean_target_logit_and_prob_batched",
        "compute_h_c_norm",
    ]
    for symbol in forbidden:
        assert symbol not in functions

    required = [
        "compute_centered_logit_metrics",
        "compute_per_token_logit_values_batched",
        "compute_cluster_mass_metrics",
    ]
    for symbol in required:
        assert symbol in functions


def test_dead_hypotheses_helpers_removed_from_core_stage7():
    functions = function_names("7c_hypotheses.py")
    imports = imported_names("7c_hypotheses.py")
    forbidden = [
        "load_sweep_config",
        "validate_h4a_dose_response",
    ]
    for symbol in forbidden:
        assert symbol not in functions

    forbidden_imports = [
        "linregress",
    ]
    for symbol in forbidden_imports:
        assert symbol not in imports

    required = [
        "_run_steering_evaluation",
        "validate_h4a_generation",
        "validate_h4c_specificity",
        "validate_h4c_cluster_mass_pairwise",
    ]
    for symbol in required:
        assert symbol in functions


def test_dead_utils_helpers_removed_from_core_stage7():
    assert "compute_all_correlations" not in function_names("7c_utils.py")


def test_dead_graph_helpers_removed_from_core_stage7():
    functions = function_names("7c_graph.py")
    forbidden = [
        "select_top_features_by_distinctiveness",
        "select_top_features_from_Hc",
        "precompute_decoder_vectors_global",
    ]
    for symbol in forbidden:
        assert symbol not in functions

    required = [
        "build_semantic_graphs_from_clustering",
        "select_top_features_by_magnitude",
        "precompute_cluster_decoder_vectors",
    ]
    for symbol in required:
        assert symbol in functions


def test_removed_helpers_have_no_active_ast_references():
    removed_symbols = {
        "compute_mean_target_logit_and_prob_batched",
        "compute_h_c_norm",
        "load_sweep_config",
        "validate_h4a_dose_response",
        "compute_all_correlations",
        "select_top_features_by_distinctiveness",
        "select_top_features_from_Hc",
        "precompute_decoder_vectors_global",
        "linregress",
    }
    offenders = {}
    for path in STAGE7.glob("*.py"):
        hits = referenced_names(path) & removed_symbols
        if hits:
            offenders[path.name] = sorted(hits)

    assert offenders == {}


def test_legacy_stage7_modules_are_not_active_entrypoints():
    active_files = {path.name for path in STAGE7.glob("7c*.py")}

    assert "7c_baseline_combined_medoid.py" in active_files
    assert "7c_baseline_single.py" in active_files
    assert "7c_baseline_kmeans.py" in active_files

    assert "7c_baseline_medoid.py" not in active_files
    assert "7c_contrastive_hc.py" not in active_files

    assert (ARCHIVE / "7c_baseline_medoid.py").exists()
    assert (ARCHIVE / "7c_contrastive_hc.py").exists()


def test_standalone_stage7_utilities_remain_active():
    required_utilities = [
        "7c_cluster_analysis.py",
        "extract_tokenwise_logit_diff.py",
        "select_h4c_manifest.py",
    ]
    for name in required_utilities:
        assert (STAGE7 / name).exists()
