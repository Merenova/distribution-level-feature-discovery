#!/usr/bin/env -S uv run python
"""Compare RD clustering against pairwise co-regularized spectral clustering.

For a fixed beta:
1. Load the RD sweep rows across gamma for each prefix.
2. Run pairwise co-regularized two-view spectral clustering across a lambda grid
   and a K grid.
3. Match each RD row to the spectral candidate with closest rate H(C).
4. Write per-prefix matched rows plus aggregate CSV/LaTeX summaries.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
GAUSSIAN_DIR = REPO_ROOT / "5_gaussian_clustering"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GAUSSIAN_DIR))


def _import_module(name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_cluster = _import_module("rd_cluster", GAUSSIAN_DIR / "cluster.py")
_em_loop = _import_module("rd_em_loop", GAUSSIAN_DIR / "em_loop.py")
_rd_objective = _import_module("rd_objective", GAUSSIAN_DIR / "rd_objective.py")

load_prefix_data = _cluster.load_prefix_data
weighted_median = _em_loop.weighted_median
compute_full_rd_statistics = _rd_objective.compute_full_rd_statistics


DEFAULT_LAMBDAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]


@dataclass(frozen=True)
class GridEntry:
    beta: float
    gamma: float
    K: int
    H: float
    D_e: float
    D_a: float


@dataclass(frozen=True)
class WorkerConfig:
    sweep_file: str
    results_dir: str
    beta: float
    gamma_values: Optional[Tuple[float, ...]]
    pooling: str
    lambda_values: Tuple[float, ...]
    k_values: Optional[Tuple[int, ...]]
    k_max: Optional[int]
    spectral_max_iter: int
    spectral_tol: float
    kmeans_seed: int
    kmeans_n_init: int
    kmeans_max_iter: int


class _SilentLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def getEffectiveLevel(self):
        return 30


def _try_import_ijson():
    try:
        import ijson  # type: ignore
    except Exception:
        return None
    return ijson


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def _iter_grid_entries(path: Path) -> Iterable[GridEntry]:
    ijson = _try_import_ijson()
    if ijson is None:
        data = json.loads(path.read_text())
        for entry in data.get("grid", []) or []:
            try:
                yield GridEntry(
                    beta=float(entry.get("beta")),
                    gamma=float(entry.get("gamma")),
                    K=int(entry.get("K", len(entry.get("components", {})) or 0)),
                    H=float(entry.get("H")),
                    D_e=float(entry.get("D_e")),
                    D_a=float(entry.get("D_a")),
                )
            except Exception:
                continue
        return

    with path.open("rb") as f:
        for entry in ijson.items(f, "grid.item"):
            try:
                yield GridEntry(
                    beta=float(entry.get("beta")),
                    gamma=float(entry.get("gamma")),
                    K=int(entry.get("K", len(entry.get("components", {})) or 0)),
                    H=float(entry.get("H")),
                    D_e=float(entry.get("D_e")),
                    D_a=float(entry.get("D_a")),
                )
            except Exception:
                continue


def _load_metric_a(path: Path) -> str:
    ijson = _try_import_ijson()
    if ijson is None:
        data = json.loads(path.read_text())
        return str(data.get("sweep_config", {}).get("metric_a", "l2"))

    with path.open("rb") as f:
        for prefix, event, value in ijson.parse(f):
            if prefix == "sweep_config.metric_a" and event in ("string", "number"):
                return str(value)
            if prefix == "grid" and event == "start_array":
                break
    return "l2"


def _parse_float_list(raw: Optional[str]) -> Optional[Tuple[float, ...]]:
    if raw is None:
        return None
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    return tuple(values)


def _parse_int_list(raw: Optional[str]) -> Optional[Tuple[int, ...]]:
    if raw is None:
        return None
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    return tuple(values)


def _beta_tag(value: float) -> str:
    return str(value).replace(".", "p")


def _weighted_mean(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = float(weights.sum())
    if total <= 0:
        return np.zeros(data.shape[1], dtype=np.float64)
    return np.sum(weights[:, None] * data, axis=0) / total


def _row_normalize(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.where(norms > eps, norms, 1.0)
    return data / norms


def _robust_sigma(distances: np.ndarray) -> float:
    if distances.shape[0] <= 1:
        return 1.0
    tri = distances[np.triu_indices(distances.shape[0], k=1)]
    positive = tri[tri > 0]
    if positive.size == 0:
        return 1.0
    sigma = float(np.median(positive))
    if not math.isfinite(sigma) or sigma <= 0:
        return 1.0
    return sigma


def _build_normalized_affinity(data: np.ndarray, metric: str) -> Tuple[np.ndarray, float]:
    if metric == "l2":
        distances = pairwise_distances(data, metric="euclidean", n_jobs=1)
        sigma = _robust_sigma(distances)
        affinity = np.exp(-(distances ** 2) / (2.0 * sigma * sigma))
    elif metric == "l1":
        distances = pairwise_distances(data, metric="manhattan", n_jobs=1)
        sigma = _robust_sigma(distances)
        affinity = np.exp(-distances / sigma)
    else:
        raise ValueError(f"Unsupported affinity metric: {metric}")

    np.fill_diagonal(affinity, 1.0)
    affinity = 0.5 * (affinity + affinity.T)

    degrees = affinity.sum(axis=1)
    inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
    normalized = inv_sqrt[:, None] * affinity * inv_sqrt[None, :]
    normalized = 0.5 * (normalized + normalized.T)
    return normalized.astype(np.float64), float(sigma)


def _top_k_eigenvectors(matrix: np.ndarray, k: int) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    n = matrix.shape[0]
    if k <= 0 or k > n:
        raise ValueError(f"Invalid k={k} for matrix with n={n}")

    if k == n:
        eigenvalues, eigenvectors = eigh(matrix)
    else:
        eigenvalues, eigenvectors = eigh(matrix, subset_by_index=[n - k, n - 1])

    order = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, order[:k]]


def _pairwise_objective(
    affinity_e: np.ndarray,
    affinity_a: np.ndarray,
    U_e: np.ndarray,
    U_a: np.ndarray,
    lambda_value: float,
) -> float:
    align = np.linalg.norm(U_e.T @ U_a, ord="fro") ** 2
    return float(
        np.trace(U_e.T @ affinity_e @ U_e)
        + np.trace(U_a.T @ affinity_a @ U_a)
        + float(lambda_value) * align
    )


def _run_pairwise_coreg(
    affinity_e: np.ndarray,
    affinity_a: np.ndarray,
    k: int,
    lambda_value: float,
    max_iter: int,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray, int, bool, float]:
    U_e = _top_k_eigenvectors(affinity_e, k)
    U_a = _top_k_eigenvectors(affinity_a, k)
    prev_obj: Optional[float] = None

    for iteration in range(1, max_iter + 1):
        U_e = _top_k_eigenvectors(affinity_e + float(lambda_value) * (U_a @ U_a.T), k)
        U_a = _top_k_eigenvectors(affinity_a + float(lambda_value) * (U_e @ U_e.T), k)
        obj = _pairwise_objective(affinity_e, affinity_a, U_e, U_a, lambda_value)

        if prev_obj is not None:
            scale = max(1.0, abs(prev_obj))
            if abs(obj - prev_obj) <= tol * scale:
                return U_e, U_a, iteration, True, obj
        prev_obj = obj

    if prev_obj is None:
        prev_obj = _pairwise_objective(affinity_e, affinity_a, U_e, U_a, lambda_value)
    return U_e, U_a, max_iter, False, prev_obj


def _cluster_embedding(
    U_e: np.ndarray,
    U_a: np.ndarray,
    requested_k: int,
    seed: int,
    n_init: int,
    max_iter: int,
) -> Tuple[np.ndarray, int]:
    if requested_k <= 1:
        labels = np.zeros(U_e.shape[0], dtype=np.int32)
        return labels, 1

    embedding = np.concatenate([U_e, U_a], axis=1)
    embedding = _row_normalize(embedding)
    kmeans = KMeans(
        n_clusters=int(requested_k),
        init="k-means++",
        n_init=int(n_init),
        max_iter=int(max_iter),
        random_state=int(seed),
    )
    labels = kmeans.fit_predict(embedding)
    actual_k = int(len(np.unique(labels)))
    return labels.astype(np.int32), actual_k


def _build_components_from_assignments(
    assignments: np.ndarray,
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    path_probs: np.ndarray,
    metric_a: str,
) -> Dict[int, Dict[str, np.ndarray]]:
    components: Dict[int, Dict[str, np.ndarray]] = {}
    for component_id in sorted(int(x) for x in np.unique(assignments)):
        mask = assignments == component_id
        if not np.any(mask):
            continue
        mu_e = _weighted_mean(embeddings_e[mask], path_probs[mask])
        if metric_a == "l1":
            mu_a = weighted_median(attributions_a[mask], path_probs[mask])
        else:
            mu_a = _weighted_mean(attributions_a[mask], path_probs[mask])
        components[int(component_id)] = {
            "mu_e": mu_e,
            "mu_a": mu_a,
        }
    return components


def _compute_assignment_stats(
    assignments: np.ndarray,
    embeddings_e: np.ndarray,
    attributions_a: np.ndarray,
    path_probs: np.ndarray,
    metric_a: str,
) -> Dict[str, float]:
    components = _build_components_from_assignments(
        assignments=assignments,
        embeddings_e=embeddings_e,
        attributions_a=attributions_a,
        path_probs=path_probs,
        metric_a=metric_a,
    )
    stats = compute_full_rd_statistics(
        embeddings_e=embeddings_e,
        attributions_a=attributions_a,
        assignments=assignments.tolist(),
        path_probs=path_probs,
        components=components,
        beta_e=1.0,
        beta_a=1.0,
        metric_a=metric_a,
    )
    return {
        "H": float(stats["H"]),
        "D_e": float(stats["D_e"]),
        "D_a": float(stats["D_a"]),
    }


def _gamma_distance(gamma: float, D_e: float, D_a: float) -> float:
    return float(gamma) * float(D_e) + (1.0 - float(gamma)) * float(D_a)


def _match_candidate(rd_entry: GridEntry, candidates: Sequence[Dict[str, float]]) -> Dict[str, float]:
    return min(
        candidates,
        key=lambda cand: (
            abs(float(cand["H"]) - rd_entry.H),
            _gamma_distance(rd_entry.gamma, float(cand["D_e"]), float(cand["D_a"])),
            float(cand["K_actual"]),
            float(cand["lambda"]),
        ),
    )


def _make_rd_row_from_match(row: Dict[str, float]) -> GridEntry:
    return GridEntry(
        beta=float(row["beta"]),
        gamma=float(row["gamma"]),
        K=int(round(float(row["rd_K"]))),
        H=float(row["rd_H"]),
        D_e=float(row["rd_D_e"]),
        D_a=float(row["rd_D_a"]),
    )


def _resolve_k_values(
    rd_entries: Sequence[GridEntry],
    explicit_k_values: Optional[Sequence[int]],
    k_max: Optional[int],
    n_samples: int,
) -> List[int]:
    if explicit_k_values:
        return sorted({k for k in explicit_k_values if 1 <= int(k) <= n_samples})
    if k_max is not None:
        upper = min(int(k_max), n_samples)
        return list(range(1, upper + 1))
    return sorted({int(entry.K) for entry in rd_entries if 1 <= int(entry.K) <= n_samples})


def _process_prefix(config: WorkerConfig) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    sweep_file = Path(config.sweep_file)
    results_dir = Path(config.results_dir)
    prefix_id = sweep_file.name.replace("_sweep_results.json", "")
    metric_a = _load_metric_a(sweep_file)
    gamma_filter = set(config.gamma_values) if config.gamma_values is not None else None

    rd_entries = [
        entry
        for entry in _iter_grid_entries(sweep_file)
        if _close(entry.beta, config.beta)
        and (gamma_filter is None or any(_close(entry.gamma, g, tol=1e-6) for g in gamma_filter))
    ]
    if not rd_entries:
        return [], []

    data = load_prefix_data(
        prefix_id=prefix_id,
        embeddings_dir=results_dir / "4_feature_extraction" / "embeddings",
        attribution_graphs_dir=results_dir / "3_attribution_graphs",
        samples_dir=results_dir / "2_branch_sampling",
        logger=_SilentLogger(),
        pooling=config.pooling,
        metric_a=metric_a,
    )

    embeddings_e = np.asarray(data["embeddings_e"], dtype=np.float64)
    attributions_a = np.asarray(data["attributions_a"], dtype=np.float64)
    path_probs = np.asarray(data["path_probs"], dtype=np.float64)

    affinity_e, sigma_e = _build_normalized_affinity(embeddings_e, metric="l2")
    affinity_a, sigma_a = _build_normalized_affinity(
        attributions_a,
        metric="l1" if metric_a == "l1" else "l2",
    )

    k_values = _resolve_k_values(rd_entries, config.k_values, config.k_max, embeddings_e.shape[0])
    if not k_values:
        return [], []

    candidates: List[Dict[str, float]] = []
    candidate_rows: List[Dict[str, float]] = []
    for requested_k in k_values:
        if requested_k == 1:
            labels = np.zeros(embeddings_e.shape[0], dtype=np.int32)
            stats = _compute_assignment_stats(labels, embeddings_e, attributions_a, path_probs, metric_a)
            for lambda_value in config.lambda_values:
                candidate = {
                    "K_requested": 1.0,
                    "K_actual": 1.0,
                    "lambda": float(lambda_value),
                    "H": stats["H"],
                    "D_e": stats["D_e"],
                    "D_a": stats["D_a"],
                    "spectral_iterations": 0.0,
                    "spectral_converged": 1.0,
                    "objective": 0.0,
                }
                candidates.append(candidate)
                candidate_rows.append(
                    {
                        "prefix_id": prefix_id,
                        "beta": float(config.beta),
                        "K_requested": float(candidate["K_requested"]),
                        "K_actual": float(candidate["K_actual"]),
                        "lambda": float(candidate["lambda"]),
                        "H": float(candidate["H"]),
                        "D_e": float(candidate["D_e"]),
                        "D_a": float(candidate["D_a"]),
                        "spectral_iterations": float(candidate["spectral_iterations"]),
                        "spectral_converged": float(candidate["spectral_converged"]),
                        "objective": float(candidate["objective"]),
                        "sigma_e": sigma_e,
                        "sigma_a": sigma_a,
                        "metric_a": metric_a,
                    }
                )
            continue

        for lambda_value in config.lambda_values:
            U_e, U_a, n_iters, converged, objective = _run_pairwise_coreg(
                affinity_e=affinity_e,
                affinity_a=affinity_a,
                k=requested_k,
                lambda_value=lambda_value,
                max_iter=config.spectral_max_iter,
                tol=config.spectral_tol,
            )
            labels, actual_k = _cluster_embedding(
                U_e=U_e,
                U_a=U_a,
                requested_k=requested_k,
                seed=config.kmeans_seed,
                n_init=config.kmeans_n_init,
                max_iter=config.kmeans_max_iter,
            )
            stats = _compute_assignment_stats(labels, embeddings_e, attributions_a, path_probs, metric_a)
            candidate = {
                "K_requested": float(requested_k),
                "K_actual": float(actual_k),
                "lambda": float(lambda_value),
                "H": stats["H"],
                "D_e": stats["D_e"],
                "D_a": stats["D_a"],
                "spectral_iterations": float(n_iters),
                "spectral_converged": 1.0 if converged else 0.0,
                "objective": float(objective),
            }
            candidates.append(candidate)
            candidate_rows.append(
                {
                    "prefix_id": prefix_id,
                    "beta": float(config.beta),
                    "K_requested": float(candidate["K_requested"]),
                    "K_actual": float(candidate["K_actual"]),
                    "lambda": float(candidate["lambda"]),
                    "H": float(candidate["H"]),
                    "D_e": float(candidate["D_e"]),
                    "D_a": float(candidate["D_a"]),
                    "spectral_iterations": float(candidate["spectral_iterations"]),
                    "spectral_converged": float(candidate["spectral_converged"]),
                    "objective": float(candidate["objective"]),
                    "sigma_e": sigma_e,
                    "sigma_a": sigma_a,
                    "metric_a": metric_a,
                }
            )

    matched_rows: List[Dict[str, float]] = []
    for rd_entry in sorted(rd_entries, key=lambda entry: entry.gamma):
        matched = _match_candidate(rd_entry, candidates)
        matched_rows.append(
            {
                "prefix_id": prefix_id,
                "beta": rd_entry.beta,
                "gamma": rd_entry.gamma,
                "rd_K": float(rd_entry.K),
                "rd_H": rd_entry.H,
                "rd_D_e": rd_entry.D_e,
                "rd_D_a": rd_entry.D_a,
                "coreg_K_requested": float(matched["K_requested"]),
                "coreg_K": float(matched["K_actual"]),
                "coreg_lambda": float(matched["lambda"]),
                "coreg_H": float(matched["H"]),
                "coreg_D_e": float(matched["D_e"]),
                "coreg_D_a": float(matched["D_a"]),
                "abs_delta_H": abs(float(matched["H"]) - rd_entry.H),
                "rd_D_gamma": _gamma_distance(rd_entry.gamma, rd_entry.D_e, rd_entry.D_a),
                "coreg_D_gamma": _gamma_distance(rd_entry.gamma, float(matched["D_e"]), float(matched["D_a"])),
                "coreg_spectral_iterations": float(matched["spectral_iterations"]),
                "coreg_spectral_converged": float(matched["spectral_converged"]),
                "sigma_e": sigma_e,
                "sigma_a": sigma_a,
                "metric_a": metric_a,
            }
        )

    return matched_rows, candidate_rows


def _aggregate_rows(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[float, Dict[str, float]] = {}
    for row in rows:
        gamma = float(row["gamma"])
        if gamma not in grouped:
            grouped[gamma] = {
                "beta": float(row["beta"]),
                "gamma": gamma,
                "rd_H": 0.0,
                "rd_D_e": 0.0,
                "rd_D_a": 0.0,
                "coreg_H": 0.0,
                "coreg_D_e": 0.0,
                "coreg_D_a": 0.0,
                "coreg_K": 0.0,
                "coreg_lambda": 0.0,
                "abs_delta_H": 0.0,
                "n": 0.0,
            }

        entry = grouped[gamma]
        entry["rd_H"] += float(row["rd_H"])
        entry["rd_D_e"] += float(row["rd_D_e"])
        entry["rd_D_a"] += float(row["rd_D_a"])
        entry["coreg_H"] += float(row["coreg_H"])
        entry["coreg_D_e"] += float(row["coreg_D_e"])
        entry["coreg_D_a"] += float(row["coreg_D_a"])
        entry["coreg_K"] += float(row["coreg_K"])
        entry["coreg_lambda"] += float(row["coreg_lambda"])
        entry["abs_delta_H"] += float(row["abs_delta_H"])
        entry["n"] += 1.0

    aggregated: List[Dict[str, float]] = []
    for gamma in sorted(grouped):
        entry = grouped[gamma]
        count = max(entry["n"], 1.0)
        aggregated.append(
            {
                "beta": entry["beta"],
                "gamma": entry["gamma"],
                "rd_H": entry["rd_H"] / count,
                "rd_D_e": entry["rd_D_e"] / count,
                "rd_D_a": entry["rd_D_a"] / count,
                "coreg_H": entry["coreg_H"] / count,
                "coreg_D_e": entry["coreg_D_e"] / count,
                "coreg_D_a": entry["coreg_D_a"] / count,
                "coreg_K": entry["coreg_K"] / count,
                "coreg_lambda": entry["coreg_lambda"] / count,
                "abs_delta_H": entry["abs_delta_H"] / count,
                "n": count,
            }
        )
    return aggregated


def _build_lambda_matched_rows(
    matched_rows: Sequence[Dict[str, float]],
    candidate_rows: Sequence[Dict[str, float]],
    beta: float,
) -> List[Dict[str, float]]:
    rd_by_prefix: Dict[str, List[GridEntry]] = {}
    for row in matched_rows:
        rd_by_prefix.setdefault(str(row["prefix_id"]), []).append(_make_rd_row_from_match(row))

    candidates_by_prefix_lambda: Dict[str, Dict[float, List[Dict[str, float]]]] = {}
    for row in candidate_rows:
        prefix_id = str(row["prefix_id"])
        lambda_value = float(row["lambda"])
        candidates_by_prefix_lambda.setdefault(prefix_id, {}).setdefault(lambda_value, []).append(
            {
                "K_requested": float(row["K_requested"]),
                "K_actual": float(row["K_actual"]),
                "lambda": lambda_value,
                "H": float(row["H"]),
                "D_e": float(row["D_e"]),
                "D_a": float(row["D_a"]),
                "spectral_iterations": float(row["spectral_iterations"]),
                "spectral_converged": float(row["spectral_converged"]),
                "objective": float(row["objective"]),
            }
        )

    lambda_matched_rows: List[Dict[str, float]] = []
    for prefix_id, rd_entries in rd_by_prefix.items():
        by_lambda = candidates_by_prefix_lambda.get(prefix_id, {})
        for rd_entry in rd_entries:
            for lambda_value, lambda_candidates in by_lambda.items():
                matched = _match_candidate(rd_entry, lambda_candidates)
                rd_d_gamma = _gamma_distance(rd_entry.gamma, rd_entry.D_e, rd_entry.D_a)
                coreg_d_gamma = _gamma_distance(
                    rd_entry.gamma,
                    float(matched["D_e"]),
                    float(matched["D_a"]),
                )
                rd_l_beta = rd_entry.H + float(beta) * rd_d_gamma
                coreg_l_beta = float(matched["H"]) + float(beta) * coreg_d_gamma
                lambda_matched_rows.append(
                    {
                        "prefix_id": prefix_id,
                        "beta": float(beta),
                        "gamma": float(rd_entry.gamma),
                        "lambda": float(lambda_value),
                        "rd_K": float(rd_entry.K),
                        "rd_H": float(rd_entry.H),
                        "rd_D_e": float(rd_entry.D_e),
                        "rd_D_a": float(rd_entry.D_a),
                        "rd_D_gamma": float(rd_d_gamma),
                        "rd_L_beta": float(rd_l_beta),
                        "coreg_K_requested": float(matched["K_requested"]),
                        "coreg_K": float(matched["K_actual"]),
                        "coreg_H": float(matched["H"]),
                        "coreg_D_e": float(matched["D_e"]),
                        "coreg_D_a": float(matched["D_a"]),
                        "coreg_D_gamma": float(coreg_d_gamma),
                        "coreg_L_beta": float(coreg_l_beta),
                        "delta_H": float(matched["H"]) - float(rd_entry.H),
                        "abs_delta_H": abs(float(matched["H"]) - float(rd_entry.H)),
                        "delta_D_e": float(matched["D_e"]) - float(rd_entry.D_e),
                        "delta_D_a": float(matched["D_a"]) - float(rd_entry.D_a),
                        "delta_D_gamma": float(coreg_d_gamma - rd_d_gamma),
                        "delta_L_beta": float(coreg_l_beta - rd_l_beta),
                    }
                )
    return lambda_matched_rows


def _aggregate_lambda_rows(
    rows: Sequence[Dict[str, float]],
    group_keys: Sequence[str],
) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[float, ...], Dict[str, float]] = {}
    for row in rows:
        key = tuple(float(row[k]) for k in group_keys)
        if key not in grouped:
            grouped[key] = {
                **{k: float(row[k]) for k in group_keys},
                "mean_delta_H": 0.0,
                "mean_abs_delta_H": 0.0,
                "mean_delta_D_e": 0.0,
                "mean_delta_D_a": 0.0,
                "mean_delta_D_gamma": 0.0,
                "mean_delta_L_beta": 0.0,
                "coreg_better_D_gamma": 0.0,
                "rd_better_D_gamma": 0.0,
                "ties_D_gamma": 0.0,
                "coreg_better_L_beta": 0.0,
                "rd_better_L_beta": 0.0,
                "ties_L_beta": 0.0,
                "n": 0.0,
            }
        entry = grouped[key]
        entry["mean_delta_H"] += float(row["delta_H"])
        entry["mean_abs_delta_H"] += float(row["abs_delta_H"])
        entry["mean_delta_D_e"] += float(row["delta_D_e"])
        entry["mean_delta_D_a"] += float(row["delta_D_a"])
        entry["mean_delta_D_gamma"] += float(row["delta_D_gamma"])
        entry["mean_delta_L_beta"] += float(row["delta_L_beta"])
        entry["n"] += 1.0

        if abs(float(row["delta_D_gamma"])) < 1e-9:
            entry["ties_D_gamma"] += 1.0
        elif float(row["delta_D_gamma"]) < 0:
            entry["coreg_better_D_gamma"] += 1.0
        else:
            entry["rd_better_D_gamma"] += 1.0

        if abs(float(row["delta_L_beta"])) < 1e-9:
            entry["ties_L_beta"] += 1.0
        elif float(row["delta_L_beta"]) < 0:
            entry["coreg_better_L_beta"] += 1.0
        else:
            entry["rd_better_L_beta"] += 1.0

    out: List[Dict[str, float]] = []
    for key in sorted(grouped):
        entry = grouped[key]
        n = max(entry["n"], 1.0)
        out.append(
            {
                **{k: entry[k] for k in group_keys},
                "n": n,
                "mean_delta_H": entry["mean_delta_H"] / n,
                "mean_abs_delta_H": entry["mean_abs_delta_H"] / n,
                "mean_delta_D_e": entry["mean_delta_D_e"] / n,
                "mean_delta_D_a": entry["mean_delta_D_a"] / n,
                "mean_delta_D_gamma": entry["mean_delta_D_gamma"] / n,
                "mean_delta_L_beta": entry["mean_delta_L_beta"] / n,
                "coreg_better_D_gamma": entry["coreg_better_D_gamma"],
                "rd_better_D_gamma": entry["rd_better_D_gamma"],
                "ties_D_gamma": entry["ties_D_gamma"],
                "coreg_better_L_beta": entry["coreg_better_L_beta"],
                "rd_better_L_beta": entry["rd_better_L_beta"],
                "ties_L_beta": entry["ties_L_beta"],
            }
        )
    return out


def _aggregate_rate_lambda_rows(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[float, float], Dict[str, float]] = {}
    for row in rows:
        gamma = float(row["gamma"])
        lambda_value = float(row["lambda"])
        key = (gamma, lambda_value)
        if key not in grouped:
            grouped[key] = {
                "beta": float(row["beta"]),
                "gamma": gamma,
                "lambda": lambda_value,
                "rd_H": 0.0,
                "coreg_H": 0.0,
                "rd_D_e": 0.0,
                "coreg_D_e": 0.0,
                "rd_D_a": 0.0,
                "coreg_D_a": 0.0,
                "rd_D_gamma": 0.0,
                "coreg_D_gamma": 0.0,
                "rd_L_beta": 0.0,
                "coreg_L_beta": 0.0,
                "mean_delta_H": 0.0,
                "mean_abs_delta_H": 0.0,
                "mean_delta_D_e": 0.0,
                "mean_delta_D_a": 0.0,
                "mean_delta_D_gamma": 0.0,
                "mean_delta_L_beta": 0.0,
                "n": 0.0,
            }
        entry = grouped[key]
        entry["rd_H"] += float(row["rd_H"])
        entry["coreg_H"] += float(row["coreg_H"])
        entry["rd_D_e"] += float(row["rd_D_e"])
        entry["coreg_D_e"] += float(row["coreg_D_e"])
        entry["rd_D_a"] += float(row["rd_D_a"])
        entry["coreg_D_a"] += float(row["coreg_D_a"])
        entry["rd_D_gamma"] += float(row["rd_D_gamma"])
        entry["coreg_D_gamma"] += float(row["coreg_D_gamma"])
        entry["rd_L_beta"] += float(row["rd_L_beta"])
        entry["coreg_L_beta"] += float(row["coreg_L_beta"])
        entry["mean_delta_H"] += float(row["delta_H"])
        entry["mean_abs_delta_H"] += float(row["abs_delta_H"])
        entry["mean_delta_D_e"] += float(row["delta_D_e"])
        entry["mean_delta_D_a"] += float(row["delta_D_a"])
        entry["mean_delta_D_gamma"] += float(row["delta_D_gamma"])
        entry["mean_delta_L_beta"] += float(row["delta_L_beta"])
        entry["n"] += 1.0

    out: List[Dict[str, float]] = []
    for key in sorted(grouped):
        entry = grouped[key]
        n = max(entry["n"], 1.0)
        out.append(
            {
                "beta": entry["beta"],
                "gamma": entry["gamma"],
                "lambda": entry["lambda"],
                "rd_H": entry["rd_H"] / n,
                "coreg_H": entry["coreg_H"] / n,
                "rd_D_e": entry["rd_D_e"] / n,
                "coreg_D_e": entry["coreg_D_e"] / n,
                "rd_D_a": entry["rd_D_a"] / n,
                "coreg_D_a": entry["coreg_D_a"] / n,
                "rd_D_gamma": entry["rd_D_gamma"] / n,
                "coreg_D_gamma": entry["coreg_D_gamma"] / n,
                "rd_L_beta": entry["rd_L_beta"] / n,
                "coreg_L_beta": entry["coreg_L_beta"] / n,
                "mean_delta_H": entry["mean_delta_H"] / n,
                "mean_abs_delta_H": entry["mean_abs_delta_H"] / n,
                "mean_delta_D_e": entry["mean_delta_D_e"] / n,
                "mean_delta_D_a": entry["mean_delta_D_a"] / n,
                "mean_delta_D_gamma": entry["mean_delta_D_gamma"] / n,
                "mean_delta_L_beta": entry["mean_delta_L_beta"] / n,
                "n": n,
            }
        )
    return out


def _aggregate_rate_gamma_rows(rows: Sequence[Dict[str, float]], beta: float) -> List[Dict[str, float]]:
    grouped: Dict[float, Dict[str, float]] = {}
    for row in rows:
        gamma = float(row["gamma"])
        if gamma not in grouped:
            grouped[gamma] = {
                "beta": float(row["beta"]),
                "gamma": gamma,
                "rd_H": 0.0,
                "coreg_H": 0.0,
                "rd_D_e": 0.0,
                "coreg_D_e": 0.0,
                "rd_D_a": 0.0,
                "coreg_D_a": 0.0,
                "rd_D_gamma": 0.0,
                "coreg_D_gamma": 0.0,
                "rd_L_beta": 0.0,
                "coreg_L_beta": 0.0,
                "coreg_lambda": 0.0,
                "coreg_K": 0.0,
                "abs_delta_H": 0.0,
                "n": 0.0,
            }
        entry = grouped[gamma]
        entry["rd_H"] += float(row["rd_H"])
        entry["coreg_H"] += float(row["coreg_H"])
        entry["rd_D_e"] += float(row["rd_D_e"])
        entry["coreg_D_e"] += float(row["coreg_D_e"])
        entry["rd_D_a"] += float(row["rd_D_a"])
        entry["coreg_D_a"] += float(row["coreg_D_a"])
        entry["rd_D_gamma"] += float(row["rd_D_gamma"])
        entry["coreg_D_gamma"] += float(row["coreg_D_gamma"])
        entry["rd_L_beta"] += float(row["rd_H"]) + float(beta) * float(row["rd_D_gamma"])
        entry["coreg_L_beta"] += float(row["coreg_H"]) + float(beta) * float(row["coreg_D_gamma"])
        entry["coreg_lambda"] += float(row["coreg_lambda"])
        entry["coreg_K"] += float(row["coreg_K"])
        entry["abs_delta_H"] += float(row["abs_delta_H"])
        entry["n"] += 1.0

    out: List[Dict[str, float]] = []
    for gamma in sorted(grouped):
        entry = grouped[gamma]
        n = max(entry["n"], 1.0)
        out.append(
            {
                "beta": entry["beta"],
                "gamma": entry["gamma"],
                "rd_H": entry["rd_H"] / n,
                "coreg_H": entry["coreg_H"] / n,
                "rd_D_e": entry["rd_D_e"] / n,
                "coreg_D_e": entry["coreg_D_e"] / n,
                "rd_D_a": entry["rd_D_a"] / n,
                "coreg_D_a": entry["coreg_D_a"] / n,
                "rd_D_gamma": entry["rd_D_gamma"] / n,
                "coreg_D_gamma": entry["coreg_D_gamma"] / n,
                "rd_L_beta": entry["rd_L_beta"] / n,
                "coreg_L_beta": entry["coreg_L_beta"] / n,
                "coreg_lambda": entry["coreg_lambda"] / n,
                "coreg_K": entry["coreg_K"] / n,
                "abs_delta_H": entry["abs_delta_H"] / n,
                "n": n,
            }
        )
    return out


def _aggregate_rate_lambda_overall_rows(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[float, Dict[str, float]] = {}
    for row in rows:
        lambda_value = float(row["lambda"])
        if lambda_value not in grouped:
            grouped[lambda_value] = {
                "beta": float(row["beta"]),
                "lambda": lambda_value,
                "rd_H": 0.0,
                "coreg_H": 0.0,
                "rd_D_e": 0.0,
                "coreg_D_e": 0.0,
                "rd_D_a": 0.0,
                "coreg_D_a": 0.0,
                "rd_D_gamma": 0.0,
                "coreg_D_gamma": 0.0,
                "rd_L_beta": 0.0,
                "coreg_L_beta": 0.0,
                "mean_delta_H": 0.0,
                "mean_abs_delta_H": 0.0,
                "mean_delta_D_e": 0.0,
                "mean_delta_D_a": 0.0,
                "mean_delta_D_gamma": 0.0,
                "mean_delta_L_beta": 0.0,
                "n": 0.0,
            }
        entry = grouped[lambda_value]
        entry["rd_H"] += float(row["rd_H"])
        entry["coreg_H"] += float(row["coreg_H"])
        entry["rd_D_e"] += float(row["rd_D_e"])
        entry["coreg_D_e"] += float(row["coreg_D_e"])
        entry["rd_D_a"] += float(row["rd_D_a"])
        entry["coreg_D_a"] += float(row["coreg_D_a"])
        entry["rd_D_gamma"] += float(row["rd_D_gamma"])
        entry["coreg_D_gamma"] += float(row["coreg_D_gamma"])
        entry["rd_L_beta"] += float(row["rd_L_beta"])
        entry["coreg_L_beta"] += float(row["coreg_L_beta"])
        entry["mean_delta_H"] += float(row["delta_H"])
        entry["mean_abs_delta_H"] += float(row["abs_delta_H"])
        entry["mean_delta_D_e"] += float(row["delta_D_e"])
        entry["mean_delta_D_a"] += float(row["delta_D_a"])
        entry["mean_delta_D_gamma"] += float(row["delta_D_gamma"])
        entry["mean_delta_L_beta"] += float(row["delta_L_beta"])
        entry["n"] += 1.0

    out: List[Dict[str, float]] = []
    for lambda_value in sorted(grouped):
        entry = grouped[lambda_value]
        n = max(entry["n"], 1.0)
        out.append(
            {
                "beta": entry["beta"],
                "lambda": entry["lambda"],
                "rd_H": entry["rd_H"] / n,
                "coreg_H": entry["coreg_H"] / n,
                "rd_D_e": entry["rd_D_e"] / n,
                "coreg_D_e": entry["coreg_D_e"] / n,
                "rd_D_a": entry["rd_D_a"] / n,
                "coreg_D_a": entry["coreg_D_a"] / n,
                "rd_D_gamma": entry["rd_D_gamma"] / n,
                "coreg_D_gamma": entry["coreg_D_gamma"] / n,
                "rd_L_beta": entry["rd_L_beta"] / n,
                "coreg_L_beta": entry["coreg_L_beta"] / n,
                "mean_delta_H": entry["mean_delta_H"] / n,
                "mean_abs_delta_H": entry["mean_abs_delta_H"] / n,
                "mean_delta_D_e": entry["mean_delta_D_e"] / n,
                "mean_delta_D_a": entry["mean_delta_D_a"] / n,
                "mean_delta_D_gamma": entry["mean_delta_D_gamma"] / n,
                "mean_delta_L_beta": entry["mean_delta_L_beta"] / n,
                "n": n,
            }
        )
    return out


def _pivot_rate_lambda_rows(
    rows: Sequence[Dict[str, float]],
    value_key: str,
    rd_key: str,
) -> List[Dict[str, float]]:
    lambdas = sorted({float(row["lambda"]) for row in rows})
    grouped: Dict[float, Dict[str, float]] = {}
    for row in rows:
        gamma = float(row["gamma"])
        if gamma not in grouped:
            grouped[gamma] = {
                "beta": float(row["beta"]),
                "gamma": gamma,
                "rd_H": float(row["rd_H"]),
                rd_key: float(row[rd_key]),
            }
        grouped[gamma][f"lambda_{_beta_tag(float(row['lambda']))}"] = float(row[value_key])

    out: List[Dict[str, float]] = []
    for gamma in sorted(grouped):
        entry = grouped[gamma]
        row = {
            "beta": entry["beta"],
            "gamma": entry["gamma"],
            "rd_H": entry["rd_H"],
            rd_key: entry[rd_key],
        }
        for lambda_value in lambdas:
            field = f"lambda_{_beta_tag(lambda_value)}"
            row[field] = entry.get(field, "")
        out.append(row)
    return out


def _write_csv(path: Path, rows: Sequence[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _write_summary_tex(path: Path, beta: float, rows: Sequence[Dict[str, float]]) -> None:
    lines = [
        "% requires in preamble:",
        "% \\usepackage{booktabs}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{c ccc ccc c c}",
        "\\toprule",
        "$\\gamma$ & $H_{RD}$ & $D^{(e)}_{RD}$ & $D^{(a)}_{RD}$ & $H_{CoReg}$ & $D^{(e)}_{CoReg}$ & $D^{(a)}_{CoReg}$ & $|\\Delta H|$ & $n$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['gamma']:.2f} & "
            f"{row['rd_H']:.3f} & {row['rd_D_e']:.3f} & {row['rd_D_a']:.3f} & "
            f"{row['coreg_H']:.3f} & {row['coreg_D_e']:.3f} & {row['coreg_D_a']:.3f} & "
            f"{row['abs_delta_H']:.3f} & {int(row['n'])} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "",
            (
                "\\caption{Fixed-rate comparison between RD clustering and pairwise "
                f"co-regularized spectral clustering at $\\beta={beta:g}$. "
                "Each CoReg row is matched to RD by closest rate $H(C)$ across the "
                "lambda/K sweep, then averaged over prefixes.}"
            ),
            f"\\label{{tab:rd-coreg-beta-{_beta_tag(beta)}}}",
            "\\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _latex_lambda_label(field_name: str) -> str:
    value = field_name.removeprefix("lambda_").replace("p", ".")
    return f"$\\lambda={float(value):g}$"


def _write_rate_lambda_pivot_tex(
    path: Path,
    beta: float,
    rows: Sequence[Dict[str, float]],
    rd_key: str,
    distortion_symbol: str,
    label_suffix: str,
) -> None:
    lambda_fields = [field for field in rows[0].keys() if field.startswith("lambda_")] if rows else []
    col_spec = "c c " + "c" * (1 + len(lambda_fields))
    header_cells = ["$\\gamma$", "$H_{RD}$", f"${distortion_symbol}_{{RD}}$"]
    header_cells.extend(_latex_lambda_label(field) for field in lambda_fields)

    lines = [
        "% requires in preamble:",
        "% \\usepackage{booktabs,graphicx}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\resizebox{\\linewidth}{!}{%",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(header_cells) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        cells = [
            f"{float(row['gamma']):.2f}",
            f"{float(row['rd_H']):.3f}",
            f"{float(row[rd_key]):.3f}",
        ]
        cells.extend(f"{float(row[field]):.3f}" if row[field] != "" else "" for field in lambda_fields)
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "",
            (
                f"\\caption{{Matched-rate {distortion_symbol} comparison across the CoReg "
                f"$\\lambda$ sweep at $\\beta={beta:g}$. Rows are indexed by the RD target "
                "rate $H_{RD}$ for each $\\gamma$; the RD column gives the matched RD "
                "baseline and each $\\lambda$ column gives the CoReg distortion under that "
                "same rate-matching rule.}}"
            ),
            f"\\label{{tab:rd-coreg-rate-lambda-{label_suffix}-beta-{_beta_tag(beta)}}}",
            "\\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _write_rate_gamma_tex(path: Path, beta: float, rows: Sequence[Dict[str, float]]) -> None:
    lines = [
        "% requires in preamble:",
        "% \\usepackage{booktabs,graphicx}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{c cc cc cc cc c}",
        "\\toprule",
        "$\\gamma$ & $H_{RD}$ & $H_{CoReg}$ & $D^{(e)}_{RD}$ & $D^{(e)}_{CoReg}$ & $D^{(a)}_{RD}$ & $D^{(a)}_{CoReg}$ & $D_{\\gamma,RD}$ & $D_{\\gamma,CoReg}$ & $\\bar{\\lambda}$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{float(row['gamma']):.2f} & "
            f"{float(row['rd_H']):.3f} & {float(row['coreg_H']):.3f} & "
            f"{float(row['rd_D_e']):.3f} & {float(row['coreg_D_e']):.3f} & "
            f"{float(row['rd_D_a']):.3f} & {float(row['coreg_D_a']):.3f} & "
            f"{float(row['rd_D_gamma']):.3f} & {float(row['coreg_D_gamma']):.3f} & "
            f"{float(row['coreg_lambda']):.3f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "",
            (
                "\\caption{Matched-rate RD versus CoReg comparison across $\\gamma$ at "
                f"$\\beta={beta:g}$. RD and CoReg are shown side by side for rate, the "
                "two distortion terms, and the combined distortion $D_\\gamma$; "
                "$\\bar{\\lambda}$ is the average selected CoReg $\\lambda$ among the "
                "matched rows.}"
            ),
            f"\\label{{tab:rd-coreg-rate-gamma-beta-{_beta_tag(beta)}}}",
            "\\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _write_rate_lambda_summary_tex(path: Path, beta: float, rows: Sequence[Dict[str, float]]) -> None:
    lines = [
        "% requires in preamble:",
        "% \\usepackage{booktabs,graphicx}",
        "",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{cc ccc ccc c}",
        "\\toprule",
        "& & \\multicolumn{3}{c}{RD} & \\multicolumn{3}{c}{CoReg} & \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}",
        "$\\gamma$ & $\\lambda$ & $H$ & $D^{(e)}$ & $D^{(a)}$ & $H$ & $D^{(e)}$ & $D^{(a)}$ & $|\\Delta H|$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{float(row['gamma']):.2f} & {float(row['lambda']):g} & "
            f"{float(row['rd_H']):.3f} & {float(row['rd_D_e']):.3f} & {float(row['rd_D_a']):.3f} & "
            f"{float(row['coreg_H']):.3f} & {float(row['coreg_D_e']):.3f} & {float(row['coreg_D_a']):.3f} & "
            f"{float(row['mean_abs_delta_H']):.3f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "",
            (
                "\\caption{Matched-rate $\\lambda$ sweep at "
                f"$\\beta={beta:g}$. Each row pairs the RD result with the CoReg result "
                "obtained under the same rate-matching rule for a fixed $(\\gamma, \\lambda)$, "
                "reporting rate $H$ and the two distortion terms $D^{(e)}$ and $D^{(a)}$.}"
            ),
            f"\\label{{tab:rd-coreg-rate-lambda-summary-beta-{_beta_tag(beta)}}}",
            "\\end{table*}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _write_rate_lambda_overall_tex(path: Path, beta: float, rows: Sequence[Dict[str, float]]) -> None:
    lines = [
        "% requires in preamble:",
        "% \\usepackage{booktabs,graphicx}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{c ccc ccc ccc}",
        "\\toprule",
        "& \\multicolumn{3}{c}{RD} & \\multicolumn{3}{c}{CoReg} & \\multicolumn{3}{c}{Loss} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
        "$\\lambda$ & $H$ & $D^{(e)}$ & $D^{(a)}$ & $H$ & $D^{(e)}$ & $D^{(a)}$ & $L_{RD}$ & $L_{CoReg}$ & $\\Delta L$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{float(row['lambda']):g} & "
            f"{float(row['rd_H']):.3f} & {float(row['rd_D_e']):.3f} & {float(row['rd_D_a']):.3f} & "
            f"{float(row['coreg_H']):.3f} & {float(row['coreg_D_e']):.3f} & {float(row['coreg_D_a']):.3f} & "
            f"{float(row['rd_L_beta']):.3f} & {float(row['coreg_L_beta']):.3f} & {float(row['mean_delta_L_beta']):.3f} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "",
            (
                "\\caption{Lambda-focused matched-rate comparison at "
                f"$\\beta={beta:g}$. Each row averages over all matched $(\\mathrm{{prefix}},\\gamma)$ "
                "pairs for that $\\lambda$, and reports the paired RD and CoReg rate $H$ and "
                "distortion terms $D^{(e)}$ and $D^{(a)}$, along with the final loss "
                "$L = H + \\beta D_\\gamma$ and the gap $\\Delta L = L_{CoReg} - L_{RD}$.}"
            ),
            f"\\label{{tab:rd-coreg-lambda-overall-beta-{_beta_tag(beta)}}}",
            "\\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _write_rate_lambda_major_tex(path: Path, beta: float, rows: Sequence[Dict[str, float]]) -> None:
    sorted_rows = sorted(rows, key=lambda row: (float(row["lambda"]), float(row["gamma"])))
    lines = [
        "% requires in preamble:",
        "% \\usepackage{booktabs,graphicx}",
        "",
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{cc ccc ccc ccc}",
        "\\toprule",
        "& & \\multicolumn{3}{c}{RD} & \\multicolumn{3}{c}{CoReg} & \\multicolumn{3}{c}{Loss} \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}",
        "$\\lambda$ & $\\gamma$ & $H$ & $D^{(e)}$ & $D^{(a)}$ & $H$ & $D^{(e)}$ & $D^{(a)}$ & $L_{RD}$ & $L_{CoReg}$ & $\\Delta L$ \\\\",
        "\\midrule",
    ]
    prev_lambda: Optional[float] = None
    for row in sorted_rows:
        lambda_value = float(row["lambda"])
        if prev_lambda is not None and not _close(lambda_value, prev_lambda, tol=1e-9):
            lines.append("\\midrule")
        lines.append(
            f"{lambda_value:g} & {float(row['gamma']):.2f} & "
            f"{float(row['rd_H']):.3f} & {float(row['rd_D_e']):.3f} & {float(row['rd_D_a']):.3f} & "
            f"{float(row['coreg_H']):.3f} & {float(row['coreg_D_e']):.3f} & {float(row['coreg_D_a']):.3f} & "
            f"{float(row['rd_L_beta']):.3f} & {float(row['coreg_L_beta']):.3f} & {float(row['mean_delta_L_beta']):.3f} \\\\"
        )
        prev_lambda = lambda_value
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "",
            (
                "\\caption{Lambda-major matched-rate comparison at "
                f"$\\beta={beta:g}$. Rows are grouped by CoReg $\\lambda$ and, within each "
                "$\\lambda$, list the paired RD and CoReg rate $H$ and distortion terms "
                "$D^{(e)}$ and $D^{(a)}$ for each matched $\\gamma$ row, together with "
                "the final loss $L = H + \\beta D_\\gamma$ and the gap "
                "$\\Delta L = L_{CoReg} - L_{RD}$.}"
            ),
            f"\\label{{tab:rd-coreg-lambda-major-beta-{_beta_tag(beta)}}}",
            "\\end{table*}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare RD clustering against pairwise co-regularized spectral clustering at fixed rate."
    )
    ap.add_argument("--results-dir", type=Path, required=True, help="Results directory containing stages 2-5")
    ap.add_argument("--beta", type=float, required=True, help="Fixed beta to compare")
    ap.add_argument("--output-dir", type=Path, required=True, help="Directory for JSON/CSV/TeX outputs")
    ap.add_argument("--gamma-values", type=str, default=None, help="Optional comma-separated gamma filter")
    ap.add_argument("--lambda-values", type=str, default=None, help="Optional comma-separated lambda grid")
    ap.add_argument("--k-values", type=str, default=None, help="Optional comma-separated K values")
    ap.add_argument("--k-max", type=int, default=None, help="Optional inclusive K cap")
    ap.add_argument("--max-prefixes", type=int, default=None, help="Optional prefix limit for smoke tests")
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "max", "sum"])
    ap.add_argument("--spectral-max-iter", type=int, default=10)
    ap.add_argument("--spectral-tol", type=float, default=1e-4)
    ap.add_argument("--kmeans-seed", type=int, default=42)
    ap.add_argument("--kmeans-n-init", type=int, default=10)
    ap.add_argument("--kmeans-max-iter", type=int, default=300)
    ap.add_argument("--n-workers", type=int, default=1)
    args = ap.parse_args()

    lambda_values = _parse_float_list(args.lambda_values)
    if lambda_values is None:
        lambda_values = tuple(DEFAULT_LAMBDAS)

    gamma_values = _parse_float_list(args.gamma_values)
    k_values = _parse_int_list(args.k_values)

    sweep_dir = args.results_dir / "5_clustering"
    sweep_files = sorted(sweep_dir.glob("*_sweep_results.json"))
    if args.max_prefixes is not None:
        sweep_files = sweep_files[: int(args.max_prefixes)]
    if not sweep_files:
        raise SystemExit(f"No sweep results found in {sweep_dir}")

    tasks = [
        WorkerConfig(
            sweep_file=str(sweep_file),
            results_dir=str(args.results_dir),
            beta=float(args.beta),
            gamma_values=gamma_values,
            pooling=args.pooling,
            lambda_values=tuple(lambda_values),
            k_values=k_values,
            k_max=args.k_max,
            spectral_max_iter=args.spectral_max_iter,
            spectral_tol=float(args.spectral_tol),
            kmeans_seed=int(args.kmeans_seed),
            kmeans_n_init=int(args.kmeans_n_init),
            kmeans_max_iter=int(args.kmeans_max_iter),
        )
        for sweep_file in sweep_files
    ]

    matched_rows: List[Dict[str, float]] = []
    candidate_rows: List[Dict[str, float]] = []
    if args.n_workers > 1:
        with ProcessPoolExecutor(max_workers=int(args.n_workers)) as executor:
            futures = {executor.submit(_process_prefix, task): task.sweep_file for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Prefixes"):
                try:
                    prefix_matched_rows, prefix_candidate_rows = future.result()
                    matched_rows.extend(prefix_matched_rows)
                    candidate_rows.extend(prefix_candidate_rows)
                except Exception as exc:
                    print(f"Skipping {futures[future]}: {exc}")
    else:
        for task in tqdm(tasks, desc="Prefixes"):
            try:
                prefix_matched_rows, prefix_candidate_rows = _process_prefix(task)
                matched_rows.extend(prefix_matched_rows)
                candidate_rows.extend(prefix_candidate_rows)
            except Exception as exc:
                print(f"Skipping {task.sweep_file}: {exc}")

    if not matched_rows:
        raise SystemExit(f"No matched rows found for beta={args.beta}")

    matched_rows.sort(key=lambda row: (row["prefix_id"], row["gamma"]))
    candidate_rows.sort(key=lambda row: (row["prefix_id"], row["lambda"], row["K_requested"], row["K_actual"]))
    summary_rows = _aggregate_rows(matched_rows)
    lambda_matched_rows = _build_lambda_matched_rows(matched_rows, candidate_rows, beta=args.beta)
    lambda_summary_rows = _aggregate_lambda_rows(lambda_matched_rows, group_keys=["lambda"])
    lambda_gamma_summary_rows = _aggregate_lambda_rows(lambda_matched_rows, group_keys=["lambda", "gamma"])
    rate_lambda_rows = _aggregate_rate_lambda_rows(lambda_matched_rows)
    rate_lambda_overall_rows = _aggregate_rate_lambda_overall_rows(lambda_matched_rows)
    rate_gamma_rows = _aggregate_rate_gamma_rows(matched_rows, beta=args.beta)
    rate_lambda_de_pivot_rows = _pivot_rate_lambda_rows(rate_lambda_rows, value_key="coreg_D_e", rd_key="rd_D_e")
    rate_lambda_da_pivot_rows = _pivot_rate_lambda_rows(rate_lambda_rows, value_key="coreg_D_a", rd_key="rd_D_a")

    beta_tag = _beta_tag(args.beta)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    matches_json = args.output_dir / f"coreg_fixed_rate_beta_{beta_tag}_matches.json"
    matches_csv = args.output_dir / f"coreg_fixed_rate_beta_{beta_tag}_matches.csv"
    summary_csv = args.output_dir / f"coreg_fixed_rate_beta_{beta_tag}_summary.csv"
    summary_tex = args.output_dir / f"coreg_fixed_rate_beta_{beta_tag}_summary.tex"
    candidates_csv = args.output_dir / f"coreg_fixed_rate_beta_{beta_tag}_candidates.csv"
    candidates_json = args.output_dir / f"coreg_fixed_rate_beta_{beta_tag}_candidates.json"
    lambda_matches_csv = args.output_dir / f"coreg_fixed_rate_beta_{beta_tag}_lambda_matched.csv"
    lambda_summary_csv = args.output_dir / f"coreg_fixed_rate_beta_{beta_tag}_lambda_summary.csv"
    lambda_gamma_summary_csv = args.output_dir / f"coreg_fixed_rate_beta_{beta_tag}_lambda_gamma_summary.csv"
    rate_lambda_csv = args.output_dir / f"coreg_fixed_rate_beta_{beta_tag}_rate_lambda_summary.csv"
    rate_lambda_overall_csv = args.output_dir / f"coreg_fixed_rate_beta_{beta_tag}_lambda_overall_summary.csv"
    rate_gamma_csv = args.output_dir / f"coreg_fixed_rate_beta_{beta_tag}_rate_gamma_summary.csv"
    rate_lambda_de_pivot_csv = args.output_dir / f"coreg_fixed_rate_beta_{beta_tag}_rate_lambda_de_pivot.csv"
    rate_lambda_da_pivot_csv = args.output_dir / f"coreg_fixed_rate_beta_{beta_tag}_rate_lambda_da_pivot.csv"
    rate_lambda_tex = args.output_dir / f"coreg_fixed_rate_beta_{beta_tag}_rate_lambda_summary.tex"
    rate_lambda_overall_tex = args.output_dir / f"coreg_fixed_rate_beta_{beta_tag}_lambda_overall_summary.tex"
    rate_lambda_major_tex = args.output_dir / f"coreg_fixed_rate_beta_{beta_tag}_lambda_major_summary.tex"
    rate_gamma_tex = args.output_dir / f"coreg_fixed_rate_beta_{beta_tag}_rate_gamma_summary.tex"
    rate_lambda_de_pivot_tex = args.output_dir / f"coreg_fixed_rate_beta_{beta_tag}_rate_lambda_de_pivot.tex"
    rate_lambda_da_pivot_tex = args.output_dir / f"coreg_fixed_rate_beta_{beta_tag}_rate_lambda_da_pivot.tex"

    _write_json(
        matches_json,
        {
            "beta": args.beta,
            "gamma_values": list(gamma_values) if gamma_values is not None else None,
            "lambda_values": list(lambda_values),
            "k_values": list(k_values) if k_values is not None else None,
            "k_max": args.k_max,
            "n_prefixes": len({row["prefix_id"] for row in matched_rows}),
            "rows": matched_rows,
        },
    )
    _write_json(
        candidates_json,
        {
            "beta": args.beta,
            "gamma_values": list(gamma_values) if gamma_values is not None else None,
            "lambda_values": list(lambda_values),
            "k_values": list(k_values) if k_values is not None else None,
            "k_max": args.k_max,
            "n_prefixes": len({row["prefix_id"] for row in candidate_rows}),
            "rows": candidate_rows,
        },
    )
    _write_csv(matches_csv, matched_rows)
    _write_csv(summary_csv, summary_rows)
    _write_summary_tex(summary_tex, args.beta, summary_rows)
    _write_csv(candidates_csv, candidate_rows)
    _write_csv(lambda_matches_csv, lambda_matched_rows)
    _write_csv(lambda_summary_csv, lambda_summary_rows)
    _write_csv(lambda_gamma_summary_csv, lambda_gamma_summary_rows)
    _write_csv(rate_lambda_csv, rate_lambda_rows)
    _write_csv(rate_lambda_overall_csv, rate_lambda_overall_rows)
    _write_csv(rate_gamma_csv, rate_gamma_rows)
    _write_csv(rate_lambda_de_pivot_csv, rate_lambda_de_pivot_rows)
    _write_csv(rate_lambda_da_pivot_csv, rate_lambda_da_pivot_rows)
    _write_rate_lambda_summary_tex(rate_lambda_tex, args.beta, rate_lambda_rows)
    _write_rate_lambda_overall_tex(rate_lambda_overall_tex, args.beta, rate_lambda_overall_rows)
    _write_rate_lambda_major_tex(rate_lambda_major_tex, args.beta, rate_lambda_rows)
    _write_rate_gamma_tex(rate_gamma_tex, args.beta, rate_gamma_rows)
    _write_rate_lambda_pivot_tex(
        rate_lambda_de_pivot_tex,
        args.beta,
        rate_lambda_de_pivot_rows,
        rd_key="rd_D_e",
        distortion_symbol="D^{(e)}",
        label_suffix="de",
    )
    _write_rate_lambda_pivot_tex(
        rate_lambda_da_pivot_tex,
        args.beta,
        rate_lambda_da_pivot_rows,
        rd_key="rd_D_a",
        distortion_symbol="D^{(a)}",
        label_suffix="da",
    )

    print(f"Wrote {matches_json}")
    print(f"Wrote {matches_csv}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {summary_tex}")
    print(f"Wrote {candidates_json}")
    print(f"Wrote {candidates_csv}")
    print(f"Wrote {lambda_matches_csv}")
    print(f"Wrote {lambda_summary_csv}")
    print(f"Wrote {lambda_gamma_summary_csv}")
    print(f"Wrote {rate_lambda_csv}")
    print(f"Wrote {rate_lambda_overall_csv}")
    print(f"Wrote {rate_gamma_csv}")
    print(f"Wrote {rate_lambda_de_pivot_csv}")
    print(f"Wrote {rate_lambda_da_pivot_csv}")
    print(f"Wrote {rate_lambda_tex}")
    print(f"Wrote {rate_lambda_overall_tex}")
    print(f"Wrote {rate_lambda_major_tex}")
    print(f"Wrote {rate_gamma_tex}")
    print(f"Wrote {rate_lambda_de_pivot_tex}")
    print(f"Wrote {rate_lambda_da_pivot_tex}")


if __name__ == "__main__":
    main()
