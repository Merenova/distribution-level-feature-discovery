"""Rate-distortion based Gaussian clustering module.

This module implements two-view Gaussian clustering using principled
rate-distortion criteria instead of hand-tuned thresholds.
"""

from .rd_objective import (
    compute_entropy,
    compute_semantic_distortion,
    compute_attribution_distortion,
    compute_rd_objective,
    compute_full_rd_statistics,
)
from .em_loop import rd_e_step, m_step, run_em_loop
from .adaptive_control import apply_adaptive_control
from .initialize import initialize_clusters

__all__ = [
    # R-D objective
    "compute_entropy",
    "compute_semantic_distortion",
    "compute_attribution_distortion",
    "compute_rd_objective",
    "compute_full_rd_statistics",
    # EM loop
    "rd_e_step",
    "m_step",
    "run_em_loop",
    # Adaptive control
    "apply_adaptive_control",
    # Initialization
    "initialize_clusters",
]
