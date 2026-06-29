"""Memory management utilities for GPU/CUDA operations."""

import gc
import time
import torch


def clear_memory():
    """Clear Python garbage and CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


_LAST_MAYBE_CLEAR_TS = 0.0


def maybe_clear_memory(
    *,
    gc_collect: bool = False,
    cuda_empty_cache: bool = False,
    min_interval_s: float = 60.0,
    force: bool = False,
    logger=None,
    tag: str = "",
):
    """Throttled memory cleanup to avoid pathological slowdowns in tight loops."""
    global _LAST_MAYBE_CLEAR_TS

    now = time.perf_counter()
    if (not force) and (now - _LAST_MAYBE_CLEAR_TS) < float(min_interval_s):
        return

    _LAST_MAYBE_CLEAR_TS = now

    t0 = time.perf_counter()
    if gc_collect:
        gc.collect()
    t1 = time.perf_counter()

    if cuda_empty_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()
    t2 = time.perf_counter()

    if logger and (t2 - t0) > 1.0:
        logger.info(
            f"maybe_clear_memory{f' ({tag})' if tag else ''}: "
            f"gc={gc_collect} ({t1 - t0:.2f}s), "
            f"empty_cache={cuda_empty_cache} ({t2 - t1:.2f}s)"
        )


def reset_cuda_state():
    """Full CUDA state reset including memory stats."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        torch.cuda.ipc_collect()
        gc.collect()

