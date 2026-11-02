"""Cooperative cancellation helpers for long-running Colortina jobs.

Neural-network kernels themselves cannot be safely killed from another Python
thread.  Instead, workers request cancellation and every expensive pipeline
stage checks the request before continuing.  This prevents new pages/tiles,
post-processing and late UI writes after the user presses Stop, without using
unsafe ``QThread.terminate()`` while PyTorch owns accelerator locks.
"""

from __future__ import annotations


class OperationCancelled(RuntimeError):
    """Raised internally when a cooperative cancellation request is observed."""


def is_cancelled(cancel_check=None) -> bool:
    if cancel_check is None:
        return False
    try:
        return bool(cancel_check())
    except Exception:
        # A broken observer must never crash the colorization pipeline.
        return False


def check_cancelled(cancel_check=None) -> None:
    if is_cancelled(cancel_check):
        raise OperationCancelled("operation cancelled")
