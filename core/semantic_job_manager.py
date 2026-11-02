"""Bounded semantic-analysis state/cache manager.

Desktop-only adaptation of the useful userscript patterns we reviewed:
- bounded concurrency instead of unbounded image jobs;
- deduplicate identical in-flight requests;
- small LRU result cache;
- explicit per-key states for diagnostics.

There is deliberately no browser/web code here.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, is_dataclass, fields
import copy
import threading
import time
from typing import Callable, Hashable, Any


@dataclass(frozen=True)
class SemanticJobState:
    key: Hashable
    status: str
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None


def _approx_bytes(value, _seen=None) -> int:
    """Conservative size estimate focused on NumPy-backed mask bundles."""
    seen = _seen if _seen is not None else set()
    oid = id(value)
    if oid in seen:
        return 0
    seen.add(oid)
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
    except Exception:
        pass
    if isinstance(value, dict):
        return sum(_approx_bytes(k, seen) + _approx_bytes(v, seen) for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return sum(_approx_bytes(v, seen) for v in value)
    if is_dataclass(value) and not isinstance(value, type):
        return sum(_approx_bytes(getattr(value, f.name), seen) for f in fields(value))
    return 0


class SemanticJobManager:
    def __init__(self, *, max_workers: int = 2, max_cache: int = 6,
                 max_cache_bytes: int = 96 * 1024 * 1024):
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)),
                                            thread_name_prefix="colortina-semantic")
        self._max_cache = max(1, int(max_cache))
        self._max_cache_bytes = max(1, int(max_cache_bytes))
        self._cache: OrderedDict[Hashable, Any] = OrderedDict()
        self._cache_sizes: dict[Hashable, int] = {}
        self._cache_bytes = 0
        self._inflight: dict[Hashable, Future] = {}
        self._states: dict[Hashable, SemanticJobState] = {}
        self._lock = threading.RLock()

    def _cache_get(self, key):
        with self._lock:
            if key not in self._cache:
                return None
            value = self._cache.pop(key)
            self._cache[key] = value
            return copy.deepcopy(value)

    def _cache_put(self, key, value):
        with self._lock:
            copied = copy.deepcopy(value)
            size = max(1, _approx_bytes(copied))
            if key in self._cache:
                self._cache.pop(key, None)
                self._cache_bytes -= self._cache_sizes.pop(key, 0)
            self._cache[key] = copied
            self._cache_sizes[key] = size
            self._cache_bytes += size
            self._cache.move_to_end(key)
            while (len(self._cache) > self._max_cache or
                   (self._cache_bytes > self._max_cache_bytes and len(self._cache) > 1)):
                old_key, _old_value = self._cache.popitem(last=False)
                self._cache_bytes -= self._cache_sizes.pop(old_key, 0)

    def get_or_compute(self, key: Hashable, fn: Callable[[], Any]) -> Any:
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        with self._lock:
            future = self._inflight.get(key)
            if future is None:
                started = time.time()
                self._states[key] = SemanticJobState(key, "running", started_at=started)

                def run():
                    try:
                        value = fn()
                        self._cache_put(key, value)
                        with self._lock:
                            self._states[key] = SemanticJobState(
                                key, "done", started_at=started, finished_at=time.time())
                        return value
                    except Exception as exc:
                        with self._lock:
                            self._states[key] = SemanticJobState(
                                key, "error", started_at=started, finished_at=time.time(),
                                error=f"{type(exc).__name__}: {exc}")
                        raise
                    finally:
                        with self._lock:
                            self._inflight.pop(key, None)

                future = self._executor.submit(run)
                self._inflight[key] = future
        return copy.deepcopy(future.result())

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._cache_sizes.clear()
            self._cache_bytes = 0
            self._states.clear()

    def state(self, key: Hashable) -> SemanticJobState | None:
        with self._lock:
            return self._states.get(key)


semantic_jobs = SemanticJobManager()
