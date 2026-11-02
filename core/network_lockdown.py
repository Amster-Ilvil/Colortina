"""Process-wide outbound network lockdown with explicit model-download windows.

Colortina is a local application: pages are never uploaded.  Normal runtime
blocks all non-loopback connections.  A model download started by an explicit
user confirmation may temporarily restore the original socket functions only
for the duration of that download, then the lock is re-installed immediately.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import socket
import threading

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}
_state_lock = threading.RLock()
_installed = False
_download_thread_state = threading.local()

# Preserve the process originals once.  Older implementations captured these
# inside install_lockdown(), making a confirmed later model download impossible
# after startup locking.
_REAL_SOCKET_CONNECT = socket.socket.connect
_REAL_SOCKET_CONNECT_EX = socket.socket.connect_ex
_REAL_CREATE_CONNECTION = socket.create_connection


class NetworkLockedError(OSError):
    def __init__(self, target: str):
        super().__init__(
            f"网络已锁定：Colortina 拒绝对外连接 ({target})。"
            "模型下载必须由界面的明确确认按钮启动。")


def _is_loopback(address) -> bool:
    try:
        host = address[0] if isinstance(address, tuple) else str(address)
    except Exception:
        return False
    host = str(host).strip("[]").split("%", 1)[0]
    if host in _LOOPBACK_HOSTS or host.startswith("127."):
        return True
    # Unix domain sockets (str/bytes path) are local by definition.  Hostname
    # strings passed to socket.create_connection are handled there separately.
    if isinstance(address, bytes):
        return True
    if isinstance(address, str) and (address.startswith("/") or address.startswith("\0")):
        return True
    return False


def network_allowed() -> bool:
    # A confirmed model download authorizes *only the calling thread*.
    # A global depth counter accidentally allowed all worker threads to
    # transmit data while any download was in progress.
    return (os.environ.get("COLORTINA_ALLOW_NETWORK", "0") == "1" or
            getattr(_download_thread_state, "depth", 0) > 0)


def lockdown_active() -> bool:
    return bool(_installed)


def required_weights_present(cfg=None) -> bool:
    """Check the hard-required model used by the startup policy.

    The denoiser and MangaLine weights are optional features. Treating either
    one as required made an otherwise runnable local installation appear to
    need network access on every launch.
    """
    if cfg is None:
        from config import Config as cfg
    paths = [cfg.GENERATOR_WEIGHTS_PATH]
    for path in paths:
        if not path:
            continue
        try:
            if not os.path.isfile(path) or os.path.getsize(path) < 64 * 1024:
                return False
        except OSError:
            return False
    return True


def apply_startup_policy(cfg=None) -> str:
    if os.environ.get("COLORTINA_ALLOW_NETWORK", "0") == "1":
        return "已按 COLORTINA_ALLOW_NETWORK=1 放开"
    install_lockdown()
    if required_weights_present(cfg):
        return "开启（本地模型就绪；仅确认后的下载线程可联网）"
    return "开启（基础模型缺失；用户确认后才能下载）"


def lock_when_weights_ready(cfg=None) -> bool:
    if os.environ.get("COLORTINA_ALLOW_NETWORK", "0") == "1" or _installed:
        return _installed
    if required_weights_present(cfg):
        install_lockdown()
        print("[colortina] 基础模型权重齐全，网络已自动锁定（完全离线）")
    return _installed


def _guarded_connect(self, address):
    if not network_allowed() and not _is_loopback(address):
        raise NetworkLockedError(repr(address))
    return _REAL_SOCKET_CONNECT(self, address)


def _guarded_connect_ex(self, address):
    if not network_allowed() and not _is_loopback(address):
        raise NetworkLockedError(repr(address))
    return _REAL_SOCKET_CONNECT_EX(self, address)


def _guarded_create_connection(address, *args, **kwargs):
    if not network_allowed() and not _is_loopback(address):
        raise NetworkLockedError(repr(address))
    return _REAL_CREATE_CONNECTION(address, *args, **kwargs)


def install_lockdown() -> bool:
    """Block all non-loopback outbound connections in normal runtime."""
    global _installed
    with _state_lock:
        if os.environ.get("COLORTINA_ALLOW_NETWORK", "0") == "1":
            return False
        if _installed:
            return True
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["DO_NOT_TRACK"] = "1"
        socket.socket.connect = _guarded_connect
        socket.socket.connect_ex = _guarded_connect_ex
        socket.create_connection = _guarded_create_connection
        _installed = True
        return True


def remove_lockdown() -> bool:
    """Restore original sockets. Intended only for confirmed model downloads."""
    global _installed
    with _state_lock:
        if not _installed:
            return False
        socket.socket.connect = _REAL_SOCKET_CONNECT
        socket.socket.connect_ex = _REAL_SOCKET_CONNECT_EX
        socket.create_connection = _REAL_CREATE_CONNECTION
        _installed = False
        return True


@contextmanager
def confirmed_model_download_window(label: str = "模型"):
    """Authorize network connections for only the confirmed download thread.

    Keep the socket guards installed throughout the window so unrelated UI,
    AI and background worker threads remain offline.  Restore environment and
    per-thread authorization even when a download fails.  Nested windows work.
    """
    with _state_lock:
        was_locked = _installed
        previous_env = {
            key: os.environ.get(key)
            for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        }
        _download_thread_state.depth = getattr(_download_thread_state, "depth", 0) + 1
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
        print(f"[colortina] 已按用户确认临时开放模型下载线程网络：{label}")
    try:
        yield
    finally:
        with _state_lock:
            _download_thread_state.depth = max(
                0, getattr(_download_thread_state, "depth", 0) - 1)
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            if was_locked and not _installed and os.environ.get("COLORTINA_ALLOW_NETWORK", "0") != "1":
                install_lockdown()
            print(f"[colortina] 模型下载线程窗口已关闭：{label}")
