"""Process-wide outbound network lockdown.

Colortina 是纯本地应用：模型权重在本地、推理在本地、页面从不上传。
本模块在 socket 层拦截一切对外连接（仅放行本机回环地址），任何依赖库
（huggingface_hub / transformers / gdown / requests / urllib …）都无法
绕过。这是硬性保证，而不是逐个库设置开关。

默认锁死。只有显式设置环境变量 COLORTINA_ALLOW_NETWORK=1 启动时才放开
（例如在新机器上首次下载模型权重时），用完即恢复默认。
"""
from __future__ import annotations

import os
import socket

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}
_installed = False


class NetworkLockedError(OSError):
    def __init__(self, target: str):
        super().__init__(
            f"网络已锁定：Colortina 拒绝对外连接 ({target})。"
            "如需临时联网下载模型，请用 COLORTINA_ALLOW_NETWORK=1 启动。")


def _is_loopback(address) -> bool:
    try:
        host = address[0] if isinstance(address, tuple) else str(address)
    except Exception:
        return False
    host = str(host).strip("[]").split("%", 1)[0]
    if host in _LOOPBACK_HOSTS:
        return True
    if host.startswith("127."):
        return True
    # Unix domain sockets (str/bytes path) are local by definition.
    if isinstance(address, (str, bytes)):
        return True
    return False


def network_allowed() -> bool:
    return os.environ.get("COLORTINA_ALLOW_NETWORK", "0") == "1"


def required_weights_present(cfg=None) -> bool:
    """Check the model weights the app needs for its features.

    Missing weights are the ONLY sanctioned reason to touch the network:
    generator (mc-v2), the shared FFDNet denoiser, and the
    MangaLineExtraction erika model used by rectangle closed-region fill.
    """
    if cfg is None:
        from config import Config as cfg
    paths = [
        cfg.GENERATOR_WEIGHTS_PATH,
        os.path.join(cfg.DENOISER_WEIGHTS_DIR, "net_rgb.pth"),
        getattr(cfg, "MANGA_LINE_WEIGHTS_PATH", ""),
    ]
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
    """没有模型权重才允许联网下载，其它情况一律锁死。

    Returns a short human-readable state string for the startup banner.
    """
    if network_allowed():
        return "已按 COLORTINA_ALLOW_NETWORK=1 放开"
    if required_weights_present(cfg):
        install_lockdown()
        return "开启（完全离线）"
    return "临时放开：检测到模型权重缺失，仅本次允许下载；下载完成后自动锁定"


def lock_when_weights_ready(cfg=None) -> bool:
    """Call after any weight download completes: lock as soon as possible."""
    if network_allowed() or _installed:
        return _installed
    if required_weights_present(cfg):
        install_lockdown()
        print("[colortina] 模型权重齐全，网络已自动锁定（完全离线）")
    return _installed


def install_lockdown() -> bool:
    """Block all non-loopback outbound connections for this process.

    Returns True when the lock is active, False when the user explicitly
    allowed network access via COLORTINA_ALLOW_NETWORK=1.
    """
    global _installed
    if network_allowed():
        return False
    if _installed:
        return True

    # 让各库“自觉”离线，避免它们在被 socket 拦截前反复重试等待。
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["NO_PROXY"] = "*"

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection

    def guarded_connect(self, address):
        if not _is_loopback(address):
            raise NetworkLockedError(repr(address))
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        if not _is_loopback(address):
            raise NetworkLockedError(repr(address))
        return real_connect_ex(self, address)

    def guarded_create_connection(address, *args, **kwargs):
        if not _is_loopback(address):
            raise NetworkLockedError(repr(address))
        return real_create_connection(address, *args, **kwargs)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.create_connection = guarded_create_connection
    _installed = True
    return True
