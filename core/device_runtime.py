"""Central accelerator detection and macOS 27 MPS compatibility helpers.

This module deliberately avoids importing torch at module import time.  Call
``configure_torch_environment`` before the first torch import, then use
``probe_torch_device`` / ``resolve_torch_device`` after torch is available.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import os
import platform
import re
import subprocess
import sys
from typing import Any


def configure_torch_environment() -> None:
    """Set conservative MPS variables before PyTorch initializes Metal."""
    # Unsupported operations may fall back to CPU rather than aborting the job.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Keep correctness-oriented defaults.  Users may explicitly override these.
    os.environ.setdefault("PYTORCH_MPS_FAST_MATH", "0")
    # Avoid accidental fp16/bf16 defaults from wrappers; Colortina runs mc-v2 fp32.
    os.environ.setdefault("COLORTINA_MPS_FP32", "1")


def _macos_version() -> str:
    value = platform.mac_ver()[0]
    if value:
        return value
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["/usr/bin/sw_vers", "-productVersion"], text=True,
                stderr=subprocess.DEVNULL).strip()
        except Exception:
            pass
    return ""


def _version_tuple(value: str, parts: int = 3) -> tuple[int, ...]:
    nums = [int(v) for v in re.findall(r"\d+", str(value or ""))[:parts]]
    return tuple(nums + [0] * (parts - len(nums)))


def _torch_version_tuple(torch_module: Any) -> tuple[int, int, int]:
    return _version_tuple(getattr(torch_module, "__version__", "0"), 3)  # type: ignore[return-value]


@dataclass(frozen=True)
class DeviceReport:
    requested: str
    selected: str
    usable: bool
    reason: str
    system: str
    architecture: str
    macos_version: str
    python_version: str
    torch_version: str
    mps_built: bool = False
    mps_available: bool = False
    mps_smoke_ok: bool = False
    cuda_available: bool = False
    rosetta: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def short_reason(self) -> str:
        return self.reason.strip() or "可用"


def _is_rosetta() -> bool:
    if sys.platform != "darwin":
        return False
    try:
        out = subprocess.check_output(
            ["/usr/sbin/sysctl", "-in", "sysctl.proc_translated"],
            text=True, stderr=subprocess.DEVNULL).strip()
        return out == "1"
    except Exception:
        return False


def _mps_smoke_test(torch_module: Any) -> tuple[bool, str]:
    """Run a tiny fp32 conv and synchronization, close to mc-v2's core ops."""
    try:
        import torch.nn.functional as F
        with torch_module.inference_mode():
            x = torch_module.arange(64, dtype=torch_module.float32,
                                    device="mps").reshape(1, 1, 8, 8) / 64.0
            weight = torch_module.ones((1, 1, 3, 3), dtype=torch_module.float32,
                                       device="mps") / 9.0
            y = F.conv2d(x, weight, padding=1)
            value = float(y.sum().detach().cpu().item())
        mps_ns = getattr(torch_module, "mps", None)
        if mps_ns is not None and hasattr(mps_ns, "synchronize"):
            mps_ns.synchronize()
        if not (value == value and abs(value) < 1e9):
            return False, "MPS 冒烟测试返回异常数值"
        return True, ""
    except Exception as exc:
        return False, f"MPS 冒烟测试失败：{type(exc).__name__}: {exc}"


def probe_torch_device(requested: str = "auto", *, torch_module: Any = None,
                       smoke_test: bool = True) -> DeviceReport:
    """Resolve CUDA/MPS/CPU and explain why an accelerator was or was not used."""
    configure_torch_environment()
    if torch_module is None:
        import torch as torch_module

    requested = str(requested or "auto").strip().lower()
    if requested not in {"auto", "cpu", "mps", "cuda"}:
        requested = "auto"

    system = platform.system()
    architecture = platform.machine().lower()
    macos = _macos_version()
    rosetta = _is_rosetta()
    torch_version = str(getattr(torch_module, "__version__", "unknown"))
    cuda_available = bool(getattr(torch_module, "cuda", None) and
                          torch_module.cuda.is_available())

    if requested == "cpu":
        return DeviceReport(requested, "cpu", True, "用户指定 CPU", system,
                            architecture, macos, platform.python_version(),
                            torch_version, cuda_available=cuda_available,
                            rosetta=rosetta)

    if requested in {"auto", "cuda"} and cuda_available:
        return DeviceReport(requested, "cuda", True, "CUDA 可用", system,
                            architecture, macos, platform.python_version(),
                            torch_version, cuda_available=True, rosetta=rosetta)
    if requested == "cuda":
        return DeviceReport(requested, "cpu", False, "CUDA 不可用，已回退 CPU",
                            system, architecture, macos, platform.python_version(),
                            torch_version, cuda_available=False, rosetta=rosetta)

    backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    try:
        built = bool(backend is not None and backend.is_built())
    except Exception:
        built = False
    try:
        available = bool(backend is not None and backend.is_available())
    except Exception:
        available = False

    reasons: list[str] = []
    if system != "Darwin":
        reasons.append("当前不是 macOS")
    if architecture != "arm64":
        if rosetta:
            reasons.append("Python 正通过 Rosetta/x86_64 运行，必须改用 arm64 Python")
        else:
            reasons.append("当前 Python 不是 arm64；官方 MPS 运行环境要求 Apple Silicon")
    if not built:
        reasons.append("当前 PyTorch 未编译 MPS 后端")
    if not available:
        reasons.append("PyTorch 报告 MPS 设备不可用")

    # macOS 27 had an upstream MPS correctness regression fixed in torch 2.13.
    mac_major = _version_tuple(macos, 1)[0]
    if mac_major >= 27 and _torch_version_tuple(torch_module) < (2, 13, 0):
        reasons.append("macOS 27 必须使用 PyTorch 2.13.0 或更新版本")

    smoke_ok = False
    if not reasons and smoke_test:
        smoke_ok, smoke_reason = _mps_smoke_test(torch_module)
        if not smoke_ok:
            reasons.append(smoke_reason)
    elif not reasons:
        smoke_ok = True

    if not reasons:
        return DeviceReport(requested, "mps", True, "MPS 可用（fp32）", system,
                            architecture, macos, platform.python_version(),
                            torch_version, built, available, smoke_ok,
                            cuda_available, rosetta)

    reason = "；".join(reasons) + "，已回退 CPU"
    return DeviceReport(requested, "cpu", False, reason, system, architecture,
                        macos, platform.python_version(), torch_version,
                        built, available, smoke_ok, cuda_available, rosetta)


def resolve_torch_device(requested: str = "auto", *, torch_module: Any = None,
                         smoke_test: bool = True):
    """Return ``(torch.device, DeviceReport)``."""
    configure_torch_environment()
    if torch_module is None:
        import torch as torch_module
    report = probe_torch_device(requested, torch_module=torch_module,
                                smoke_test=smoke_test)
    return torch_module.device(report.selected), report
