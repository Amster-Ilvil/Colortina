"""Colortina — desktop editor entry point.

Run with:
    python main.py
"""

import os
import sys

# Configure MPS before any UI import can initialize torch.
from core.device_runtime import configure_torch_environment
configure_torch_environment()


def _configure_low_disk_runtime():
    """Keep third-party libraries from creating large user-cache folders."""
    os.environ.setdefault("PIP_NO_CACHE_DIR", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    if os.environ.get("COLORTINA_ALLOW_CLIP_DOWNLOAD", "0") != "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


_configure_low_disk_runtime()

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from config import Config
from ui.main_window import MainWindow


def main():
    # 启动时始终禁止外联；模型缺失不会放开全局网络。
    # 仅用户在界面中确认下载后，对该下载线程临时授予网络权限。
    from core.network_lockdown import apply_startup_policy
    print(f"[colortina] 网络锁定: {apply_startup_policy()}")
    if getattr(sys, "frozen", False):
        from core.bundled_models import install_bundled_weights
        seeded = install_bundled_weights(Config.WEIGHTS_DIR)
        if seeded:
            print(f"[colortina] 已从安装包部署 {len(seeded)} 个本地模型文件")
    # ── 降本增效: cap worker threads so long batch jobs stay responsive ──
    # OpenCV and PyTorch both default to using EVERY core, which pegs the
    # whole machine (fans, UI stutter, other apps starved) for marginal
    # speedup on our workload.  Half the cores is the sweet spot.
    half_cores = max(1, (os.cpu_count() or 4) // 2)
    try:
        import cv2
        cv2.setNumThreads(half_cores)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", str(half_cores))
    try:
        import torch
        torch.set_num_threads(half_cores)
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName("Colortina")
    app.setApplicationVersion(Config.APP_VERSION)
    app.setOrganizationName("Colortina")

    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.png")
    if os.path.isfile(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
