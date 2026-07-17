"""Colortina — desktop editor entry point.

Run with:
    python main.py
"""

import os
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main():
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
    app.setOrganizationName("Colortina")

    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.png")
    if os.path.isfile(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
