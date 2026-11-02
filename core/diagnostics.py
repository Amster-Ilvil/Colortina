"""Privacy-minimal headless support summary; intentionally excludes file paths."""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def snapshot() -> dict:
    from config import Config
    candidates = {
        "generator": Config.GENERATOR_WEIGHTS_PATH,
        "semantic_head": Config.GENERATOR_SEMANTIC_HEAD_PATH,
        "tunnel2_adapter": Config.TUNNEL2_SEMANTIC_ADAPTER_PATH,
        "denoiser": str(Path(Config.DENOISER_WEIGHTS_DIR) / "net_rgb.pth"),
    }
    weights = {}
    for label, value in candidates.items():
        try:
            path = Path(value)
            weights[label] = {"present": path.is_file(),
                              "bytes": path.stat().st_size if path.is_file() else 0}
        except OSError:
            weights[label] = {"present": False, "bytes": 0}
    return {"app_version": Config.APP_VERSION_LABEL,
            "python": platform.python_version(),
            "platform": sys.platform, "architecture": platform.machine(),
            "weights": weights,
            "note": "No personal file paths, page pixels, projects or model bytes included."}


def export_bundle(path: str) -> str:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as z:
        z.writestr("diagnostics.json", json.dumps(snapshot(), ensure_ascii=False, indent=2))
        z.writestr("README.txt", "Local Colortina workspace diagnostics (sanitized).\n"
                   "This bundle intentionally contains no raw logs, image content, "
                   "model weights or local paths.\n")
    return str(output)
