"""Copy signed/read-only bundled model assets to the writable model cache.

For macOS .app releases, shipped weights under PyInstaller _MEIPASS or
Contents/Resources cannot be updated in place. This helper copies only files
we actually ship, never downloads, and never overwrites user models.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import tempfile

_BUNDLED_NAMES = (
    "generator.zip", "extractor.pth", "region_control_adapter.pth",
    "generator_semantic_head.pth", "tunnel2_semantic_color_adapter.pth",
    "denoiser/net_rgb.pth", "erika.pth", "isnetis.onnx",
    "comictextdetector.pt.onnx", "realesrgan_anime6b.pth",
)


def _candidate_asset_dirs() -> list[Path]:
    roots: list[Path] = []
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            roots.append(Path(meipass))
        exe = Path(sys.executable).resolve()
        roots.append(exe.parent.parent / "Resources")
        roots.append(exe.parent)
    return [root / "models" / "weights" for root in roots]


def install_bundled_weights(destination: str | Path, *, sources=None) -> list[str]:
    """Atomic first-run seed; never overwrite newer user-supplied weights.

    Actual checkpoint authenticity still requires a signed release manifest;
    this routine only addresses immutability of packaged resources.
    """
    target = Path(destination).expanduser().resolve()
    dirs = [Path(p).resolve() for p in (sources or _candidate_asset_dirs())]
    installed: list[str] = []
    for relative in _BUNDLED_NAMES:
        dest = target / relative
        if dest.is_file():
            continue
        src = next((p / relative for p in dirs if (p / relative).is_file()), None)
        if src is None or src.resolve() == dest:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=dest.name + ".", suffix=".part",
                                    dir=dest.parent)
        os.close(fd)
        try:
            shutil.copyfile(src, name)
            # Recheck if another thread populated it while copying.
            if not dest.exists():
                os.replace(name, dest)
                installed.append(relative)
        finally:
            if os.path.exists(name):
                os.unlink(name)
    return installed
