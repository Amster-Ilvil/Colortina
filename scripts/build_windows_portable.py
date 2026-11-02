"""Build an offline-ready Windows portable package that runs editable Python sources.

No monolithic EXE, and no model checkpoint is bundled. Python, torch, PySide6
and other runtime dependencies are present in the portable runtime directory.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import sys
import sysconfig
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "Colortina"
DEFAULT_EMBED_VERSION = "3.11.9"
SOURCE_DIRS = ("core", "models", "ui", "vendor", "assets")
SOURCE_FILES = (
    "main.py", "pipeline.py", "config.py",
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
    "LICENSE_ColorComic_MIT.txt", "LICENSE_lbpcascade_animeface_MIT.txt",
)


def version() -> str:
    text = (ROOT / "config.py").read_text(encoding="utf-8")
    match = re.search(r'^\s*APP_VERSION\s*=\s*["\x27]([^"\x27]+)', text, re.M)
    if not match:
        raise RuntimeError("Config.APP_VERSION is missing")
    return match.group(1)


def copy_sources(target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_DIRS:
        shutil.copytree(
            ROOT / name, target / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo",
                                          "*.pth", "*.pt", "*.onnx", "*.zip",
                                          "*.part", ".DS_Store"),
        )
    for name in SOURCE_FILES:
        shutil.copy2(ROOT / name, target / name)


def install_runtime(runtime: Path, *, embedded_zip: Path,
                    site_packages: Path) -> None:
    runtime.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(embedded_zip) as archive:
        archive.extractall(runtime)
    pth_files = list(runtime.glob("python*._pth"))
    if len(pth_files) != 1:
        raise RuntimeError("Expected exactly one Python embeddable ._pth file")
    standard_lib_zip = f"python{sys.version_info.major}{sys.version_info.minor}.zip"
    (runtime / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        site_packages, runtime / "Lib" / "site-packages",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", "*.pth"),
    )
    # The embeddable interpreter uses explicit import paths. Keep source code
    # in app/ editable and dependencies isolated from users' system Python.
    pth_files[0].write_text(
        standard_lib_zip + "\n.\nLib\\site-packages\n..\\app\nimport site\n",
        encoding="utf-8",
    )
    if not (runtime / "python.exe").is_file():
        raise RuntimeError("Embedded python.exe is missing")


def write_launchers(target: Path) -> None:
    offline = r"""@echo off
setlocal EnableExtensions
cd /d "%~dp0"
if not defined COLORTINA_DATA_DIR set "COLORTINA_DATA_DIR=%~dp0"
if not defined COLORTINA_WEIGHTS_DIR set "COLORTINA_WEIGHTS_DIR=%~dp0models\weights"
if not defined COLORTINA_OFFLINE set "COLORTINA_OFFLINE=1"
set "PYTHONNOUSERSITE=1"
set "PYTHONDONTWRITEBYTECODE=1"
"%~dp0runtime\python.exe" "%~dp0app\main.py" %*
if errorlevel 1 (
  echo Colortina failed to start. See WINDOWS_OFFLINE.md.
  pause
)
"""
    online = r"""@echo off
setlocal
rem Online mode still requires confirmation in the user interface.
set "COLORTINA_OFFLINE=0"
call "%~dp0Run_Colortina.bat" %*
"""
    open_dir = r"""@echo off
setlocal
set "MODEL_DIR=%~dp0models\weights"
if not exist "%MODEL_DIR%" mkdir "%MODEL_DIR%"
explorer "%MODEL_DIR%"
"""
    (target / "Run_Colortina.bat").write_text(offline, encoding="utf-8")
    (target / "Run_Colortina_Allow_Download.bat").write_text(
        online, encoding="utf-8")
    (target / "Open_Models_Folder.bat").write_text(
        open_dir, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--python-embed-version", default=DEFAULT_EMBED_VERSION)
    ap.add_argument("--embed-zip", type=Path)
    args = ap.parse_args()
    if sys.platform != "win32" or sys.version_info[:2] != (3, 11):
        raise SystemExit("Build on Windows with Python 3.11 x64")
    if sys.maxsize <= 2**32:
        raise SystemExit("x64 Python is required")
    if args.clean and DIST.exists():
        shutil.rmtree(DIST)
    source_site_packages = Path(sysconfig.get_path("purelib")).resolve()
    if not (source_site_packages / "torch").exists():
        raise SystemExit("Install requirements and CPU PyTorch before building")
    embedded_zip = args.embed_zip
    if embedded_zip is None:
        embedded_zip = ROOT / "build" / "python-embed-amd64.zip"
        embedded_zip.parent.mkdir(parents=True, exist_ok=True)
        if not embedded_zip.exists():
            url = (f"https://www.python.org/ftp/python/"
                   f"{args.python_embed_version}/"
                   f"python-{args.python_embed_version}-embed-amd64.zip")
            print("Downloading CPython embedded distribution:", url, flush=True)
            urllib.request.urlretrieve(url, embedded_zip)
    if not embedded_zip.exists():
        raise SystemExit(f"Embedded Python ZIP not found: {embedded_zip}")

    print("Preparing editable Python sources...", flush=True)
    copy_sources(DIST / "app")
    print("Copying offline runtime dependencies...", flush=True)
    install_runtime(DIST / "runtime", embedded_zip=embedded_zip,
                    site_packages=source_site_packages)
    weights = DIST / "models" / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    (weights / "PLACE_GENERATOR_ZIP_HERE.txt").write_text(
        "Place generator.zip here. Optional: extractor.pth, "
        "denoiser/net_rgb.pth and other feature models.\n"
        "Do NOT unzip generator.zip. Run Run_Colortina.bat offline.\n",
        encoding="utf-8",
    )
    (DIST / "WINDOWS_OFFLINE.md").write_text(
        (ROOT / "docs" / "WINDOWS_OFFLINE.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (DIST / "BUILD-INFO.txt").write_text(
        f"Colortina {version()} - Windows x64 CPU source-layout portable\n"
        f"Git commit: {os.environ.get('GITHUB_SHA', 'local')}\n"
        "Python source: app/ (editable)\n"
        "Python interpreter and dependencies: runtime/\n"
        "Model weights bundled: NO; user-managed models/weights/\n",
        encoding="utf-8",
    )
    write_launchers(DIST)
    print(f"Built Windows portable-source distribution: {DIST}", flush=True)


if __name__ == "__main__":
    main()
