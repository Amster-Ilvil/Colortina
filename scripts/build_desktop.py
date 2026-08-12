from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
BUILD = ROOT / "build"
APP_NAME = "Colortina"
BUNDLE_ID = "io.github.amsterilvil.colortina"


def run(args: list[str]) -> None:
    print("+", " ".join(str(x) for x in args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def read_version() -> str:
    scope: dict[str, object] = {}
    config_text = (ROOT / "config.py").read_text(encoding="utf-8")
    # Avoid importing the application just to read the release version.
    import re

    m = re.search(r'^\s*APP_VERSION\s*=\s*["\']([^"\']+)["\']', config_text, re.MULTILINE)
    if not m:
        raise SystemExit("Could not find Config.APP_VERSION in config.py")
    return m.group(1)


def add_data_args() -> list[str]:
    sep = ";" if os.name == "nt" else ":"
    result: list[str] = []

    assets = ROOT / "assets"
    if assets.is_dir():
        result += ["--add-data", f"{assets}{sep}assets"]

    for filename in (
        "THIRD_PARTY_NOTICES.md",
        "LICENSE_ColorComic_MIT.txt",
        "LICENSE_lbpcascade_animeface_MIT.txt",
        "README.md",
    ):
        path = ROOT / filename
        if path.is_file():
            result += ["--add-data", f"{path}{sep}."]
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Build standalone Colortina desktop distribution")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    system = platform.system()
    if system not in {"Windows", "Darwin", "Linux"}:
        raise SystemExit(f"Unsupported packaging platform: {system}")

    if args.clean:
        shutil.rmtree(DIST, ignore_errors=True)
        shutil.rmtree(BUILD, ignore_errors=True)

    version = read_version()
    print(f"Building {APP_NAME} {version} for {system} {platform.machine()}", flush=True)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name",
        APP_NAME,
        "--exclude-module",
        "pytest",
        "--collect-all",
        "transformers",
        "--collect-all",
        "timm",
        "--collect-all",
        "onnxruntime",
        "--collect-all",
        "gdown",
        *add_data_args(),
    ]

    icon = ROOT / "assets" / "icon.png"
    if icon.is_file():
        cmd += ["--icon", str(icon)]

    if system == "Darwin":
        cmd += ["--osx-bundle-identifier", BUNDLE_ID]

    cmd += [str(ROOT / "main.py")]
    run(cmd)

    if system == "Darwin":
        app = DIST / f"{APP_NAME}.app"
        if not app.is_dir():
            raise SystemExit(f"Expected app bundle not found: {app}")
        plist = app / "Contents" / "Info.plist"
        if plist.exists():
            subprocess.run(
                ["/usr/libexec/PlistBuddy", "-c", f"Set :CFBundleShortVersionString {version}", str(plist)],
                check=False,
            )
            subprocess.run(
                ["/usr/libexec/PlistBuddy", "-c", f"Set :CFBundleVersion {version}", str(plist)],
                check=False,
            )
            subprocess.run(["plutil", "-lint", str(plist)], check=True)
    else:
        app_dir = DIST / APP_NAME
        if not app_dir.is_dir():
            raise SystemExit(f"Expected distribution directory not found: {app_dir}")

    print(f"Build complete: {DIST}")


if __name__ == "__main__":
    main()
