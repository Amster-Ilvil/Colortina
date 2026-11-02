"""Minimal local, opt-in diagnostics logging for a source-only workspace.

Does not automatically capture page images, project content or model tensors.
"""
from __future__ import annotations

import logging
from pathlib import Path
import os


def _logs_dir() -> Path:
    # Avoid forcing users to keep growing logs inside the repository.
    return Path(os.environ.get("COLORTINA_LOGS_DIR", "~/.colortina/logs")).expanduser()


def log_paths() -> dict[str, str]:
    return {"text": str(_logs_dir() / "colortina.log")}


def setup_logging(app_version: str = "") -> None:
    folder = _logs_dir()
    folder.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("colortina")
    if any(getattr(h, "_colortina_local", False) for h in logger.handlers):
        return
    from logging.handlers import RotatingFileHandler
    handler = RotatingFileHandler(folder / "colortina.log", maxBytes=1_000_000,
                                  backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    handler._colortina_local = True
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info("Diagnostic CLI initialized; version=%s", app_version)
