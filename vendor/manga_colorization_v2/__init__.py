"""External integration point for qweasdd/manga-colorization-v2.

The upstream repository currently does not publish an explicit software license.
To avoid redistributing upstream source without a clear grant, Colortina does not
vendor that source in this public repository.

Users who have obtained an upstream checkout directly and have the right to use
it can point Colortina at the checkout with COLORTINA_MANGA_COLORIZATION_V2_PATH.
A few conventional per-user locations are also checked.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

UPSTREAM_PATH: Path | None = None


def _candidates():
    env = os.environ.get("COLORTINA_MANGA_COLORIZATION_V2_PATH", "").strip()
    if env:
        yield Path(env).expanduser()

    home = Path.home()
    yield home / ".colortina" / "upstream" / "manga-colorization-v2"

    if sys.platform == "darwin":
        yield home / "Library" / "Application Support" / "Colortina" / "upstream" / "manga-colorization-v2"
    elif os.name == "nt":
        local = os.environ.get("LOCALAPPDATA", "").strip()
        if local:
            yield Path(local) / "Colortina" / "upstream" / "manga-colorization-v2"
    else:
        xdg = os.environ.get("XDG_DATA_HOME", "").strip()
        base = Path(xdg).expanduser() if xdg else home / ".local" / "share"
        yield base / "Colortina" / "upstream" / "manga-colorization-v2"


for _root in _candidates():
    try:
        _root = _root.resolve()
    except OSError:
        continue
    if (_root / "colorizator.py").is_file() and (_root / "networks").is_dir():
        # Extend this package's search path so existing imports such as
        # vendor.manga_colorization_v2.colorizator resolve to the user-provided
        # upstream checkout rather than a redistributed copy in this repository.
        __path__.append(str(_root))
        UPSTREAM_PATH = _root
        break
