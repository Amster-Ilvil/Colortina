"""Color director — decides the book's palette (fully local).

Produces a "color script": one hex color per semantic region class
(skin, hair, metal, sky, ...) from a curated static palette.  One call
per job so the palette stays stable across the whole book.  No network
access, no external APIs.
"""


# Curated defaults — natural manhwa-leaning material colors (hex RGB).
DEFAULT_PALETTE: dict[str, str] = {
    "skin":               "#f0c8a0",
    "hair":               "#6b4a32",
    "eyes":               "#4f7397",
    "clothing_primary":   "#5b7fa6",
    "clothing_secondary": "#a65e50",
    "clothing_accent":    "#c9a227",
    "metal":              "#9aa2ab",
    "wood":               "#8a6a48",
    "sky":                "#a5c6e8",
    "foliage":            "#6f9e5f",
    "stone":              "#8f8f92",
    "water":              "#6f9ec8",
    "fire":               "#e8862e",
    "background":         "#d9cfc0",
}


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    s = hex_color.lstrip("#")
    return int(s[4:6], 16), int(s[2:4], 16), int(s[0:2], 16)


class ColorDirector:
    """Builds the per-job color script from the static palette."""

    def __init__(self, config=None):
        self._cfg = config

    def build_script(self, tag_summary: dict | None = None) -> dict:
        """Return {"palette": {...}, "mood": str, "source": "static"}."""
        return {"palette": dict(DEFAULT_PALETTE),
                "mood": "neutral daylight", "source": "static"}
