"""Hint Manager — unifies auto-generated and manually-drawn color hints.

mc-v2's native hint channel takes a flat list of sparse points:
    HintPoint = (x_norm, y_norm, (r, g, b))

ColorComic's GuidedColorist produces "auto" points (priority 0).
The editor (ported from MangaColorer) will produce "manual" points the
user draws with a brush/eyedropper (priority 100).

Merge rule: manual always wins. Any auto point that falls within
`suppress_radius_norm` of a manual point is dropped, so the user's
brush stroke isn't fighting an auto-hint sitting right underneath it.
Manual points are always kept in full.
"""

from __future__ import annotations

from dataclasses import dataclass, field

HintPoint = tuple[float, float, tuple[int, int, int], float]
# (x_norm, y_norm, (r, g, b), radius_norm)
# radius_norm is the dot radius as a fraction of image width — this is
# what makes brush size actually reach the model. Without it, every hint
# collapses to a fixed tiny dot regardless of how big a brush the user drew.

_DEFAULT_AUTO_RADIUS_NORM = 0.006  # ~4px at mc-v2's default 768px working size


@dataclass
class Hint:
    """A single color hint with provenance."""
    x_norm: float
    y_norm: float
    color: tuple[int, int, int]  # (r, g, b)
    radius_norm: float = _DEFAULT_AUTO_RADIUS_NORM
    priority: int = 0            # 0 = auto, 100 = manual

    def as_point(self) -> HintPoint:
        return (self.x_norm, self.y_norm, self.color, self.radius_norm)


@dataclass
class HintManager:
    """Holds one page's auto + manual hints and merges them on demand."""
    auto_hints: list[Hint] = field(default_factory=list)
    manual_hints: list[Hint] = field(default_factory=list)

    # ── Population ─────────────────────────────────────────────────────

    def set_auto_hints(self, points: list) -> None:
        """Replace the auto layer (e.g. after GuidedColorist.hints_for_page).

        Accepts either 4-tuples (x, y, rgb, radius_norm) or legacy 3-tuples
        (x, y, rgb) — the latter get the default auto-hint radius.
        """
        hints = []
        for p in points:
            if len(p) == 4:
                x, y, c, r = p
            else:
                x, y, c = p
                r = _DEFAULT_AUTO_RADIUS_NORM
            hints.append(Hint(x, y, c, radius_norm=r, priority=0))
        self.auto_hints = hints

    def add_manual_hint(self, x_norm: float, y_norm: float,
                         color: tuple[int, int, int],
                         radius_norm: float = 0.015) -> None:
        """Add one manually-drawn hint point (brush dab, eyedropper, etc.).

        `radius_norm` should come straight from the UI's brush-size control
        (converted to a fraction of image width) so the drawn dot size
        actually reaches the model.
        """
        self.manual_hints.append(
            Hint(x_norm, y_norm, color, radius_norm=radius_norm, priority=100))

    def clear_manual_hints(self) -> None:
        self.manual_hints = []

    def undo_last_manual(self) -> None:
        if self.manual_hints:
            self.manual_hints.pop()

    # ── Merge ─────────────────────────────────────────────────────────

    def merge(self, suppress_radius_norm: float = 0.02) -> list[HintPoint]:
        """Return the final point list to feed into MangaColorizer.colorize().

        Manual hints are always included. Auto hints are dropped if they
        fall within `suppress_radius_norm` (normalized page-diagonal-ish
        distance) of any manual hint, so hand-drawn corrections aren't
        immediately overridden by the automatic layer sitting underneath.
        """
        manual_pts = [h.as_point() for h in self.manual_hints]
        if not self.manual_hints:
            return [h.as_point() for h in self.auto_hints]

        kept_auto: list[HintPoint] = []
        r2 = suppress_radius_norm * suppress_radius_norm
        for a in self.auto_hints:
            suppressed = False
            for m in self.manual_hints:
                dx = a.x_norm - m.x_norm
                dy = a.y_norm - m.y_norm
                if dx * dx + dy * dy <= r2:
                    suppressed = True
                    break
            if not suppressed:
                kept_auto.append(a.as_point())

        return kept_auto + manual_pts

    def reset(self) -> None:
        self.auto_hints = []
        self.manual_hints = []
