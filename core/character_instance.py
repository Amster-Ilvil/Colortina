"""Page-local character instances.

This lightweight v4 implementation still uses semantic regions as candidates,
but it no longer exposes a naked hair region as the complete identity object.
The structure is intentionally ready for a future dedicated face detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CharacterInstance:
    instance_id: str
    panel_id: int | None = None
    head_bbox: tuple[int, int, int, int] | None = None
    body_bbox: tuple[int, int, int, int] | None = None
    face_embedding: list[float] = field(default_factory=list)
    lineart_embedding: list[float] = field(default_factory=list)
    hair_regions: list[int] = field(default_factory=list)
    skin_regions: list[int] = field(default_factory=list)
    eye_regions: list[int] = field(default_factory=list)
    clothing_regions: list[int] = field(default_factory=list)
    clothing_parts: dict[int, str] = field(default_factory=dict)
    confidence: float = 0.0
    matched_character_id: int | None = None
    top1_score: float | None = None
    top2_score: float | None = None
    margin: float = 0.0
    lock_allowed: bool = False
