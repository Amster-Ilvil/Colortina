"""Explicit page analysis state passed through the colour pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.hint_spec import HintSpec


@dataclass
class PageColorContext:
    segmentation: object | None = None
    semantic_labels: list = field(default_factory=list)
    character_instances: list = field(default_factory=list)
    identity_assignments: dict = field(default_factory=dict)
    hints: list[HintSpec] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
