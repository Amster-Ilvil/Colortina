"""Explicit schema upgrades for Colortina artifact files."""

from __future__ import annotations


def migrate_ccstyle(data: dict) -> tuple[dict, list[str]]:
    out = dict(data or {})
    version = int(out.get("version", 1) or 1)
    notes: list[str] = []
    if version < 4:
        out["version"] = 4
        out.setdefault("style_scope", {
            "character_rendering": True,
            "background_rendering": True,
            "global_ambience": 0.20,
        })
        out.setdefault("revision", 0)
        notes.append(f"ccstyle v{version} -> v4: absolute reference palette retained for preview only")
    return out, notes


def migrate_ccpalette(data: dict) -> tuple[dict, list[str]]:
    out = dict(data or {})
    version = int(out.get("version", 1) or 1)
    notes: list[str] = []
    if version < 4:
        out["version"] = 4
        out.setdefault("min_match_score", 0.48)
        out.setdefault("min_margin", 0.055)
        out.setdefault("min_semantic_conf", 0.32)
        out.setdefault("revision", 0)
        notes.append(f"ccpalette v{version} -> v4: confidence and ambiguity gates added")
    if int(out.get("version", 4) or 4) < 5:
        for character in out.get("characters", []):
            colors = dict(character.get("colors", {}) or {})
            slots = dict(character.get("color_slots", {}) or {})
            for attr, value in colors.items():
                if attr not in slots and isinstance(value, str):
                    slots[attr] = [value.lower()]
            character["color_slots"] = slots
        out["version"] = 5
        notes.append("ccpalette v4 -> v5: multi-slot clothing palette added")
    return out, notes


def migrate_ccproject(data: dict) -> tuple[dict, list[str]]:
    out = dict(data or {})
    version = int(out.get("version", 1) or 1)
    notes: list[str] = []
    if version < 2:
        out["version"] = 2
        out.setdefault("scene_palette", None)
        for page in out.get("pages", []):
            page.setdefault("diagnostics", {})
            page.setdefault("forced_character_matches", {})
            page.setdefault("diagnostics_file", None)
        notes.append(f"ccproject v{version} -> v2: scene palette and page diagnostics added")
    return out, notes
