"""Persistence for Colortina project sessions (.ccproject)."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict


from core.character_library import CharacterLibrary
from core.character_memory import CharacterMemory
from core.hint_manager import HintManager
from core.imageio import imwrite
from core.style_descriptor import StyleDescriptor
from core.style_engine import StyleProfile, _descriptor_to_profile
from core.scene_palette import ScenePalette

PROJECT_VERSION = 3


def _style_to_dict(profile) -> dict | None:
    if profile is None:
        return None
    descriptor = getattr(profile, "_descriptor", None)
    if descriptor is not None:
        return {"kind": "descriptor", "data": descriptor._to_json()}
    data = asdict(profile)
    data.pop("_descriptor", None)
    return {"kind": "profile", "data": data}


def style_from_dict(payload: dict | None):
    if not payload:
        return None
    data = payload.get("data", {})
    if payload.get("kind") == "descriptor":
        desc = StyleDescriptor._from_dict(data)
        profile = _descriptor_to_profile(desc)
        profile._descriptor = desc
        return profile
    known = {k: v for k, v in data.items()
             if k in StyleProfile.__dataclass_fields__ and k != "_descriptor"}
    return StyleProfile(**known)


def save_project(path: str, *, pages: list, style_profile=None,
                 character_library: CharacterLibrary | None = None,
                 character_memories: dict | None = None,
                 scene_palette: ScenePalette | None = None,
                 settings: dict | None = None,
                 semantic_scribbles: dict | None = None) -> str:
    if not path.endswith(".ccproject"):
        path += ".ccproject"
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    asset_dir = os.path.splitext(path)[0] + "_assets"
    os.makedirs(asset_dir, exist_ok=True)

    page_records = []
    for index, state in enumerate(pages):
        record = {
            "path": os.path.abspath(state.path),
            "hints": state.hint_manager.to_dict(),
            "ai_result": None,
            "result": None,
            "filter_base": None,
            "diagnostics": dict(getattr(state, "pipeline_diagnostics", {}) or
                                getattr(state.hint_manager, "last_diagnostics", {}) or {}),
            "forced_character_matches": {
                str(k): int(v) for k, v in
                (getattr(state, "forced_character_matches", {}) or {}).items()
            },
            "diagnostics_file": None,
            "semantic_hair_positive": None,
            "semantic_hair_negative": None,
        }
        if state.ai_result_bgr is not None:
            name = f"page_{index:04d}_ai.png"
            imwrite(os.path.join(asset_dir, name), state.ai_result_bgr)
            record["ai_result"] = os.path.relpath(os.path.join(asset_dir, name),
                                                   os.path.dirname(path))
        if state.result_bgr is not None:
            name = f"page_{index:04d}_edited.png"
            imwrite(os.path.join(asset_dir, name), state.result_bgr)
            record["result"] = os.path.relpath(os.path.join(asset_dir, name),
                                                os.path.dirname(path))
        if getattr(state, "filter_base_bgr", None) is not None:
            name = f"page_{index:04d}_filter_base.png"
            imwrite(os.path.join(asset_dir, name), state.filter_base_bgr)
            record["filter_base"] = os.path.relpath(
                os.path.join(asset_dir, name), os.path.dirname(path))
        scribbles = dict((semantic_scribbles or {}).get(os.path.abspath(state.path), {}) or {})
        for semantic_key, record_key in (("hair_positive", "semantic_hair_positive"),
                                         ("hair_negative", "semantic_hair_negative")):
            mask = scribbles.get(semantic_key)
            if mask is None:
                continue
            import numpy as np
            arr = np.asarray(mask)
            if arr.ndim == 3:
                arr = np.max(arr, axis=2)
            arr = np.where(arr > 0, 255, 0).astype(np.uint8)
            if not np.any(arr):
                continue
            name = f"page_{index:04d}_{semantic_key}.png"
            mask_path = os.path.join(asset_dir, name)
            imwrite(mask_path, arr)
            record[record_key] = os.path.relpath(mask_path, os.path.dirname(path))

        if record["diagnostics"]:
            diag_name = f"page_{index:04d}.diagnostics.json"
            diag_path = os.path.join(asset_dir, diag_name)
            with open(diag_path, "w", encoding="utf-8") as diag_file:
                json.dump(record["diagnostics"], diag_file,
                          ensure_ascii=False, indent=2)
            record["diagnostics_file"] = os.path.relpath(
                diag_path, os.path.dirname(path))
        page_records.append(record)
        # Pages loaded from Colortina's lossless session cache should not stay
        # decoded merely because a project save touched them.
        release_cached = getattr(state, "release_cached_layers", None)
        if callable(release_cached):
            release_cached()

    payload = {
        "version": PROJECT_VERSION,
        "pages": page_records,
        "style": _style_to_dict(style_profile),
        "character_library": (character_library.to_dict()
                              if character_library is not None else None),
        "character_memories": {
            key: value.to_dict() for key, value in (character_memories or {}).items()
        },
        "scene_palette": scene_palette.to_dict() if scene_palette is not None else None,
        "settings": settings or {},
    }
    # Write the manifest to a same-volume temp file so an interrupted save
    # does not turn a previously loadable project JSON into truncated data.
    fd, temp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp",
        dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        raise
    return path


def load_project(path: str) -> dict:
    path = os.path.abspath(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    from core.schema_migration import migrate_ccproject
    payload, migration_log = migrate_ccproject(payload)
    base = os.path.dirname(path)
    pages = []
    for record in payload.get("pages", []):
        item = dict(record)
        for key in ("ai_result", "result", "filter_base",
                    "semantic_hair_positive", "semantic_hair_negative"):
            value = item.get(key)
            if value:
                item[key] = os.path.abspath(os.path.join(base, value))
        item["hint_manager"] = HintManager.from_dict(item.get("hints", {}))
        item["forced_character_matches"] = {
            int(k): int(v) for k, v in
            (item.get("forced_character_matches", {}) or {}).items()
        }
        diag_file = item.get("diagnostics_file")
        if not item.get("diagnostics") and diag_file:
            diag_path = os.path.abspath(os.path.join(base, diag_file))
            if os.path.isfile(diag_path):
                try:
                    with open(diag_path, "r", encoding="utf-8") as f:
                        item["diagnostics"] = json.load(f)
                except Exception:
                    item["diagnostics"] = {}
        pages.append(item)
    return {
        "pages": pages,
        "style_profile": style_from_dict(payload.get("style")),
        "character_library": (CharacterLibrary.from_dict(payload["character_library"])
                              if payload.get("character_library") else None),
        "character_memories": {
            key: CharacterMemory.from_dict(value)
            for key, value in payload.get("character_memories", {}).items()
        },
        "scene_palette": ScenePalette.from_dict(payload.get("scene_palette")),
        "settings": payload.get("settings", {}),
        "migration_log": migration_log,
    }
