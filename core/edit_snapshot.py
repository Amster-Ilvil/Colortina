"""Full or field-scoped page edit snapshots used by undo/redo.

Field-scoped snapshots avoid decoding/copying untouched cached AI layers for
ordinary brush, bucket and hint edits.  This is important after inactive pages
are losslessly spooled to disk: the first click must not synchronously decode
three full-resolution PNGs just to create an undo checkpoint.
"""
from __future__ import annotations

import numpy as np

from core.hint_manager import HintManager


IMAGE_FIELDS = (
    "result_bgr",
    "ai_result_bgr",
    "filter_base_bgr",
    "bias_brush_reference_bgr",
    "bias_brush_reference_filter_bgr",
)
FULL_FIELDS = IMAGE_FIELDS + (
    "hint_manager",
    "pipeline_diagnostics",
    "quality_report",
)


def copy_image(image):
    return None if image is None else image.copy()


def image_equal(left, right) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return (left.shape == right.shape and left.dtype == right.dtype and
            np.array_equal(left, right))


def _snapshot_fields(snapshot: dict) -> tuple[str, ...]:
    raw = snapshot.get("__fields__")
    if raw is None:
        return FULL_FIELDS
    return tuple(str(field) for field in raw)


def capture_edit_state(page, fields=None) -> dict:
    selected = tuple(fields or FULL_FIELDS)
    state: dict = {"__fields__": selected}
    if "result_bgr" in selected:
        state["result_bgr"] = copy_image(page.result_bgr)
    if "ai_result_bgr" in selected:
        state["ai_result_bgr"] = copy_image(page.ai_result_bgr)
    if "filter_base_bgr" in selected:
        state["filter_base_bgr"] = copy_image(getattr(page, "filter_base_bgr", None))
    if "bias_brush_reference_bgr" in selected:
        state["bias_brush_reference_bgr"] = copy_image(
            getattr(page, "bias_brush_reference_bgr", None))
    if "bias_brush_reference_filter_bgr" in selected:
        state["bias_brush_reference_filter_bgr"] = copy_image(
            getattr(page, "bias_brush_reference_filter_bgr", None))
    if "hint_manager" in selected:
        state["hint_manager"] = page.hint_manager.to_dict()
    if "pipeline_diagnostics" in selected:
        state["pipeline_diagnostics"] = dict(page.pipeline_diagnostics or {})
    if "quality_report" in selected:
        state["quality_report"] = page.quality_report
    return state


def snapshots_equal(left: dict, right: dict) -> bool:
    fields = tuple(dict.fromkeys(_snapshot_fields(left) + _snapshot_fields(right)))
    for field in fields:
        if field in IMAGE_FIELDS:
            if not image_equal(left.get(field), right.get(field)):
                return False
        elif field == "hint_manager":
            if left.get(field, {}) != right.get(field, {}):
                return False
        elif field == "pipeline_diagnostics":
            if left.get(field, {}) != right.get(field, {}):
                return False
        elif field == "quality_report":
            if left.get(field) != right.get(field):
                return False
    return True


def restore_edit_state(page, snapshot: dict) -> None:
    fields = set(_snapshot_fields(snapshot))
    if "result_bgr" in fields:
        page.result_bgr = copy_image(snapshot.get("result_bgr"))
    if "ai_result_bgr" in fields:
        page.ai_result_bgr = copy_image(snapshot.get("ai_result_bgr"))
    if "filter_base_bgr" in fields:
        page.filter_base_bgr = copy_image(snapshot.get("filter_base_bgr"))
    if "bias_brush_reference_bgr" in fields:
        page.bias_brush_reference_bgr = copy_image(
            snapshot.get("bias_brush_reference_bgr"))
    if "bias_brush_reference_filter_bgr" in fields:
        page.bias_brush_reference_filter_bgr = copy_image(
            snapshot.get("bias_brush_reference_filter_bgr"))
    if "hint_manager" in fields:
        page.hint_manager = HintManager.from_dict(snapshot.get("hint_manager", {}))
    if "pipeline_diagnostics" in fields:
        page.pipeline_diagnostics = dict(snapshot.get("pipeline_diagnostics", {}) or {})
    if "quality_report" in fields:
        page.quality_report = snapshot.get("quality_report")
