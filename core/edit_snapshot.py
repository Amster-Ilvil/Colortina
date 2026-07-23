"""Full-page edit snapshots used by force undo/redo."""
from __future__ import annotations

import numpy as np

from core.hint_manager import HintManager


def copy_image(image):
    return None if image is None else image.copy()


def image_equal(left, right) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return (left.shape == right.shape and left.dtype == right.dtype and
            np.array_equal(left, right))


def capture_edit_state(page) -> dict:
    return {
        "result_bgr": copy_image(page.result_bgr),
        "ai_result_bgr": copy_image(page.ai_result_bgr),
        "filter_base_bgr": copy_image(getattr(page, "filter_base_bgr", None)),
        "hint_manager": page.hint_manager.to_dict(),
        "pipeline_diagnostics": dict(page.pipeline_diagnostics or {}),
        "quality_report": page.quality_report,
    }


def snapshots_equal(left: dict, right: dict) -> bool:
    return (image_equal(left.get("result_bgr"), right.get("result_bgr")) and
            image_equal(left.get("ai_result_bgr"), right.get("ai_result_bgr")) and
            image_equal(left.get("filter_base_bgr"), right.get("filter_base_bgr")) and
            left.get("hint_manager", {}) == right.get("hint_manager", {}))


def restore_edit_state(page, snapshot: dict) -> None:
    page.result_bgr = copy_image(snapshot.get("result_bgr"))
    page.ai_result_bgr = copy_image(snapshot.get("ai_result_bgr"))
    page.filter_base_bgr = copy_image(snapshot.get("filter_base_bgr"))
    page.hint_manager = HintManager.from_dict(snapshot.get("hint_manager", {}))
    page.pipeline_diagnostics = dict(snapshot.get("pipeline_diagnostics", {}) or {})
    page.quality_report = snapshot.get("quality_report")
