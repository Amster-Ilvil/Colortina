from pathlib import Path

import cv2
import numpy as np

from core.manual_edit import build_selection_edit_mask
from core.structural_line_detector import (
    _component_shape_metrics,
    _is_slender_noise_component,
)


ROOT = Path(__file__).resolve().parents[1]


def _component_stats(mask: np.ndarray):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8)
    assert count > 1
    return labels, stats


def test_slender_filter_removes_large_thin_strip_but_keeps_compact_detail():
    mask = np.zeros((130, 190), np.uint8)
    # The strip has slightly more area than the compact square, so area-only
    # filtering cannot solve this case.
    cv2.rectangle(mask, (10, 12), (165, 18), 255, -1)
    cv2.rectangle(mask, (25, 60), (56, 91), 255, -1)
    labels, stats = _component_stats(mask)

    strip_label = int(labels[15, 80])
    compact_label = int(labels[75, 40])
    strip_metrics = _component_shape_metrics(labels, stats, strip_label)
    compact_metrics = _component_shape_metrics(labels, stats, compact_label)

    assert stats[strip_label, cv2.CC_STAT_AREA] > stats[compact_label, cv2.CC_STAT_AREA]
    assert strip_metrics[0] > 10.0
    assert compact_metrics[0] < 1.2
    assert _is_slender_noise_component(labels, stats, strip_label, 8)
    assert not _is_slender_noise_component(labels, stats, compact_label, 8)


def test_slender_filter_detects_winding_string_like_component():
    mask = np.zeros((150, 150), np.uint8)
    points = np.array([
        [15, 20], [115, 20], [115, 65], [45, 65], [45, 120]
    ], np.int32)
    cv2.polylines(mask, [points], False, 255, thickness=5)
    labels, stats = _component_stats(mask)
    label = int(labels[20, 40])
    aspect, thickness, compactness, fill_ratio = _component_shape_metrics(
        labels, stats, label)

    assert thickness < 7
    assert compactness < 0.20 or fill_ratio < 0.64
    assert _is_slender_noise_component(labels, stats, label, 7)
    assert not _is_slender_noise_component(labels, stats, label, 0)


def test_public_closed_selection_filter_is_independent_from_area_filter():
    height, width = 150, 190
    source = np.full((height, width, 3), 255, np.uint8)
    probability = np.zeros((height, width), np.float32)
    # One long narrow cavity and one compact cavity.
    for p1, p2 in [((10, 10), (170, 24)), ((24, 62), (62, 100))]:
        cv2.rectangle(source, p1, p2, (0, 0, 0), 1)
        cv2.rectangle(probability, p1, p2, 1.0, 1)
    selection = np.full((height, width), 255, np.uint8)

    unfiltered = build_selection_edit_mask(
        source, selection, closed_only=True,
        extra_probability=probability,
        min_area=0, min_thickness=0)
    filtered = build_selection_edit_mask(
        source, selection, closed_only=True,
        extra_probability=probability,
        min_area=0, min_thickness=12)

    assert unfiltered[17, 90] == 255
    assert unfiltered[80, 42] == 255
    assert filtered[17, 90] == 0
    assert filtered[80, 42] == 255


def test_ui_exposes_and_persists_slender_filter():
    main = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    i18n = (ROOT / "ui" / "i18n.py").read_text(encoding="utf-8")

    assert "_selection_closed_min_thickness_slider" in main
    assert '"selection_closed_min_thickness"' in i18n
    assert ('"selection_closed_min_thickness": '
            'self._selection_closed_min_thickness_slider.value()') in main
    assert "min_thickness=min_thickness" in main
    assert "min_thickness=closed_min_thickness" in main
