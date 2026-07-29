from pathlib import Path

import cv2
import numpy as np

from core.manual_edit import (
    apply_selection_edit,
    build_closed_region_selection_mask,
    build_rect_selection_mask,
)
from core.structural_line_detector import detect_structural_lines

ROOT = Path(__file__).resolve().parents[1]


def _panel_with_pale_closed_cells() -> np.ndarray:
    source = np.full((160, 180, 3), 255, np.uint8)
    cv2.rectangle(source, (12, 12), (167, 147), (0, 0, 0), 2,
                  lineType=cv2.LINE_AA)
    # Real scans often turn fine antialiased contours into mid grey. These two
    # regions must still be recognised as structural closed line areas.
    cv2.rectangle(source, (35, 38), (75, 95), (172, 172, 172), 1,
                  lineType=cv2.LINE_AA)
    cv2.ellipse(source, (125, 76), (24, 30), 0, 0, 360,
                (185, 185, 185), 1, lineType=cv2.LINE_AA)
    return source


def test_structural_detector_recovers_pale_antialiased_contours():
    source = _panel_with_pale_closed_cells()
    selection = build_rect_selection_mask(source.shape[:2], 5, 5, 174, 154)
    analysis = detect_structural_lines(source, selection, gap_close=4)

    # Pale contour centres are barriers despite being much lighter than the old
    # fixed gray<=108 threshold.
    assert analysis.barrier[38, 55] == 255
    assert analysis.barrier[76, 149] == 255


def test_closed_rectangle_keeps_cells_but_rejects_panel_background():
    source = _panel_with_pale_closed_cells()
    selection = build_rect_selection_mask(source.shape[:2], 5, 5, 174, 154)
    closed = build_closed_region_selection_mask(
        source, selection, reject_dominant=True)

    assert closed[60, 55] == 255       # first small closed cell
    assert closed[76, 125] == 255      # ellipse interior
    assert closed[120, 90] == 0        # dominant panel/background cavity
    assert closed[6, 6] == 0           # never the raw rectangle
    assert np.count_nonzero(closed) < np.count_nonzero(selection) * 0.45


def test_single_near_whole_panel_is_rejected_instead_of_full_box_fill():
    source = np.full((120, 140, 3), 255, np.uint8)
    cv2.rectangle(source, (8, 8), (131, 111), (0, 0, 0), 2,
                  lineType=cv2.LINE_AA)
    selection = build_rect_selection_mask(source.shape[:2], 3, 3, 136, 116)
    closed = build_closed_region_selection_mask(
        source, selection, reject_dominant=True)

    # An ambiguous near-whole-frame cavity is safer as no-op than recolouring a
    # complete panel. The user can use ordinary rectangle fill when desired.
    assert np.count_nonzero(closed) == 0


def test_authoritative_closed_mask_never_reexpands_to_rectangle():
    source = _panel_with_pale_closed_cells()
    result = np.full_like(source, (175, 185, 200))
    selection = build_rect_selection_mask(source.shape[:2], 5, 5, 174, 154)
    closed = build_closed_region_selection_mask(
        source, selection, reject_dominant=True)
    edited, _base, used, changed = apply_selection_edit(
        source, result, result.copy(), closed, '#ef5a70', feather=2,
        closed_only=False, mode='shift')

    assert changed
    assert np.array_equal(used, closed)
    changed_pixels = np.any(edited != result, axis=2)
    assert np.count_nonzero(changed_pixels & (closed == 0)) == 0
    assert not changed_pixels[120, 90]


def test_ui_filters_rectangle_at_release_and_applies_exact_preview_mask():
    text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
    assert "self._set_pending_selection(combined, 'rect_closed', combine=False)" in text
    assert "already_closed = str(selection_kind or '').endswith('_closed')" in text
    assert "mask, hex_color, feather=feather, closed_only=False" in text
    assert "kind = self._pending_selection_kind" in text
