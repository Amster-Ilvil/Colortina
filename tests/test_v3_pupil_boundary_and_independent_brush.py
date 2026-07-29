from pathlib import Path

import cv2
import numpy as np

from core.manual_edit import (
    apply_brush_edit,
    apply_selection_edit,
    build_closed_region_selection_mask,
    build_rect_selection_mask,
)

ROOT = Path(__file__).resolve().parents[1]


def _eye_source(size: int = 96) -> np.ndarray:
    source = np.full((size, size, 3), 255, np.uint8)
    centre = (size // 2, size // 2)
    cv2.circle(source, centre, 20, (0, 0, 0), 2, lineType=cv2.LINE_AA)
    cv2.circle(source, centre, 7, (0, 0, 0), -1, lineType=cv2.LINE_AA)
    return source


def test_free_region_brush_works_with_snap_and_pupil_blend_disabled():
    source = np.full((80, 80, 3), 255, np.uint8)
    result = np.full((80, 80, 3), (170, 180, 190), np.uint8)
    edited, base, mask, changed = apply_brush_edit(
        source, result, result.copy(), 40, 40, 10, (230, 70, 80),
        snap_to_lineart=False, pupil_blend=False, mode="shift")

    assert changed
    assert np.count_nonzero(mask) > 80
    assert not np.array_equal(edited, result)
    assert not np.array_equal(base, result)
    yy, xx = np.ogrid[:80, :80]
    outside = (xx - 40) ** 2 + (yy - 40) ** 2 > 11 ** 2
    assert np.array_equal(edited[outside], result[outside])


def test_natural_pupil_brush_never_crosses_outer_iris_line():
    source = _eye_source()
    result = np.full_like(source, (150, 160, 170))
    edited, _base, mask, changed = apply_brush_edit(
        source, result, result.copy(), 48, 48, 30, (40, 160, 220),
        snap_to_lineart=False, pupil_blend=True, mode="shift")

    yy, xx = np.ogrid[:96, :96]
    outside_iris = (xx - 48) ** 2 + (yy - 48) ** 2 > 22 ** 2
    iris_body = (((xx - 48) ** 2 + (yy - 48) ** 2 < 18 ** 2) &
                 ((xx - 48) ** 2 + (yy - 48) ** 2 > 9 ** 2))
    assert changed
    assert np.count_nonzero((mask > 0) & outside_iris) == 0
    assert np.count_nonzero((edited != result).any(axis=2) & outside_iris) == 0
    assert np.count_nonzero((mask > 0) & iris_body) > 300


def test_finished_stroke_selection_path_uses_same_pupil_boundary_guard():
    source = _eye_source()
    result = np.full_like(source, (150, 160, 170))
    raw = np.zeros(source.shape[:2], np.uint8)
    cv2.circle(raw, (48, 48), 30, 255, -1)
    edited, _base, used, changed = apply_selection_edit(
        source, result, result.copy(), raw, '#28a0dc', feather=1,
        closed_only=False, mode='shift', pupil_blend=True)

    yy, xx = np.ogrid[:96, :96]
    outside_iris = (xx - 48) ** 2 + (yy - 48) ** 2 > 22 ** 2
    assert changed
    assert np.count_nonzero((used > 0) & outside_iris) == 0
    assert np.count_nonzero((edited != result).any(axis=2) & outside_iris) == 0


def test_closed_rectangle_recognises_antialiased_grey_line_as_boundary():
    source = np.full((100, 100, 3), 255, np.uint8)
    cv2.rectangle(source, (30, 30), (70, 70), (70, 70, 70), 1,
                  lineType=cv2.LINE_AA)
    selection = build_rect_selection_mask((100, 100), 10, 10, 90, 90)
    closed = build_closed_region_selection_mask(source, selection)

    assert closed[50, 50] == 255
    assert closed[20, 20] == 0
    assert np.count_nonzero(closed) > 1000


def test_ui_removes_optional_brush_feature_switches_and_dead_state():
    main = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
    assert '_chk_brush_snap_lineart' not in main
    assert '_chk_pupil_natural_blend' not in main
    assert 'brush_snap_lineart' not in main
    assert 'pupil_natural_blend' not in main
    assert 'snap_brush_stroke_mask' not in main
