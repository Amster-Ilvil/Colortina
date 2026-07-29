from pathlib import Path

import cv2
import numpy as np

from core.selection_snap import snap_selection_mask_to_lineart

ROOT = Path(__file__).resolve().parents[1]


def _closed_box(size=96):
    image = np.full((size, size, 3), 255, np.uint8)
    cv2.rectangle(image, (18, 18), (76, 76), (0, 0, 0), 3)
    return image


def test_conservative_snap_can_reach_nearby_closed_lineart_region():
    source = _closed_box()
    rough = np.zeros(source.shape[:2], np.uint8)
    cv2.rectangle(rough, (25, 25), (69, 69), 255, -1)
    snapped, diagnostics = snap_selection_mask_to_lineart(
        source, rough, gap_close=2, max_distance=8, max_growth_ratio=1.8)
    assert snapped[48, 48] == 255
    assert snapped[15, 15] == 0
    assert diagnostics.snapped_area <= int(diagnostics.raw_area * 1.8)


def test_snap_does_not_cross_ink_divider_or_select_background():
    source = np.full((80, 120, 3), 255, np.uint8)
    cv2.rectangle(source, (8, 8), (111, 71), (0, 0, 0), 3)
    cv2.line(source, (60, 9), (60, 70), (0, 0, 0), 3)
    rough = np.zeros(source.shape[:2], np.uint8)
    cv2.rectangle(rough, (15, 18), (52, 62), 255, -1)
    snapped, diagnostics = snap_selection_mask_to_lineart(
        source, rough, gap_close=1, max_distance=8)
    assert np.count_nonzero(snapped[:, 66:]) == 0
    assert diagnostics.snapped_area <= int(diagnostics.raw_area * 1.8)


def test_ambiguous_open_page_falls_back_without_expansion():
    source = np.full((80, 100, 3), 255, np.uint8)
    rough = np.zeros((80, 100), np.uint8)
    cv2.rectangle(rough, (20, 20), (70, 60), 255, -1)
    snapped, diagnostics = snap_selection_mask_to_lineart(
        source, rough, max_distance=12)
    assert np.array_equal(snapped, rough)
    assert diagnostics.used_fallback


def test_zero_distance_returns_same_binary_mask():
    source = _closed_box()
    rough = np.zeros(source.shape[:2], np.uint8)
    rough[30:50, 35:60] = 255
    snapped, diagnostics = snap_selection_mask_to_lineart(
        source, rough, max_distance=0)
    assert np.array_equal(snapped, rough)
    assert diagnostics.max_distance == 0


def test_ui_uses_snap_only_for_lasso_and_defaults_off():
    main = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
    assert 'from core.selection_snap import snap_selection_mask_to_lineart' in main
    assert '_chk_selection_snap_lineart = QCheckBox' in main
    assert '_selection_snap_distance_slider = QSlider' in main
    assert 'def _snap_selection_to_lineart' in main
    assert "mask = self._snap_selection_to_lineart(mask)" in main
    assert "snapped_mask = self._snap_selection_to_lineart(raw_mask)" not in main
    assert 'mask = self._snap_selection_to_lineart(mask, status=False)' not in main
    # V5.4.x: 套索贴线默认关闭，用户需要时手动开启。
    assert 'settings.get("selection_snap_lineart", False)' in main
    assert 'self._chk_selection_snap_lineart.setChecked(False)' in main


def test_i18n_keeps_selection_boundary_snap_without_brush_snap_copy():
    text = (ROOT / 'ui' / 'i18n.py').read_text(encoding='utf-8')
    assert '"selection_snap_lineart_checkbox": "套索选区边界自动贴线"' in text
    assert '"selection_snap_lineart_checkbox": "Snap lasso selection boundary to ink"' in text
    assert 'brush_snap_lineart_checkbox' not in text


def test_snap_can_contract_rough_lasso_drawn_outside_ink():
    source = _closed_box()
    rough = np.zeros(source.shape[:2], np.uint8)
    cv2.rectangle(rough, (12, 12), (82, 82), 255, -1)
    snapped, diagnostics = snap_selection_mask_to_lineart(
        source, rough, gap_close=2, max_distance=10, max_growth_ratio=1.8)
    assert not diagnostics.used_fallback
    assert diagnostics.snapped_area < diagnostics.raw_area
    x, y, w, h = cv2.boundingRect((snapped > 0).astype(np.uint8))
    assert x >= 20 and y >= 20
    assert x + w <= 74 and y + h <= 74
