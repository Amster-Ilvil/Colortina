from pathlib import Path

import cv2
import numpy as np

from config import Config
from core.manga_line_extractor import _output_to_line_probability
from core.manual_edit import build_rect_selection_mask, build_selection_edit_mask
from core.structural_line_detector import (
    _expand_closed_regions_within_paintable,
    detect_structural_lines,
)
from vendor.manga_line_extraction import res_skip

ROOT = Path(__file__).resolve().parents[1]


def test_official_network_layout_is_vendored_with_expected_state_keys():
    model = res_skip()
    keys = list(model.state_dict())
    assert keys[0] == "block0.model.0.conv1.model.0.weight"
    assert "block4.model.11.residual.model.2.weight" in keys
    assert keys[-1] == "conv15.model.2.bias"


def test_model_output_orientation_converts_black_lines_to_high_probability():
    gray = np.full((48, 48), 255, np.uint8)
    output = np.full((48, 48), 255, np.float32)
    cv2.rectangle(gray, (12, 10), (35, 37), 20, 2)
    cv2.rectangle(output, (12, 10), (35, 37), 0, 2)
    probability = _output_to_line_probability(output, gray)
    assert float(probability[10, 24]) > 0.85
    assert float(probability[24, 24]) < 0.15


def test_ai_probability_is_primary_and_produces_real_closed_region():
    source = np.full((120, 140, 3), 255, np.uint8)
    # Texture that should not become a structural barrier in AI-primary mode.
    for y in range(15, 105, 8):
        for x in range(12, 128, 9):
            source[y, x] = (165, 165, 165)

    extra = np.zeros(source.shape[:2], np.float32)
    cv2.rectangle(extra, (42, 28), (102, 94), 1.0, 2)
    selection = build_rect_selection_mask(source.shape[:2], 20, 12, 122, 108)

    analysis = detect_structural_lines(
        source, selection, gap_close=3, extra_probability=extra)
    assert analysis.barrier[28, 70] == 255
    assert analysis.barrier[60, 42] == 255
    # Isolated gray print texture is not promoted to a wall merely by Canny.
    assert analysis.barrier[47, 57] == 0

    closed = build_selection_edit_mask(
        source, selection, closed_only=True, reject_dominant=False,
        extra_probability=extra)
    assert closed[60, 70] == 255
    assert closed[16, 25] == 0
    assert np.count_nonzero(closed) > 2500


def test_closed_rectangle_ui_runs_ai_worker_and_never_plain_box_fallback():
    main = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    worker = (ROOT / "ui" / "worker.py").read_text(encoding="utf-8")
    detector = (ROOT / "core" / "structural_line_detector.py").read_text(encoding="utf-8")
    assert "MangaLineExtractionWorker" in main
    assert "extra_probability=probability" in main
    assert "没有使用普通矩形" in main
    assert "ensure_manga_line_model_downloaded" in worker
    assert "AI-primary fusion" in detector


def test_official_weight_url_and_separate_weight_path_are_configured():
    assert Config.MANGA_LINE_WEIGHTS_PATH.endswith("models/weights/erika.pth")
    assert Config.MANGA_LINE_MODEL_URL.endswith("/releases/download/v1/erika.pth")
    assert Config.MANGA_LINE_MAX_SIDE >= 512


def test_closed_region_expansion_grows_outward_without_crossing_barrier():
    source = np.full((80, 90, 3), 255, np.uint8)
    selection = build_rect_selection_mask(source.shape[:2], 8, 8, 82, 72)
    extra = np.zeros(source.shape[:2], np.float32)
    cv2.rectangle(extra, (24, 20), (66, 58), 1.0, 2)

    no_expand = build_selection_edit_mask(
        source, selection, closed_only=True,
        reject_dominant=False, extra_probability=extra, expand_px=0)
    expanded = build_selection_edit_mask(
        source, selection, closed_only=True,
        reject_dominant=False, extra_probability=extra, expand_px=3)

    assert np.count_nonzero(expanded) > np.count_nonzero(no_expand)
    # The expanded mask still cannot cross the recognised structural line.
    assert expanded[19, 45] == 0
    assert expanded[59, 45] == 0
    # Interior grows outward near the boundary.
    assert no_expand[22, 45] == 0
    assert expanded[22, 45] == 255


def test_ui_exposes_closed_region_expansion_slider_and_persists_setting():
    main = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    i18n = (ROOT / "ui" / "i18n.py").read_text(encoding="utf-8")
    assert "_selection_closed_expand_slider" in main
    assert '"selection_closed_expand"' in i18n
    assert '"selection_closed_expand_hint"' in i18n
    assert '"selection_closed_expand": self._selection_closed_expand_slider.value()' in main


def test_larger_closed_expansion_values_keep_growing_until_source_ink():
    source = np.full((100, 110, 3), 255, np.uint8)
    # Actual source ink is outside the AI-predicted inner contour.
    cv2.rectangle(source, (24, 18), (86, 80), (20, 20, 20), 2)
    selection = build_rect_selection_mask(source.shape[:2], 8, 6, 102, 92)
    extra = np.zeros(source.shape[:2], np.float32)
    cv2.rectangle(extra, (31, 25), (79, 73), 1.0, 2)

    e0 = build_selection_edit_mask(
        source, selection, closed_only=True,
        reject_dominant=False, extra_probability=extra, expand_px=0)
    e3 = build_selection_edit_mask(
        source, selection, closed_only=True,
        reject_dominant=False, extra_probability=extra, expand_px=3)
    e8 = build_selection_edit_mask(
        source, selection, closed_only=True,
        reject_dominant=False, extra_probability=extra, expand_px=8)
    assert np.count_nonzero(e3) > np.count_nonzero(e0)
    assert np.count_nonzero(e8) > np.count_nonzero(e3)
    # The real dark source line remains protected.
    assert e8[18, 55] == 0
    assert e8[50, 24] == 0


def test_closed_expand_slider_uses_cached_probability_for_live_preview():
    main = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    assert "_closed_preview_probability" in main
    assert "_closed_expand_preview_timer" in main
    assert "_schedule_closed_preview_refresh" in main
    assert "_refresh_closed_selection_preview" in main
    assert "Moving the slider must never re-download/re-run MangaLineExtraction" in main


def test_small_closed_region_filter_removes_only_components_below_threshold():
    source = np.full((120, 150, 3), 255, np.uint8)
    selection = build_rect_selection_mask(source.shape[:2], 6, 6, 144, 114)
    extra = np.zeros(source.shape[:2], np.float32)
    cv2.rectangle(extra, (24, 24), (94, 94), 1.0, 2)
    cv2.rectangle(extra, (112, 42), (123, 53), 1.0, 2)

    unfiltered = build_selection_edit_mask(
        source, selection, closed_only=True, reject_dominant=False,
        extra_probability=extra, min_area=1)
    filtered = build_selection_edit_mask(
        source, selection, closed_only=True, reject_dominant=False,
        extra_probability=extra, min_area=300)

    assert unfiltered[47, 117] == 255
    assert filtered[47, 117] == 0
    assert filtered[60, 60] == 255


def test_geodesic_expansion_never_drops_seed_pixels_blocked_by_guard():
    closed = np.zeros((30, 30), np.uint8)
    closed[12:18, 12:18] = 255
    allowed = np.ones((30, 30), np.uint8) * 255
    # Simulate gray shading/antialias evidence overlapping the accepted seed.
    allowed[13:17, 13:17] = 0

    expanded = _expand_closed_regions_within_paintable(
        closed, allowed, expand_px=4)

    assert np.all(expanded[closed > 0] == 255)
    assert np.count_nonzero(expanded) > np.count_nonzero(closed)


def test_gray_small_region_can_expand_until_true_black_ink():
    source = np.full((90, 100, 3), 255, np.uint8)
    source[25:65, 30:70] = (104, 104, 104)
    cv2.rectangle(source, (20, 15), (80, 75), (18, 18, 18), 2)
    selection = build_rect_selection_mask(source.shape[:2], 8, 6, 92, 84)
    extra = np.zeros(source.shape[:2], np.float32)
    cv2.rectangle(extra, (32, 27), (68, 63), 1.0, 2)

    e0 = build_selection_edit_mask(
        source, selection, closed_only=True, reject_dominant=False,
        extra_probability=extra, expand_px=0)
    e8 = build_selection_edit_mask(
        source, selection, closed_only=True, reject_dominant=False,
        extra_probability=extra, expand_px=8)

    assert np.count_nonzero(e8) > np.count_nonzero(e0)
    assert e8[15, 50] == 0


def test_ui_exposes_small_region_filter_and_persists_setting():
    main = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    i18n = (ROOT / "ui" / "i18n.py").read_text(encoding="utf-8")
    assert "_selection_closed_min_area_slider" in main
    assert '"selection_closed_min_area"' in i18n
    assert '"selection_closed_min_area": self._selection_closed_min_area_slider.value()' in main
    assert "min_area=min_area" in main
