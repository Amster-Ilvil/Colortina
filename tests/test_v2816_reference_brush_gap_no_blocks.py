from pathlib import Path

import cv2
import numpy as np

from core.manual_edit import build_region_edit_mask, apply_selection_edit
from core.region_map import build_region_map

ROOT = Path(__file__).resolve().parents[1]


def test_brush_size_and_dab_spacing_match_reference_package():
    main = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    canvas = (ROOT / "ui" / "canvas.py").read_text(encoding="utf-8")
    assert "_BRUSH_PX_MIN, _BRUSH_PX_MAX = 2, 60" in main
    assert "min_step = max(2, int(self._brush_radius * 0.45))" in canvas
    assert "def _emit_dab" not in canvas


def test_larger_gap_setting_closes_broken_block_and_shrinks_from_page_to_one_block():
    source = np.full((200, 240, 3), 255, np.uint8)
    cv2.rectangle(source, (50, 40), (190, 160), (0, 0, 0), 2)
    # Eight-pixel break in the top edge.
    cv2.rectangle(source, (120, 38), (127, 42), (255, 255, 255), -1)
    result = np.full_like(source, (180, 190, 205))

    open_map = build_region_map(source, gap_close=0)
    open_mask = build_region_edit_mask(
        source, result, 100, 100, gap_close=0, region_map=open_map)
    closed_map = build_region_map(source, gap_close=8)
    closed_mask = build_region_edit_mask(
        source, result, 100, 100, gap_close=8, region_map=closed_map)

    # With no repair the component is open/page-connected and is rejected.
    assert np.count_nonzero(open_mask) == 0
    # Raising the maximum gap repairs only the short break and selects the
    # intended inner block, not a page-wide area.
    area = np.count_nonzero(closed_mask)
    assert 12000 < area < 19000
    assert area < int(closed_mask.size * 0.45)


def test_uniform_hue_is_full_resolution_and_has_no_32px_block_steps():
    h, w = 192, 224
    source = np.full((h, w, 3), 255, np.uint8)
    result = np.zeros((h, w, 3), np.uint8)
    for y in range(h):
        value = 70 + int(150 * y / (h - 1))
        result[y, :, :] = (value, min(255, value + 18), min(255, value + 34))
    mask = np.zeros((h, w), np.uint8)
    mask[12:-12, 12:-12] = 255
    out, _base, used, changed = apply_selection_edit(
        source, result, result.copy(), mask, '#397bd7',
        feather=0, closed_only=False, mode='shading')
    assert changed
    active = used > 0
    hls = cv2.cvtColor(out.astype(np.float32) / 255.0, cv2.COLOR_BGR2HLS)
    # Hue is unified, while the smooth source lightness gradient remains smooth.
    assert float(np.std(hls[..., 0][active])) < 1.0
    row_means = hls[12:-12, 20:-20, 1].mean(axis=1)
    adjacent = np.abs(np.diff(row_means))
    assert float(adjacent.max()) < 0.025
    # No periodic jumps at the model's common 32-pixel grid boundaries.
    grid_jumps = adjacent[[i - 1 for i in range(32, len(row_means), 32)]]
    assert float(grid_jumps.max(initial=0.0)) < 0.02


def test_manual_region_modes_do_not_start_local_crop_model_jobs():
    source = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    region_start = source.index("    def _on_region_fill_requested")
    selection_start = source.index("    def _apply_selection_fill_mask", region_start)
    polygon_start = source.index("    def _on_polygon_fill_requested", selection_start)
    region_body = source[region_start:selection_start]
    selection_body = source[selection_start:polygon_start]
    assert "_start_model_recolor" not in region_body
    assert "_start_model_recolor" not in selection_body
    assert "mode=fill_mode" in region_body
    assert "mode=fill_mode" in selection_body
