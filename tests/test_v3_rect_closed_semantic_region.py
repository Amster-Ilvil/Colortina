from pathlib import Path

import cv2
import numpy as np

from core.manual_edit import (
    build_closed_region_selection_mask,
    build_rect_selection_mask,
)

ROOT = Path(__file__).resolve().parents[1]


def _garment_with_texture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.full((180, 180, 3), 255, np.uint8)
    garment = np.array([
        [70, 25], [115, 25], [132, 55], [122, 150],
        [88, 165], [55, 145], [58, 60],
    ], np.int32)
    cv2.polylines(source, [garment], True, (0, 0, 0), 2,
                  lineType=cv2.LINE_AA)

    # Manga screentone / print dots inside the garment must not become a wall
    # network that leaves only a few tiny selectable islands.
    for y in range(55, 145, 12):
        for x in range(70, 120, 13):
            cv2.circle(source, (x, y), 1, (175, 175, 175), -1,
                       lineType=cv2.LINE_AA)

    # A real small closed ornament is present as well. The large garment region
    # must not be discarded merely because this smaller component exists.
    cv2.circle(source, (92, 75), 7, (40, 40, 40), 1,
               lineType=cv2.LINE_AA)

    truth = np.zeros(source.shape[:2], np.uint8)
    cv2.fillPoly(truth, [garment], 255)
    selection = build_rect_selection_mask(source.shape[:2], 48, 18, 139, 171)
    return source, selection, truth


def test_tight_rectangle_keeps_main_garment_not_only_tiny_details():
    source, selection, truth = _garment_with_texture()
    closed = build_closed_region_selection_mask(
        source, selection, reject_dominant=False)

    selection_area = int(np.count_nonzero(selection))
    main_coverage = int(np.count_nonzero((closed > 0) & (truth > 0)))
    truth_area = int(np.count_nonzero(truth))

    assert closed[120, 90] == 255       # main garment body
    assert closed[30, 50] == 0         # outside the outlined garment
    assert np.count_nonzero(closed) > selection_area * 0.30
    assert main_coverage > truth_area * 0.72


def test_screentone_dots_do_not_reduce_preview_to_specks():
    source, selection, _truth = _garment_with_texture()
    closed = build_closed_region_selection_mask(
        source, selection, reject_dominant=False)

    # A horizontal scan through the body should contain a broad continuous fill,
    # not just one or two isolated cyan preview marks.
    row = closed[120] > 0
    runs = []
    start = None
    for idx, value in enumerate(row):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append(idx - start)
            start = None
    if start is not None:
        runs.append(len(row) - start)
    assert runs and max(runs) >= 18


def test_rectangle_ui_accepts_largest_valid_closed_region():
    text = (ROOT / 'ui' / 'main_window.py').read_text(encoding='utf-8')
    assert 'reject_dominant=True' in text
