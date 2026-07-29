import cv2
import numpy as np

from core.manual_edit import build_region_edit_mask
from core.region_map import build_region_map


def _closed_rect(height=180, width=220):
    image = np.full((height, width, 3), 255, np.uint8)
    cv2.rectangle(image, (40, 35), (180, 145), (0, 0, 0), 3)
    return image


def test_bucket_accepts_closed_line_block():
    source = _closed_rect()
    result = np.full_like(source, 240)
    region_map = build_region_map(source, gap_close=4)
    mask = build_region_edit_mask(source, result, 100, 90, gap_close=4, region_map=region_map)
    assert np.count_nonzero(mask) > 8000
    assert not np.any(mask[:, 0]) and not np.any(mask[:, -1])


def test_bucket_rejects_open_background_region_instead_of_flooding_page():
    source = _closed_rect()
    result = np.full_like(source, 240)
    region_map = build_region_map(source, gap_close=4)
    # Click in the page background / open region.
    mask = build_region_edit_mask(source, result, 10, 10, gap_close=4, region_map=region_map)
    assert np.count_nonzero(mask) == 0
