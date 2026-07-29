import cv2
import numpy as np

from core.manual_edit import build_selection_edit_mask


def test_closed_only_rejects_near_whole_rectangle_component():
    source = np.full((100, 120, 3), 255, np.uint8)
    selection = np.zeros((100, 120), np.uint8)
    cv2.rectangle(selection, (10, 10), (109, 89), 255, -1)
    # Strong artificial border around almost the whole selection. A failed
    # structural topology should return empty rather than paint the full box.
    probability = np.zeros((100, 120), np.float32)
    cv2.rectangle(probability, (12, 12), (107, 87), 1.0, 2)
    mask = build_selection_edit_mask(
        source, selection, closed_only=True, reject_dominant=True,
        extra_probability=probability, expand_px=0, min_area=0,
        min_thickness=0)
    assert np.count_nonzero(mask) < int(np.count_nonzero(selection) * 0.68)
