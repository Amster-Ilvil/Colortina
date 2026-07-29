import cv2
import numpy as np

from core.hint_manager import HintManager
from core.ml_colorizer import build_hint_arrays
from core.hint_rasterizer import model_geometry


def test_mixed_region_hint_handles_resized_height_padding_817_to_832():
    # User image is 1009x712. At model size 576 it becomes rh=817 and is
    # padded to ph=832, which previously caused a 832-vs-817 boolean index.
    h, w, size = 1009, 712, 576
    ph, pw, rh, rw = model_geometry(h, w, size)
    assert (ph, rh) == (832, 817)

    page = np.full((h, w), 245, np.uint8)
    cv2.rectangle(page, (250, 40), (580, 360), 0, 3)
    hm = HintManager()
    hm.bind_source_image(cv2.cvtColor(page, cv2.COLOR_GRAY2BGR))
    hm.add_manual_hint(0.58, 0.16, (30, 80, 230), 0.02, source="manual")
    specs = hm.merge_specs(image_bgr=cv2.cvtColor(page, cv2.COLOR_GRAY2BGR))

    hint, alpha = build_hint_arrays(
        h, w, size, specs, label_map=hm.region_map, page_gray=page,
        render_mode="mixed")

    assert hint.shape == (ph, pw, 3)
    assert alpha.shape == (ph, pw)
    assert np.count_nonzero(alpha[:rh, :rw]) > 0
    # Padding remains untouched.
    assert np.count_nonzero(alpha[rh:, :]) == 0


def test_mixed_region_hint_handles_width_padding_too():
    # Landscape geometry can pad the right edge instead of the bottom edge.
    h, w, size = 500, 901, 576
    ph, pw, rh, rw = model_geometry(h, w, size)
    assert pw >= rw and ph >= rh

    page = np.full((h, w), 245, np.uint8)
    cv2.rectangle(page, (100, 100), (500, 400), 0, 3)
    hm = HintManager()
    hm.bind_source_image(cv2.cvtColor(page, cv2.COLOR_GRAY2BGR))
    hm.add_manual_hint(0.30, 0.45, (220, 50, 40), 0.02, source="manual")
    specs = hm.merge_specs(image_bgr=cv2.cvtColor(page, cv2.COLOR_GRAY2BGR))

    hint, alpha = build_hint_arrays(
        h, w, size, specs, label_map=hm.region_map, page_gray=page,
        render_mode="mixed")

    assert hint.shape == (ph, pw, 3)
    assert alpha.shape == (ph, pw)
    assert np.count_nonzero(alpha[:rh, :rw]) > 0
    if pw > rw:
        assert np.count_nonzero(alpha[:, rw:]) == 0
