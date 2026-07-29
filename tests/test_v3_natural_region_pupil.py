from pathlib import Path

import cv2
import numpy as np

from core.manual_edit import build_region_edit_mask
from core.perceptual_recolor import recolor_with_mode

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_iris() -> tuple[np.ndarray, np.ndarray]:
    image = np.full((64, 64, 3), (160, 140, 120), np.uint8)
    cv2.circle(image, (32, 32), 19, (120, 100, 80), -1)
    cv2.circle(image, (32, 32), 7, (25, 20, 15), -1)
    cv2.circle(image, (25, 25), 4, (245, 245, 245), -1)
    alpha = np.zeros((64, 64), np.float32)
    cv2.circle(alpha, (32, 32), 19, 1.0, -1)
    return image, alpha


def test_v5_version_label_is_kept_while_pupil_toggle_state_is_removed():
    main = (ROOT / "ui" / "main_window.py").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    i18n = (ROOT / "ui" / "i18n.py").read_text(encoding="utf-8")

    assert "Colortina V5" in readme
    assert "## V5.0.0" in changelog
    assert 'self._version_label = QLabel(Config.APP_VERSION_LABEL)' in main
    assert "_chk_pupil_natural_blend" not in main
    assert '"pupil_natural_blend"' not in main
    assert "瞳色自然融合" not in i18n


def test_pupil_blend_preserves_dark_pupil_and_bright_highlight():
    image, alpha = _synthetic_iris()
    target = (40, 160, 220)
    plain = recolor_with_mode(
        image, target, alpha, active=alpha > 0, mode="shift",
        pupil_blend=False)
    natural = recolor_with_mode(
        image, target, alpha, active=alpha > 0, mode="shift",
        pupil_blend=True)

    pupil = (32, 32)
    highlight = (25, 25)
    iris_mid = (45, 32)

    def delta(at, output):
        x, y = at
        return float(np.abs(
            output[y, x].astype(np.int16) - image[y, x].astype(np.int16)
        ).mean())

    assert delta(pupil, natural) < delta(pupil, plain) * 0.65
    assert delta(highlight, natural) < delta(highlight, plain) * 0.55
    assert delta(iris_mid, natural) > 8.0

    flat = recolor_with_mode(
        image, target, alpha, active=alpha > 0, mode="flat",
        pupil_blend=True)
    assert np.all(flat[alpha > 0] == np.array(target[::-1], np.uint8))


def test_background_click_and_oversized_region_return_empty_masks():
    source = np.full((200, 240, 3), 255, np.uint8)
    cv2.rectangle(source, (70, 55), (170, 145), (0, 0, 0), 3)
    result = np.full_like(source, (180, 190, 205))

    background = build_region_edit_mask(source, result, 10, 10, gap_close=4)
    enclosed = build_region_edit_mask(source, result, 110, 100, gap_close=4)
    assert np.count_nonzero(background) == 0
    assert 0 < np.count_nonzero(enclosed) < int(enclosed.size * 0.35)

    huge = np.full((200, 240, 3), 255, np.uint8)
    cv2.rectangle(huge, (12, 12), (228, 188), (0, 0, 0), 3)
    huge_mask = build_region_edit_mask(huge, result, 110, 100, gap_close=4)
    assert np.count_nonzero(huge_mask) == 0


def test_natural_shift_retains_luminance_texture_and_is_not_flat():
    image = np.zeros((96, 120, 3), np.uint8)
    for y in range(image.shape[0]):
        value = 70 + int(150 * y / (image.shape[0] - 1))
        image[y, :] = (value, min(255, value + 16), min(255, value + 30))
    alpha = np.ones(image.shape[:2], np.float32)
    shifted = recolor_with_mode(
        image, (35, 130, 215), alpha, mode="shift")

    before_l = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    after_l = cv2.cvtColor(shifted, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    correlation = float(np.corrcoef(before_l.ravel(), after_l.ravel())[0, 1])
    assert correlation > 0.995
    assert float(after_l.std()) > float(before_l.std()) * 0.75
    assert len(np.unique(shifted.reshape(-1, 3), axis=0)) > 40
