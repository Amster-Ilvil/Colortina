import unittest

import cv2
import numpy as np

from core.image_filter import apply_image_filter


class TestV24NaturalFilter(unittest.TestCase):
    def test_default_filter_is_exact_identity(self):
        rng = np.random.default_rng(42)
        image = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        out = apply_image_filter(image, {})
        self.assertTrue(np.array_equal(image, out))

    def test_shadow_control_is_zone_selective(self):
        image = np.zeros((80, 80, 3), np.uint8)
        image[:40] = (45, 55, 65)
        image[40:] = (205, 215, 225)
        source = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        out = apply_image_filter(
            image, {"shadow_lift": 165}, source_bw_bgr=source)
        before = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        after = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dark_change = float((after[:40] - before[:40]).mean())
        bright_change = float((after[40:] - before[40:]).mean())
        self.assertGreater(dark_change, bright_change + 3.0)

    def test_line_and_paper_are_protected(self):
        image = np.full((96, 96, 3), 248, np.uint8)
        cv2.line(image, (48, 0), (48, 95), (7, 7, 7), 3)
        # Colored patch makes the input representative of a colorized manga page.
        cv2.rectangle(image, (8, 18), (38, 78), (120, 150, 205), -1)
        source = np.full_like(image, 255)
        cv2.line(source, (48, 0), (48, 95), (0, 0, 0), 3)
        out = apply_image_filter(image, {
            "brightness": 165, "warmth": 170, "saturation": 165,
            "shadow_lift": 150, "highlight": 150,
        }, source_bw_bgr=source)
        # Paper and line anchors should remain close to their original values.
        self.assertLess(float(np.abs(out[10, 80].astype(np.int16) - image[10, 80]).mean()), 6.0)
        self.assertLess(float(np.abs(out[48, 48].astype(np.int16) - image[48, 48]).mean()), 9.0)

    def test_fine_detail_survives_tone_adjustment(self):
        yy, xx = np.mgrid[0:96, 0:96]
        texture = ((xx + yy) % 6 - 3).astype(np.float32) * 3.0
        base = np.clip(128.0 + texture, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        source = image.copy()
        out = apply_image_filter(image, {
            "brightness": 145, "contrast": 140, "shadow_lift": 125,
        }, source_bw_bgr=source)
        before = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        after = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
        before_detail = before - cv2.GaussianBlur(before, (0, 0), 2.0)
        after_detail = after - cv2.GaussianBlur(after, (0, 0), 2.0)
        ratio = float(after_detail.std() / max(before_detail.std(), 1e-5))
        self.assertGreater(ratio, 0.72)
        self.assertLess(ratio, 1.35)

    def test_vibrance_avoids_overboosting_saturated_color(self):
        image = np.zeros((40, 80, 3), np.uint8)
        image[:, :40] = (130, 145, 160)  # low chroma
        image[:, 40:] = (20, 30, 230)    # already high chroma
        source = np.full_like(image, 180)
        out = apply_image_filter(image, {"saturation": 170}, source_bw_bgr=source)
        lab0 = cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
        lab1 = cv2.cvtColor(out.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
        c0 = np.sqrt(lab0[..., 1] ** 2 + lab0[..., 2] ** 2)
        c1 = np.sqrt(lab1[..., 1] ** 2 + lab1[..., 2] ** 2)
        low_ratio = float(c1[:, :40].mean() / max(c0[:, :40].mean(), 1e-5))
        high_ratio = float(c1[:, 40:].mean() / max(c0[:, 40:].mean(), 1e-5))
        self.assertGreater(low_ratio, high_ratio)


if __name__ == "__main__":
    unittest.main()
