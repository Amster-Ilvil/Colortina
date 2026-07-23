import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from core.presets import get_style
from core.style_post import apply_style_grade


class _Region:
    def __init__(self, label_id):
        self.label_id = label_id


class TestMonochromePastelV62(unittest.TestCase):
    @staticmethod
    def _chroma(img_bgr):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        a = lab[..., 1] - 128.0
        b = lab[..., 2] - 128.0
        return np.sqrt(a * a + b * b)

    def test_monochrome_people_environment_is_grayscale(self):
        h, w = 24, 24
        source = np.full((h, w, 3), 245, np.uint8)
        source[:, :12] = 205   # person side
        source[:, 12:] = 242   # environment side

        colorized = source.copy()
        colorized[:, :12] = (60, 115, 180)   # clearly coloured person region
        colorized[:, 12:] = (120, 200, 70)   # intentionally coloured background

        seg = SimpleNamespace(
            labels=np.hstack([
                np.ones((h, 12), np.int32),
                np.full((h, 12), 2, np.int32),
            ]),
            regions=[_Region(1), _Region(2)],
        )
        ctx = SimpleNamespace(
            segmentation=seg,
            semantic_labels=[('hair', 1.0), ('background', 1.0)],
            identity_assignments={},
            character_instances=[SimpleNamespace(body_bbox=(0, 0, 12, h), head_bbox=(0, 0, 12, h))],
        )

        out = apply_style_grade(colorized, source, get_style('monochrome_people'), context=ctx)
        chroma = self._chroma(out)
        # Person region must keep visible colour.
        self.assertGreater(float(chroma[:, :12].mean()), 8.0)
        # Environment must be effectively grayscale.
        env = out[:, 12:]
        self.assertLess(float(chroma[:, 12:].mean()), 1.5)
        self.assertLess(float(np.abs(env[..., 0].astype(np.int16) - env[..., 1].astype(np.int16)).mean()), 1.0)
        self.assertLess(float(np.abs(env[..., 1].astype(np.int16) - env[..., 2].astype(np.int16)).mean()), 1.0)

    def test_monochrome_page_keeps_color_across_bright_hair(self):
        h, w = 20, 20
        source = np.full((h, w, 3), 248, np.uint8)
        # Entire top band is hair, but the left half is very bright highlight.
        source[:10, :10] = 247
        source[:10, 10:] = 190
        source[10:, :] = 245

        colorized = source.copy()
        colorized[:10, :] = (70, 120, 185)  # coloured hair

        labels = np.zeros((h, w), np.int32)
        labels[:10, :] = 1
        labels[10:, :] = 2
        seg = SimpleNamespace(labels=labels, regions=[_Region(1), _Region(2)])
        ctx = SimpleNamespace(
            segmentation=seg,
            semantic_labels=[('hair', 1.0), ('background', 1.0)],
            identity_assignments={},
            character_instances=[SimpleNamespace(body_bbox=(0, 0, w, 10), head_bbox=(0, 0, w, 10))],
        )

        out = apply_style_grade(colorized, source, get_style('monochrome_page'), context=ctx)
        chroma = self._chroma(out)
        bright_hair = float(chroma[:10, :10].mean())
        dark_hair = float(chroma[:10, 10:].mean())
        # Both bright and darker parts of the same hair mass should retain colour.
        self.assertGreater(bright_hair, 6.0)
        self.assertGreater(dark_hair, 6.0)
        self.assertGreater(bright_hair / max(dark_hair, 1e-6), 0.45)


if __name__ == '__main__':
    unittest.main()
