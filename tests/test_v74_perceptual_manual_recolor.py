import unittest
import cv2
import numpy as np

from core.perceptual_recolor import perceptual_target_ab
from core.presets import STYLE_PRESETS, get_style


class PerceptualManualRecolorTests(unittest.TestCase):
    def test_recolor_retains_local_chroma_variation(self):
        lab = np.zeros((20, 20, 3), np.float32)
        lab[..., 0] = np.linspace(70, 210, 20)[None, :]
        lab[..., 1] = 145 + np.linspace(-8, 8, 20)[:, None]
        lab[..., 2] = 132 + np.linspace(-6, 6, 20)[None, :]
        active = np.ones((20, 20), bool)
        desired = perceptual_target_ab(lab, active, np.array([110, 155], np.float32))
        self.assertGreater(float(np.std(desired[..., 0])), 1.5)
        self.assertGreater(float(np.std(desired[..., 1])), 1.5)
        self.assertLess(np.linalg.norm(np.median(desired.reshape(-1, 2), axis=0) - np.array([110, 155])), 3.0)

    def test_black_white_pastel_presets_removed(self):
        self.assertNotIn('monochrome', STYLE_PRESETS)
        self.assertNotIn('monochrome_people', STYLE_PRESETS)
        self.assertNotIn('monochrome_page', STYLE_PRESETS)
        self.assertEqual(get_style('monochrome').key, 'none')


if __name__ == '__main__':
    unittest.main()
