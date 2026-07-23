import unittest
import numpy as np
from core.line_boundary import analyze_boundaries


class TestV27LineBoundaryRegression(unittest.TestCase):
    def test_gap_one_does_not_crash_on_sparse_page(self):
        page = np.full((96, 128, 3), 255, np.uint8)
        page[20:22, 10:70] = 0
        result = analyze_boundaries(page, gap_close=1)
        self.assertEqual(result.barrier.shape, page.shape[:2])


if __name__ == '__main__':
    unittest.main()
