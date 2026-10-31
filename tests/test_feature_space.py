import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from feature_space import auc, cell_weights, knn_share, topk_sim  # noqa: E402


class FeatureSpaceTests(unittest.TestCase):
    def test_cell_weights_counts_pixels_per_cell(self):
        m = np.zeros((32, 32), bool)
        m[:16, :16] = True
        m[20, 30] = True
        np.testing.assert_array_equal(cell_weights(m, 2, 2), [[256, 0], [0, 1]])

    def test_auc(self):
        self.assertEqual(auc(np.array([3., 4., 1., 2.]), np.array([1, 1, 0, 0], bool)), 1.0)
        self.assertEqual(auc(np.array([1., 1.]), np.array([1, 0], bool)), 0.5)

    def test_similarity_and_knn_share(self):
        q = np.array([[1., 0.]])
        hard = np.array([[1., .1], [.9, 0.]])
        oil = np.array([[0., 1.], [-1., 0.]])
        self.assertAlmostEqual(knn_share(q, hard, oil, k=2)[0], 1.0)
        self.assertAlmostEqual(knn_share(q, hard, oil, k=4)[0], 0.5)
        self.assertAlmostEqual(topk_sim(q, oil, k=1)[0], 0.0)


if __name__ == "__main__":
    unittest.main()
