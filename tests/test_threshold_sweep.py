import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_threshold_sweep import at_target, curves, load_fold, tail  # noqa: E402
from threshold_sweep import EDGES, NBINS, bin_counts  # noqa: E402


class BinTests(unittest.TestCase):
    def test_half_is_an_edge_matching_argmax(self):
        half = NBINS // 2
        self.assertEqual(EDGES[half], 0.5)
        h = bin_counts(np.array([0.0, 0.4999999, 0.5, 0.75, 1.0], np.float32))
        self.assertEqual(h.sum(), 5)
        self.assertEqual(h[:half].sum(), 2)        # below 0.5: background under argmax
        self.assertEqual(h[half:].sum(), 3)        # 0.5 and above: oil
        self.assertEqual(h[-1], 1)                 # P == 1 lands in the last bin


class SweepTests(unittest.TestCase):
    def test_tail_counts_at_or_above(self):
        np.testing.assert_array_equal(tail(np.array([[1, 2, 3]])), [[6, 5, 3]])

    def test_at_target_interpolates_on_a_decreasing_curve(self):
        c = {"recall": np.array([[1.0, 0.8, 0.4, 0.0]]), "fp": np.array([[100., 50., 10., 0.]])}
        self.assertAlmostEqual(at_target(c, "recall", np.array([0.6]), "fp")[0], 30.0)
        self.assertAlmostEqual(at_target(c, "recall", np.array([0.8]), "fp")[0], 50.0)
        # only "every pixel is oil" (the lowest bin) reaches 0.9: unreachable
        self.assertTrue(np.isnan(at_target(c, "recall", np.array([0.9]), "fp")[0]))

    def test_load_fold_and_curves(self):
        nb = 4
        edges = np.linspace(0, 1, nb + 1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "fold_1.npz"
            np.savez(path, edges=edges, recon_scenes=np.array(["s1"]),
                     recon_oil=np.array([[0, 2, 0, 2]]), recon_other=np.array([[4, 0, 0, 0]]),
                     bg_scenes=np.array(["s1", "s2"]),
                     bg_hist=np.array([[10, 0, 0, 0], [8, 0, 2, 0]]))
            scenes, m = load_fold(path)
        self.assertEqual(scenes, ["s1", "s2"])
        c = curves(m, np.ones((1, 2)))
        np.testing.assert_allclose(c["recall"][0], [1, 1, 0.5, 0.5])
        np.testing.assert_allclose(c["bg_tile_rate"][0], [1, 0.5, 0.5, 0])
        np.testing.assert_allclose(c["bg_fp_per_M"][0] * 20 / 1e6, [20, 2, 2, 0])


if __name__ == "__main__":
    unittest.main()
