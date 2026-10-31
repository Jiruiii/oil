import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main"))
from metrics_extra import bg_tile_counts, boundary_band, boundary_counts  # noqa: E402


def square(shape, y0, y1, x0, x1):
    a = np.zeros(shape, bool)
    a[y0:y1, x0:x1] = True
    return a


class BoundaryTests(unittest.TestCase):
    def test_band_is_the_inner_ring_of_width_d(self):
        valid = np.ones((40, 40), bool)
        band = boundary_band(square((40, 40), 10, 30, 10, 30), valid, 3)
        self.assertTrue(band[10, 20] and band[12, 20])     # within 3 px of the contour
        self.assertFalse(band[20, 20])                      # interior
        self.assertFalse(band[5, 5])                        # outside the region
        self.assertEqual(int(band.sum()), 20 * 20 - 14 * 14)

    def test_identical_masks_score_one_and_disjoint_zero(self):
        valid = np.ones((40, 40), bool)
        a = square((40, 40), 5, 20, 5, 20)
        self.assertEqual(boundary_counts(a, a, valid, 2)[0], boundary_counts(a, a, valid, 2)[1])
        inter, union = boundary_counts(a, square((40, 40), 25, 35, 25, 35), valid, 2)
        self.assertEqual(inter, 0)
        self.assertGreater(union, 0)

    def test_edge_of_scored_area_is_not_a_contour(self):
        valid = square((40, 40), 0, 40, 0, 20)             # right half unscored
        region = square((40, 40), 10, 30, 0, 40)           # crosses the scored edge
        band = boundary_band(region, valid, 2)
        self.assertFalse(band[20, 19])                     # next to unscored, not a contour
        self.assertTrue(band[10, 19])                      # real top contour
        self.assertFalse(band[:, 20:].any())               # nothing outside the scored area

    def test_large_object_boundary_shift_hurts_more_than_mask_iou(self):
        valid = np.ones((200, 200), bool)
        gt = square((200, 200), 20, 180, 20, 180)
        pred = square((200, 200), 23, 183, 23, 183)        # shifted by 3 px
        mask_iou = (gt & pred).sum() / (gt | pred).sum()
        inter, union = boundary_counts(gt, pred, valid, 4)
        self.assertLess(inter / union, mask_iou)


class BackgroundTileTests(unittest.TestCase):
    def test_counts_and_rejects_oil(self):
        pred = np.zeros((4, 4), bool)
        pred[0, :2] = True
        self.assertEqual(bg_tile_counts(pred, np.ones((4, 4), np.uint8)), (2, 16))
        gt = np.ones((4, 4), np.uint8)
        gt[3, 3] = 0
        with self.assertRaisesRegex(ValueError, "ground-truth oil"):
            bg_tile_counts(pred, gt)


if __name__ == "__main__":
    unittest.main()
