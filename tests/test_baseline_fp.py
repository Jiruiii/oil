import importlib.util
import unittest
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "analyze_baseline_fp", Path(__file__).resolve().parents[1] / "tools" / "analyze_baseline_fp.py")
fp_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fp_tool)

SEA = [.03, .03, .03, .02, .01, .01, .005, .004]       # B01..B12 reflectance
LAND = [.05, .05, .08, .06, .3, .3, .2, .12]           # vegetation
CLOUD = [.3, .3, .3, .3, .3, .3, .2, .15]
HAZE = [.07, .07, .06, .06, .05, .05, .05, .04]


def raw_from(refl_by_pixel, shape):
    """(8, H, W) DN with the +1000 offset from a per-pixel reflectance list."""
    raw = np.empty((8, *shape), np.float32)
    for y in range(shape[0]):
        for x in range(shape[1]):
            raw[:, y, x] = 1000 + 10000 * np.asarray(refl_by_pixel(y, x))
    return raw


class SpectralTests(unittest.TestCase):
    def test_classes(self):
        refl = np.stack([np.array(v, np.float32) for v in (SEA, LAND, CLOUD, HAZE)],
                        axis=1)[:, None, :]
        valid = np.ones((1, 4), bool)
        np.testing.assert_array_equal(fp_tool.spectral_class(refl, valid),
                                      [[fp_tool.WATER, fp_tool.LAND, fp_tool.CLOUD, fp_tool.HAZE]])
        np.testing.assert_array_equal(fp_tool.spectral_class(refl, ~valid), [[fp_tool.NODATA] * 4])


class WindowTests(unittest.TestCase):
    def test_categories_precedence_and_regions(self):
        h, w = 96, 96
        mask = np.ones((h, w), np.uint8)
        mask[:, :4] = 0                                   # GT oil on the left edge
        mask[70:, 70:] = 255
        land = np.zeros((h, w), bool)
        land[20:30, 30:40] = True
        raw = raw_from(lambda y, x: LAND if land[y, x] else SEA, (h, w))
        raw[:, 60:, :2] = 0                               # nodata corner
        pred = np.zeros((h, w), bool)
        pred[:, 4:20] = True                              # hugs the oil, then extends past N
        pred[20:30, 30:40] = True                         # land
        pred[80:88, 80:88] = True                         # open sea on unlabeled pixels
        pred[60:64, 26:30] = True                         # sea pixels next to land -> coast
        scored = np.ones((h, w), bool)
        cat, dist, regions = fp_tool.analyse_window(pred, scored, mask, raw, 10000., 1000., 8)
        name = lambda y, x: fp_tool.CATEGORIES[cat[y, x]]
        self.assertEqual(name(10, 5), "near_oil")
        self.assertEqual(name(10, 18), "oil_attached")
        self.assertEqual(name(25, 35), "land_coast")
        self.assertEqual(name(62, 28), "land_coast")
        self.assertEqual(name(84, 84), "other_sea")
        self.assertEqual(cat[0, 0], -1)                   # correct oil is not FP
        self.assertEqual(dist[0, 0], 0)
        self.assertEqual(len(regions), 4)
        sea = [r for r in regions if r["category"] == "other_sea"][0]
        self.assertEqual((sea["area"], sea["n_unlabeled"], sea["n_confirmed_bg"]), (64, 64, 0))

    def test_bright_small_region(self):
        h, w = 48, 48
        mask = np.ones((h, w), np.uint8)
        ship = np.zeros((h, w), bool)
        ship[20:24, 20:24] = True
        bright = [v + .05 for v in SEA]
        raw = raw_from(lambda y, x: bright if ship[y, x] else SEA, (h, w))
        _, _, regions = fp_tool.analyse_window(ship, np.ones((h, w), bool), mask, raw,
                                               10000., 1000., 8)
        self.assertEqual(regions[0]["category"], "bright_small")
        self.assertGreater(regions[0]["contrast"], .02)


if __name__ == "__main__":
    unittest.main()
