import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyze_lost_oil import classify, tp_at  # noqa: E402
from oil_component_prob import CROP_PAD, scene_records  # noqa: E402
from threshold_sweep import EDGES  # noqa: E402


class ComponentTests(unittest.TestCase):
    def test_records_hist_and_crop_bits(self):
        gt = np.ones((60, 60), np.uint8)
        gt[10:20, 10:20] = 0            # 100 px component
        gt[40:42, 40:45] = 0            # 10 px: below MIN_PX, dropped
        tested = np.ones_like(gt, bool)
        tested[:, 55:] = False
        prob = np.zeros(gt.shape, np.float32)
        prob[10:15, 10:20] = 0.9        # half the component above 0.5
        recs = list(scene_records("s", gt, prob, tested))
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual((r["area"], r["bbox"]), (100, (10, 10, 10, 10)))
        self.assertEqual(tp_at(r["hist"][None], EDGES, 0.5)[0], 50)
        self.assertEqual(r["origin"], (0, 0))
        self.assertEqual(r["crop"].shape, (20 + CROP_PAD, 20 + CROP_PAD))
        bits = r["bits"]
        self.assertEqual(int((bits & 1).sum()), 100)
        self.assertEqual(int(((bits & 2) > 0).sum()), 108)    # + 8 px of the small one (crop ends at 44)
        self.assertEqual(r["crop"][12, 12], round(255 * 0.9))


class ClassifyTests(unittest.TestCase):
    def test_classes_and_unrecoverable(self):
        df = pd.DataFrame(dict(area=[100] * 5, tp_A=[90, 90, 5, 60, 0],
                               tp_C=[5, 5, 80, 40, 0], tplow_C=[50, 5, 90, 50, 0]))
        cls, unrec = classify(df, "C", "A")
        self.assertEqual(list(cls), ["lost", "lost", "gained", "shrunk", "both_missed"])
        self.assertEqual(list(unrec), [False, True, False, False, False])


if __name__ == "__main__":
    unittest.main()
