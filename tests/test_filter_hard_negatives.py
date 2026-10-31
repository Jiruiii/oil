import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from filter_hard_negatives import build_arms, train_share  # noqa: E402


class FilterHardNegativesTests(unittest.TestCase):
    def test_train_share_excludes_self(self):
        train = np.array([[0., 1.], [0., 1.]])
        hard = np.array([[1., 0.], [.9, .1]])
        q = np.array([[1., 0.]])
        self.assertEqual(train_share(q, train, hard, np.array([-1]), k=2)[0], 0.0)
        # Without itself, the nearest two are the other hard vector and a train vector.
        self.assertEqual(train_share(q, train, hard, np.array([0]), k=2)[0], 0.5)

    def test_build_arms_keeps_counts_and_matches_scenes(self):
        sel = {"a": [("a", 0, i) for i in range(4)], "b": [("b", 0, i) for i in range(3)]}
        uns = {"a": [("a", 1, i) for i in range(3)], "b": [("b", 1, i) for i in range(5)],
               "c": [("c", 1, i) for i in range(4)]}
        score = {k: 0.0 for v in list(sel.values()) + list(uns.values()) for k in v}
        score.update({("a", 0, 0): .5, ("a", 0, 1): .3, ("b", 0, 0): .9,   # too close: dropped
                      ("a", 0, 2): np.nan,                                  # unscored: kept
                      ("a", 1, 0): .4, ("a", 1, 1): np.nan})               # not addable
        knn, swap, stats = build_arms(sel, uns, score, fold=1, tau=.3)
        self.assertEqual(len(knn), 7)
        self.assertEqual(len(set(knn)), 7)
        for k in [("a", 0, 0), ("a", 0, 1), ("b", 0, 0)]:
            self.assertNotIn(k, knn)
        self.assertIn(("a", 0, 2), knn)
        self.assertTrue(all(not score[k] >= .3 or k in sel.get(k[0], []) for k in knn))
        self.assertNotIn(("a", 1, 0), knn)
        self.assertNotIn(("a", 1, 1), knn)
        # Scene a has one addable window for two drops: one slot filled from another scene.
        self.assertEqual(stats["a"], {"selected": 4, "dropped": 2, "added_same_scene": 1})
        self.assertEqual(sum(v.get("added_other_scene", 0) for v in stats.values()), 1)
        self.assertEqual(Counter(k[0] for k in knn), Counter(k[0] for k in swap))
        self.assertEqual(len(set(swap)), 7)
        self.assertEqual(sum(k in sel["a"] for k in swap), 2)
        self.assertEqual(sum(k in sel["b"] for k in swap), 2)

    def test_build_arms_is_deterministic(self):
        sel = {"a": [("a", 0, i) for i in range(6)]}
        uns = {"a": [("a", 1, i) for i in range(10)]}
        score = {("a", 0, i): (.9 if i < 3 else 0.) for i in range(6)}
        score.update({("a", 1, i): 0. for i in range(10)})
        self.assertEqual(build_arms(sel, uns, score, 2), build_arms(sel, uns, score, 2))


if __name__ == "__main__":
    unittest.main()
