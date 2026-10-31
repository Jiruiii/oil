import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn

import tests.test_hnm_si as base
from main.hnm_config import apply_hnm_to_fold
from preprocess.hnm import models
from preprocess.hnm.data import Patch, read_background
from preprocess.hnm_si.oof import (load_scores, mine_oof, negative_rows, read_group, row_groups,
                                   scene_inner_folds)
from preprocess.hnm_si.pipeline import select_fold


class BandMiner(nn.Module):
    """P(oil) rises with the normalized B03 mean, so some windows pass 0.5 and some do not."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        z = x[:, 2].mean((1, 2)) * 4 + 0 * self.w.sum()
        return torch.stack([torch.zeros_like(z), z], 1)


class ReadingTests(unittest.TestCase):
    def test_row_groups_cover_each_window_once_in_short_strips(self):
        coords = [(0, 0), (32, 0), (0, 16), (0, 40), (16, 100)]
        groups = row_groups(coords, 16)
        self.assertEqual(sorted(c for g in groups for c in g), sorted(coords))
        for g in groups:
            self.assertLessEqual(max(y for _, y in g) - g[0][1], 16)

    def test_read_group_matches_single_window_reads(self):
        class Src:
            def __init__(self):
                self.a = np.arange(2 * 64 * 64, dtype=np.float32).reshape(2, 64, 64)

            def read(self, window, boundless, fill_value):
                r, c = window.row_off, window.col_off
                return self.a[:, r:r + window.height, c:c + window.width]
        src = Src()
        for (x, y), w in read_group(src, [(0, 0), (40, 8), (16, 20)], 16, 0).items():
            np.testing.assert_array_equal(w, src.a[:, y:y + 16, x:x + 16])

    def test_negative_rows_take_whole_rows_spread_over_scenes(self):
        pool = pd.DataFrame([(s, x, y) for s in ("a", "b", "c") for y in (0, 16, 32)
                             for x in (0, 16)], columns=["scene", "x", "y"])
        picked = negative_rows(pool, 6, 0)
        self.assertEqual(len(picked), 6)
        self.assertEqual(sorted({p.scene for p in picked}), ["a", "b", "c"])
        rows = {(p.scene, p.y) for p in picked}
        self.assertEqual(len(rows), 3)            # three complete rows of two windows
        self.assertEqual(negative_rows(pool, 100, 0).__len__(), len(pool))

    def test_scene_cannot_be_held_out_twice(self):
        records = [Patch("a", 0, 0, 1), Patch("a", 16, 0, 0)]
        with self.assertRaisesRegex(ValueError, "two inner folds"):
            scene_inner_folds(records, [([1], [0]), ([0], [1])])


class OofPipelineTests(unittest.TestCase):
    setUp = base.PipelineTests.setUp
    write_config = base.PipelineTests.write_config
    mine = base.PipelineTests.mine
    fold_config = base.PipelineTests.fold_config
    arm_dir = base.PipelineTests.arm_dir
    N_TRAIN = base.PipelineTests.N_TRAIN

    def mine_oof(self):
        self.write_config({"oof": {"inner_folds": 3, "epochs": 1, "batch_size": 4, "workers": 0,
                                   "pretrained": False, "neg_pool": 40, "infer_batch_size": 5}})
        with patch.object(models, "create_model", side_effect=lambda *a, **k: BandMiner()):
            return mine_oof(self.config, 1, torch.device("cpu"), progress=False)

    def test_every_pool_window_scored_by_a_classifier_that_never_saw_its_scene(self):
        self.mine()
        manifest = self.mine_oof()
        self.assertEqual(manifest["status"], "complete")
        out = self.root / "out" / "oof" / "fold_1"
        scores = pd.read_csv(out / "scores.tsv", sep="\t")
        pool = pd.read_csv(self.root / "out" / "mining" / "fold_1" / "candidates.tsv", sep="\t")
        self.assertEqual(len(scores), len(pool))
        plan = json.loads((out / "plan.json").read_text())
        self.assertTrue((scores["inner_fold"] == scores["scene"].map(plan["scene_inner_fold"])).all())
        self.assertEqual(sorted(set(plan["scene_inner_fold"].values())), [1, 2, 3])
        self.assertTrue(scores["p_oil"].between(0, 1).all())
        # window SI is computed over mask==1 pixels only; every pool window has enough of them
        self.assertTrue(scores[["w_mndwi", "w_ndi", "w_sr"]].notna().all().all())
        self.assertTrue((scores["n_si_px"] >= 20).all())

    def test_oof_arms_select_hard_windows_and_the_trainer_accepts_them(self):
        self.mine()
        self.mine_oof()
        for arm in ("oof", "oof_si"):
            manifest = select_fold(self.config, 1, arm)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["probe"]["scope"], "scene_oof_within_fold_train")
            table = pd.read_csv(self.arm_dir(arm) / "candidates_scored.tsv", sep="\t")
            chosen = table[table["status"] == "selected"]
            self.assertTrue((chosen["p_oil"] >= .5).all())
            self.assertTrue((table.loc[table["p_oil"] < .5, "status"] == "oof_not_oil").all())
            cfg = self.fold_config(arm)
            apply_hnm_to_fold(cfg, 1)
            selected = read_background(self.arm_dir(arm) / "selected_train_bg.tsv")
            self.assertEqual(len(selected), manifest["selected_count"])
        si = pd.read_csv(self.arm_dir("oof_si") / "candidates_scored.tsv", sep="\t")
        self.assertTrue((si.loc[si["scene"] == "train5_20200101_S2_T5", "status"]
                         .isin(["no_oil_reference", "oof_not_oil"])).all())
        self.assertIn("too_close_to_oil", set(si["status"]))

    def test_selection_refuses_tampered_oof_scores(self):
        self.mine()
        self.mine_oof()
        path = self.root / "out" / "oof" / "fold_1" / "scores.tsv"
        path.write_text(path.read_text().replace("\t1\t", "\t2\t", 1))
        with self.assertRaisesRegex(ValueError, "changed after completion"):
            select_fold(self.config, 1, "oof")

    def test_oof_requires_completed_mining(self):
        with self.assertRaises(FileNotFoundError):
            load_scores(self.config, 1, {})


if __name__ == "__main__":
    unittest.main()
