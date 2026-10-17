import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.transform import from_origin

from main.hnm_config import apply_hnm_to_fold
from preprocess.hnm.data import Patch, read_background, sha256, write_background
from preprocess.hnm_si import mining
from preprocess.hnm_si.config import load_config
from preprocess.hnm_si.indices import (band_index, compute_si, model_input, model_valid_mask,
                                       reflectance, sensor_of)
from preprocess.hnm_si.pipeline import mine_fold, select_fold
from preprocess.hnm_si.stats import (ledoit_wolf_reference, lower_cutoff, mahalanobis,
                                     outlier_scenes, stratified_select)

NAMES = ["B01", "B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
MEAN = [0.275] * 8
STD = [0.05] * 8


def fake_probe(batch):
    """P(oil) is high where the normalized B03 channel is positive: ~half the pixels."""
    return np.where(batch[:, 2] > 0, .9, .1).astype(np.float32)


class IndexTests(unittest.TestCase):
    def test_si_formulas_and_band_mapping(self):
        refl = np.full((8, 2, 2), .2, dtype=np.float32)
        refl[0], refl[1], refl[2], refl[5], refl[6] = .1, .2, .3, .4, .1
        si, ok = compute_si(refl, band_index(NAMES))
        self.assertTrue(ok.all())
        np.testing.assert_allclose(si[0], (.3 - .1) / (.3 + .1), rtol=1e-5)    # MNDWI
        np.testing.assert_allclose(si[1], 0.0, atol=1e-6)                        # (B01-B11)/(B01+B11)
        np.testing.assert_allclose(si[2], (.1 + .2) / (.3 + .4), rtol=1e-5)    # SR

    def test_reflectance_removes_additive_offset_but_model_input_keeps_it(self):
        raw = np.full((8, 1, 1), 1500., dtype=np.float32)
        np.testing.assert_allclose(reflectance(raw, 10000.0, 1000.0), 0.05)
        np.testing.assert_allclose(model_input(raw, 10000.0, [0.] * 8, [1.] * 8), 0.15)
        # Nodata (DN 0) falls below the offset and is therefore invalid for indices.
        _, ok = compute_si(reflectance(np.zeros((8, 1, 1), np.float32), 10000.0, 1000.0),
                           band_index(NAMES))
        self.assertFalse(ok.any())

    def test_invalid_pixels_are_nan_not_zero(self):
        refl = np.full((8, 1, 4), .2, dtype=np.float32)
        refl[6, 0, 0] = 0          # nodata band -> zero denominator share
        refl[0, 0, 1] = np.nan
        refl[2, 0, 2] = 1e37       # float nodata fill
        si, ok = compute_si(refl, band_index(NAMES))
        self.assertEqual(ok.tolist(), [[False, False, False, True]])
        self.assertTrue(np.isnan(si[:, ~ok]).all())
        self.assertTrue(np.isfinite(si[:, ok]).all())

    def test_unknown_or_duplicate_bands_rejected(self):
        with self.assertRaises(ValueError):
            band_index(["B01", "B02"])
        with self.assertRaises(ValueError):
            band_index(NAMES + ["B01"])

    def test_model_input_matches_segmentation_loader(self):
        raw = np.array([[[1500., np.nan, 0., 2e9]]] * 8, dtype=np.float32)
        out = model_input(raw, 10000.0, MEAN, STD)
        np.testing.assert_allclose(out[0, 0], [(0.15 - .275) / .05, (0 - .275) / .05,
                                               (0 - .275) / .05, (0 - .275) / .05], rtol=1e-5)
        self.assertEqual(model_valid_mask(raw).tolist(), [[True, False, False, True]])

    def test_sensor_parsing(self):
        self.assertEqual(sensor_of("NOAA_Gulf_High_Confidence_20190320_L8_15RUL"), "L8")
        self.assertEqual(sensor_of("2023年事件_20230801_L9_50QRL"), "L9")
        self.assertEqual(sensor_of("x_20200101_S2_16RCT"), "S2")
        self.assertEqual(sensor_of("weird"), "unknown")


class StatsTests(unittest.TestCase):
    def test_mahalanobis_reduces_to_euclidean_and_keeps_nan(self):
        d = mahalanobis([[3., 4., 0.], [np.nan, 0, 0]], [[0, 0, 0]] * 2, np.eye(3))
        self.assertAlmostEqual(d[0], 5.0)
        self.assertTrue(np.isnan(d[1]))

    def test_ledoit_wolf_reference(self):
        rng = np.random.default_rng(0)
        ref = ledoit_wolf_reference(rng.normal(size=(500, 3)))
        self.assertEqual(ref["n"], 500)
        self.assertEqual(ref["precision"].shape, (3, 3))
        self.assertTrue(0 <= ref["shrinkage"] <= 1)
        with self.assertRaises(ValueError):
            ledoit_wolf_reference(np.array([[1., np.nan, 0.]]))

    def test_cutoff_disabled_and_nan_ignored(self):
        self.assertIsNone(lower_cutoff([1, 2, 3], 0))
        self.assertAlmostEqual(lower_cutoff([np.nan, 1, 2, 3, 4], .5), 2.5)

    def test_outlier_scenes(self):
        scenes = ["a"] * 2 + ["b"] * 2 + ["c"] * 2 + ["d"] * 30
        self.assertEqual(outlier_scenes(scenes, 1.0), {"d"})
        self.assertEqual(outlier_scenes(scenes, None), set())

    def test_stratified_select_exact_unique_deterministic(self):
        pool = {"a": [("a", i, 0) for i in range(2)], "b": [("b", i, 0) for i in range(10)],
                "c": [("c", i, 0) for i in range(3)]}
        got = stratified_select(pool, 9, 1)
        self.assertEqual(len(got), 9)
        self.assertEqual(len(set(got)), 9)
        self.assertEqual(got, stratified_select(pool, 9, 1))
        counts = {s: sum(k[0] == s for k in got) for s in pool}
        self.assertEqual((counts["a"], counts["c"]), (2, 3))     # small scenes fully used
        self.assertEqual(len(stratified_select(pool, 100, 1)), 15)  # never replaces


class WindowTests(unittest.TestCase):
    def test_eligibility_rules(self):
        m = np.ones((32, 32), dtype=np.uint8)
        m[:16, :16] = 0           # oil window
        m[16:, 16:] = 255         # unlabeled window
        rules = {"min_mask1_ratio": .5, "max_mask255_ratio": .3}
        got = {(x, y) for y, x, *_ in mining.eligible_windows(m, 16, 16, rules)}
        self.assertEqual(got, {(16, 0), (0, 16)})
        got = {(x, y) for y, x, *_ in mining.eligible_windows(m, 16, 16, rules, {(16, 0)})}
        self.assertEqual(got, {(0, 16)})

    def test_others_are_not_confirmed_background(self):
        m = np.full((16, 16), 2, dtype=np.uint8)
        m[:4] = 1
        rules = {"min_mask1_ratio": .5, "max_mask255_ratio": .3}
        self.assertEqual(mining.eligible_windows(m, 16, 16, rules), [])

    def test_edge_window_included_and_small_raster_empty(self):
        self.assertEqual(mining.window_positions(40, 40, 16, 16), ([0, 16, 24], [0, 16, 24]))
        self.assertEqual(mining.window_positions(8, 40, 16, 16), ([], []))

    def test_bad_mask_encoding_rejected(self):
        with self.assertRaisesRegex(ValueError, "mask encoding"):
            mining.check_mask(np.full((4, 4), 7, dtype=np.uint8), "s")


class PipelineTests(unittest.TestCase):
    N_TRAIN = 6

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for d in ("splits", "rasters", "masks", "run1/weights"):
            (self.root / d).mkdir(parents=True)
        self.train = [f"train{i}_20200101_{'S2' if i % 2 else 'L8'}_T{i}" for i in range(self.N_TRAIN)]
        self.scenes = self.train + ["val_20200101_S2_TV", "test_20200101_S2_TT"]
        rng = np.random.default_rng(3)
        meta = dict(driver="GTiff", height=64, width=64, transform=from_origin(0, 640, 10, 10),
                    crs="EPSG:32631")
        for scene in self.scenes:
            image = rng.integers(1500, 4000, (8, 64, 64)).astype(np.uint16)
            mask = np.ones((64, 64), dtype=np.uint8)
            mask[:16, :16] = 0          # oil
            mask[48:, 32:48] = 255      # unlabeled
            if scene == "train5_20200101_S2_T5":
                mask[:16, :16] = 1      # a training scene without any oil reference
            with rasterio.open(self.root / "rasters" / f"{scene}.tif", "w", count=8,
                               dtype="uint16", **meta) as ds:
                ds.write(image)
            with rasterio.open(self.root / "masks" / f"{scene}.tif", "w", count=1,
                               dtype="uint8", **meta) as ds:
                ds.write(mask, 1)
        for split, scenes in (("train", self.train), ("val", ["val_20200101_S2_TV"]),
                              ("test", ["test_20200101_S2_TT"])):
            (self.root / "splits" / f"{split}_fold1.txt").write_text(
                "".join(f"{s}_patch_x0_y0\n" for s in scenes))
        baseline = [Patch(s, 16, 16, 0) for s in self.scenes]
        write_background(self.root / "bg.tsv", baseline)
        (self.root / "run1" / "weights" / "best.pt").write_bytes(b"probe")
        (self.root / "run1" / "args.yaml").write_text(yaml.safe_dump(
            {"architecture_cfg": {"in_channels": 8, "input_band_names": NAMES},
             "dataset": {"mean": MEAN, "std": STD, "target_class_index": 0}}))
        self.config_path = self.root / "config.yaml"
        self.write_config({})
        self.config = load_config(self.config_path)

    def write_config(self, overrides):
        config = {"split_dir": "splits", "baseline_bg_tsv": "bg.tsv", "vrt_dir": "rasters",
                  "mask_dir": "masks", "output_dir": "out", "outer_folds": [1],
                  "probe_run_dirs": {1: "run1"}, "patch_size": 16, "stride": 16,
                  "batch_size": 4, "device": "cpu", "amp": False,
                  "si": {"min_fp_pixels": 20, "min_oil_pool": 500, "top_q": .1}}
        for k, v in overrides.items():
            config[k] = config.get(k, {}) | v if isinstance(v, dict) and isinstance(config.get(k), dict) else v
        self.config_path.write_text(yaml.safe_dump(config))
        self.config = load_config(self.config_path)

    def mine(self, **kw):
        return mine_fold(self.config, 1, None, probe_factory=lambda *a: fake_probe,
                         progress=False, **kw)

    def fold_config(self, arm):
        return {"hnm": {"enabled": True, "artifacts_dir": str(self.root / "out" / "arms" / arm)},
                "cross_validation": {"split_dir": str(self.root / "splits")},
                "dataset": {"bg_coord_txt": str(self.root / "bg.tsv"),
                            "vrt_dir": str(self.root / "rasters"),
                            "labels_dir": str(self.root / "masks"), "patch_size": 16,
                            **{f"{s}_txt": str(self.root / "splits" / f"{s}_fold1.txt")
                               for s in ("train", "val", "test")}}}

    def arm_dir(self, arm):
        return self.root / "out" / "arms" / arm / "fold_1"

    def test_mining_scans_only_training_scenes_with_train_only_reference(self):
        manifest = self.mine()
        self.assertEqual(manifest["status"], "complete")
        mdir = self.root / "out" / "mining" / "fold_1"
        table = pd.read_csv(mdir / "candidates.tsv", sep="\t")
        refs = pd.read_csv(mdir / "scene_reference.csv")
        self.assertEqual(set(table["scene"]), set(self.train))
        self.assertEqual(set(refs["scene"]), set(self.train))
        # oil window (0,0) is GT-centric and contains oil; unlabeled window is gated out
        self.assertEqual(len(table[table["scene"] == self.train[0]]), 14)
        self.assertFalse(((table["x"] == 0) & (table["y"] == 0)).any())
        self.assertFalse(((table["x"] == 32) & (table["y"] == 48)).any())
        no_ref = refs.set_index("scene").loc["train5_20200101_S2_T5"]
        self.assertFalse(bool(no_ref["reference_valid"]))
        self.assertEqual(np.load(mdir / "oil_pool.npy").shape, (5 * 256, 3))

    def test_all_arms_produce_valid_artifacts_the_trainer_accepts(self):
        self.mine()
        for arm in ("hnm_si", "hnm", "random"):
            manifest = select_fold(self.config, 1, arm)
            self.assertEqual((manifest["status"], manifest["selected_count"]), ("complete", 6))
            cfg = self.fold_config(arm)
            apply_hnm_to_fold(cfg, 1)
            combined = read_background(cfg["dataset"]["bg_coord_txt"])
            baseline = read_background(self.root / "bg.tsv")
            self.assertEqual([r for r in combined if r.scene not in self.train],
                             [r for r in baseline if r.scene not in self.train])
            selected = read_background(self.arm_dir(arm) / "selected_train_bg.tsv")
            self.assertTrue(all(r.scene in self.train and (r.x, r.y) != (0, 0) for r in selected))
            self.assertEqual(len({r.key for r in selected}), 6)

    def test_si_arm_excludes_scene_without_reference_and_closest_candidates(self):
        self.mine()
        select_fold(self.config, 1, "hnm_si")
        scored = pd.read_csv(self.arm_dir("hnm_si") / "candidates_scored.tsv", sep="\t")
        no_ref = scored[scored["scene"] == "train5_20200101_S2_T5"]
        self.assertTrue(set(no_ref["status"]) <= {"no_oil_reference", "no_fp",
                                                   "fp_region_below_si_min"})
        ref = json.loads((self.arm_dir("hnm_si") / "si_reference.json").read_text())
        dropped = scored[scored["status"] == "too_close_to_oil"]
        kept = scored[scored["status"].isin(["selected", "not_selected_quota"])]
        self.assertGreater(len(dropped), 0)
        self.assertTrue((dropped["d_si"] <= ref["cutoff"]).all())
        self.assertTrue((kept["d_si"] > ref["cutoff"]).all())
        self.assertEqual(ref["pool_pixels"], 5 * 256)

    def test_hnm_arm_only_takes_windows_with_false_positives(self):
        self.mine()
        select_fold(self.config, 1, "hnm")
        scored = pd.read_csv(self.arm_dir("hnm") / "candidates_scored.tsv", sep="\t")
        self.assertTrue((scored.loc[scored["status"] == "selected", "n_fp"] >= 1).all())

    def test_insufficient_candidates_fail_loudly_unless_allowed(self):
        self.mine()
        self.write_config({"selection": {"target_count": 10_000}})
        with self.assertRaisesRegex(ValueError, "qualified candidates"):
            select_fold(self.config, 1, "hnm")
        report = json.loads((self.arm_dir("hnm") / "selection_report.json").read_text())
        self.assertEqual(report["outcome"], "insufficient_candidates")
        self.assertFalse((self.arm_dir("hnm") / "manifest.json").exists())
        self.write_config({"selection": {"target_count": 10_000, "allow_shortfall": True}})
        manifest = select_fold(self.config, 1, "hnm")
        self.assertEqual(manifest["shortfall"], 10_000 - manifest["selected_count"])
        apply_hnm_to_fold(self.fold_config("hnm"), 1)

    def test_partial_mining_cannot_be_selected_or_trained(self):
        manifest = self.mine(max_scenes=2)
        self.assertEqual(manifest["status"], "partial")
        with self.assertRaisesRegex(ValueError, "partial"):
            select_fold(self.config, 1, "random")
        self.write_config({"selection": {"target_count": 3}})
        result = select_fold(self.config, 1, "random", allow_partial=True)
        self.assertEqual(result["status"], "partial")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            apply_hnm_to_fold(self.fold_config("random"), 1)

    def test_resume_skips_finished_scenes_and_rejects_changed_settings(self):
        self.mine()
        calls = []

        def counting_probe(batch):
            calls.append(len(batch))
            return fake_probe(batch)

        mine_fold(self.config, 1, None, probe_factory=lambda *a: counting_probe, progress=False)
        self.assertEqual(calls, [])      # every scene was already complete
        self.write_config({"rules": {"prob_threshold": .7}})
        with self.assertRaisesRegex(ValueError, "differs"):
            self.mine()

    def test_selection_only_settings_do_not_require_remining(self):
        self.mine()
        self.write_config({"si": {"top_q": .3, "min_oil_pool": 100}})
        select_fold(self.config, 1, "hnm_si")
        self.write_config({"si": {"reflectance_offset": 0.0}})
        with self.assertRaisesRegex(ValueError, "Scan settings differ"):
            select_fold(self.config, 1, "random")

    def test_selection_refuses_changed_inputs(self):
        self.mine()
        (self.root / "out" / "mining" / "fold_1" / "candidates.tsv").write_text("scene\n")
        with self.assertRaisesRegex(ValueError, "changed after completion"):
            select_fold(self.config, 1, "hnm")

    def test_selection_refuses_changed_split(self):
        self.mine()
        split = self.root / "splits" / "val_fold1.txt"
        split.write_text(split.read_text() + "val_20200101_S2_TV_patch_x16_y16\n")
        with self.assertRaisesRegex(ValueError, "Frozen splits"):
            select_fold(self.config, 1, "hnm")

    def test_trainer_rejects_tampered_hnm_si_records(self):
        self.mine()
        select_fold(self.config, 1, "hnm_si")
        out = self.arm_dir("hnm_si")
        apply_hnm_to_fold(self.fold_config("hnm_si"), 1)
        original = (out / "candidates_scored.tsv").read_text()
        (out / "candidates_scored.tsv").write_text(original.replace("\tselected\t", "\tqualified\t", 1))
        with self.assertRaisesRegex(ValueError, "candidate records checksum"):
            apply_hnm_to_fold(self.fold_config("hnm_si"), 1)
        # Recomputing the hash must not hide a record/selection disagreement.
        manifest = json.loads((out / "manifest.json").read_text())
        manifest["candidates_sha256"] = sha256(out / "candidates_scored.tsv")
        (out / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "differ from the candidate records"):
            apply_hnm_to_fold(self.fold_config("hnm_si"), 1)

    def test_reselecting_into_finished_output_is_refused(self):
        self.mine()
        select_fold(self.config, 1, "hnm")
        with self.assertRaises(FileExistsError):
            select_fold(self.config, 1, "hnm")

    def test_oil_class_index_must_be_zero(self):
        args = yaml.safe_load((self.root / "run1" / "args.yaml").read_text())
        args["dataset"]["target_class_index"] = 1
        (self.root / "run1" / "args.yaml").write_text(yaml.safe_dump(args))
        with self.assertRaisesRegex(ValueError, "class is index 0"):
            self.mine()

    def test_config_validation(self):
        for bad in ({"arm": "nope"}, {"si": {"top_q": 1.5}}, {"rules": {"typo": 1}},
                    {"selection": {"target_count": 0}}, {"outer_folds": [2]}, {"unknown": 1}):
            with self.assertRaises(ValueError, msg=str(bad)):
                self.write_config(bad)


if __name__ == "__main__":
    unittest.main()
