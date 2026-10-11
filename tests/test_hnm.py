import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch
from torch import nn
import yaml

from main.hnm_config import apply_hnm_to_fold
from preprocess.hnm.data import (Patch, WindowDataset, oof_splits, read_background,
                                 read_patches, sha256, split_manifest, validate_labels,
                                 write_background)
from preprocess.hnm.pipeline import load_config, mine_fold, prepare_fold
from preprocess.hnm.sampling import facility_location, select_negatives


class TinyMiner(nn.Module):
    """A fast real training model for exercising the complete I/O pipeline."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(8, 4, 1)
        self.fc = nn.Linear(4, 2)

    def forward_features(self, x):
        return self.conv(x).relu()

    def forward_head(self, x, pre_logits=False):
        x = x.mean((2, 3))
        return x if pre_logits else self.fc(x)

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


class SamplingTests(unittest.TestCase):
    def test_zero_quota_selects_nothing(self):
        f = np.eye(3, dtype=np.float32)
        self.assertEqual(facility_location(f, 0), [])
        self.assertEqual(select_negatives([.1, .2, .3], f, 0), ([], []))

    def test_exact_unique_count_and_highest_scores(self):
        rng = np.random.default_rng(42)
        f = rng.normal(size=(20, 5))
        scores = np.arange(20, dtype=float)
        hard, diverse = select_negatives(scores, f, 10)
        self.assertEqual(hard, [19, 18, 17, 16, 15])
        self.assertEqual(len(set(hard + diverse)), 10)
        self.assertEqual((hard, diverse), select_negatives(scores, f, 10))
        h, d = select_negatives(scores, f, 100)
        self.assertEqual(len(set(h + d)), 20)

    def test_oof_never_sees_heldout_group(self):
        records = [Patch(f"scene{i}", label, 0, label) for i in range(10) for label in (0, 1)]
        groups = {f"scene{i}": f"event{i // 2}" for i in range(10)}
        splits = oof_splits(records, 5, groups=groups)
        coverage = []
        for train, heldout in splits:
            self.assertFalse({groups[records[i].scene] for i in train} &
                             {groups[records[i].scene] for i in heldout})
            coverage.extend(heldout)
        self.assertEqual(sorted(coverage), list(range(len(records))))

    def test_reject_nonfinite_embeddings_even_when_keeping_all(self):
        with self.assertRaises(ValueError):
            select_negatives([.1, .2], [[0., 1.], [float('nan'), 0.]], 2)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ("splits", "rasters", "masks"):
            (self.root / name).mkdir()
        self.train_scenes = [f"train{i}" for i in range(6)]
        self.scenes = self.train_scenes + ["val", "test"]
        rng = np.random.default_rng(7)
        for scene in self.scenes:
            meta = dict(driver="GTiff", height=16, width=32,
                        transform=from_origin(100, 100, 10, 10), crs="EPSG:32631")
            with rasterio.open(self.root / "rasters" / f"{scene}.tif", "w",
                               count=8, dtype="uint16", **meta) as ds:
                a = rng.integers(100, 9000, (8, 16, 32), dtype=np.uint16)
                ds.write(a)
            with rasterio.open(self.root / "masks" / f"{scene}.tif", "w",
                               count=1, dtype="uint8", **meta) as ds:
                a = np.ones((16, 32), dtype=np.uint8)
                a[:, :16] = 0
                ds.write(a, 1)
        for split, scenes in (("train", self.train_scenes), ("val", ["val"]), ("test", ["test"])):
            (self.root / "splits" / f"{split}_fold1.txt").write_text(
                "".join(f"{s}_patch_x0_y0\n" for s in scenes))
        write_background(self.root / "bg.tsv", [Patch(s, 16, 0, 0) for s in self.scenes])
        self.config_path = self.root / "config.yaml"
        self.config_path.write_text(yaml.safe_dump({
            "split_dir": "splits", "baseline_bg_tsv": "bg.tsv",
            "vrt_dir": "rasters", "mask_dir": "masks", "output_dir": "output",
            "outer_folds": [1], "inner_folds": 3, "patch_size": 16,
            "mean": [0.] * 8, "std": [1.] * 8, "epochs": 1, "pretrained": False,
            "workers": 0, "batch_size": 2, "device": "cpu",
        }))
        self.config = load_config(self.config_path)

    def fold_config(self):
        return {"hnm": {"enabled": True, "artifacts_dir": str(self.root / "output")},
                "cross_validation": {"split_dir": str(self.root / "splits")},
                "dataset": {"bg_coord_txt": str(self.root / "bg.tsv"),
                            "vrt_dir": str(self.root / "rasters"),
                            "labels_dir": str(self.root / "masks"), "patch_size": 16,
                            **{f"{s}_txt": str(self.root / "splits" / f"{s}_fold1.txt")
                               for s in ("train", "val", "test")}}}

    def test_plan_uses_train_only_and_preserves_eight_bands(self):
        plan, records, bg, _, index, _ = prepare_fold(self.config, 1)
        self.assertEqual((plan["positive_count"], len(bg)), (6, 6))
        self.assertEqual({r.scene for r in records}, set(self.train_scenes))
        x, label = WindowDataset(records, index, self.config)[0]
        self.assertEqual(x.shape, (8, 16, 16))
        self.assertEqual(label, 1)
        self.assertFalse(torch.equal(x[0], x[1]))
        validate_labels(records, self.config["mask_dir"], 16)

    def test_fail_on_oil_in_background(self):
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            validate_labels([Patch("train0", 0, 0, 0)], self.config["mask_dir"], 16)

    def test_gt_tiles_without_oil_are_not_mislabeled_as_positive(self):
        with rasterio.open(self.root / "masks" / "train0.tif", "r+") as ds:
            ds.write(np.ones((16, 32), dtype=np.uint8), 1)
        plan, records, _, _, _, _ = prepare_fold(self.config, 1)
        self.assertEqual(plan["positive_count"], 5)
        self.assertEqual(plan["fixed_gt_background_count"], 1)
        self.assertEqual(next(r.label for r in records if r.key == ("train0", 0, 0)), 0)

    def test_reject_duplicate_and_cross_split_leakage(self):
        p = self.root / "splits" / "val_fold1.txt"
        p.write_text("train0_patch_x0_y0\n")
        with self.assertRaisesRegex(ValueError, "Scene leakage"):
            split_manifest(self.root / "splits", 1)
        p.write_text("val_patch_x0_y0\nval_patch_x0_y0\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_patches(p)

    def test_reject_event_leakage(self):
        groups = {s: s for s in self.scenes}
        groups["train0"] = groups["val"] = "shared_event"
        with self.assertRaisesRegex(ValueError, "Group leakage"):
            split_manifest(self.root / "splits", 1, groups)

    def test_disabled_hnm_does_not_change_config(self):
        cfg = {"dataset": {"bg_coord_txt": "original"}}
        original = copy.deepcopy(cfg)
        apply_hnm_to_fold(cfg, 1)
        self.assertEqual(cfg, original)

    def test_incomplete_artifacts_cannot_start_training(self):
        out = self.root / "output" / "fold_1"
        out.mkdir(parents=True)
        (out / "plan.json").write_text("{}")
        with self.assertRaises(FileNotFoundError):
            apply_hnm_to_fold(self.fold_config(), 1)

    def test_end_to_end_and_artifact_guards(self):
        from preprocess.hnm import models
        actual_train = models.train_miner
        training_seen = []
        # A fixed GT tile is oil-free, as in the real dataset. It may train the
        # miner but must never shift candidate score/embedding indices.
        with rasterio.open(self.root / "masks" / "train0.tif", "r+") as ds:
            ds.write(np.ones((16, 32), dtype=np.uint8), 1)

        def record_train(records, *args, **kwargs):
            training_seen.append({r.scene for r in records})
            return actual_train(records, *args, **kwargs)

        with patch.object(models, "create_model", side_effect=lambda *a, **kw: TinyMiner()), \
                patch.object(models, "train_miner", side_effect=record_train):
            mine_fold(self.config, 1, torch.device("cpu"))
        out = self.root / "output" / "fold_1"
        manifest = json.loads((out / "manifest.json").read_text())
        for actual, spec in zip(training_seen[:-1], manifest["inner_folds"]):
            self.assertEqual(actual, set(spec["train_scenes"]))
            self.assertFalse(actual & set(spec["heldout_scenes"]))
        self.assertEqual(training_seen[-1], set(self.train_scenes))
        self.assertEqual(manifest["selected_count"], 3)
        cfg = self.fold_config()
        apply_hnm_to_fold(cfg, 1)
        combined = read_background(cfg["dataset"]["bg_coord_txt"])
        self.assertEqual([r for r in combined if r.scene in {"val", "test"}],
                         [Patch("val", 16, 0, 0), Patch("test", 16, 0, 0)])
        with self.assertRaises(FileExistsError):
            mine_fold(self.config, 1, torch.device("cpu"))
        # Tampered coordinates are rejected before the segmentation model is created.
        original = (out / "bg_coords.tsv").read_text()
        (out / "bg_coords.tsv").write_text(original + "bad\t0\t0\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            apply_hnm_to_fold(self.fold_config(), 1)
        (out / "bg_coords.tsv").write_text(original)
        # Even recomputing a checksum must not allow altered validation data.
        changed = [r for r in combined if r.scene != "val"]
        write_background(out / "bg_coords.tsv", changed)
        manifest["bg_coords_sha256"] = sha256(out / "bg_coords.tsv")
        (out / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "non-training"):
            apply_hnm_to_fold(self.fold_config(), 1)
        (out / "bg_coords.tsv").write_text(original)
        manifest["bg_coords_sha256"] = sha256(out / "bg_coords.tsv")
        (out / "manifest.json").write_text(json.dumps(manifest))
        split = self.root / "splits" / "train_fold1.txt"
        split.write_text(split.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "split differs"):
            apply_hnm_to_fold(self.fold_config(), 1)


if __name__ == "__main__":
    unittest.main()
