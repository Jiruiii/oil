"""Coordinate manifests, split checks and windowed multiband input."""
import csv
import hashlib
import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

PATCH_RE = re.compile(r"^(.+)_patch_x(\d+)_y(\d+)$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Patch:
    scene: str
    x: int
    y: int
    label: int

    @property
    def key(self):
        return self.scene, self.x, self.y


def read_patches(path):
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = PATCH_RE.fullmatch(line.strip())
        if not m:
            raise ValueError(f"Invalid patch name in {path}: {line}")
        records.append(Patch(m[1], int(m[2]), int(m[3]), 1))
    _unique(records, path)
    if not records:
        raise ValueError(f"Empty split: {path}")
    return records


def read_background(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f, delimiter="\t")
        if not {"scene", "x", "y"}.issubset(rows.fieldnames or []):
            raise ValueError(f"Expected scene/x/y columns: {path}")
        records = [Patch(r["scene"], int(r["x"]), int(r["y"]), 0) for r in rows]
    _unique(records, path)
    if any(not r.scene or r.x < 0 or r.y < 0 for r in records):
        raise ValueError(f"Invalid background coordinates: {path}")
    return records


def _unique(records, source):
    if len({r.key for r in records}) != len(records):
        raise ValueError(f"Duplicate patch coordinates: {source}")


def write_background(path, records):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["scene", "x", "y"])
        writer.writerows(r.key for r in records)


def read_groups(path):
    if not path:
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    groups = {r["scene"]: r["group"] for r in rows}
    if len(groups) != len(rows) or any(not g for g in groups.values()):
        raise ValueError("Group TSV requires exactly one nonempty group per scene")
    return groups


def split_manifest(split_dir, fold, groups=None):
    splits = {s: read_patches(Path(split_dir) / f"{s}_fold{fold}.txt")
              for s in ("train", "val", "test")}
    scenes = {s: {p.scene for p in rows} for s, rows in splits.items()}
    if groups is not None:
        missing = set.union(*scenes.values()) - groups.keys()
        if missing:
            raise ValueError(f"Missing scene groups: {sorted(missing)[:5]}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if scenes[a] & scenes[b]:
            raise ValueError(f"Scene leakage between {a}/{b} in fold {fold}")
        if groups is not None and ({groups[s] for s in scenes[a]} &
                                   {groups[s] for s in scenes[b]}):
            raise ValueError(f"Group leakage between {a}/{b} in fold {fold}; "
                             "the frozen outer split does not support this grouping")
    return splits, scenes


def oof_splits(records, k=5, seed=42, groups=None):
    y = np.array([r.label for r in records])
    g = np.array([groups[r.scene] if groups is not None else r.scene for r in records])
    if k < 2 or len(set(g)) < k:
        raise ValueError(f"Need at least {k} independent groups for OOF")
    result = []
    coverage = np.zeros(len(records), dtype=int)
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for train, heldout in splitter.split(np.zeros(len(records)), y, g):
        if set(g[train]) & set(g[heldout]):
            raise ValueError("OOF group leakage")
        if set(y[train]) != {0, 1} or set(y[heldout]) != {0, 1}:
            raise ValueError("Each OOF training/held-out fold must contain both classes")
        coverage[heldout] += 1
        result.append((train, heldout))
    if not np.all(coverage == 1):
        raise ValueError("Each sample must be scored exactly once")
    return result


def raster_index(vrt_dir):
    paths = sorted(p for p in Path(vrt_dir).rglob("*")
                   if p.suffix.lower() in (".vrt", ".tif", ".tiff"))
    result = {}
    for p in paths:
        if p.stem in result:
            raise ValueError(f"Ambiguous scene raster: {p.stem}")
        result[p.stem] = p
    return result


def inspect_rasters(records, index, mask_dir, channels):
    import rasterio
    metadata = {}
    for scene in sorted({r.scene for r in records}):
        if scene not in index:
            raise FileNotFoundError(f"Missing scene raster: {scene}")
        mask = Path(mask_dir) / f"{scene}.tif"
        with rasterio.open(index[scene]) as src, rasterio.open(mask) as gt:
            if src.count != channels:
                raise ValueError(f"Expected {channels} bands, found {src.count}: {scene}")
            if (src.shape, src.crs, src.transform) != (gt.shape, gt.crs, gt.transform):
                raise ValueError(f"Image/mask alignment mismatch: {scene}")
            metadata[scene] = src.shape
    for r in records:
        h, w = metadata[r.scene]
        if r.x >= w or r.y >= h:
            raise ValueError(f"Patch origin is outside raster: {r}")


def label_gt_patches(records, mask_dir, patch_size):
    """GT-centric tiles can be oil-free; derive classifier labels from raw masks."""
    import rasterio
    from rasterio.windows import Window
    labels = {}
    by_scene = {}
    for r in records:
        by_scene.setdefault(r.scene, []).append(r)
    for scene, patches in by_scene.items():
        with rasterio.open(Path(mask_dir) / f"{scene}.tif") as src:
            for p in patches:
                mask = src.read(1, window=Window(p.x, p.y, patch_size, patch_size),
                                boundless=True, fill_value=255)
                if not np.isin(mask, [0, 1, 2, 255]).all():
                    raise ValueError(f"Unexpected mask encoding: {p.key}")
                labels[p.key] = int((mask == 0).any())
    return [replace(r, label=labels[r.key]) for r in records]


def validate_labels(records, mask_dir, patch_size, candidate_keys=None):
    """Fail before training if a positive has no oil or a background contains oil."""
    import rasterio
    from rasterio.windows import Window
    by_scene = {}
    for r in records:
        by_scene.setdefault(r.scene, []).append(r)
    for scene, patches in by_scene.items():
        with rasterio.open(Path(mask_dir) / f"{scene}.tif") as src:
            for p in patches:
                mask = src.read(1, window=Window(p.x, p.y, patch_size, patch_size),
                                boundless=True, fill_value=255)
                if not np.isin(mask, [0, 1, 2, 255]).all():
                    raise ValueError(f"Unexpected mask encoding: {p.key}")
                if bool((mask == 0).any()) != bool(p.label):
                    raise ValueError(f"Oil/background label mismatch: {p.key}")
                check_bg = not p.label and (candidate_keys is None or p.key in candidate_keys)
                if check_bg and ((np.isin(mask, [1, 2]).mean() < .30) or
                                    ((mask == 255).mean() >= .50)):
                    raise ValueError(f"Background lacks sufficient annotated area: {p.key}")


class WindowDataset:
    """Read all bands, preserving the existing reflectance scale and normalization."""
    def __init__(self, records, index, config, augment=False):
        self.records, self.index, self.augment = records, index, augment
        self.size = config["patch_size"]
        self.mean = np.array(config["mean"], dtype=np.float32)[:, None, None]
        self.std = np.array(config["std"], dtype=np.float32)[:, None, None]
        self.scale = config["scale"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        import rasterio
        import torch
        from rasterio.windows import Window
        p = self.records[i]
        with rasterio.open(self.index[p.scene]) as src:
            a = src.read(window=Window(p.x, p.y, self.size, self.size),
                         boundless=True, fill_value=0).astype(np.float32) / self.scale
        if not np.isfinite(a).all():
            raise ValueError(f"Nonfinite imagery: {p.key}")
        a[(a > 100) | (a < -100)] = 0  # match existing reflectance loader
        a = (a - self.mean) / self.std
        if self.augment:
            if torch.rand(()) < .5:
                a = a[:, :, ::-1]
            if torch.rand(()) < .5:
                a = a[:, ::-1, :]
            a = np.rot90(a, int(torch.randint(4, ()).item()), axes=(1, 2))
        return torch.from_numpy(a.copy()), p.label
