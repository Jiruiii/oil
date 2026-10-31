"""OOF classifier mining on the HNM_SI candidate pool (arms ``oof`` and ``oof_si``).

Ports the EfficientNetV2-S out-of-fold idea of preprocess/hnm onto the same eligible-window
pool, target count and scene-balanced sampling as the HNM_SI arms, so only the judge of
"hard" changes:

* the fold's train scenes are split into ``oof.inner_folds`` scene groups; every pool window
  is scored only by the classifier that never trained on its scene (scene-level OOF);
* positives are GT-centric train tiles that contain oil; negatives are the oil-free GT tiles
  plus a fixed sample of the pool taken as whole grid rows per scene (``oof.neg_pool``),
  read into memory once. The rasters are LZW strips, so one 256 px window decodes 256 full
  rows; reading whole rows of windows together keeps the I/O bounded;
* every pool window gets P(oil) and the median spectral indices of its valid mask==1 pixels,
  which the oof_si arm filters against the fold's HNM_SI oil reference.

Needs the fold's completed HNM_SI mining. Output: <output_dir>/oof/fold_N/; selection then
runs through pipeline.select_fold with --arm oof / oof_si.
"""
import csv
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from preprocess.hnm.data import Patch, WindowDataset, label_gt_patches, oof_splits, sha256

from .indices import band_index, compute_si, model_valid_mask, reflectance

SCHEMA_VERSION = 1
SCORE_COLUMNS = ("scene", "x", "y", "inner_fold", "p_oil", "n_si_px", "w_mndwi", "w_ndi", "w_sr")
OOF_FILES = ("oof.py", "indices.py")


def oof_dir(config, fold):
    return Path(config["output_dir"]) / "oof" / f"fold_{fold}"


# --------------------------------------------------------------------------- reading
def row_groups(coords, size):
    """Group (x, y) origins into strips at most 2 * size tall, each read once."""
    groups, cur = [], []
    for x, y in sorted(coords, key=lambda c: (c[1], c[0])):
        if cur and y > cur[0][1] + size:
            groups.append(cur)
            cur = []
        cur.append((x, y))
    if cur:
        groups.append(cur)
    return groups


def read_group(src, group, size, fill):
    """{(x, y): (C, size, size)} for one strip of windows."""
    x0, x1 = min(x for x, _ in group), max(x for x, _ in group) + size
    y0, y1 = group[0][1], max(y for _, y in group) + size
    strip = src.read(window=Window(x0, y0, x1 - x0, y1 - y0), boundless=True, fill_value=fill)
    return {(x, y): strip[..., y - y0:y - y0 + size, x - x0:x - x0 + size] for x, y in group}


def as_uint16(a, key):
    """Delivered DN are integral uint16 values exposed as float32 by the VRT."""
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 65535 or not np.array_equal(a, np.round(a)):
        raise ValueError(f"Imagery is not integral uint16 DN: {key}")
    return a.astype(np.uint16)


def _read_scene_windows(job):
    path, scene, coords, size = job
    out = []
    with rasterio.open(path) as src:
        for group in row_groups(coords, size):
            for (x, y), a in read_group(src, group, size, 0).items():
                out.append(((scene, x, y), as_uint16(a, (scene, x, y))))
    return out


def build_cache(records, index, size, workers, progress=False):
    """Read every record's window once into one (N, C, size, size) uint16 array."""
    by_scene = {}
    for r in records:
        by_scene.setdefault(r.scene, []).append((r.x, r.y))
    jobs = [(str(index[s]), s, c, size) for s, c in sorted(by_scene.items())]
    rows, chunks = {}, []
    ctx = get_context("fork")
    with ctx.Pool(max(1, workers)) as pool:
        it = pool.imap_unordered(_read_scene_windows, jobs)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, total=len(jobs), desc="cache windows")
        for result in it:
            for key, a in result:
                rows[key] = len(chunks)
                chunks.append(a)
    array = np.stack(chunks)
    missing = [r.key for r in records if r.key not in rows]
    if missing:
        raise ValueError(f"Cache misses {len(missing)} windows, e.g. {missing[:2]}")
    return array, rows


class CachedWindows(WindowDataset):
    """WindowDataset over cached windows: ``index`` is (array, {key: row})."""

    def _read(self, p):
        array, rows = self.index
        return array[rows[p.key]]


# --------------------------------------------------------------------------- sampling
def negative_rows(pool, target, seed):
    """Whole grid rows of pool windows, round-robin over scenes, until ``target`` windows."""
    rng = np.random.default_rng(seed)
    rows = {}
    for (scene, y), g in pool.groupby(["scene", "y"], sort=True):
        rows.setdefault(scene, []).append([(scene, int(x), int(y)) for x in sorted(g["x"])])
    scenes = sorted(rows)
    queues = {s: [rows[s][i] for i in rng.permutation(len(rows[s]))] for s in scenes}
    order = [scenes[i] for i in rng.permutation(len(scenes))]
    picked, depth = [], 0
    while len(picked) < target:
        progressed = False
        for s in order:
            if depth < len(queues[s]):
                picked += queues[s][depth]
                progressed = True
                if len(picked) >= target:
                    break
        if not progressed:
            break
        depth += 1
    return [Patch(s, x, y, 0) for s, x, y in picked]


def scene_inner_folds(records, splits):
    """scene -> inner fold whose classifier never trained on it."""
    mapping = {}
    for k, (_, heldout) in enumerate(splits, 1):
        for i in heldout:
            if mapping.setdefault(records[i].scene, k) != k:
                raise ValueError(f"Scene {records[i].scene} is held out by two inner folds")
    return mapping


# --------------------------------------------------------------------------- scoring
class _ScoreItems:
    """One item per strip: raw windows plus per-window SI medians on valid mask==1 pixels."""

    def __init__(self, jobs, mask_dir, size, scale, si_cfg, bands):
        self.jobs, self.mask_dir, self.size = jobs, mask_dir, size
        self.scale, self.si_cfg, self.bands = scale, si_cfg, bands

    def __len__(self):
        return len(self.jobs)

    def __getitem__(self, i):
        scene, path, group = self.jobs[i]
        with rasterio.open(path) as src:
            raws = read_group(src, group, self.size, 0)
        with rasterio.open(Path(self.mask_dir) / f"{scene}.tif") as msrc:
            masks = read_group(msrc, group, self.size, 255)
        keys, arrays, si = [], [], []
        s = self.si_cfg
        for x, y in group:
            raw = raws[(x, y)]
            arrays.append(as_uint16(raw, (scene, x, y)))
            vals, ok = compute_si(reflectance(raw, self.scale, s["reflectance_offset"]),
                                  self.bands, s["den_floor"], s["clip"])
            sel = ok & model_valid_mask(raw) & (masks[(x, y)][0] == 1)
            n = int(sel.sum())
            med = (np.median(vals[:, sel], axis=1) if n >= s["min_fp_pixels"]
                   else np.full(3, np.nan, np.float32))
            keys.append((scene, x, y))
            si.append([n, *med])
        return scene, keys, np.stack(arrays), np.asarray(si, np.float64)


def score_pool(pool, scene_fold, checkpoints, config, cls_cfg, index, bands, device, progress):
    import torch
    from preprocess.hnm import models
    size, scale = config["patch_size"], config["scale"]
    nets = {}
    for k, path in checkpoints.items():
        net = models.create_model(cls_cfg, pretrained=False).to(device)
        net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        nets[k] = net.eval()
    mean = torch.tensor(cls_cfg["mean"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    std = torch.tensor(cls_cfg["std"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
    jobs = []
    for scene, g in pool.groupby("scene", sort=True):
        for group in row_groups(list(zip(g["x"].astype(int), g["y"].astype(int))), size):
            jobs.append((scene, str(index[scene]), group))
    loader = torch.utils.data.DataLoader(
        _ScoreItems(jobs, config["mask_dir"], size, scale, config["si"], bands),
        batch_size=None, num_workers=cls_cfg["workers"], collate_fn=lambda item: item)
    if progress:
        from tqdm import tqdm
        loader = tqdm(loader, total=len(jobs), desc="score pool")
    rows = []
    amp = device.type == "cuda"
    bs = cls_cfg["infer_batch_size"]
    with torch.inference_mode():
        for scene, keys, raw, si in loader:
            k = scene_fold[scene]
            probs = []
            for b in range(0, len(keys), bs):
                # Same input as WindowDataset: DN / scale, |value| > 100 -> 0, then mean/std.
                x = torch.as_tensor(raw[b:b + bs].astype(np.float32), device=device) / scale
                x = torch.where(x.abs() > 100, torch.zeros_like(x), x)
                with torch.autocast(device.type, enabled=amp):
                    logits = nets[k]((x - mean) / std)
                probs.append(logits.float().softmax(1)[:, 1].cpu().numpy())
            for key, p, s in zip(keys, np.concatenate(probs), si):
                rows.append((*key, k, float(p), int(s[0]), *(float(v) for v in s[1:])))
    return pd.DataFrame(rows, columns=SCORE_COLUMNS)


# --------------------------------------------------------------------------- stage
def _write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def classifier_config(config, desc):
    """preprocess/hnm classifier settings with the probe's own normalization."""
    return {**config["oof"], "patch_size": config["patch_size"], "scale": config["scale"],
            "mean": desc["mean"], "std": desc["std"]}


def mine_oof(config, fold, device, progress=True):
    from preprocess.hnm import models
    from .pipeline import _load_mining, _table_inputs, fold_inputs, probe_description
    ctx = fold_inputs(config, fold)
    mdir, mm, _ = _load_mining(config, fold, ctx, allow_partial=False)
    pool = pd.read_csv(mdir / "candidates.tsv", sep="\t", usecols=["scene", "x", "y"])
    index = _table_inputs(config, ctx["train_scenes"])
    desc = probe_description(config, fold)
    bands = band_index(desc["band_names"])
    cls_cfg = classifier_config(config, desc)
    o, size = config["oof"], config["patch_size"]
    gt = label_gt_patches(ctx["splits"]["train"], config["mask_dir"], size)
    neg = negative_rows(pool, o["neg_pool"], config["seed"] + 100 * fold)
    records = gt + neg
    if len({r.key for r in records}) != len(records):
        raise ValueError("A GT tile coincides with a pool window")
    splits = oof_splits(records, o["inner_folds"], config["seed"] + fold)
    scene_fold = scene_inner_folds(records, splits)
    missing = sorted(set(pool["scene"]) - set(scene_fold))
    if missing:
        raise ValueError(f"Pool scenes without an inner fold: {missing[:3]}")
    out = oof_dir(config, fold)
    out.mkdir(parents=True, exist_ok=True)
    plan = {"schema_version": SCHEMA_VERSION, "outer_fold": fold,
            "mining_manifest_sha256": sha256(mdir / "mining_manifest.json"),
            "split_sha256": ctx["split_sha256"], "classifier": cls_cfg, "seed": config["seed"],
            "positives": sum(r.label for r in gt), "fixed_gt_negatives": sum(not r.label for r in gt),
            "pool_negatives": len(neg), "pool_windows": int(len(pool)),
            "scene_inner_fold": scene_fold,
            "inner_folds": [{"fold": k, "train": len(tr), "heldout": len(ho)}
                            for k, (tr, ho) in enumerate(splits, 1)],
            "implementation_sha256": {n: sha256(Path(__file__).parent / n) for n in OOF_FILES}
            | {f"hnm/{n}": sha256(Path(models.__file__).parent / n) for n in ("models.py", "data.py")}}
    plan_path = out / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != json.loads(json.dumps(plan)):
            raise ValueError(f"{plan_path} differs from this run; use a new output_dir")
    else:
        _write_json(plan_path, plan)
    checkpoints = {k: out / f"miner_oof_{k}.pt" for k in range(1, o["inner_folds"] + 1)}
    histories_path = out / "histories.json"
    histories = json.loads(histories_path.read_text()) if histories_path.exists() else {}
    todo = [k for k, p in checkpoints.items() if not p.exists()]
    if todo:
        cache = build_cache(records, index, size, o["workers"], progress)
        for k in todo:
            train, _ = splits[k - 1]
            histories[str(k)] = models.train_miner(
                [records[i] for i in train], cache, cls_cfg, checkpoints[k],
                config["seed"] + 1000 * fold + k, device, dataset_cls=CachedWindows)
            _write_json(histories_path, histories)
        del cache
    scores = score_pool(pool, scene_fold, checkpoints, config, cls_cfg, index, bands, device, progress)
    if set(map(tuple, scores[["scene", "x", "y"]].itertuples(index=False))) != set(
            map(tuple, pool.itertuples(index=False))) or len(scores) != len(pool):
        raise ValueError("Scores do not cover the mined pool exactly")
    scores = pool.merge(scores, on=["scene", "x", "y"], how="left", validate="one_to_one")
    scores.to_csv(out / "scores.tsv", sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    manifest = {"schema_version": SCHEMA_VERSION, "status": "complete", "outer_fold": fold,
                "plan_sha256": sha256(plan_path), "scores_sha256": sha256(out / "scores.tsv"),
                "mining_manifest_sha256": plan["mining_manifest_sha256"],
                "miner_sha256": {str(k): sha256(p) for k, p in checkpoints.items()},
                "scored_windows": int(len(scores)),
                "p_oil_ge_0_5": int((scores["p_oil"] >= .5).sum())}
    _write_json(out / "oof_manifest.json", manifest)
    print(f"Fold {fold} OOF scoring complete: {len(scores)} windows, "
          f"{manifest['p_oil_ge_0_5']} with P(oil) >= 0.5", flush=True)
    return manifest


def load_scores(config, fold, mm):
    """Completed, unmodified OOF scores of this fold, tied to the same mining run."""
    out = oof_dir(config, fold)
    path = out / "oof_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Run --stage oof for fold {fold} first: {path}")
    m = json.loads(path.read_text(encoding="utf-8"))
    if m.get("status") != "complete":
        raise ValueError(f"OOF scoring for fold {fold} is incomplete")
    if sha256(out / "scores.tsv") != m["scores_sha256"] or sha256(out / "plan.json") != m["plan_sha256"]:
        raise ValueError("OOF artifacts changed after completion")
    mining_sha = sha256(Path(config["output_dir"]) / "mining" / f"fold_{fold}" / "mining_manifest.json")
    if m["mining_manifest_sha256"] != mining_sha:
        raise ValueError("OOF scores belong to a different mining run")
    return pd.read_csv(out / "scores.tsv", sep="\t"), m, sha256(path)
