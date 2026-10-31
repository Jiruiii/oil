"""Step 1 of HNM_EXPERIMENT_PLAN.md: where do the baseline's false positives fall?

Nothing is trained; only each fold's baseline best.pt and its finished outputs are used.

Sources per outer fold:
  recon     GT-aware reconstruction artifacts (``*.pred.json``) of the fold's test scenes,
            i.e. exactly the pixels the pooled Oil IoU scores. That metric maps every non-oil
            GT value to background (1/2/255 -> 1), so FP pixels are also split by GT label.
  bg_patch  the fold's frozen test background tiles (``bg_coords.tsv`` rows in test scenes),
            which the reconstruction never scores. Inferred here with the reconstruction's
            own 3-way flip TTA, FP32, argmax.

Each FP pixel gets one category, first match wins:
  nodata        model input invalid
  near_oil      <= N px from GT oil (annotation boundary)
  oil_attached  farther than N px, but its FP region comes within N px of GT oil
                (slick continuation over unlabeled pixels, not an independent lookalike)
  land_coast    on land, or in an FP region that touches land (coarse spectral land)
  cloud         thick cloud (bright from blue to SWIR)
  haze_glint    bright, flat spectrum with NIR/SWIR clearly above water (haze, thin cloud,
                broad sun glint)
  bright_small  small FP region brighter than its surroundings (ship, wake, small glint)
  other_sea     everything else: open water, typically dark lookalikes
land / cloud / haze_glint come from coarse spectral rules (RULES); no land or cloud mask
exists, so they are rough estimates to be checked against the crops.

Also reported: the same land / cloud shares for the selected HNM backgrounds next to the
baseline train backgrounds, and SR_B1B2_B3B8A clip saturation in the HNM_SI mining records.

Output (git-ignored): hnm_runs/diagnostics/. Run from the repo root:
  .venv/bin/python tools/analyze_baseline_fp.py --config preprocess/hnm_si/config.example.yaml
Any other finished run (e.g. a step 2 arm) can be analysed the same way with --run-dir.
"""
import argparse
import heapq
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, REPO_ROOT / "main"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from metrics_extra import boundary_counts  # noqa: E402
from predmask_codec import decode_scene_coverage, decode_scene_final, load_scene_document  # noqa: E402
from preprocess.export_hnm_selection import short_name, true_colour  # noqa: E402
from preprocess.hnm.data import raster_index, read_background, split_manifest  # noqa: E402
from preprocess.hnm_si.config import load_config  # noqa: E402
from preprocess.hnm_si.indices import (model_input, model_valid_mask, reflectance,  # noqa: E402
                                       sensor_of)

CATEGORIES = ("nodata", "near_oil", "oil_attached", "land_coast", "cloud", "haze_glint",
              "bright_small", "other_sea")
# Plan step 1 groups: what HNM can target vs annotation-boundary FP.
HNM_TARGETS = ("land_coast", "cloud", "haze_glint", "bright_small")
GT_LABELS = {1: "confirmed_bg", 2: "others", 255: "unlabeled"}
BAND = {n: i for i, n in enumerate(("B01", "B02", "B03", "B04", "B08", "B8A", "B11", "B12"))}
WATER, LAND, CLOUD, HAZE, NODATA = range(5)

# Coarse rules on reflectance (DN - offset) / scale. Rough estimates, not validated:
# floating vegetation (e.g. sargassum) passes as land; clear water has NIR/SWIR near 0.
RULES = {
    "veg_ndvi": 0.3, "veg_nir": 0.08,            # (B8A-B04)/(B8A+B04) and B8A
    "soil_mndwi": -0.15, "soil_swir": 0.10,      # bare soil / built-up: B11 well above B03
    "cloud_blue": 0.15, "cloud_nir": 0.12, "cloud_swir": 0.08,  # thick cloud
    "haze_nir": 0.03, "haze_swir": 0.025,        # haze / glint: NIR and SWIR above water
    "coast_px": 32, "coast_min_land_px": 20,     # FP region within coast_px of land
    "small_max_px": 400, "small_contrast": 0.02, "ring_px": 8,  # bright_small regions
    "tile_land_frac": 0.01, "tile_cloud_frac": 0.05, "tile_haze_frac": 0.5,  # tile flags
}
NEAR_OIL_SWEEP = (4, 8, 16)
PAD = 64           # context read around scored pixels; distances are exact up to PAD
CROP_PAD = 48


# --------------------------------------------------------------------------- pixel rules
def spectral_class(refl, valid):
    """Per-pixel coarse class: WATER, LAND, CLOUD, HAZE (haze/glint) or NODATA."""
    g, r, nir, swir, blue = (refl[BAND[k]] for k in ("B03", "B04", "B8A", "B11", "B02"))
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - r) / (nir + r)
        mndwi = (g - swir) / (g + swir)
    land = (((ndvi >= RULES["veg_ndvi"]) & (nir >= RULES["veg_nir"])) |
            ((mndwi <= RULES["soil_mndwi"]) & (swir >= RULES["soil_swir"])))
    cloud = (~land & (blue >= RULES["cloud_blue"]) & (nir >= RULES["cloud_nir"]) &
             (swir >= RULES["cloud_swir"]))
    haze = ~land & ~cloud & (nir >= RULES["haze_nir"]) & (swir >= RULES["haze_swir"])
    out = np.full(valid.shape, WATER, np.uint8)
    out[land] = LAND
    out[cloud] = CLOUD
    out[haze] = HAZE
    out[~valid] = NODATA
    return out


def oil_distance(mask):
    """Euclidean px distance to the nearest GT oil pixel, capped at PAD."""
    oil = mask == 0
    if not oil.any():
        return np.full(mask.shape, float(PAD), np.float32)
    return np.minimum(cv2.distanceTransform((~oil).astype(np.uint8), cv2.DIST_L2, 5), PAD)


def brightness(refl):
    return refl[[BAND["B02"], BAND["B03"], BAND["B04"], BAND["B8A"]]].mean(axis=0)


def analyse_window(pred_oil, scored, mask, raw, scale, offset, near_n):
    """Categorise the FP pixels of one context window.

    All inputs cover the same window; ``scored`` marks the pixels the model predicted.
    Returns (category (H, W) int8 with -1 off FP, dist, regions).
    """
    valid = model_valid_mask(raw)
    refl = reflectance(raw, scale, offset)
    dist = oil_distance(mask)
    fp = scored & pred_oil & (mask != 0)
    spec = spectral_class(refl, valid)
    cat = np.full(mask.shape, -1, np.int8)
    idx = CATEGORIES.index
    order = [("other_sea", fp), ("haze_glint", fp & (spec == HAZE)),
             ("cloud", fp & (spec == CLOUD)), ("land_coast", fp & (spec == LAND)),
             ("near_oil", fp & (dist <= near_n)), ("nodata", fp & ~valid)]
    for name, sel in order:                  # later entries take precedence
        cat[sel] = idx(name)
    regions = []
    if not fp.any():
        return cat, dist, regions
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fp.astype(np.uint8), connectivity=8)
    land = cv2.morphologyEx((spec == LAND).astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8)).astype(bool)
    bright = brightness(refl)
    margin = max(RULES["coast_px"], RULES["ring_px"])
    reassignable = [idx(c) for c in ("land_coast", "cloud", "haze_glint", "other_sea")]
    for k in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[k])
        y0, y1 = max(0, y - margin), min(mask.shape[0], y + h + margin)
        x0, x1 = max(0, x - margin), min(mask.shape[1], x + w + margin)
        reg = labels[y0:y1, x0:x1] == k
        sub_cat = cat[y0:y1, x0:x1]          # a view: edits land in ``cat``
        d = dist[y0:y1, x0:x1][reg]
        movable = reg & np.isin(sub_cat, reassignable)
        contrast, land_px = np.nan, 0
        if d.min() <= near_n:
            sub_cat[movable] = idx("oil_attached")
        else:
            near = cv2.dilate(reg.astype(np.uint8), np.ones((2 * RULES["coast_px"] + 1,) * 2,
                                                            np.uint8)).astype(bool)
            land_px = int((near & land[y0:y1, x0:x1]).sum())
            if land_px >= RULES["coast_min_land_px"]:
                sub_cat[movable] = idx("land_coast")
            elif area <= RULES["small_max_px"]:
                size = 2 * RULES["ring_px"] + 1
                ring = (cv2.dilate(reg.astype(np.uint8), np.ones((size, size), np.uint8))
                        .astype(bool) & ~fp[y0:y1, x0:x1] & (mask[y0:y1, x0:x1] != 0)
                        & valid[y0:y1, x0:x1])
                inner = reg & valid[y0:y1, x0:x1]
                b = bright[y0:y1, x0:x1]
                if ring.any() and inner.any():
                    contrast = float(np.median(b[inner]) - np.median(b[ring]))
                    if contrast >= RULES["small_contrast"]:
                        sub_cat[reg & np.isin(sub_cat, [idx("haze_glint"), idx("other_sea")])] = \
                            idx("bright_small")
        counts = np.bincount(sub_cat[reg], minlength=len(CATEGORIES))
        m = mask[y0:y1, x0:x1][reg]
        r = refl[:, y0:y1, x0:x1][:, reg]
        sp = np.bincount(spec[y0:y1, x0:x1][reg], minlength=5) / area
        regions.append({
            "x": x, "y": y, "w": w, "h": h, "area": area,
            "category": CATEGORIES[int(counts.argmax())],
            **{f"n_{v}": int((m == k_).sum()) for k_, v in GT_LABELS.items()},
            "min_dist": float(d.min()), "median_dist": float(np.median(d)),
            "land_px_near": land_px, "contrast": contrast,
            "cloud_frac": float(sp[CLOUD]), "haze_frac": float(sp[HAZE]),
            **{f"med_{b_}": float(np.nanmedian(r[BAND[b_]]))
               for b_ in ("B02", "B04", "B8A", "B11")}})
    return cat, dist, regions


# --------------------------------------------------------------------------- bookkeeping
class Collector:
    """Scene-level tables plus the top-k largest FP regions per (source, category) as crops."""

    def __init__(self, out, top_k, scale, offset, boundary_px=8):
        self.out, self.top_k, self.scale, self.offset = out, top_k, scale, offset
        self.boundary_px = boundary_px
        self.totals, self.fp_rows, self.sweep_rows, self.region_rows = [], [], [], []
        self.heaps = defaultdict(list)
        self._tie = 0

    def add(self, meta, pred_oil, scored, mask, raw, near_n, origin):
        """Accumulate one context window; ``origin`` is its (x, y) in scene coordinates."""
        cat, dist, regions = analyse_window(pred_oil, scored, mask, raw, self.scale,
                                               self.offset, near_n)
        fp = cat >= 0
        gt_s = mask[scored]
        row = dict(meta, scored_px=int(scored.sum()),
                   tp=int((scored & pred_oil & (mask == 0)).sum()),
                   fn=int((scored & ~pred_oil & (mask == 0)).sum()),
                   gt_oil=int((gt_s == 0).sum()),
                   **{f"gt_{v}": int((gt_s == k).sum()) for k, v in GT_LABELS.items()},
                   **{f"fp_{v}": int((fp & (mask == k)).sum()) for k, v in GT_LABELS.items()})
        # Boundary IoU atoms (Cheng et al. 2021), same code as the reconstruction's option.
        row["b_inter"], row["b_union"] = boundary_counts(mask == 0, pred_oil & scored, scored,
                                                         self.boundary_px)
        self.totals.append(row)
        for k, v in GT_LABELS.items():
            counts = np.bincount(cat[fp & (mask == k)], minlength=len(CATEGORIES))
            for c, n in zip(CATEGORIES, counts):
                if n:
                    self.fp_rows.append(dict(meta, gt_label=v, category=c, n_px=int(n)))
        for n_ in NEAR_OIL_SWEEP:
            self.sweep_rows.append(dict(meta, near_n=n_, n_near=int((fp & (dist <= n_)).sum()),
                                        n_fp=int(fp.sum())))
        ox, oy = origin
        for r in regions:
            rec = dict(meta, **r)
            rec["x"], rec["y"] = r["x"] + ox, r["y"] + oy
            self.region_rows.append(rec)
            self._keep_crop(rec, r, pred_oil & scored, fp, mask, raw)

    def _keep_crop(self, rec, r, pred, fp, mask, raw):
        heap = self.heaps[(rec["source"], rec["category"])]
        key = (r["area"], self._tie)
        self._tie += 1
        if len(heap) >= self.top_k and key <= heap[0][:2]:
            return
        y0, y1 = max(0, r["y"] - CROP_PAD), min(mask.shape[0], r["y"] + r["h"] + CROP_PAD)
        x0, x1 = max(0, r["x"] - CROP_PAD), min(mask.shape[1], r["x"] + r["w"] + CROP_PAD)
        image = render_crop(raw[:, y0:y1, x0:x1], pred[y0:y1, x0:x1], fp[y0:y1, x0:x1],
                            mask[y0:y1, x0:x1], self.scale, self.offset)
        item = (*key, rec, image)
        if len(heap) < self.top_k:
            heapq.heappush(heap, item)
        else:
            heapq.heapreplace(heap, item)

    def write(self):
        self.out.mkdir(parents=True, exist_ok=True)
        tables = {"scene_totals": self.totals, "fp_pixels": self.fp_rows,
                  "near_oil_sweep": self.sweep_rows, "fp_regions": self.region_rows}
        frames = {k: pd.DataFrame(v) for k, v in tables.items()}
        for k, df in frames.items():
            df.to_csv(self.out / f"{k}.csv", index=False, encoding="utf-8-sig")
        crops = self.out / "crops"
        for (source, category), heap in self.heaps.items():
            d = crops / source / category
            d.mkdir(parents=True, exist_ok=True)
            for f in d.glob("*.png"):
                f.unlink()
            for rank, (_, _, rec, image) in enumerate(sorted(heap, key=lambda t: -t[0]), 1):
                lines = [f"#{rank} f{rec['fold']} {short_name(rec['scene'])} {category}",
                         f"x{rec['x']} y{rec['y']} area {rec['area']} dmin {rec['min_dist']:.0f} "
                         f"bg{rec['n_confirmed_bg']}/oth{rec['n_others']}/unl{rec['n_unlabeled']}"]
                cv2.imwrite(str(d / f"{rank:02d}_fold{rec['fold']}_{short_name(rec['scene'])}"
                                    f"_x{rec['x']}_y{rec['y']}.png"), captioned(image, lines))
        return frames


def render_crop(raw, pred, fp, mask, scale, offset, min_side=256):
    """Left: true colour. Right: FP red, correct oil prediction green, GT oil outline cyan,
    unlabeled (255) tinted magenta."""
    bgr = np.ascontiguousarray(true_colour(raw, scale, offset, .15, .6))
    over = bgr.astype(np.float32)
    for sel, colour in ((mask == 255, (255, 0, 255)), (pred & ~fp, (0, 200, 0)), (fp, (0, 0, 255))):
        a = .25 if colour == (255, 0, 255) else .55
        over[sel] = (1 - a) * over[sel] + a * np.array(colour, np.float32)
    over = np.ascontiguousarray(over.astype(np.uint8))
    ctrs, _ = cv2.findContours((mask == 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, ctrs, -1, (255, 255, 0), 1)
    pair = np.hstack([bgr, np.full((bgr.shape[0], 2, 3), 80, np.uint8), over])
    f = max(1, int(np.ceil(min_side / max(1, min(bgr.shape[:2])))))
    f = min(f, 4)
    return cv2.resize(pair, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST) if f > 1 else pair


def captioned(image, lines):
    bar = np.full((16 * len(lines) + 6, image.shape[1], 3), 32, np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(bar, text, (4, 15 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, .42,
                    (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([image, bar])


# --------------------------------------------------------------------------- sources
def scene_size_bins(split_dir):
    """Scene oil-size tertiles (small/medium/large) as in resplit_stratified_v2.py.

    The original bin_edges.txt was not delivered; edges are recomputed from the frozen
    per-component oil counts over every scene of the three test folds.
    """
    stats = pd.read_csv(Path(split_dir) / "stats.csv")
    oil = stats.groupby("scene_name")["n_oil"].sum()
    scenes = set()
    for f in (1, 2, 3):
        scenes |= split_manifest(split_dir, f)[1]["test"]
    oil = oil.reindex(sorted(scenes)).fillna(0)
    e1, e2 = np.quantile(oil.to_numpy(), [1 / 3, 2 / 3])
    bins = {s: ("small" if v < e1 else "medium" if v < e2 else "large") for s, v in oil.items()}
    return bins, {"small_upper": float(e1), "medium_upper": float(e2)}


def recon_artifacts(run_dir):
    found = sorted(Path(run_dir).glob("post_test_*/gt_aware_recon/prediction_artifacts/*.pred.json"))
    if not found:
        raise FileNotFoundError(f"No GT-aware prediction artifacts under {run_dir}")
    return found


def read_window(src, x0, y0, w, h, fill):
    return src.read(window=Window(x0, y0, w, h), boundless=True, fill_value=fill)


def run_recon(col, config, fold, test_scenes, bins, index, near_n, progress):
    paths = recon_artifacts(config["probe_run_dirs"][fold])
    scenes_seen = set()
    it = paths
    if progress:
        from tqdm import tqdm
        it = tqdm(paths, desc=f"fold {fold} recon")
    for path in it:
        doc = load_scene_document(path)
        scene = doc["scene_id"]
        if scene not in test_scenes:
            raise ValueError(f"Artifact scene {scene} is not a fold {fold} test scene")
        scenes_seen.add(scene)
        final = decode_scene_final(doc)
        coverage = decode_scene_coverage(doc, final)
        pred_full = final == int(doc["labels"]["target_class_index"])
        meta = {"fold": fold, "source": "recon", "scene": scene, "sensor": sensor_of(scene),
                "size_bin": bins.get(scene, "unknown")}
        n, comp, stats, _ = cv2.connectedComponentsWithStats(coverage.astype(np.uint8),
                                                             connectivity=8)
        with rasterio.open(index[scene]) as src, \
                rasterio.open(Path(config["mask_dir"]) / f"{scene}.tif") as msrc:
            for k in range(1, n):
                x, y, w, h, _ = (int(v) for v in stats[k])
                x0, y0 = max(0, x - PAD), max(0, y - PAD)
                x1, y1 = min(src.width, x + w + PAD), min(src.height, y + h + PAD)
                raw = read_window(src, x0, y0, x1 - x0, y1 - y0, 0)
                mask = read_window(msrc, x0, y0, x1 - x0, y1 - y0, 255)[0]
                # Only this component counts as scored: padding may reach a neighbour.
                col.add(meta, pred_full[y0:y1, x0:x1], comp[y0:y1, x0:x1] == k, mask, raw,
                        near_n, (x0, y0))
    missing = test_scenes - scenes_seen
    if missing:
        print(f"  fold {fold}: {len(missing)} test scenes have no recon artifact "
              f"(no GT-centric test tile), e.g. {sorted(missing)[:2]}", flush=True)
    return len(scenes_seen)


def tta_oil(model, batch, device):
    """Reconstruction TTA: mean logits of identity, hflip, vflip; argmax == oil (index 0)."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(batch)).to(device)
    with torch.no_grad():
        out = model(x)["out"].float()
        out = out + torch.flip(model(torch.flip(x, dims=[3]))["out"].float(), dims=[3])
        out = out + torch.flip(model(torch.flip(x, dims=[2]))["out"].float(), dims=[2])
    return (out.argmax(1) == 0).cpu().numpy()


def run_bg_patches(col, config, fold, test_scenes, bins, index, near_n, device, progress):
    from preprocess.hnm_si.probe import load_probe, probe_settings, read_run
    args, _ = read_run(config["probe_run_dirs"][fold])
    mean, std = probe_settings(args)
    model = load_probe(config["probe_run_dirs"][fold], device, amp=False).model
    size, scale = config["patch_size"], config["scale"]
    rows = [r for r in read_background(config["baseline_bg_tsv"]) if r.scene in test_scenes]
    by_scene = defaultdict(list)
    for r in rows:
        by_scene[r.scene].append(r)
    it = sorted(by_scene)
    if progress:
        from tqdm import tqdm
        it = tqdm(it, desc=f"fold {fold} bg_patch")
    for scene in it:
        meta = {"fold": fold, "source": "bg_patch", "scene": scene, "sensor": sensor_of(scene),
                "size_bin": bins.get(scene, "unknown")}
        with rasterio.open(index[scene]) as src, \
                rasterio.open(Path(config["mask_dir"]) / f"{scene}.tif") as msrc:
            items = []
            for r in by_scene[scene]:
                x0, y0, side = r.x - PAD, r.y - PAD, size + 2 * PAD
                items.append((x0, y0, read_window(src, x0, y0, side, side, 0),
                              read_window(msrc, x0, y0, side, side, 255)[0]))
        for b in range(0, len(items), config["batch_size"]):
            chunk = items[b:b + config["batch_size"]]
            batch = np.stack([model_input(raw[:, PAD:PAD + size, PAD:PAD + size], scale, mean, std)
                              for _, _, raw, _ in chunk])
            preds = tta_oil(model, batch, device)
            for (x0, y0, raw, mask), p in zip(chunk, preds):
                pred = np.zeros(mask.shape, bool)
                scored = np.zeros(mask.shape, bool)
                pred[PAD:PAD + size, PAD:PAD + size] = p
                scored[PAD:PAD + size, PAD:PAD + size] = True
                col.add(meta, pred, scored, mask, raw, near_n, (x0, y0))
    return len(rows)


# --------------------------------------------------------------------------- backgrounds
def background_sets(config, fold, scenes):
    """Baseline train backgrounds plus every completed HNM arm for this fold."""
    train = scenes["train"]
    sets = {"baseline": [r for r in read_background(config["baseline_bg_tsv"]) if r.scene in train]}
    for arm_dir in sorted((Path(config["output_dir"]) / "arms").glob(f"*/fold_{fold}")):
        m = arm_dir / "manifest.json"
        if m.exists() and json.loads(m.read_text(encoding="utf-8")).get("status") == "complete":
            sets[arm_dir.parent.name] = read_background(arm_dir / "selected_train_bg.tsv")
    return sets


def background_composition(config, fold, sets, index, progress):
    size, scale = config["patch_size"], config["scale"]
    offset = config["si"]["reflectance_offset"]
    out = []
    for name, records in sets.items():
        by_scene = defaultdict(list)
        for r in records:
            by_scene[r.scene].append(r)
        it = sorted(by_scene)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, desc=f"fold {fold} bg set {name}")
        for scene in it:
            with rasterio.open(index[scene]) as src, \
                    rasterio.open(Path(config["mask_dir"]) / f"{scene}.tif") as msrc:
                tiles = [(r, read_window(src, r.x, r.y, size, size, 0),
                          read_window(msrc, r.x, r.y, size, size, 255)[0]) for r in by_scene[scene]]
            for r, raw, mask in tiles:
                out.append(_tile_composition(fold, name, r, raw, mask, scale, offset))
    return out


def _tile_composition(fold, name, r, raw, mask, scale, offset):
    valid = model_valid_mask(raw)
    spec = spectral_class(reflectance(raw, scale, offset), valid)
    n = spec.size
    return {"fold": fold, "set": name, "scene": r.scene, "x": r.x, "y": r.y,
            "sensor": sensor_of(r.scene),
            "land_frac": float((spec == LAND).sum() / n),
            "cloud_frac": float((spec == CLOUD).sum() / n),
            "haze_glint_frac": float((spec == HAZE).sum() / n),
            "nodata_frac": float((spec == NODATA).sum() / n),
            "mask1_frac": float((mask == 1).mean()),
            "others_frac": float((mask == 2).mean()),
            "unlabeled_frac": float((mask == 255).mean())}


def sr_saturation(config, fold):
    """Share of FP-region SR_B1B2_B3B8A medians stuck at the +-clip bound."""
    clip = config["si"]["clip"]
    rows = []
    cand = Path(config["output_dir"]) / "mining" / f"fold_{fold}" / "candidates.tsv"
    if cand.exists():
        df = pd.read_csv(cand, sep="\t", usecols=["p_sr"])
        v = df["p_sr"].dropna()
        rows.append({"fold": fold, "what": "mined_fp_regions", "n": int(len(v)),
                     "saturated_frac": float((v.abs() >= clip - 1e-6).mean()) if len(v) else np.nan})
    ref = Path(config["output_dir"]) / "mining" / f"fold_{fold}" / "scene_reference.csv"
    if ref.exists():
        v = pd.read_csv(ref)["oil_sr"].dropna()
        rows.append({"fold": fold, "what": "scene_oil_reference", "n": int(len(v)),
                     "saturated_frac": float((v.abs() >= clip - 1e-6).mean()) if len(v) else np.nan})
    scored = Path(config["output_dir"]) / "arms" / "hnm_si" / f"fold_{fold}" / "candidates_scored.tsv"
    if scored.exists():
        df = pd.read_csv(scored, sep="\t", usecols=["status", "p_sr"])
        v = df.loc[df["status"] == "selected", "p_sr"].dropna()
        rows.append({"fold": fold, "what": "hnm_si_selected", "n": int(len(v)),
                     "saturated_frac": float((v.abs() >= clip - 1e-6).mean()) if len(v) else np.nan})
    return rows


# --------------------------------------------------------------------------- summary
def md_table(df, floatfmt="{:.4g}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summarise(frames, comp, sat, edges, near_n, out):
    tot, fpx, sweep, reg = (frames[k] for k in ("scene_totals", "fp_pixels", "near_oil_sweep",
                                                 "fp_regions"))
    md = [f"# Baseline FP diagnostic (step 1)\n",
          f"near_oil N = {near_n} px; land_coast / cloud / haze_glint are coarse spectral estimates (see RULES in "
          f"tools/analyze_baseline_fp.py). Oil-size tertile edges (recomputed): {edges}.\n"]
    summary = {"near_oil_n": near_n, "rules": RULES, "size_edges": edges}

    # 1. FP volume per fold / source, split by GT label
    g = tot.groupby(["source", "fold"]).sum(numeric_only=True)
    bg_px = g["gt_confirmed_bg"] + g["gt_others"] + g["gt_unlabeled"]
    fp_all = g["fp_confirmed_bg"] + g["fp_others"] + g["fp_unlabeled"]
    vol = pd.DataFrame({
        "scored_px": g["scored_px"], "bg_px": bg_px, "fp_px": fp_all,
        "fp_per_M_bg": 1e6 * fp_all / bg_px.where(bg_px > 0),
        "fp_on_mask1_pct": 100 * g["fp_confirmed_bg"] / fp_all.where(fp_all > 0),
        "fp_on_255_pct": 100 * g["fp_unlabeled"] / fp_all.where(fp_all > 0),
        "oil_iou": g["tp"] / (g["tp"] + g["fn"] + fp_all),
        "precision": g["tp"] / (g["tp"] + fp_all).where(g["tp"] + fp_all > 0),
        "recall": g["tp"] / (g["tp"] + g["fn"]).where(g["tp"] + g["fn"] > 0)}).reset_index()
    md += ["## 1. FP volume by GT label\n",
           "recon oil_iou here must reproduce the run's pooled Oil IoU (sanity check).\n",
           md_table(vol), ""]
    summary["volume"] = vol.to_dict(orient="records")

    # 2. category shares
    def shares(df, by):
        p = df.pivot_table(index=by, columns="category", values="n_px", aggfunc="sum", fill_value=0)
        p = p.reindex(columns=[c for c in CATEGORIES if c in p.columns])
        pct = 100 * p.div(p.sum(axis=1), axis=0)
        pct.insert(0, "hnm_target", pct[[c for c in HNM_TARGETS if c in pct]].sum(axis=1))
        pct.insert(0, "boundary", pct[[c for c in ("near_oil", "oil_attached") if c in pct]]
                   .sum(axis=1))
        pct.insert(0, "fp_px", p.sum(axis=1))
        return pct.reset_index()
    for title, df in (("all FP pixels", fpx),
                      ("FP on confirmed background (mask==1, the HNM mining definition)",
                       fpx[fpx["gt_label"] == "confirmed_bg"])):
        if df.empty:
            continue
        a = shares(df, ["source", "fold"])
        b = shares(df.assign(fold="all"), ["source", "fold"])
        md += [f"## 2. Category share (%) - {title}\n",
               "boundary = near_oil + oil_attached; hnm_target = " + " + ".join(HNM_TARGETS) + ".\n",
               md_table(pd.concat([a, b])), ""]
        summary.setdefault("category_share", {})[title] = pd.concat([a, b]).to_dict(orient="records")

    # 3. near-oil sensitivity
    s = sweep.groupby(["source", "fold", "near_n"])[["n_near", "n_fp"]].sum().reset_index()
    s["near_pct"] = 100 * s["n_near"] / s["n_fp"].where(s["n_fp"] > 0)
    s = s.pivot_table(index=["source", "fold"], columns="near_n", values="near_pct").reset_index()
    s.columns = [f"near<={c}px_%" if isinstance(c, (int, np.integer)) else c for c in s.columns]
    md += ["## 3. Share of FP within N px of GT oil\n", md_table(s), ""]
    summary["near_oil_sweep"] = s.to_dict(orient="records")

    # 4. strata: sensor and oil size (pooled over folds)
    for key in ("sensor", "size_bin"):
        t = tot.groupby(["source", key]).sum(numeric_only=True)
        fp_t = t["fp_confirmed_bg"] + t["fp_others"] + t["fp_unlabeled"]
        bg_t = t["gt_confirmed_bg"] + t["gt_others"] + t["gt_unlabeled"]
        st = pd.DataFrame({"scenes": tot.groupby(["source", key])["scene"].nunique(),
                           "fp_px": fp_t, "fp_per_M_bg": 1e6 * fp_t / bg_t.where(bg_t > 0),
                           "oil_iou": t["tp"] / (t["tp"] + t["fn"] + fp_t).where(t["tp"] + t["fn"] + fp_t > 0),
                           "recall": t["tp"] / (t["tp"] + t["fn"]).where(t["tp"] + t["fn"] > 0)})
        cat = shares(fpx, ["source", key]).set_index(["source", key])
        st = st.join(cat.drop(columns="fp_px")).reset_index()
        md += [f"## 4. By {key} (pooled over folds; category columns are % of FP)\n",
               md_table(st), ""]
        summary[f"by_{key}"] = st.to_dict(orient="records")

    # 5. regions
    if not reg.empty:
        r = reg.groupby(["source", "category"]).agg(
            regions=("area", "size"), px=("area", "sum"), median_area=("area", "median"),
            p90_area=("area", lambda v: float(np.quantile(v, .9)))).reset_index()
        md += ["## 5. FP connected regions (8-connected, category = pixel majority)\n",
               md_table(r), ""]
        summary["regions"] = r.to_dict(orient="records")

    # 6. background sets
    if comp is not None and not comp.empty:
        flags = comp.assign(
            has_land=comp["land_frac"] >= RULES["tile_land_frac"],
            has_cloud=comp["cloud_frac"] >= RULES["tile_cloud_frac"],
            mostly_haze=comp["haze_glint_frac"] >= RULES["tile_haze_frac"])

        def agg(df):
            return df.groupby(["set", "fold"]).agg(
                tiles=("x", "size"), has_land_pct=("has_land", lambda v: 100 * v.mean()),
                has_cloud_pct=("has_cloud", lambda v: 100 * v.mean()),
                mostly_haze_glint_pct=("mostly_haze", lambda v: 100 * v.mean()),
                mean_land_pct=("land_frac", lambda v: 100 * v.mean()),
                mean_cloud_pct=("cloud_frac", lambda v: 100 * v.mean()),
                mean_haze_glint_pct=("haze_glint_frac", lambda v: 100 * v.mean()),
                mean_unlabeled_pct=("unlabeled_frac", lambda v: 100 * v.mean()),
                mean_others_pct=("others_frac", lambda v: 100 * v.mean())).reset_index()
        b = pd.concat([agg(flags), agg(flags.assign(fold="all"))])
        md += [f"## 6. Train background sets: land / cloud / haze content\n",
               f"has_land = land pixels >= {RULES['tile_land_frac']:.0%} of the tile; "
               f"has_cloud = cloud pixels >= {RULES['tile_cloud_frac']:.0%}; "
               f"mostly_haze_glint = haze/glint pixels >= {RULES['tile_haze_frac']:.0%}. "
               "Per-fold sets overlap across folds (a scene is train in two folds).\n",
               md_table(b), ""]
        summary["background_sets"] = b.to_dict(orient="records")

    # 7. SR saturation
    if sat:
        sdf = pd.DataFrame(sat)
        md += ["## 7. SR_B1B2_B3B8A at the clip bound\n", md_table(sdf), ""]
        summary["sr_saturation"] = sdf.to_dict(orient="records")

    md += ["## Crops\n", "`crops/<source>/<category>/`: the 20 largest FP regions per category. "
           "Left true colour; right FP red, correct oil prediction green, GT oil outline cyan, "
           "unlabeled (255) tinted magenta. Use them to check the coarse spectral calls."]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float),
                                      encoding="utf-8")


# --------------------------------------------------------------------------- CLI
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="HNM_SI config (paths and probe runs)")
    parser.add_argument("--fold", type=int, action="append", help="default: all outer folds")
    parser.add_argument("--out", default=str(REPO_ROOT / "hnm_runs" / "diagnostics"))
    parser.add_argument("--near-oil", type=int, default=8)
    parser.add_argument("--boundary-px", type=int, default=8, help="Boundary IoU band width d")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--skip-bg-patches", action="store_true", help="no GPU inference")
    parser.add_argument("--skip-backgrounds", action="store_true",
                        help="skip land/cloud content of the train background sets")
    parser.add_argument("--run-dir", action="append", metavar="FOLD=DIR",
                        help="analyse another finished run instead of the config's baseline, "
                             "e.g. --run-dir 1=result-seg/.../..._fold1 (repeat per fold)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    for item in args.run_dir or []:
        fold, _, path = item.partition("=")
        if not path or not (Path(path) / "weights" / "best.pt").is_file():
            parser.error(f"--run-dir {item!r}: expected FOLD=DIR holding weights/best.pt")
        config["probe_run_dirs"][int(fold)] = str(Path(path).resolve())
    folds = args.fold or config["outer_folds"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = raster_index(config["vrt_dir"])
    bins, edges = scene_size_bins(config["split_dir"])
    offset = config["si"]["reflectance_offset"]
    col = Collector(out, args.top_k, config["scale"], offset, args.boundary_px)
    device = None
    if not args.skip_bg_patches:
        import torch
        device = torch.device(args.device or config["device"])
    comp, sat, counts = [], [], {}
    for fold in folds:
        _, scenes = split_manifest(config["split_dir"], fold)
        test = scenes["test"]
        counts[fold] = {"recon_scenes": run_recon(col, config, fold, test, bins, index,
                                                  args.near_oil, not args.quiet)}
        if not args.skip_bg_patches:
            counts[fold]["bg_patches"] = run_bg_patches(col, config, fold, test, bins, index,
                                                        args.near_oil, device, not args.quiet)
        if not args.skip_backgrounds:
            comp += background_composition(config, fold, background_sets(config, fold, scenes),
                                           index, not args.quiet)
        sat += sr_saturation(config, fold)
    frames = col.write()
    comp_df = pd.DataFrame(comp)
    if not comp_df.empty:
        comp_df.to_csv(out / "background_composition.csv", index=False, encoding="utf-8-sig")
    summarise(frames, comp_df, sat, edges, args.near_oil, out)
    (out / "run_info.json").write_text(json.dumps(
        {"folds": folds, "counts": counts, "near_oil": args.near_oil, "rules": RULES,
         "probe_run_dirs": {str(k): v for k, v in config["probe_run_dirs"].items()}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
