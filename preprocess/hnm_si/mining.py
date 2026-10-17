"""Per-scene scan: eligible windows, probe false positives, FP-region SI, oil reference.

Imagery is read one window-row strip at a time, so memory stays bounded for the
10980 x 10980 scenes even though every eligible window of the scene is scored.
"""
import numpy as np
import rasterio
from rasterio.windows import Window

from .indices import (compute_si, model_input, model_valid_mask, reflectance, sensor_of)

COLUMNS = ("scene", "x", "y", "sensor", "mask1_ratio", "others_ratio", "unlabeled_ratio",
           "nodata_ratio", "n_fp", "n_fp_si", "mean_oil_prob", "max_oil_prob",
           "p_mndwi", "p_ndi", "p_sr")
VALID_MASK_VALUES = {0, 1, 2, 255}


def window_positions(height, width, size, stride):
    """Original grid: regular stride plus a final window flush with the far edge."""
    if height < size or width < size:
        return [], []
    ys = list(dict.fromkeys(list(range(0, height - size, stride)) + [height - size]))
    xs = list(dict.fromkeys(list(range(0, width - size, stride)) + [width - size]))
    return ys, xs


def check_mask(mask, scene):
    present = {int(v) for v in np.flatnonzero(np.bincount(mask.ravel(), minlength=256))}
    if not present <= VALID_MASK_VALUES:
        raise ValueError(f"Unexpected mask encoding in {scene}: {sorted(present - VALID_MASK_VALUES)}")


def eligible_windows(mask, size, stride, rules, exclude=frozenset()):
    """Mask-only gate: no oil, enough confirmed background, limited unlabeled area.

    ``exclude`` holds (x, y) origins that duplicate existing GT-centric tiles.
    ``others_ratio`` (mask==2) is recorded but never counted as confirmed background.
    """
    ys, xs = window_positions(*mask.shape, size, stride)
    n = size * size
    out = []
    for y in ys:
        for x in xs:
            if (x, y) in exclude:
                continue
            w = mask[y:y + size, x:x + size]
            if np.count_nonzero(w == 0):
                continue
            m1, m2, m255 = (np.count_nonzero(w == v) / n for v in (1, 2, 255))
            if m1 < rules["min_mask1_ratio"] or m255 > rules["max_mask255_ratio"]:
                continue
            out.append((y, x, m1, m2, m255))
    return out


def scene_oil_reference(src, mask, bands, scale, si_cfg, rng, chunk_rows=512):
    """Median SI over valid ground-truth oil pixels plus a random pixel sample.

    Only called for train scenes of one fold, so the reference never sees val/test labels.
    """
    oil = mask == 0
    n_oil = int(oil.sum())
    result = {"n_oil_px": n_oil, "n_valid_oil_px": 0, "fingerprint": None, "pool": None}
    if n_oil == 0:
        return result
    rows = np.flatnonzero(oil.any(axis=1))
    cols = np.flatnonzero(oil.any(axis=0))
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    chunks = []
    for r0 in range(int(rows[0]), int(rows[-1]) + 1, chunk_rows):
        r1 = min(r0 + chunk_rows, int(rows[-1]) + 1)
        sel_oil = oil[r0:r1, x0:x1]
        if not sel_oil.any():
            continue
        raw = src.read(window=Window(x0, r0, x1 - x0, r1 - r0))
        si, ok = compute_si(reflectance(raw, scale, si_cfg["reflectance_offset"]), bands,
                            si_cfg["den_floor"], si_cfg["clip"])
        sel = sel_oil & ok
        if sel.any():
            chunks.append(si[:, sel].T)
    if not chunks:
        return result
    values = np.concatenate(chunks)
    result["n_valid_oil_px"] = int(len(values))
    if len(values) < si_cfg["min_oil_pixels"]:
        return result
    result["fingerprint"] = np.median(values, axis=0)
    take = min(si_cfg["oil_pool_samples_per_scene"], len(values))
    result["pool"] = values[rng.choice(len(values), size=take, replace=False)].astype(np.float32)
    return result


def scan_scene(scene, raster_path, mask_path, config, probe, mean, std, bands,
               exclude, rng, with_reference=True):
    """Score every eligible window of one scene. Returns (rows, summary, reference)."""
    size, stride, scale = config["patch_size"], config["stride"], config["scale"]
    rules, si_cfg = config["rules"], config["si"]
    with rasterio.open(mask_path) as m_src:
        mask = m_src.read(1)
    check_mask(mask, scene)
    sensor = sensor_of(scene)
    rows, n_nodata = [], 0
    with rasterio.open(raster_path) as src:
        if (src.height, src.width) != mask.shape:
            raise ValueError(f"Image/mask shape mismatch: {scene}")
        windows = eligible_windows(mask, size, stride, rules, exclude)
        by_row = {}
        for w in windows:
            by_row.setdefault(w[0], []).append(w)
        for y, row in sorted(by_row.items()):
            x_lo, x_hi = min(w[1] for w in row), max(w[1] for w in row) + size
            strip = src.read(window=Window(x_lo, y, x_hi - x_lo, size))
            crops, meta = [], []
            for _, x, m1, m2, m255 in row:
                raw = strip[:, :, x - x_lo:x - x_lo + size]
                valid = model_valid_mask(raw)
                nodata = 1.0 - float(valid.mean())
                if nodata > rules["max_nodata_ratio"]:
                    n_nodata += 1
                    continue
                crops.append((raw, valid))
                meta.append((x, m1, m2, m255, nodata))
            for b in range(0, len(crops), config["batch_size"]):
                batch = crops[b:b + config["batch_size"]]
                probs = probe(np.stack([model_input(r, scale, mean, std) for r, _ in batch]))
                for (raw, valid), p, (x, m1, m2, m255, nodata) in zip(
                        batch, probs, meta[b:b + config["batch_size"]]):
                    m = mask[y:y + size, x:x + size]
                    fp = (p > rules["prob_threshold"]) & (m == 1) & valid
                    n_fp = int(fp.sum())
                    n_fp_si, med = 0, (np.nan,) * 3
                    if n_fp >= si_cfg["min_fp_pixels"]:
                        si, ok = compute_si(reflectance(raw, scale, si_cfg["reflectance_offset"]),
                                            bands, si_cfg["den_floor"], si_cfg["clip"])
                        sel = fp & ok
                        n_fp_si = int(sel.sum())
                        if n_fp_si >= si_cfg["min_fp_pixels"]:
                            med = tuple(float(v) for v in np.median(si[:, sel], axis=1))
                    rows.append((scene, x, y, sensor, m1, m2, m255, nodata, n_fp, n_fp_si,
                                 float(p.mean()), float(p.max()), *med))
        reference = (scene_oil_reference(src, mask, bands, scale, si_cfg, rng)
                     if with_reference else None)
    summary = {"scene": scene, "sensor": sensor, "n_windows_total": int(
                   np.prod([len(a) for a in window_positions(*mask.shape, size, stride)])),
               "n_mask_eligible": len(windows), "n_nodata_rejected": n_nodata,
               "n_scored": len(rows), "n_with_fp": sum(r[8] > 0 for r in rows)}
    return rows, summary, reference
