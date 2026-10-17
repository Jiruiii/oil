"""Export the backgrounds an HNM run selected for training as viewable PNGs.

Works on any fold artifact directory that holds ``selected_train_bg.tsv`` and
``manifest.json`` (both the EfficientNet OOF run and HNM_SI arms). Per-candidate
scores are merged in from ``candidates_scored.tsv`` or ``scores.tsv`` when present.

Output (default ``<fold dir>/preview``): ``images/`` one PNG per patch (true colour on
the left, the same patch with unlabeled/Others mask pixels tinted on the right),
``sheets/`` contact sheets, ``index.csv`` and ``README.txt``.
Run from the repo root: python -m preprocess.export_hnm_selection --fold-dir DIR
"""
import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

from preprocess.hnm.data import raster_index, read_background, sha256

RGB_BANDS = (3, 2, 1)       # B04, B03, B02 in the eight-band VRT order
CAPTION_H = 34


def true_colour(raw, scale, offset, stretch, gamma):
    """(8, H, W) DN -> uint8 BGR. Reflectance offset removed; nodata stays black."""
    refl = (raw[list(RGB_BANDS)].astype(np.float32) - offset) / scale
    valid = np.isfinite(raw).all(axis=0) & (raw != 0).any(axis=0)
    rgb = np.clip(np.nan_to_num(refl) / stretch, 0, 1) ** gamma
    rgb[:, ~valid] = 0
    return (rgb[::-1].transpose(1, 2, 0) * 255).astype(np.uint8)   # RGB -> BGR for cv2


def tint_mask(bgr, mask):
    """Magenta = unlabeled (255), yellow = Others (2); red would mean oil (0) and should not occur."""
    out = bgr.astype(np.float32)
    for value, colour in ((255, (255, 0, 255)), (2, (0, 255, 255)), (0, (0, 0, 255))):
        sel = mask == value
        out[sel] = .55 * out[sel] + .45 * np.array(colour, dtype=np.float32)
    return out.astype(np.uint8)


def load_scores(fold_dir):
    for name in ("candidates_scored.tsv", "scores.tsv"):
        path = fold_dir / name
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            if name == "candidates_scored.tsv":
                rows = [r for r in rows if r.get("status") == "selected"]
            return {(r["scene"], int(r["x"]), int(r["y"])): {
                k: v for k, v in r.items() if k not in ("scene", "x", "y")} for r in rows}
    return {}


def short_name(scene):
    """ASCII tail 'YYYYMMDD_SENSOR_TILE' (scene names start with a CJK event title)."""
    m = re.search(r"(\d{8}_[A-Z0-9]+_[0-9A-Za-z]+)$", scene)
    return m[1] if m else "".join(c for c in scene if c.isascii())[-17:]


def caption(scene, x, y, info, number):
    tail = short_name(scene)
    extra = " ".join(f"{k}={float(info[k]):.3g}" for k in ("n_fp", "d_si", "el2n") if info.get(k) not in (None, ""))
    return f"#{number} {tail}", f"x{x} y{y} {extra}".strip()


def tile(bgr, lines):
    canvas = np.zeros((bgr.shape[0] + CAPTION_H, bgr.shape[1], 3), np.uint8)
    canvas[:bgr.shape[0]] = bgr
    for i, text in enumerate(lines):
        cv2.putText(canvas, text, (3, bgr.shape[0] + 13 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    .38, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def export(fold_dir, out=None, limit=None, individual=True, cols=5, rows=4,
           stretch=.15, gamma=.6, config=None):
    fold_dir = Path(fold_dir)
    manifest = json.loads((fold_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"Manifest status is {manifest.get('status')!r}; export only completed runs")
    cfg = config or manifest["config"]
    vrt_dir, mask_dir, size = cfg["vrt_dir"], cfg["mask_dir"], cfg["patch_size"]
    scale = cfg.get("scale", 10000.0)
    offset = (cfg.get("si") or {}).get("reflectance_offset", 1000.0)
    selected = read_background(fold_dir / "selected_train_bg.tsv")
    if sha256(fold_dir / "selected_train_bg.tsv") != manifest["selected_train_bg_sha256"]:
        raise ValueError("selected_train_bg.tsv does not match its manifest")
    if limit:
        selected = selected[:limit]
    scores = load_scores(fold_dir)
    index = raster_index(vrt_dir)
    out = Path(out) if out else fold_dir / "preview"
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    if individual:
        (out / "images").mkdir(exist_ok=True)
    extra_cols = sorted({k for v in scores.values() for k in v})
    records, tiles, sheet_no = [], [], 0

    def flush():
        nonlocal tiles, sheet_no
        if not tiles:
            return
        sheet_no += 1
        blank = np.zeros_like(tiles[0])
        tiles += [blank] * (cols * rows - len(tiles))
        grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
        cv2.imwrite(str(out / "sheets" / f"sheet_{sheet_no:03d}.png"), grid)
        tiles = []

    by_scene = {}
    for number, p in enumerate(selected, 1):
        by_scene.setdefault(p.scene, []).append((number, p))
    ordered = []
    for scene in sorted(by_scene):
        ordered += by_scene[scene]
    for number, p in ordered:
        with rasterio.open(index[p.scene]) as src:
            raw = src.read(window=Window(p.x, p.y, size, size), boundless=True, fill_value=0)
        with rasterio.open(Path(mask_dir) / f"{p.scene}.tif") as src:
            mask = src.read(1, window=Window(p.x, p.y, size, size), boundless=True, fill_value=255)
        bgr = true_colour(raw, scale, offset, stretch, gamma)
        info = scores.get(p.key, {})
        name = f"{number:05d}_{short_name(p.scene)}_x{p.x}_y{p.y}.png"
        if individual:
            cv2.imwrite(str(out / "images" / name), np.hstack([bgr, tint_mask(bgr, mask)]))
        tiles.append(tile(bgr, caption(p.scene, p.x, p.y, info, number)))
        if len(tiles) == cols * rows:
            flush()
        records.append({"number": number, "file": name if individual else "", "scene": p.scene,
                        "x": p.x, "y": p.y,
                        "mask1_pct": round(100 * float((mask == 1).mean()), 1),
                        "others_pct": round(100 * float((mask == 2).mean()), 1),
                        "unlabeled_pct": round(100 * float((mask == 255).mean()), 1),
                        "oil_pct": round(100 * float((mask == 0).mean()), 1),
                        **{k: info.get(k, "") for k in extra_cols}})
    flush()
    records.sort(key=lambda r: r["number"])
    with open(out / "index.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]) if records else ["number"])
        writer.writeheader()
        writer.writerows(records)
    (out / "README.txt").write_text(
        f"HNM selected training backgrounds\n"
        f"method: {manifest.get('method', 'unknown')}  arm: {manifest.get('arm', '-')}  "
        f"fold: {manifest['outer_fold']}\n"
        f"patches exported: {len(records)} of {manifest['selected_count']}\n"
        f"source: {fold_dir}\n"
        f"selected_train_bg.tsv sha256: {manifest['selected_train_bg_sha256']}\n\n"
        f"images/  left = true colour (B04,B03,B02; reflectance (DN-{offset:g})/{scale:g}, "
        f"stretch 0-{stretch}, gamma {gamma});\n"
        f"         right = same patch with unlabeled (255) tinted magenta and Others (2) yellow.\n"
        f"         Every patch is trained as all-background, so tinted areas are not confirmed sea.\n"
        f"sheets/  contact sheets of {cols}x{rows}; caption = #number, date_sensor_tile, x/y, scores.\n"
        f"index.csv  full scene names, coordinates, mask composition and per-patch scores.\n",
        encoding="utf-8")
    return len(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", required=True,
                        help="e.g. hnm_runs/hnm_si_v1/arms/hnm_si/fold_1")
    parser.add_argument("--out", help="default: <fold-dir>/preview")
    parser.add_argument("--limit", type=int, help="export only the first N selected patches")
    parser.add_argument("--no-individual", action="store_true", help="contact sheets only")
    args = parser.parse_args(argv)
    n = export(args.fold_dir, args.out, args.limit, not args.no_individual)
    print(f"Exported {n} patches -> {args.out or Path(args.fold_dir) / 'preview'}")


if __name__ == "__main__":
    main()
