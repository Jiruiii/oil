"""Render true colour next to the GT mask overlay, to check what each label value means.

Overlay: 0 oil = red, 1 labelled background = blue tint, 2 others = yellow, 255 = untouched.
  .venv/bin/python \\
      report_scripts/2026-10-22/analysis/label_check.py OUT_NAME scene:x:y:size [scene:x:y:size ...]
Writes analysis/label_check/OUT_NAME.png (one row per crop) plus per-crop label shares.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

REPO = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO.parent / "reports" / "2026-10-22" / "analysis"   # outputs stay outside git
sys.path.insert(0, str(REPO))

from preprocess.export_hnm_selection import true_colour  # noqa: E402
from preprocess.hnm.data import raster_index  # noqa: E402
from preprocess.hnm_si.config import load_config  # noqa: E402

COLOURS = {0: (0, 0, 255), 1: (255, 120, 0), 2: (0, 220, 255)}   # BGR
ALPHA = {0: .55, 1: .30, 2: .70}
T = 360


def render(config, index, scene, x, y, size):
    with rasterio.open(index[scene]) as src:
        raw = src.read(window=Window(x, y, size, size), boundless=True, fill_value=0)
    with rasterio.open(Path(config["mask_dir"]) / f"{scene}.tif") as m:
        mask = m.read(1, window=Window(x, y, size, size), boundless=True, fill_value=255)
    bgr = true_colour(raw, config["scale"], config["si"]["reflectance_offset"], .15, .6)
    over = bgr.astype(np.float32)
    for v, c in COLOURS.items():
        sel = mask == v
        over[sel] = (1 - ALPHA[v]) * over[sel] + ALPHA[v] * np.array(c, np.float32)
    over = np.ascontiguousarray(over.astype(np.uint8))
    ctrs, _ = cv2.findContours((mask == 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, ctrs, -1, (0, 0, 255), 1)
    pair = [cv2.resize(im, (T, T), interpolation=cv2.INTER_NEAREST) for im in (bgr, over)]
    row = np.hstack([pair[0], np.full((T, 4, 3), 60, np.uint8), pair[1]])
    shares = {v: float((mask == v).mean()) for v in (0, 1, 2, 255)}
    cap = np.full((26, row.shape[1], 3), 30, np.uint8)
    txt = f"{scene[-30:]} x{x} y{y} s{size} | 0:{shares[0]:.0%} 1:{shares[1]:.0%} 2:{shares[2]:.0%} 255:{shares[255]:.0%}"
    cv2.putText(cap, txt, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([row, cap]), shares


def main():
    name, specs = sys.argv[1], sys.argv[2:]
    config = load_config(REPO / "preprocess/hnm_si/config.example.yaml")
    index = raster_index(config["vrt_dir"])
    out = OUT_ROOT / "label_check"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in specs:
        scene, x, y, size = spec.rsplit(":", 3)
        img, shares = render(config, index, scene, int(x), int(y), int(size))
        rows.append(img)
        print(spec, {k: round(v, 3) for k, v in shares.items()})
    cv2.imwrite(str(out / f"{name}.png"), np.vstack(rows))
    print(out / f"{name}.png")


if __name__ == "__main__":
    main()
