"""Random sample of an arm's selected train backgrounds (three folds pooled) -> numbered
contact sheets for visual labelling against the papers' look-alike categories.

  .venv/bin/python \\
      report_scripts/2026-10-22/analysis/sample_arm_sheets.py ARM N SEED
Writes analysis/hnm_vs_literature/label_<ARM>/sheet_XX.png, tiles/NNN.png and sample.csv.
"""
import csv
import random
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
from preprocess.hnm.data import raster_index, read_background  # noqa: E402
from preprocess.hnm_si.config import load_config  # noqa: E402

arm, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
config = load_config(REPO / "preprocess/hnm_si/config.example.yaml")
index = raster_index(config["vrt_dir"])
size, scale = config["patch_size"], config["scale"]
offset = config["si"]["reflectance_offset"]
rows = []
for f in config["outer_folds"]:
    rows += [(f, r) for r in read_background(Path(config["output_dir"]) / "arms" / arm /
                                             f"fold_{f}" / "selected_train_bg.tsv")]
random.Random(seed).shuffle(rows)
rows = rows[:n]
out = OUT_ROOT / "hnm_vs_literature" / f"label_{arm}"
(out / "tiles").mkdir(parents=True, exist_ok=True)
T, COLS, ROWS = 300, 4, 3
tiles = []
for i, (f, r) in enumerate(rows, 1):
    with rasterio.open(index[r.scene]) as src:
        raw = src.read(window=Window(r.x, r.y, size, size), boundless=True, fill_value=0)
    bgr = true_colour(raw, scale, offset, .15, .6)
    cv2.imwrite(str(out / "tiles" / f"{i:03d}.png"), bgr)
    tiles.append(cv2.resize(bgr, (T, T), interpolation=cv2.INTER_AREA))
for s in range(0, len(tiles), COLS * ROWS):
    sheet = np.zeros((ROWS * (T + 30), COLS * T, 3), np.uint8)
    for k, t in enumerate(tiles[s:s + COLS * ROWS]):
        y, x = (k // COLS) * (T + 30), (k % COLS) * T
        sheet[y:y + T, x:x + T] = t
        cv2.putText(sheet, str(s + k + 1), (x + 6, y + T + 22), cv2.FONT_HERSHEY_SIMPLEX, .7,
                    (255, 255, 255), 2)
    cv2.imwrite(str(out / f"sheet_{s // (COLS * ROWS) + 1:02d}.png"), sheet)
with open(out / "sample.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["id", "fold", "scene", "x", "y"])
    for i, (f, r) in enumerate(rows, 1):
        w.writerow([i, f, r.scene, r.x, r.y])
print(f"{len(rows)} -> {out}")
