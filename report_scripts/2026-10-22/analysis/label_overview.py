"""Whole-scene overview (downsampled): true colour | mask overlay (0 red, 1 blue, 2 yellow, 255 none).
  .venv/bin/python report_scripts/2026-10-22/analysis/label_overview.py OUT_NAME SCENE [SCENE ...]
Also prints, per scene, how far labelled background (1) and unlabelled (255) sit from GT oil."""
import sys
from pathlib import Path
import cv2, numpy as np, rasterio
from rasterio.enums import Resampling
REPO = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO.parent / "reports" / "2026-10-22" / "analysis"   # outputs stay outside git
sys.path.insert(0, str(REPO))
from preprocess.export_hnm_selection import true_colour  # noqa: E402
from preprocess.hnm.data import raster_index  # noqa: E402
from preprocess.hnm_si.config import load_config  # noqa: E402
C = {0: ((0, 0, 255), .7), 1: ((255, 120, 0), .35), 2: ((0, 220, 255), .8)}
name, scenes = sys.argv[1], sys.argv[2:]
cfg = load_config(REPO / "preprocess/hnm_si/config.example.yaml"); idx = raster_index(cfg["vrt_dir"])
out = OUT_ROOT / "label_check"; out.mkdir(exist_ok=True)
rows = []
for s in scenes:
    with rasterio.open(Path(cfg["mask_dir"]) / f"{s}.tif") as m:
        full = m.read(1)
        f = max(1, max(m.height, m.width) // 700)
        small = m.read(1, out_shape=(m.height // f, m.width // f), resampling=Resampling.nearest)
    with rasterio.open(idx[s]) as src:
        raw = src.read([2, 3, 4], out_shape=(3, small.shape[0], small.shape[1]), resampling=Resampling.average)
    pad = np.zeros((8,) + raw.shape[1:], raw.dtype); pad[[1, 2, 3]] = raw[[0, 1, 2]]
    bgr = true_colour(pad, cfg["scale"], cfg["si"]["reflectance_offset"], .15, .6)
    over = bgr.astype(np.float32)
    for v, (c, a) in C.items():
        sel = small == v; over[sel] = (1 - a) * over[sel] + a * np.array(c, np.float32)
    rows.append(np.hstack([bgr, np.full((bgr.shape[0], 6, 3), 60, np.uint8), over.astype(np.uint8)]))
    oil = full == 0
    d = cv2.distanceTransform((~oil).astype(np.uint8), cv2.DIST_L2, 5)
    def q(sel): return np.percentile(d[sel], [10, 50]).round(0).tolist() if sel.any() else None
    near = d <= 64
    print(f"{s[-34:]}: dist-to-oil p10/p50 for 1:{q(full == 1)} 255:{q(full == 255)} | within 64px of oil: "
          f"1={(full[near] == 1).mean():.1%} 255={(full[near] == 255).mean():.1%} 0={(full[near] == 0).mean():.1%}")
w = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 10), (0, w - r.shape[1]), (0, 0))) for r in rows]
cv2.imwrite(str(out / f"{name}.png"), np.vstack(rows)); print(out / f"{name}.png")
