"""A vs R prediction on the same window: true colour | A | R. Green TP, red FP, orange FN, cyan GT outline.
Optional 5th arg "pct": stretch true colour to the window's 2-98 percentile (for bright scenes that saturate the fixed stretch)."""
import sys
from pathlib import Path
import cv2, numpy as np, rasterio
REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO), str(REPO / "main"), str(REPO / "tools")]
from analyze_baseline_fp import recon_artifacts, read_window
from predmask_codec import decode_scene_coverage, decode_scene_final, load_scene_document
from preprocess.export_hnm_selection import RGB_BANDS, true_colour
from preprocess.hnm.data import raster_index
from preprocess.hnm_si.config import load_config

fold, scene, out, side = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
pct = len(sys.argv) > 5 and sys.argv[5] == "pct"
cfg = load_config(REPO / "preprocess/hnm_si/config.example.yaml")
runs = {"A": cfg["probe_run_dirs"][fold],
        "R": next((REPO / "result-seg/CV_gt_expand_ring8").glob(f"*_run_fold{fold}"))}
preds = {}
for k, d in runs.items():
    p = next(x for x in recon_artifacts(d) if load_scene_document(x)["scene_id"] == scene)
    doc = load_scene_document(p); fin = decode_scene_final(doc)
    preds[k] = (fin == int(doc["labels"]["target_class_index"])), decode_scene_coverage(doc, fin)
cov = preds["A"][1] & preds["R"][1]
extra = preds["R"][0] & ~preds["A"][0] & cov
# centre on the densest area of R's extra prediction
with rasterio.open(Path(cfg["mask_dir"]) / f"{scene}.tif") as ms:
    gt_full = ms.read(1) == 0
near = cv2.dilate(gt_full.astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
dens = cv2.boxFilter((extra & near & ~gt_full).astype(np.float32), -1, (side, side), normalize=False)
cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
idx = raster_index(cfg["vrt_dir"])
with rasterio.open(idx[scene]) as src, rasterio.open(Path(cfg["mask_dir"]) / f"{scene}.tif") as ms:
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    x0, y0 = min(x0, src.width - side), min(y0, src.height - side)
    raw = read_window(src, x0, y0, side, side, 0); mask = read_window(ms, x0, y0, side, side, 255)[0]
sl = np.s_[y0:y0 + side, x0:x0 + side]
c = cov[sl]
if pct:
    refl = (raw[list(RGB_BANDS)].astype(np.float32) - cfg["si"]["reflectance_offset"]) / cfg["scale"]
    lo, hi = np.percentile(refl, [2, 98])
    bgr = np.ascontiguousarray((np.clip((refl - lo) / (hi - lo), 0, 1)[::-1].transpose(1, 2, 0) * 255).astype(np.uint8))
else:
    bgr = np.ascontiguousarray(true_colour(raw, cfg["scale"], cfg["si"]["reflectance_offset"], .15, .6))
gt = mask == 0
ctrs, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
panels = [bgr.copy()]
cv2.drawContours(panels[0], ctrs, -1, (255, 255, 0), 1)
for k in ("A", "R"):
    p = preds[k][0][sl] & c
    o = bgr.astype(np.float32)
    for sel, col in ((p & gt, (60, 200, 60)), (p & ~gt, (40, 40, 230)), (~p & gt & c, (0, 165, 255))):
        o[sel] = .4 * o[sel] + .6 * np.array(col, np.float32)
    o[~c] *= .35
    o = np.ascontiguousarray(o.astype(np.uint8)); cv2.drawContours(o, ctrs, -1, (255, 255, 0), 1)
    panels.append(o)
    print(k, "TP", int((p & gt).sum()), "FP", int((p & ~gt).sum()), "FN", int((~p & gt & c).sum()))
f = max(1, 720 // side)
for name, im in zip(("rgb", "A", "R"), panels):
    cv2.imwrite(f"{out}_{name}.png", cv2.resize(im, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST))
print("window", x0, y0, side, "unscored px", int((~c).sum()))
