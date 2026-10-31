"""Pilot label audit: 10 suspicious labels per class, rendered for human review.

A  GT oil components the baseline (A) missed almost entirely (recall < 10%, area >= 300 px)
B  train background windows (mask==1 >= 50%) the scene-OOF EfficientNet calls oil (p >= 0.95)
C  baseline false positives next to GT oil on unlabelled (255) pixels (step-1 crops)

  .venv/bin/python -W ignore report_scripts/2026-10-22/analysis/build_review.py
Writes analysis/review/{A,B,C}_NN.png and 判讀表.csv.
"""
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT_ROOT = REPO.parent / "reports" / "2026-10-22" / "analysis"   # outputs stay outside git
for p in (REPO, REPO / "main", HERE):
    sys.path.insert(0, str(p))

from predmask_codec import decode_scene_coverage, decode_scene_final, load_scene_document  # noqa: E402
from preprocess.export_hnm_selection import true_colour  # noqa: E402
from preprocess.hnm.data import raster_index  # noqa: E402
from preprocess.hnm_si.config import load_config  # noqa: E402

N, T = 10, 420
OUT = OUT_ROOT / "review"
cfg = load_config(REPO / "preprocess/hnm_si/config.example.yaml")
index = raster_index(cfg["vrt_dir"])
scale, offset = cfg["scale"], cfg["si"]["reflectance_offset"]


def render(scene, x, y, size, pred=None, box=None, title=""):
    with rasterio.open(index[scene]) as src:
        raw = src.read(window=Window(x, y, size, size), boundless=True, fill_value=0)
    with rasterio.open(Path(cfg["mask_dir"]) / f"{scene}.tif") as m:
        mask = m.read(1, window=Window(x, y, size, size), boundless=True, fill_value=255)
    bgr = np.ascontiguousarray(true_colour(raw, scale, offset, .15, .6))
    over = bgr.astype(np.float32)
    for v, c, a in ((1, (255, 120, 0), .30), (2, (0, 220, 255), .7)):
        sel = mask == v
        over[sel] = (1 - a) * over[sel] + a * np.array(c, np.float32)
    if pred is not None:
        sel = pred & (mask != 0)
        over[sel] = .45 * over[sel] + .55 * np.array((0, 0, 255), np.float32)
    over = np.ascontiguousarray(over.astype(np.uint8))
    ctrs, _ = cv2.findContours((mask == 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(over, ctrs, -1, (255, 255, 0), 2)
    for im in (bgr, over):
        if box is not None:
            cv2.rectangle(im, box[:2], box[2:], (255, 255, 255), 2)
    pair = [cv2.resize(im, (T, T), interpolation=cv2.INTER_AREA) for im in (bgr, over)]
    row = np.hstack([pair[0], np.full((T, 6, 3), 60, np.uint8), pair[1]])
    cap = np.full((30, row.shape[1], 3), 30, np.uint8)
    cv2.putText(cap, title, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([row, cap])


def class_a(rng):
    comp = pd.read_csv(OUT_ROOT / "contrast_recall_6/components.csv")
    a = comp[(comp.arm == "A") & (comp.area >= 300) & (comp.tp < 0.1 * comp.area)]
    a = a.sample(frac=1, random_state=1).drop_duplicates("scene").head(N)
    runs = {int(k): v for k, v in json.loads(
        (REPO / "hnm_runs/diagnostics/run_info.json").read_text())["probe_run_dirs"].items()}
    rows = []
    for r in a.itertuples():
        art = next(Path(runs[r.fold]).glob(f"post_test_*/gt_aware_recon/prediction_artifacts/{r.scene}.pred.json"))
        doc = load_scene_document(art, verify_sidecar=False)
        final = decode_scene_final(doc)
        cov = decode_scene_coverage(doc, final)
        pred = (final == int(doc["labels"]["target_class_index"])) & cov
        with rasterio.open(Path(cfg["mask_dir"]) / f"{r.scene}.tif") as m:
            mask = m.read(1)
        n, lab, st, _ = cv2.connectedComponentsWithStats(((mask == 0) & cov).astype(np.uint8), connectivity=8)
        x, y, w, h, area = (int(v) for v in st[r.k])
        size = int(min(1024, max(256, 1.6 * max(w, h))))
        x0, y0 = x + w // 2 - size // 2, y + h // 2 - size // 2
        p = np.zeros((size, size), bool)
        ys, xs = slice(max(0, y0), y0 + size), slice(max(0, x0), x0 + size)
        sub = pred[ys, xs]
        p[ys.start - y0:ys.start - y0 + sub.shape[0], xs.start - x0:xs.start - x0 + sub.shape[1]] = sub
        rows.append(dict(cls="A", scene=r.scene, fold=r.fold, x=x0, y=y0, size=size, pred=p,
                         info=f"GT 油 {area} px，A 抓到 {r.tp / r.area:.0%}，{r.sign}",
                         cap=f"GT oil {area} px, A recall {r.tp / r.area:.0%}, {r.sign} contrast; cyan line = GT oil, red = A predicted oil"))
    return rows


def class_b(rng):
    frames = []
    for f in (1, 2, 3):
        d = pd.read_csv(REPO / f"hnm_runs/hnm_si_v1/arms/oof/fold_{f}/candidates_scored.tsv", sep="\t")
        d["fold"] = f
        frames.append(d[d.p_oil >= .95])
    b = pd.concat(frames).sample(frac=1, random_state=2).drop_duplicates("scene").head(N)
    return [dict(cls="B", scene=r.scene, fold=r.fold, x=int(r.x) - 128, y=int(r.y) - 128, size=512,
                 box=(128, 128, 384, 384), info=f"P(oil) = {r.p_oil:.2f}，白框為被評分的窗",
                 cap=f"OOF P(oil) = {r.p_oil:.2f}; white box = scored window; blue = labelled background")
            for r in b.itertuples()]


def class_c():
    rows = []
    for cat in ("near_oil", "oil_attached"):
        files = sorted((REPO / "hnm_runs/diagnostics/crops/recon" / cat).glob("*.png"))[:5]
        rows += [dict(cls="C", file=f, info=f"{cat}；紅 = A 誤報，綠 = 抓對，青框 = GT 油，洋紅 = 255") for f in files]
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    rng = random.Random(0)
    items = class_a(rng) + class_b(rng) + class_c()
    table = []
    counter = {"A": 0, "B": 0, "C": 0}
    for it in items:
        counter[it["cls"]] += 1
        name = f"{it['cls']}_{counter[it['cls']]:02d}"
        if it["cls"] == "C":
            shutil.copy(it["file"], OUT / f"{name}.png")
            scene = it["file"].stem
            loc = ""
        else:
            img = render(it["scene"], it["x"], it["y"], it["size"], it.get("pred"), it.get("box"),
                         f"{name}  {it['cap']}")
            cv2.imwrite(str(OUT / f"{name}.png"), img)
            scene, loc = it["scene"], f"x{it['x']} y{it['y']} s{it['size']}"
        table.append({"編號": name, "類別": it["cls"], "場景": scene, "位置": loc, "說明": it["info"],
                      "判讀（油／look-alike／不確定）": "", "備註": ""})
    with open(OUT / "判讀表.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"{len(table)} items -> {OUT}")


if __name__ == "__main__":
    main()
