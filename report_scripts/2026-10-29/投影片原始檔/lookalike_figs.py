"""Figures for the look-alike pages of the 10/29 deck (pages: sweep, lost oil, feature space).

Run from the repo root (OIL_PROJECT_MutiBand_GT_expand) with .venv/bin/python:
  .venv/bin/python report_scripts/2026-10-29/投影片原始檔/lookalike_figs.py
Reads hnm_runs/threshold_sweep, hnm_runs/lost_oil, hnm_runs/feature_space and the
HNM selection under hnm_runs/hnm_si_v1/arms; writes into img/ next to this file.
"""
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path.cwd()
sys.path.insert(0, str(REPO / "tools"))
from analyze_baseline_fp import raster_index, true_colour  # noqa: E402
from preprocess.hnm_si.config import load_config  # noqa: E402

IMG = REPO.parent / "reports" / "2026-10-29" / "投影片原始檔" / "img"   # outputs stay outside git
COL = {"A": "#5B6472", "B": "#2F6FB0", "C": "#A3561A", "D": "#E3A869"}
NAME = {"A": "A 基準", "B": "B 隨機（同池）", "C": "C HNM-A", "D": "D HNM-A+SI"}
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Noto Sans CJK TC", "Noto Sans CJK HK", "DejaVu Sans"],
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 12})
CFG = load_config(str(REPO / "preprocess/hnm_si/config.example.yaml"))
INDEX = raster_index(CFG["vrt_dir"])
SCALE, OFFSET = CFG["scale"], CFG["si"]["reflectance_offset"]


def sweep():
    cur = pd.read_csv(REPO / "hnm_runs/threshold_sweep/sweep_curves.csv")
    pts = pd.read_csv(REPO / "hnm_runs/threshold_sweep/matched_points.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9), sharey=True)
    for ax, f in zip(axes, (1, 2, 3)):
        for a in "ABCD":
            g = cur[(cur.arm == a) & (cur.fold == f) & (cur.threshold > 0)]
            ax.plot(g.recall, 100 * g.bg_tile_rate, color=COL[a], lw=2.2 if a == "C" else 1.6,
                    label=NAME[a])
            p = g.iloc[(g.threshold - 0.5).abs().argmin()]
            ax.plot(p.recall, 100 * p.bg_tile_rate, "o", color=COL[a], ms=6)
        t = pts[(pts.match == "recall") & (pts.arm == "A") & (pts.fold.astype(str) == str(f))].target.iloc[0]
        ax.axvline(t, color="#888888", lw=1, ls="--")
        ax.set_title(f"fold {f}", fontsize=13)
        ax.set_xlabel("recall（油汙周邊重組）")
        ax.set_xlim(0.25, 0.8)
        ax.set_ylim(0, 12)
        ax.grid(alpha=.3)
    axes[0].set_ylabel("背景小圖警報率（%）")
    axes[-1].legend(frameon=False, fontsize=11, loc="upper left")
    fig.tight_layout()
    fig.savefig(IMG / "sweep.png", dpi=200)
    plt.close(fig)


def crop(z, i, field):
    a, b = z["crop_offset"][i], z["crop_offset"][i + 1]
    return z[field][a:b].reshape(z["crop_shape"][i])


def square(img, fill):
    h, w = img.shape[:2]
    s = max(h, w)
    out = np.full((s, s, 3), fill, np.uint8)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    return cv2.resize(out, (480, 480), interpolation=cv2.INTER_NEAREST)


def lost_examples():
    picks = [("ex34", 2, "34SGH", 27020), ("ex16", 2, "16RCT", 4935), ("ex19", 2, "19TDE", 5852)]
    for tag, fold, site, area in picks:
        za = np.load(REPO / f"hnm_runs/lost_oil/A/fold_{fold}.npz")
        zc = np.load(REPO / f"hnm_runs/lost_oil/C/fold_{fold}.npz")
        i = int(np.flatnonzero(np.char.endswith(za["scene"].astype(str), site) & (za["area"] == area))[0])
        bits = crop(za, i, "crop_mask")
        (x0, y0), (h, w) = za["crop_origin"][i], bits.shape
        with rasterio.open(INDEX[str(za["scene"][i])]) as src:
            raw = src.read(window=Window(int(x0), int(y0), w, h)).astype(np.float32)
        ctrs, _ = cv2.findContours((bits & 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        rgb = np.ascontiguousarray(true_colour(raw, SCALE, OFFSET, .12, .6))
        panels = {"rgb": rgb}
        for k, z in (("A", za), ("C", zc)):
            p = cv2.applyColorMap(crop(z, i, "crop_prob"), cv2.COLORMAP_VIRIDIS)
            p[(bits & 4) == 0] = 60
            panels[k] = p
        for k, p in panels.items():
            cv2.drawContours(p, ctrs, -1, (255, 255, 255), 1)
            cv2.imwrite(str(IMG / f"{tag}_{k}.png"), square(p, (242, 246, 247)))
        print(tag, za["scene"][i], area)


def hard_negatives():
    picks = [("hn1", 2, "34SGH", 3), ("hn2", 2, "34SGH", 4), ("hn3", 2, "16RCT", 2), ("hn4", 2, "16RCT", 3)]
    for tag, fold, site, rank in picks:
        d = REPO / f"hnm_runs/hnm_si_v1/arms/hnm/fold_{fold}"
        sel = pd.read_csv(d / "selected_train_bg.tsv", sep="\t").merge(
            pd.read_csv(d / "candidates_scored.tsv", sep="\t"), on=["scene", "x", "y"])
        r = sel[sel.scene.str.endswith("_" + site)].sort_values("n_fp", ascending=False).iloc[rank]
        with rasterio.open(INDEX[r.scene]) as src:
            raw = src.read(window=Window(int(r.x), int(r.y), 256, 256), boundless=True,
                           fill_value=0).astype(np.float32)
        t = true_colour(raw, SCALE, OFFSET, .12, .6)
        cv2.imwrite(str(IMG / f"{tag}.png"), cv2.resize(t, (480, 480), interpolation=cv2.INTER_AREA))
        print(tag, r.scene, int(r.n_fp))


def pca():
    out = REPO / "hnm_runs/feature_space"
    sim = pd.read_csv(out / "similarities.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.9))
    for ax, f in zip(axes, (1, 2, 3)):
        z = np.load(out / f"fold_{f}.npz")
        ref = np.vstack([z[f"{g}_enc"] for g in ("train", "random", "hard_C")])
        mu, sd = ref.mean(axis=0), ref.std(axis=0) + 1e-6
        n = (ref - mu) / sd
        c0 = n.mean(axis=0)
        _, _, vt = np.linalg.svd(n - c0, full_matrices=False)
        proj = lambda x: ((x - mu) / sd - c0) @ vt[:2].T  # noqa: E731
        for g, c, lab in (("random", "#C9CDD3", "隨機背景"), ("hard_C", "#E3A869", "C 的硬負樣本"),
                          ("train", "#8FB3DA", "訓練油")):
            p = proj(z[f"{g}_enc"])
            ax.scatter(p[:, 0], p[:, 1], s=5, c=c, alpha=.45, label=lab, linewidths=0)
        keys = pd.read_csv(out / f"fold_{f}_oil.csv")
        d = keys.merge(sim[(sim.fold == f) & (sim.space == "enc")], on=["scene", "label"], how="left")
        p = proj(z["oil_enc"])
        lost = (d.class_C == "lost").values
        kept = ((d.tp_A / d.area >= .5) & (d.tp_C / d.area >= .5)).values
        ax.scatter(p[kept, 0], p[kept, 1], s=14, facecolors="none", edgecolors="#2B7A3F",
                   linewidths=.8, label="test 油：A、C 都抓到")
        ax.scatter(p[lost, 0], p[lost, 1], s=36, c="#C0392B", marker="x", linewidths=1.5,
                   label="test 油：C 漏掉")
        ax.set_title(f"fold {f}", fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_ylabel("PC2")
    for ax in axes:
        ax.set_xlabel("PC1")
    axes[-1].legend(frameon=True, fontsize=9.5, loc="lower left", markerscale=1.6)
    fig.tight_layout()
    fig.savefig(IMG / "pca_enc.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    sweep()
    lost_examples()
    hard_negatives()
    pca()
