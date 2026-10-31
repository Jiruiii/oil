"""Which GT oil do the HNM arms lose that the baseline finds?

Reads tools/oil_component_prob.py output (P(oil) per GT oil component, same components for
every arm) and asks, against the reference arm (A):
  1. Is the recall loss whole slicks missed or slicks shrinking?  Component classes:
       lost     ref recall >= FOUND, arm recall < MISSED (both at P(oil) >= 0.5)
       shrunk   both >= MISSED and the arm finds fewer pixels
     "unrecoverable": a lost component whose recall is still < MISSED at P(oil) >= LOW.
  2. What is the lost oil like?  Recall per stratum (size, thickness, contrast sign,
     glint proxy, sensor, coast) with paired scene-bootstrap differences.
  3. Is it concentrated in a few events?
  4. Does it sit where the arm trained on hard negatives?  exposure = the arm's selected
     train background tiles at the same MGRS tile (same place, other dates) on which the
     baseline probe predicted oil (n_fp > 0); bins < 50 / 50-199 / 200-499 / >= 500 tiles.
Attributes (arm-independent, cached in attributes.csv; coarse, from the eight-band VRT):
  thickness  2 * area / perimeter (px)
  contrast   B8A (slick median - ring median) / ring median, ring 4-16 px of non-oil water,
             as reports/2026-10-22/analysis/contrast_recall.py; bright >= +10 %, dark <= -10 %
  nir_ring   ring median B8A reflectance (sun glint / haze raise it)
  coast      >= 20 land px (analyze_baseline_fp.spectral_class) within CROP_PAD px

  .venv/bin/python tools/analyze_lost_oil.py --out hnm_runs/lost_oil \\
      --arm A=hnm_runs/lost_oil/A --arm B=hnm_runs/lost_oil/B ... --ref A
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_baseline_fp import (BAND, LAND, REPO_ROOT, raster_index, short_name,  # noqa: E402
                                 spectral_class, true_colour)
from preprocess.hnm_si.config import load_config  # noqa: E402
from preprocess.hnm_si.indices import model_valid_mask, reflectance, sensor_of  # noqa: E402

FOLDS = (1, 2, 3)
FOUND, MISSED = 0.5, 0.1
HALF, LOW = 0.5, 0.05
RING_IN, RING_OUT = 4, 16
BRIGHT, DARK = 0.10, -0.10
COAST_MIN_LAND = 20
STRATA = {
    "size": ([0, 500, 5000, 50000, np.inf], ["<500", "500-5k", "5k-50k", ">=50k"]),
    "thick": ([0, 6, 15, np.inf], ["<6px", "6-15px", ">=15px"]),
    "glint": ([-np.inf, 0.01, 0.03, 0.06, np.inf], ["<0.01", "0.01-0.03", "0.03-0.06", ">=0.06"]),
}
PAIRS = (("B", "A"), ("C", "A"), ("D", "A"), ("C", "B"), ("D", "B"),
         ("G", "C"), ("G", "G0"), ("G", "B"), ("G0", "C"), ("G", "A"), ("G0", "A"))
SELECTION = {"B": "random", "C": "hnm", "D": "hnm_si",     # arm -> hnm_runs/.../arms/<name>
             "G": "hnm_knn", "G0": "hnm_swap"}
EXPOSURE = ([0, 50, 200, 500, np.inf], ["<50", "50-199", "200-499", ">=500"])   # ~ scene quartiles


def load_arm(path):
    z = np.load(path)
    keys = pd.DataFrame(dict(scene=z["scene"], label=z["label"], area=z["area"]))
    return keys, z


def tp_at(hist, edges, t):
    return hist[:, int(np.searchsorted(edges, t)):].sum(axis=1)


def crop(z, i, field):
    a, b = z["crop_offset"][i], z["crop_offset"][i + 1]
    return z[field][a:b].reshape(z["crop_shape"][i])


def attributes(z, keys, index, scale, offset):
    """Arm-independent attributes of every component (reads the VRT crop of each)."""
    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING_IN + 1,) * 2)
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING_OUT + 1,) * 2)
    rows, open_scene, src = [], None, None
    for i, r in enumerate(keys.itertuples()):
        if r.scene != open_scene:
            if src is not None:
                src.close()
            src, open_scene = rasterio.open(index[r.scene]), r.scene
        bits = crop(z, i, "crop_mask")
        (x0, y0), (h, w) = z["crop_origin"][i], bits.shape
        raw = src.read(window=Window(int(x0), int(y0), w, h)).astype(np.float32)
        valid = model_valid_mask(raw)
        refl = reflectance(raw, scale, offset)
        comp = (bits & 1).astype(np.uint8)
        anyoil = (bits & 2) > 0
        ring = (cv2.dilate(comp, k_out) > 0) & ~(cv2.dilate(comp, k_in) > 0) & valid & ~anyoil
        inside = (comp > 0) & valid
        ctrs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perim = sum(cv2.arcLength(c, True) for c in ctrs)
        nir = refl[BAND["B8A"]]
        row = dict(thickness=2 * r.area / max(perim, 1.0), contrast=np.nan, nir_ring=np.nan,
                   coast=int((spectral_class(refl, valid) == LAND).sum() >= COAST_MIN_LAND))
        if ring.sum() >= 20 and inside.sum() >= 20:
            nir_o, nir_r = np.median(nir[inside]), np.median(nir[ring])
            row.update(nir_ring=nir_r, contrast=(nir_o - nir_r) / max(abs(nir_r), 0.005))
        rows.append(row)
    if src is not None:
        src.close()
    df = pd.DataFrame(rows, index=keys.index)
    df["sign"] = np.select([df.contrast >= BRIGHT, df.contrast <= DARK], ["bright", "dark"], "weak")
    df.loc[df.contrast.isna(), "sign"] = "unknown"
    return df


def build(arms, ref, config, out):
    index = raster_index(config["vrt_dir"])
    scale, offset = config["scale"], config["si"]["reflectance_offset"]
    frames, store = [], {}
    for f in FOLDS:
        keys, zr = load_arm(Path(arms[ref]) / f"fold_{f}.npz")
        zs = {ref: zr}
        for a, d in arms.items():
            if a != ref:
                k, zs[a] = load_arm(Path(d) / f"fold_{f}.npz")
                if not k.equals(keys):
                    raise ValueError(f"arm {a} fold {f}: components differ from {ref}")
        cache = out / f"attributes_fold{f}.csv"
        if cache.exists():
            attr = pd.read_csv(cache, index_col=0)
        else:
            print(f"fold {f}: attributes of {len(keys)} components", flush=True)
            attr = attributes(zr, keys, index, scale, offset)
            attr.to_csv(cache)
        df = pd.concat([keys, attr], axis=1)
        df["fold"] = f
        edges = zr["edges"]
        for a, z in zs.items():
            df[f"tp_{a}"] = tp_at(z["hist"], edges, HALF)
            df[f"tplow_{a}"] = tp_at(z["hist"], edges, LOW)
            occupied = z["hist"] > 0
            df[f"pmax_{a}"] = edges[z["hist"].shape[1] - np.argmax(occupied[:, ::-1], axis=1)]
        frames.append(df)
        store[f] = zs
    df = pd.concat(frames, ignore_index=True)
    df["sensor"] = df.scene.map(sensor_of)
    df["event"] = df.scene.str.rsplit("_", n=3).str[0]
    for name, (bins, labels) in STRATA.items():
        col = {"size": "area", "thick": "thickness", "glint": "nir_ring"}[name]
        df[name] = pd.cut(df[col], bins, labels=labels, right=False).astype(str)
    df["coast"] = np.where(df.coast == 1, "coast", "open_sea")
    return df, store


def mgrs(scene):
    return scene.rsplit("_", 1)[1]


def exposure(df, arms, sel_dir):
    """Per component: the arm's selected train tiles at the same MGRS tile with probe FP."""
    for a in [x for x in arms if x in SELECTION]:
        col = pd.Series(0, index=df.index)
        for f in FOLDS:
            d = Path(sel_dir) / SELECTION[a] / f"fold_{f}"
            sel = pd.read_csv(d / "selected_train_bg.tsv", sep="\t").merge(
                pd.read_csv(d / "candidates_scored.tsv", sep="\t")[["scene", "x", "y", "n_fp"]],
                on=["scene", "x", "y"], how="left", validate="one_to_one")
            if sel.n_fp.isna().any():
                raise ValueError(f"{d}: selected tiles missing from candidates_scored.tsv")
            hard = sel[sel.n_fp > 0].scene.map(mgrs).value_counts()
            rows = df.fold == f
            col[rows] = df.loc[rows, "scene"].map(mgrs).map(hard).fillna(0).astype(int)
        df[f"hard_{a}"] = col
        df[f"exposure_{a}"] = pd.cut(col, EXPOSURE[0], labels=EXPOSURE[1], right=False).astype(str)


def classify(df, a, ref):
    r_ref, r_a, r_low = (df[f"tp_{ref}"] / df.area, df[f"tp_{a}"] / df.area,
                         df[f"tplow_{a}"] / df.area)
    cls = np.select([(r_ref >= FOUND) & (r_a < MISSED),
                     (r_a >= FOUND) & (r_ref < MISSED),
                     (r_ref < MISSED) & (r_a < MISSED),
                     df[f"tp_{a}"] < df[f"tp_{ref}"]],
                    ["lost", "gained", "both_missed", "shrunk"], "grown_or_same")
    return pd.Series(cls, index=df.index), (cls == "lost") & (r_low < MISSED)


def boot_by_stratum(df, col, a, b, draws=2000, seed=0):
    """Pooled recall(a) - recall(b) per stratum; three-fold mean, scenes resampled per fold."""
    rng = np.random.default_rng(seed)
    levels = sorted(df[col].unique())
    pts, mean_draws = {lv: [] for lv in levels}, {lv: 0.0 for lv in levels}
    for f in FOLDS:
        g = df[df.fold == f]
        scenes = sorted(g.scene.unique())
        w = np.vstack([np.ones(len(scenes)),
                       rng.multinomial(len(scenes), np.full(len(scenes), 1 / len(scenes)), draws)])
        for lv in levels:
            s = (g[g[col] == lv].groupby("scene")[[f"tp_{a}", f"tp_{b}", "area"]].sum()
                 .reindex(scenes, fill_value=0))
            with np.errstate(invalid="ignore", divide="ignore"):
                d = (w @ (s[f"tp_{a}"].values - s[f"tp_{b}"].values)) / (w @ s.area.values)
            pts[lv].append(d[0])
            mean_draws[lv] = mean_draws[lv] + d / len(FOLDS)
    rows = []
    for lv in levels:
        lo, hi = np.nanpercentile(mean_draws[lv][1:], [2.5, 97.5])
        rows.append({"pair": f"{a}-{b}", col: lv,
                     **{f"fold{f}": p for f, p in zip(FOLDS, pts[lv])},
                     "mean": mean_draws[lv][0], "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def md_table(df, fmt="{:.4f}"):
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def summarise(df, arms, ref, out):
    hnm = [a for a in ("C", "D", "G", "G0") if a in arms]
    md = ["# Which oil do the HNM arms lose?", "",
          f"Components: GT oil, 8-connected, >= 20 px, in the GT-aware reconstruction coverage "
          f"({len(df)} over three folds). Recall at P(oil) >= {HALF}. lost = {ref} recall >= "
          f"{FOUND} and arm recall < {MISSED}; unrecoverable = lost and still < {MISSED} at "
          f"P(oil) >= {LOW}. Single seed; CIs from 2,000 paired scene-bootstrap draws.", ""]
    # 1. decomposition of the TP difference
    rows = []
    for a in [x for x in arms if x != ref]:
        cls, unrec = classify(df, a, ref)
        d = df[f"tp_{a}"] - df[f"tp_{ref}"]
        total = df.area.sum()
        row = {"arm": a, "recall_diff": d.sum() / total}
        for c in ("lost", "shrunk", "gained", "grown_or_same"):
            row[f"{c}_px"] = d[cls == c].sum() / total
        row["lost_n"] = int((cls == "lost").sum())
        row["unrecoverable_n"] = int(unrec.sum())
        row["unrecoverable_px"] = d[unrec].sum() / total
        rows.append(row)
        df[f"class_{a}"], df[f"unrec_{a}"] = cls, unrec
    dec = pd.DataFrame(rows)
    md += ["## 1. Recall difference vs " + ref + ", split by component outcome", "",
           "Each column is (arm TP - ref TP) over all GT oil px, summed over the components of "
           "that class; the classes add up to recall_diff.", "", md_table(dec), ""]
    # 2. recall by stratum
    # components C loses: do the kNN-filtered arms get them back? (G-C favours G by
    # construction, since C's own misses define the stratum; G-G0 does not)
    strata = ["size", "thick", "sign", "glint", "sensor", "coast"]
    if "C" in arms and "G" in arms:
        df["lost_by_C"] = np.where(df.class_C == "lost", "lost", "not_lost")
        strata.append("lost_by_C")
    for col in strata:
        share = df.groupby(col).area.sum() / df.area.sum()
        rec = pd.DataFrame({a: df.groupby(col)[f"tp_{a}"].sum() / df.groupby(col).area.sum()
                            for a in arms})
        rec.insert(0, "oil_share", share)
        rec.insert(1, "components", df.groupby(col).size())
        md += [f"## 2. By {col}", "", md_table(rec.reset_index()), ""]
        diffs = pd.concat([boot_by_stratum(df, col, a, b) for a, b in PAIRS
                           if a in arms and b in arms])
        diffs.to_csv(out / f"recall_diff_by_{col}.csv", index=False)
        diffs["ci95"] = [f"[{lo:+.4f}, {hi:+.4f}]" for lo, hi in zip(diffs.ci_lo, diffs.ci_hi)]
        md += [md_table(diffs.drop(columns=["ci_lo", "ci_hi"])), ""]
    # 3. lost components: attributes vs components both arms find, and events
    for a in hnm:
        lost = df[df[f"class_{a}"] == "lost"]
        kept = df[(df[f"tp_{ref}"] / df.area >= FOUND) & (df[f"tp_{a}"] / df.area >= FOUND)]
        cmp = pd.DataFrame({
            "group": [f"lost by {a}", f"found by {ref} and {a}"],
            "n": [len(lost), len(kept)],
            "median_area": [lost.area.median(), kept.area.median()],
            "median_thickness": [lost.thickness.median(), kept.thickness.median()],
            "median_contrast": [lost.contrast.median(), kept.contrast.median()],
            "median_nir_ring": [lost.nir_ring.median(), kept.nir_ring.median()],
            "share_bright": [(lost.sign == "bright").mean(), (kept.sign == "bright").mean()],
            "share_coast": [(lost.coast == "coast").mean(), (kept.coast == "coast").mean()],
            f"median_pmax_{a}": [lost[f"pmax_{a}"].median(), kept[f"pmax_{a}"].median()]})
        ev = (lost.assign(px=lost[f"tp_{ref}"] - lost[f"tp_{a}"]).groupby("event")
              .agg(components=("px", "size"), lost_px=("px", "sum"))
              .sort_values("lost_px", ascending=False))
        ev["share"] = ev.lost_px / ev.lost_px.sum()
        md += [f"## 3. Components lost by {a}", "", md_table(cmp, "{:.4g}"), "",
               f"Events by oil px lost ({a}): {len(ev)} events; top 8:", "",
               md_table(ev.head(8).reset_index(), "{:.3f}"), ""]
    # 4. exposure to same-place hard negatives
    for a in [x for x in hnm if f"exposure_{a}" in df]:
        col = f"exposure_{a}"
        rec = pd.DataFrame({x: df.groupby(col)[f"tp_{x}"].sum() / df.groupby(col).area.sum()
                            for x in arms})
        rec.insert(0, "oil_share", df.groupby(col).area.sum() / df.area.sum())
        rec.insert(1, "scenes", df.groupby(col).scene.nunique())
        diffs = pd.concat([boot_by_stratum(df, col, a, b) for b in (ref, "B") if b in arms])
        diffs.to_csv(out / f"recall_diff_by_{col}.csv", index=False)
        diffs["ci95"] = [f"[{lo:+.4f}, {hi:+.4f}]" for lo, hi in zip(diffs.ci_lo, diffs.ci_hi)]
        lost = df[f"class_{a}"] == "lost"
        share = (df[lost].groupby(col).apply(lambda g: (g[f"tp_{ref}"] - g[f"tp_{a}"]).sum())
                 / (df[lost][f"tp_{ref}"] - df[lost][f"tp_{a}"]).sum())
        md += [f"## 4. By exposure of {a} ({SELECTION[a]}): selected train tiles with probe FP "
               "at the same MGRS tile", "", md_table(rec.reset_index()), "",
               md_table(diffs.drop(columns=["ci_lo", "ci_hi"])), "",
               "Share of px in components lost by " + a + ": " +
               ", ".join(f"{k} {v:.1%}" for k, v in share.items()), ""]
    df.to_csv(out / "components.csv", index=False)
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'summary.md'}", flush=True)


def heat(p):
    return cv2.applyColorMap(p, cv2.COLORMAP_VIRIDIS)


def gallery(df, store, arms, ref, config, out, n=12):
    """True colour | P(oil) per arm (GT component outlined), top lost components by px."""
    index = raster_index(config["vrt_dir"])
    scale, offset = config["scale"], config["si"]["reflectance_offset"]
    hnm = [a for a in ("C", "D") if a in arms]
    lost = df[np.logical_or.reduce([df[f"class_{a}"] == "lost" for a in hnm])].copy()
    lost["loss"] = lost[f"tp_{ref}"] - lost[[f"tp_{a}" for a in hnm]].min(axis=1)
    show = [ref] + [a for a in ("B", "C", "D") if a in arms and a != ref]
    (out / "gallery").mkdir(exist_ok=True)
    tiles = []
    for rank, r in enumerate(lost.sort_values("loss", ascending=False).head(n).itertuples(), 1):
        zs = store[r.fold]
        keys = pd.DataFrame(dict(scene=zs[ref]["scene"], label=zs[ref]["label"]))
        i = int(np.flatnonzero((keys.scene == r.scene) & (keys.label == r.label))[0])
        bits = crop(zs[ref], i, "crop_mask")
        (x0, y0), (h, w) = zs[ref]["crop_origin"][i], bits.shape
        with rasterio.open(index[r.scene]) as src:
            raw = src.read(window=Window(int(x0), int(y0), w, h)).astype(np.float32)
        ctrs, _ = cv2.findContours((bits & 1).astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
        panels = [np.ascontiguousarray(true_colour(raw, scale, offset, .15, .6))]
        for a in show:
            p = heat(crop(zs[a], i, "crop_prob"))
            p[(bits & 4) == 0] = 40
            panels.append(p)
        for p in panels:
            cv2.drawContours(p, ctrs, -1, (255, 255, 255), 1)
        f = min(4, max(1, int(np.ceil(160 / max(1, min(h, w))))))
        row = np.hstack([np.pad(cv2.resize(p, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST),
                                ((0, 0), (0, 3), (0, 0)), constant_values=255) for p in panels])
        rec = " ".join(f"{a}={getattr(r, f'tp_{a}') / r.area:.2f}" for a in show)
        cap = (f"#{rank} fold {r.fold} {short_name(r.scene)} area {r.area} px "
               f"{r.sign} glint {r.nir_ring:.3f} {r.coast} | recall@0.5 {rec}")
        bar = np.full((22, row.shape[1], 3), 32, np.uint8)
        cv2.putText(bar, cap, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (240, 240, 240), 1,
                    cv2.LINE_AA)
        img = np.vstack([bar, row])
        cv2.imwrite(str(out / "gallery" / f"{rank:02d}.png"), img)
        tiles.append(img)
    head = "columns: true colour | " + " | ".join(f"P(oil) {a}" for a in show) + \
           " (white = GT component; grey = not scored)"
    if tiles:
        width = max(t.shape[1] for t in tiles)
        tiles = [np.pad(t, ((0, 6), (0, width - t.shape[1]), (0, 0)), constant_values=255)
                 for t in tiles]
        bar = np.full((24, width, 3), 255, np.uint8)
        cv2.putText(bar, head, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / "gallery.png"), np.vstack([bar] + tiles))
        print(f"Wrote {out / 'gallery.png'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "preprocess/hnm_si/config.example.yaml"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--ref", default="A")
    parser.add_argument("--gallery", type=int, default=12)
    parser.add_argument("--selection-dir", default=str(REPO_ROOT / "hnm_runs/hnm_si_v1/arms"))
    args = parser.parse_args(argv)
    arms = dict(a.split("=", 1) for a in args.arm)
    config = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, store = build(arms, args.ref, config, out)
    exposure(df, arms, args.selection_dir)
    summarise(df, arms, args.ref, out)
    gallery(df, store, arms, args.ref, config, out, args.gallery)


if __name__ == "__main__":
    main()
