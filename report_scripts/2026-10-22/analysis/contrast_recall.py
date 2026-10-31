"""Oil recall split by oil-water contrast sign and a sun-glint proxy, per background arm.

Idea from the two papers read on 10/21: Pisano et al. 2015 (oil looks brighter than water
under glint, darker outside it) and Sun et al. 2024 (bright vs dark contrast slicks).
Question: does the HNM recall loss sit in bright-contrast (glint) oil?

Every GT oil component (mask == 0) inside the scored coverage of the fold's test scenes:
  contrast  (median B8A of the slick - median B8A of a water ring 4..16 px around it)
            / ring median; bright >= +10 %, dark <= -10 %, else weak
  glint     ring median B8A reflectance (clear water ~0; sun glint / haze raise it)
Recall per stratum = sum TP / sum oil px, pooled; paired arm differences with a scene
bootstrap (2000 draws), three-fold mean resampling each fold independently.

Nothing is trained or inferred; reads the GT-aware recon artifacts each run already wrote.
  nice -n 19 .venv/bin/python \\
      report_scripts/2026-10-22/analysis/contrast_recall.py \\
      --arm A=hnm_runs/diagnostics --arm B=hnm_runs/diagnostics_step2/random ...
Each --arm points at a directory holding the run_info.json of an analyze_baseline_fp run.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

REPO = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO.parent / "reports" / "2026-10-22" / "analysis"   # outputs stay outside git
for p in (REPO, REPO / "main"):
    sys.path.insert(0, str(p))

from predmask_codec import decode_scene_coverage, decode_scene_final, load_scene_document  # noqa: E402
from preprocess.hnm.data import raster_index, split_manifest  # noqa: E402
from preprocess.hnm_si.config import load_config  # noqa: E402

B04, B8A = 3, 5            # eight-band VRT order B01 B02 B03 B04 B08 B8A B11 B12
RING_IN, RING_OUT, PAD = 4, 16, 24
BRIGHT, DARK = 0.10, -0.10
GLINT_BINS = [-np.inf, 0.01, 0.03, 0.06, np.inf]
GLINT_LABELS = ["<0.01", "0.01-0.03", "0.03-0.06", ">=0.06"]
MIN_PX = 20


def artifacts(run_dir):
    found = sorted(Path(run_dir).glob("post_test_*/gt_aware_recon/prediction_artifacts/*.pred.json"))
    return {p.name[:-len(".pred.json")]: p for p in found}


def components(scene, cov, mask, src, scale, offset):
    oil = (mask == 0) & cov
    n, lab, stats, _ = cv2.connectedComponentsWithStats(oil.astype(np.uint8), connectivity=8)
    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING_IN + 1,) * 2)
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * RING_OUT + 1,) * 2)
    rows = []
    for k in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[k])
        if area < MIN_PX:
            continue
        x0, y0 = max(0, x - PAD), max(0, y - PAD)
        x1, y1 = min(mask.shape[1], x + w + PAD), min(mask.shape[0], y + h + PAD)
        comp = (lab[y0:y1, x0:x1] == k).astype(np.uint8)
        raw = src.read([B04 + 1, B8A + 1], window=Window(x0, y0, x1 - x0, y1 - y0)).astype(np.float32)
        valid = (raw != 0).any(axis=0)
        refl = (raw - offset) / scale
        ring = (cv2.dilate(comp, k_out) > 0) & ~(cv2.dilate(comp, k_in) > 0)
        ring &= valid & (mask[y0:y1, x0:x1] != 0)
        inside = (comp > 0) & valid
        if ring.sum() < MIN_PX or inside.sum() < MIN_PX:
            rows.append({"k": k, "area": area, "nir_oil": np.nan, "nir_ring": np.nan,
                         "red_contrast": np.nan, "contrast": np.nan})
            continue
        nir_o, nir_r = np.median(refl[1][inside]), np.median(refl[1][ring])
        red_o, red_r = np.median(refl[0][inside]), np.median(refl[0][ring])
        rows.append({"k": k, "area": area, "nir_oil": nir_o, "nir_ring": nir_r,
                     "contrast": (nir_o - nir_r) / max(abs(nir_r), 0.005),
                     "red_contrast": (red_o - red_r) / max(abs(red_r), 0.005)})
    return lab, pd.DataFrame(rows)


def run(config, arms, out):
    index = raster_index(config["vrt_dir"])
    scale, offset = config["scale"], config["si"]["reflectance_offset"]
    recs = []
    for fold in config["outer_folds"]:
        _, scenes = split_manifest(config["split_dir"], fold)
        per_arm = {a: artifacts(d[fold]) for a, d in arms.items()}
        common = sorted(set.intersection(*(set(v) for v in per_arm.values())) & scenes["test"])
        print(f"fold {fold}: {len(common)} scenes in all arms", flush=True)
        for i, scene in enumerate(common):
            docs = {a: load_scene_document(per_arm[a][scene], verify_sidecar=False) for a in arms}
            first = next(iter(arms))
            final0 = decode_scene_final(docs[first])
            cov = decode_scene_coverage(docs[first], final0)
            with rasterio.open(Path(config["mask_dir"]) / f"{scene}.tif") as msrc:
                mask = msrc.read(1)
            with rasterio.open(index[scene]) as src:
                lab, comp = components(scene, cov, mask, src, scale, offset)
            if comp.empty:
                continue
            for a in arms:
                final = final0 if a == first else decode_scene_final(docs[a])
                cov_a = cov if a == first else decode_scene_coverage(docs[a], final)
                pred = (final == int(docs[a]["labels"]["target_class_index"])) & cov_a
                tp = np.bincount(lab[pred], minlength=lab.max() + 1)
                for r in comp.itertuples():
                    recs.append({"arm": a, "fold": fold, "scene": scene, "k": r.k, "area": r.area,
                                 "tp": int(tp[r.k]), "contrast": r.contrast,
                                 "red_contrast": r.red_contrast, "nir_ring": r.nir_ring})
            if (i + 1) % 20 == 0:
                print(f"  fold {fold}: {i + 1}/{len(common)}", flush=True)
    df = pd.DataFrame(recs)
    df["sign"] = np.select([df.contrast >= BRIGHT, df.contrast <= DARK], ["bright", "dark"], "weak")
    df.loc[df.contrast.isna(), "sign"] = "unknown"
    df["glint"] = pd.cut(df.nir_ring, GLINT_BINS, labels=GLINT_LABELS).astype(str)
    df.to_csv(out / "components.csv", index=False)
    return df


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |"
              for row in df.itertuples(index=False)]
    return "\n".join(lines)


def boot_diff(g, a, b, n=2000, seed=0):
    """Paired scene bootstrap of recall(a) - recall(b) for one fold and stratum."""
    s = g.pivot_table(index="scene", columns="arm", values=["tp", "area"], aggfunc="sum").fillna(0)
    tp_a, tp_b, area = s["tp"][a].values, s["tp"][b].values, s["area"][a].values
    if area.sum() == 0:
        return np.nan, np.full(n, np.nan)
    point = tp_a.sum() / area.sum() - tp_b.sum() / area.sum()
    idx = np.random.default_rng(seed).integers(0, len(area), (n, len(area)))
    ar = area[idx].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = (tp_a[idx].sum(1) - tp_b[idx].sum(1)) / ar
    return point, draws


def summarise(df, arms, pairs, out):
    md = ["# Oil recall by contrast sign and glint proxy\n",
          f"contrast = B8A (slick median - ring median) / ring median, ring {RING_IN}-{RING_OUT} px; "
          f"bright >= {BRIGHT:+.0%}, dark <= {DARK:+.0%}. glint = ring median B8A reflectance. "
          f"Components < {MIN_PX} px dropped. Recall pooled over pixels.\n"]
    for strat in ("sign", "glint"):
        base = df[df.arm == next(iter(arms))]
        share = base.groupby(strat).area.sum() / base.area.sum()
        md += [f"## Oil pixel share by {strat} (same for every arm)\n",
               "| " + strat + " | share | components |", "|---|---|---|"]
        md += [f"| {k} | {v:.1%} | {int((base[strat] == k).sum())} |" for k, v in share.items()]
        rec = df.groupby(["arm", strat, "fold"]).agg(tp=("tp", "sum"), area=("area", "sum"))
        rec["recall"] = rec.tp / rec.area
        tab = rec.recall.unstack("fold")
        tab["mean"] = tab.mean(axis=1)
        md += ["", f"## Recall by {strat}\n", md_table(tab.reset_index()), ""]
        rows = []
        for a, b in pairs:
            for k in sorted(df[strat].unique()):
                pts, draws = [], []
                for f in sorted(df.fold.unique()):
                    g = df[(df[strat] == k) & (df.fold == f) & df.arm.isin([a, b])]
                    p, d = boot_diff(g, a, b, seed=f)
                    pts.append(p)
                    draws.append(d)
                mean_d = np.nanmean(np.vstack(draws), axis=0)
                lo, hi = np.nanpercentile(mean_d, [2.5, 97.5])
                rows.append({"pair": f"{a}-{b}", strat: k,
                             **{f"fold{f}": round(p, 4) for f, p in zip(sorted(df.fold.unique()), pts)},
                             "mean": round(float(np.nanmean(pts)), 4),
                             "ci95": f"[{lo:+.4f}, {hi:+.4f}]"})
        pdf = pd.DataFrame(rows)
        pdf.to_csv(out / f"recall_diff_by_{strat}.csv", index=False)
        md += [f"## Paired recall difference by {strat}\n", md_table(pdf), ""]
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'summary.md'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "preprocess/hnm_si/config.example.yaml"))
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=DIAG_DIR")
    ap.add_argument("--pair", action="append", metavar="A-B", help="default: every arm minus the first")
    ap.add_argument("--out", default=str(OUT_ROOT / "contrast_recall"))
    ap.add_argument("--reuse", action="store_true", help="re-summarise an existing components.csv")
    args = ap.parse_args()
    config = load_config(args.config)
    arms = {}
    for item in args.arm:
        name, _, d = item.partition("=")
        info = json.loads((Path(d) / "run_info.json").read_text(encoding="utf-8"))
        arms[name] = {int(k): v for k, v in info["probe_run_dirs"].items()}
    names = list(arms)
    pairs = [tuple(p.split("-")) for p in args.pair] if args.pair else [(n, names[0]) for n in names[1:]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(out / "components.csv") if args.reuse else run(config, arms, out)
    df = df[df.arm.isin(names)]
    summarise(df, arms, pairs, out)


if __name__ == "__main__":
    main()
