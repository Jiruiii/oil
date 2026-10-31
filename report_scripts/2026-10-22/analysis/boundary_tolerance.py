"""Oil IoU with a two-sided boundary tolerance band, and where the missed oil sits.

Per fold test scene, inside each run's scored coverage (GT oil = mask 0, the rest background,
exactly as the pooled Oil IoU):
  band   pixels within TOL px of the GT oil boundary on either side (oil within TOL of
         non-oil, or non-oil within TOL of oil). IoU_tol drops them from TP, FN and FP.
  FN split per GT oil component (8-connected):
    whole  every FN pixel of a component whose own recall < WHOLE_RECALL (slick missed)
    edge   FN within TOL px inside the boundary of a detected component
    core   the remaining FN of detected components (interior holes)
  Recall = 1 - (whole + edge + core) / oil, so the three shares add up to the recall loss.
Paired differences use the scene bootstrap of compare_hnm_arms.py (2000 draws, folds
resampled independently, three-fold mean).

  nice -n 10 .venv/bin/python -W ignore \\
      report_scripts/2026-10-22/analysis/boundary_tolerance.py --arm A=hnm_runs/diagnostics ...
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio

REPO = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO.parent / "reports" / "2026-10-22" / "analysis"   # outputs stay outside git
for p in (REPO, REPO / "main"):
    sys.path.insert(0, str(p))

from predmask_codec import decode_scene_coverage, decode_scene_final, load_scene_document  # noqa: E402
from preprocess.hnm.data import split_manifest  # noqa: E402
from preprocess.hnm_si.config import load_config  # noqa: E402

TOL = 8
WHOLE_RECALL = 0.10
SUMS = ["oil", "tp", "fn", "fp", "tp_t", "fn_t", "fp_t", "fn_whole", "fn_edge", "fn_core", "band_px"]


def artifacts(run_dir):
    found = sorted(Path(run_dir).glob("post_test_*/gt_aware_recon/prediction_artifacts/*.pred.json"))
    return {p.name[:-len(".pred.json")]: p for p in found}


def scene_rows(mask, docs, arms):
    oil_all = mask == 0
    d_out = cv2.distanceTransform((~oil_all).astype(np.uint8), cv2.DIST_L2, 5)  # non-oil -> oil
    d_in = cv2.distanceTransform(oil_all.astype(np.uint8), cv2.DIST_L2, 5)      # oil -> non-oil
    band = np.where(oil_all, d_in <= TOL, d_out <= TOL)
    del d_out
    edge = oil_all & (d_in <= TOL)
    del d_in
    rows, comp_cache = [], {}
    for a in arms:
        doc = docs[a]
        final = decode_scene_final(doc)
        cov = decode_scene_coverage(doc, final)
        pred = (final == int(doc["labels"]["target_class_index"])) & cov
        del final
        oil = oil_all & cov
        key = hash(np.packbits(cov).tobytes())
        if key not in comp_cache:                      # coverage is the same for every arm
            comp_cache[key] = cv2.connectedComponents(oil.astype(np.uint8), connectivity=8)
        n, lab = comp_cache[key]
        tp_px, fn_px = pred & oil, ~pred & oil
        fp_px = pred & cov & ~oil_all
        area = np.bincount(lab[oil], minlength=n)
        tp_c = np.bincount(lab[tp_px], minlength=n)
        with np.errstate(invalid="ignore", divide="ignore"):
            whole_c = (tp_c / area) < WHOLE_RECALL
        whole_c[0] = False
        fn_whole = fn_px & whole_c[lab]
        fn_rest = fn_px & ~fn_whole
        keep = cov & ~band
        rows.append({"arm": a, "oil": int(oil.sum()), "tp": int(tp_px.sum()), "fn": int(fn_px.sum()),
                     "fp": int(fp_px.sum()), "tp_t": int((tp_px & keep).sum()),
                     "fn_t": int((fn_px & keep).sum()), "fp_t": int((fp_px & keep).sum()),
                     "fn_whole": int(fn_whole.sum()), "fn_edge": int((fn_rest & edge).sum()),
                     "fn_core": int((fn_rest & ~edge).sum()), "band_px": int((band & cov).sum()),
                     "n_comp": int(n - 1), "n_whole": int(whole_c[1:].sum())})
    return rows


def run(config, arms, out):
    recs = []
    for fold in config["outer_folds"]:
        _, scenes = split_manifest(config["split_dir"], fold)
        per_arm = {a: artifacts(d[fold]) for a, d in arms.items()}
        common = sorted(set.intersection(*(set(v) for v in per_arm.values())) & scenes["test"])
        print(f"fold {fold}: {len(common)} scenes in all arms", flush=True)
        for i, scene in enumerate(common):
            docs = {a: load_scene_document(per_arm[a][scene], verify_sidecar=False) for a in arms}
            with rasterio.open(Path(config["mask_dir"]) / f"{scene}.tif") as msrc:
                mask = msrc.read(1)
            for r in scene_rows(mask, docs, arms):
                recs.append({"fold": fold, "scene": scene, **r})
            if (i + 1) % 20 == 0:
                print(f"  fold {fold}: {i + 1}/{len(common)}", flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(out / "scenes.csv", index=False)
    return df


def stats(s):
    """Statistics from summed counts (works on Series or on arrays of bootstrap sums)."""
    return {"iou": s["tp"] / (s["tp"] + s["fn"] + s["fp"]),
            "iou_tol": s["tp_t"] / (s["tp_t"] + s["fn_t"] + s["fp_t"]),
            "recall": s["tp"] / s["oil"],
            "prec_tol": s["tp_t"] / (s["tp_t"] + s["fp_t"]),
            "recall_tol": s["tp_t"] / (s["tp_t"] + s["fn_t"]),
            "fn_whole": s["fn_whole"] / s["oil"], "fn_edge": s["fn_edge"] / s["oil"],
            "fn_core": s["fn_core"] / s["oil"]}


def boot(df, a, b, n=2000, seed=42):
    """Three-fold mean of stat(a) - stat(b): point and 95% CI per statistic."""
    pts, draws = [], []
    rng = np.random.default_rng(seed)
    for f in sorted(df.fold.unique()):
        g = df[df.fold == f]
        sa = g[g.arm == a].set_index("scene")[SUMS]
        sb = g[g.arm == b].set_index("scene")[SUMS].loc[sa.index]
        pa, pb = stats(sa.sum()), stats(sb.sum())
        pts.append({k: pa[k] - pb[k] for k in pa})
        idx = rng.integers(0, len(sa), (n, len(sa)))
        A = {c: sa[c].to_numpy()[idx].sum(1) for c in SUMS}
        B = {c: sb[c].to_numpy()[idx].sum(1) for c in SUMS}
        da, db = stats(A), stats(B)
        draws.append({k: da[k] - db[k] for k in da})
    out = {}
    for k in pts[0]:
        mean_draw = np.mean([d[k] for d in draws], axis=0)
        lo, hi = np.percentile(mean_draw, [2.5, 97.5])
        out[k] = (float(np.mean([p[k] for p in pts])), float(lo), float(hi))
    return out


def summarise(df, names, ref, out):
    per = []
    for (a, f), g in df.groupby(["arm", "fold"]):
        s = stats(g[SUMS].sum())
        per.append({"arm": a, "fold": f, **{k: float(v) for k, v in s.items()},
                    "band_share": g.band_px.sum() / (g.tp + g.fn + g.fp).sum(),
                    "n_comp": int(g.n_comp.sum()), "n_whole": int(g.n_whole.sum())})
    per = pd.DataFrame(per)
    mean = per.groupby("arm").mean(numeric_only=True).reindex(names)
    per.to_csv(out / "per_fold.csv", index=False)
    diffs = []
    for a in names:
        if a == ref:
            continue
        for k, (p, lo, hi) in boot(df, a, ref).items():
            diffs.append({"pair": f"{a}-{ref}", "stat": k, "diff": p, "lo": lo, "hi": hi,
                          "sig": not (lo <= 0 <= hi)})
    diffs = pd.DataFrame(diffs)
    diffs.to_csv(out / "paired_vs_ref.csv", index=False)
    cols = ["iou", "iou_tol", "prec_tol", "recall_tol", "recall", "fn_whole", "fn_edge", "fn_core"]
    md = [f"# Boundary tolerance +-{TOL} px and missed-oil split\n",
          f"Whole-slick miss = GT oil component with recall < {WHOLE_RECALL:.0%}. "
          "Shares of oil px; recall = 1 - whole - edge - core.\n",
          "## Three-fold mean\n", "| arm | " + " | ".join(cols) + " | slicks | whole-missed |",
          "|" + "---|" * (len(cols) + 3)]
    for a, r in mean.iterrows():
        md.append(f"| {a} | " + " | ".join(f"{r[c]:.4f}" for c in cols) +
                  f" | {r.n_comp:.0f} | {r.n_whole:.0f} |")
    md += ["", f"## Paired difference vs {ref} (three-fold mean [95% CI])\n",
           "| pair | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for pair, g in diffs.groupby("pair", sort=False):
        g = g.set_index("stat")
        md.append(f"| {pair} | " + " | ".join(
            ("**" if g.loc[c, "sig"] else "") + f"{g.loc[c, 'diff']:+.4f} [{g.loc[c, 'lo']:+.4f}, "
            f"{g.loc[c, 'hi']:+.4f}]" + ("**" if g.loc[c, "sig"] else "") for c in cols) + " |")
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "preprocess/hnm_si/config.example.yaml"))
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=DIAG_DIR")
    ap.add_argument("--ref", default="B")
    ap.add_argument("--out", default=str(OUT_ROOT / "boundary_tolerance"))
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    config = load_config(args.config)
    arms = {}
    for item in args.arm:
        name, _, d = item.partition("=")
        info = json.loads((Path(d) / "run_info.json").read_text(encoding="utf-8"))
        arms[name] = {int(k): v for k, v in info["probe_run_dirs"].items()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(out / "scenes.csv") if args.reuse else run(config, arms, out)
    summarise(df, list(arms), args.ref, out)


if __name__ == "__main__":
    main()
