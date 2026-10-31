"""Step 2 of HNM_EXPERIMENT_PLAN.md: paired comparison of the background arms.

Reads the per-scene tables that tools/analyze_baseline_fp.py wrote for each arm (baseline
plus one --run-dir analysis per arm), so every arm is scored by exactly the same code:
  recon     GT-aware reconstruction pixels (pooled Oil IoU, precision, recall, FP split)
  bg_patch  the frozen test background tiles (FP the main metric never scores)
  combined  both together per scene: the background-inclusive Oil IoU
            TP / (TP + FN + FP_reconstruction + FP_background_tiles)
Boundary IoU (Cheng et al., CVPR 2021) comes from the reconstruction rows.

Differences are paired per fold (same test scenes) with a scene bootstrap: test scenes are
resampled with replacement, the same draw is applied to both arms, and the statistic is
recomputed. The three-fold mean resamples each fold independently.

  .venv/bin/python tools/compare_hnm_arms.py \\
      --arm A=hnm_runs/diagnostics --arm B=hnm_runs/diagnostics_step2/random \\
      --arm C=hnm_runs/diagnostics_step2/hnm --arm D=hnm_runs/diagnostics_step2/hnm_si
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

GT = ("confirmed_bg", "others", "unlabeled")
BOUNDARY = ("near_oil", "oil_attached")
PAIRS = (("C", "B"), ("D", "C"), ("D", "B"), ("B", "A"), ("C", "A"), ("D", "A"),
         ("E", "B"), ("F", "E"), ("F", "B"), ("E", "C"), ("F", "D"), ("E", "A"), ("F", "A"),
         ("R", "A"), ("R", "B"),      # R: baseline retrained with oil_ignore_ring
         ("G", "C"), ("G", "G0"), ("G", "B"), ("G0", "C"), ("G", "A"), ("G0", "A"))      # G/G0: kNN-filtered / swapped C


def load_arm(path):
    path = Path(path)
    tot = pd.read_csv(path / "scene_totals.csv")
    fpx = pd.read_csv(path / "fp_pixels.csv")
    tot["fp"] = sum(tot[f"fp_{g}"] for g in GT)
    tot["bg"] = sum(tot[f"gt_{g}"] for g in GT)
    # Recon rows are per coverage component and bg rows per tile; sum to scenes.
    keys = ["source", "fold", "scene", "sensor", "size_bin"]
    if "b_inter" not in tot:
        raise ValueError(f"{path} predates Boundary IoU; rerun tools/analyze_baseline_fp.py")
    scenes = tot.groupby(keys, as_index=False)[["tp", "fn", "fp", "bg", "fp_confirmed_bg",
                                                 "fp_unlabeled", "b_inter", "b_union"]].sum()
    tiles = tot[tot.source == "bg_patch"].groupby(["fold", "scene"]).agg(
        tiles=("fp", "size"), tiles_fp=("fp", lambda v: int((v > 0).sum())))
    scenes = scenes.merge(tiles.reset_index(), on=["fold", "scene"], how="left")
    fpx["boundary"] = fpx["category"].isin(BOUNDARY)
    split = fpx.pivot_table(index=["source", "fold", "scene"], columns="boundary",
                            values="n_px", aggfunc="sum", fill_value=0)
    split = split.rename(columns={True: "fp_boundary", False: "fp_other"}).reset_index()
    for c in ("fp_boundary", "fp_other"):
        if c not in split:
            split[c] = 0
    scenes = scenes.merge(split, on=["source", "fold", "scene"], how="left").fillna(
        {"fp_boundary": 0, "fp_other": 0})
    combined = scenes.groupby(["fold", "scene", "sensor", "size_bin"], as_index=False)[
        [c for c in SUM_COLS if c in scenes]].sum(min_count=1)
    scenes = pd.concat([scenes, combined.assign(source="combined")], ignore_index=True)
    info = json.loads((path / "run_info.json").read_text(encoding="utf-8"))
    return scenes, info


SUM_COLS = ("tp", "fn", "fp", "bg", "fp_boundary", "fp_other", "tiles", "tiles_fp",
            "b_inter", "b_union")


def totals(df, weights=None):
    """Column sums; with ``weights`` (n_boot, n_scenes) one sum per bootstrap draw."""
    out = {}
    for c in SUM_COLS:
        v = df[c].fillna(0).to_numpy(float)
        out[c] = v.sum() if weights is None else weights @ v
    return out


def recon_stats(t):
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"oil_iou": t["tp"] / (t["tp"] + t["fn"] + t["fp"]),
                "precision": t["tp"] / (t["tp"] + t["fp"]),
                "recall": t["tp"] / (t["tp"] + t["fn"]),
                "fp_per_M_bg": 1e6 * t["fp"] / t["bg"],
                "fp_boundary": t["fp_boundary"], "fp_other": t["fp_other"], "fn": t["fn"],
                "boundary_iou": t["b_inter"] / t["b_union"]}


def combined_stats(t):
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"oil_iou_with_bg": t["tp"] / (t["tp"] + t["fn"] + t["fp"]),
                "precision_with_bg": t["tp"] / (t["tp"] + t["fp"])}


def bg_stats(t):
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"bg_fp_per_M": 1e6 * t["fp"] / t["bg"],
                "bg_tiles_fp_pct": 100 * t["tiles_fp"] / t["tiles"]}


def run_meta(info):
    """Epochs and optimizer steps of each fold's run (early stopping makes them differ)."""
    out = {}
    for fold, run_dir in info["probe_run_dirs"].items():
        run_dir = Path(run_dir)
        epochs = sum(1 for _ in open(run_dir / "results.csv", encoding="utf-8")) - 1
        metrics = json.loads((run_dir / "run_metrics.json").read_text(encoding="utf-8"))
        out[int(fold)] = {"epochs": epochs, "steps": metrics.get("global_optimizer_step"),
                          "train_min": metrics.get("training_time_minutes"),
                          "run": run_dir.name}
    return out


def paired_bootstrap(a, b, stat, metric, folds, n_boot, seed):
    """Per fold and three-fold-mean differences (a - b) with 95% scene-bootstrap intervals."""
    rng = np.random.default_rng(seed)
    rows, boots = [], {}
    for f in folds:
        fa = a[a.fold == f].sort_values("scene")
        fb = b[b.fold == f].sort_values("scene")
        if list(fa["scene"]) != list(fb["scene"]):
            raise ValueError(f"Fold {f}: arms were scored on different scenes")
        n = len(fa)
        idx = rng.integers(0, n, (n_boot, n))
        w = np.zeros((n_boot, n))
        np.add.at(w, (np.arange(n_boot)[:, None], idx), 1)
        point = stat(totals(fa))[metric] - stat(totals(fb))[metric]
        draws = stat(totals(fa, w))[metric] - stat(totals(fb, w))[metric]
        boots[f] = draws
        rows.append({"fold": f, "diff": float(point), "ci_lo": np.nanquantile(draws, .025),
                     "ci_hi": np.nanquantile(draws, .975), "p_gt0": float((draws > 0).mean())})
    draws = np.mean([boots[f] for f in folds], axis=0)
    rows.append({"fold": "mean", "diff": float(np.mean([r["diff"] for r in rows])),
                 "ci_lo": np.nanquantile(draws, .025), "ci_hi": np.nanquantile(draws, .975),
                 "p_gt0": float((draws > 0).mean())})
    return rows


def md_table(df, fmt="{:.4f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(
            (f"{v:.0f}" if v.is_integer() and abs(v) >= 10 else fmt.format(v))
            if isinstance(v, float) else str(v) for v in row) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIAG_DIR")
    parser.add_argument("--out", default="hnm_runs/diagnostics_step2")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    arms, metas = {}, {}
    for item in args.arm:
        name, _, path = item.partition("=")
        arms[name], info = load_arm(path)
        metas[name] = run_meta(info)
    folds = sorted(next(iter(arms.values()))["fold"].unique())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Step 2 comparison\n",
          "Same scoring code for every arm (tools/analyze_baseline_fp.py). Single seed (42); "
          f"95% intervals from {args.n_boot} paired scene-bootstrap draws.\n"]

    # 1. per arm and fold
    rows = []
    for name, df in arms.items():
        for f in folds + ["mean"]:
            if f == "mean":
                per = [r for r in rows if r["arm"] == name]
                rows.append({"arm": name, "fold": "mean", **{k: float(np.mean([r[k] for r in per]))
                             for k in per[0] if k not in ("arm", "fold", "run")}})
                continue
            rec, bg = df[(df.source == "recon") & (df.fold == f)], df[(df.source == "bg_patch") & (df.fold == f)]
            comb = df[(df.source == "combined") & (df.fold == f)]
            m = metas[name][f]
            rows.append({"arm": name, "fold": f, "epochs": float(m["epochs"]),
                         "steps": float(m["steps"] or np.nan),
                         **{k: float(v) for k, v in recon_stats(totals(rec)).items()},
                         **{k: float(v) for k, v in bg_stats(totals(bg)).items()},
                         **{k: float(v) for k, v in combined_stats(totals(comb)).items()}})
    table = pd.DataFrame(rows)
    table.to_csv(out / "comparison_by_fold.csv", index=False)
    view = table[["arm", "fold", "epochs", "steps", "oil_iou", "oil_iou_with_bg", "boundary_iou",
                  "precision", "recall", "fp_per_M_bg", "fp_boundary", "fp_other", "fn",
                  "bg_fp_per_M", "bg_tiles_fp_pct"]]
    md += ["## 1. Per arm and fold\n",
           "recon: oil_iou / precision / recall / fp_per_M_bg over GT-aware reconstruction pixels; "
           "fp_boundary = near_oil + oil_attached FP px, fp_other = the rest. "
           "bg_*: FP on the test background tiles. oil_iou_with_bg adds those FP to the oil union. "
           "boundary_iou: Boundary IoU over the reconstruction pixels.\n",
           md_table(view.assign(**{c: view[c].round(0) for c in
                                   ("epochs", "steps", "fp_per_M_bg", "fp_boundary", "fp_other",
                                    "fn", "bg_fp_per_M")})), ""]

    # 2. paired differences
    diff_rows = []
    for metric, source, stat in (("oil_iou", "recon", recon_stats),
                                 ("oil_iou_with_bg", "combined", combined_stats),
                                 ("boundary_iou", "recon", recon_stats),
                                 ("precision", "recon", recon_stats),
                                 ("recall", "recon", recon_stats),
                                 ("fp_per_M_bg", "recon", recon_stats),
                                 ("bg_fp_per_M", "bg_patch", bg_stats)):
        for x, y in PAIRS:
            if x not in arms or y not in arms:
                continue
            a = arms[x][arms[x].source == source]
            b = arms[y][arms[y].source == source]
            for r in paired_bootstrap(a, b, stat, metric, folds, args.n_boot, args.seed):
                diff_rows.append({"metric": metric, "pair": f"{x}-{y}", **r})
    diffs = pd.DataFrame(diff_rows)
    diffs.to_csv(out / "paired_differences.csv", index=False)
    for metric in diffs["metric"].unique():
        d = diffs[diffs.metric == metric].copy()
        d["cell"] = d.apply(lambda r: f"{r['diff']:+.4g} [{r['ci_lo']:+.4g}, {r['ci_hi']:+.4g}]", axis=1)
        piv = d.pivot(index="pair", columns="fold", values="cell")
        piv = piv[[c for c in folds if c in piv] + ["mean"]].reset_index()
        piv.columns = [str(c) for c in piv.columns]
        md += [f"## 2. Paired difference: {metric}\n", "diff [95% CI]; mean = three-fold mean.\n",
               md_table(piv), ""]

    # 3. strata (pooled over folds)
    for key in ("sensor", "size_bin"):
        srows = []
        for name, df in arms.items():
            rec, bg = df[df.source == "recon"], df[df.source == "bg_patch"]
            for v in sorted(set(rec[key]) | set(bg[key])):
                r = recon_stats(totals(rec[rec[key] == v]))
                srows.append({"arm": name, key: v,
                              **{k: float(r[k]) for k in ("oil_iou", "precision", "recall")},
                              "fp_per_M_bg": round(float(r["fp_per_M_bg"])),
                              "bg_fp_per_M": round(float(bg_stats(totals(bg[bg[key] == v]))["bg_fp_per_M"]))})
        st = pd.DataFrame(srows).sort_values([key, "arm"])
        st.to_csv(out / f"strata_{key}.csv", index=False)
        md += [f"## 3. By {key} (recon pooled over folds; bg_fp_per_M from test background tiles)\n",
               md_table(st), ""]
    (out / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'comparison.md'}")


if __name__ == "__main__":
    main()
