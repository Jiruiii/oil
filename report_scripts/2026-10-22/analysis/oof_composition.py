"""Land / cloud / haze-glint content of the E (oof) and F (oof_si) train background sets,
with the same coarse spectral rules as tools/analyze_baseline_fp.py, plus overlap with C/D.

  nice -n 19 .venv/bin/python \\
      report_scripts/2026-10-22/analysis/oof_composition.py
"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO.parent / "reports" / "2026-10-22" / "analysis"   # outputs stay outside git
for p in (REPO, REPO / "main", REPO / "tools"):
    sys.path.insert(0, str(p))

from analyze_baseline_fp import RULES, background_composition  # noqa: E402
from preprocess.hnm.data import raster_index, read_background  # noqa: E402
from preprocess.hnm_si.config import load_config  # noqa: E402

OUT = OUT_ROOT / "hnm_vs_literature"
NAMES = {"baseline": "A", "random": "B", "hnm": "C", "hnm_si": "D", "oof": "E", "oof_si": "F"}


def main():
    config = load_config(REPO / "preprocess/hnm_si/config.example.yaml")
    index = raster_index(config["vrt_dir"])
    arms = Path(config["output_dir"]) / "arms"
    OUT.mkdir(parents=True, exist_ok=True)
    comp, overlap = [], []
    for fold in config["outer_folds"]:
        sets = {a: read_background(arms / a / f"fold_{fold}" / "selected_train_bg.tsv")
                for a in ("oof", "oof_si")}
        comp += background_composition(config, fold, sets, index, progress=False)
        keys = {a: {(r.scene, r.x, r.y) for r in read_background(arms / a / f"fold_{fold}" /
                                                                 "selected_train_bg.tsv")}
                for a in ("random", "hnm", "hnm_si", "oof", "oof_si")}
        for a in ("oof", "oof_si"):
            for b in ("random", "hnm", "hnm_si", "oof", "oof_si"):
                if a != b:
                    overlap.append({"fold": fold, "set": a, "vs": b, "n": len(keys[a]),
                                    "shared": len(keys[a] & keys[b])})
        print(f"fold {fold} done", flush=True)
    old = pd.read_csv(REPO / "hnm_runs/diagnostics/background_composition.csv", encoding="utf-8-sig")
    df = pd.concat([old, pd.DataFrame(comp)], ignore_index=True)
    df.to_csv(OUT / "background_composition_all.csv", index=False, encoding="utf-8-sig")
    df["has_land"] = df.land_frac >= RULES["tile_land_frac"]
    df["has_cloud"] = df.cloud_frac >= RULES["tile_cloud_frac"]
    df["mostly_haze"] = df.haze_glint_frac >= RULES["tile_haze_frac"]
    df["unlab"] = df.unlabeled_frac > 0
    t = df.groupby("set").agg(tiles=("x", "size"), has_land=("has_land", "mean"),
                              has_cloud=("has_cloud", "mean"), mostly_haze=("mostly_haze", "mean"),
                              unlabeled=("unlabeled_frac", "mean"))
    t.index = [f"{NAMES.get(i, i)} {i}" for i in t.index]
    t = t.sort_index()
    ov = pd.DataFrame(overlap).groupby(["set", "vs"])[["n", "shared"]].sum()
    ov["shared_pct"] = 100 * ov.shared / ov.n
    lines = ["# Train background sets: coarse spectral content (三折合計)\n",
             "has_land = land >= 1% of tile; has_cloud = cloud >= 5%; mostly_haze = haze/glint >= 50%; "
             "unlabeled = mean share of unlabeled (255) pixels.\n",
             "| set | tiles | has_land | has_cloud | mostly_haze | unlabeled |", "|---|---|---|---|---|---|"]
    lines += [f"| {i} | {r.tiles} | {r.has_land:.1%} | {r.has_cloud:.1%} | {r.mostly_haze:.1%} | "
              f"{r.unlabeled:.1%} |" for i, r in t.iterrows()]
    lines += ["", "## Overlap of selected tiles (same scene, x, y)\n", "| set | vs | n | shared | % |",
              "|---|---|---|---|---|"]
    lines += [f"| {a} | {b} | {r.n} | {r.shared} | {r.shared_pct:.1f}% |" for (a, b), r in ov.iterrows()]
    (OUT / "composition.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
