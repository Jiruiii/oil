"""True-colour strip of an arm's selected train background tiles at given MGRS tiles.

For looking at what the hard negatives near a lost slick look like (tools/analyze_lost_oil.py
gallery): per (fold, MGRS tile), the selected tiles with the highest baseline-probe FP count.

  .venv/bin/python tools/show_hard_negative_tiles.py --arm-dir hnm_runs/hnm_si_v1/arms/hnm \\
      --site 2:34SGH --site 2:16RCT --site 3:15RVM --out hnm_runs/lost_oil/hard_negative_tiles.png
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
from analyze_baseline_fp import REPO_ROOT, raster_index, true_colour  # noqa: E402
from preprocess.hnm_si.config import load_config  # noqa: E402


def strip(arm_dir, fold, site, index, config, n):
    d = Path(arm_dir) / f"fold_{fold}"
    sel = pd.read_csv(d / "selected_train_bg.tsv", sep="\t").merge(
        pd.read_csv(d / "candidates_scored.tsv", sep="\t"), on=["scene", "x", "y"])
    sel = sel[sel.scene.str.endswith("_" + site)].sort_values("n_fp", ascending=False).head(n)
    size, scale = config["patch_size"], config["scale"]
    offset = config["si"]["reflectance_offset"]
    tiles = []
    for r in sel.itertuples():
        with rasterio.open(index[r.scene]) as src:
            raw = src.read(window=Window(r.x, r.y, size, size), boundless=True,
                           fill_value=0).astype(np.float32)
        t = np.ascontiguousarray(true_colour(raw, scale, offset, .15, .6))
        cv2.putText(t, f"{r.scene[-21:-13]} n_fp {int(r.n_fp)}", (3, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.pad(t, ((0, 0), (0, 3), (0, 0)), constant_values=255))
    row = np.hstack(tiles)
    bar = np.full((20, row.shape[1], 3), 32, np.uint8)
    cv2.putText(bar, f"fold {fold} {Path(arm_dir).name}: selected train background tiles at "
                f"{site}, top {len(tiles)} by baseline n_fp", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([bar, row])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO_ROOT / "preprocess/hnm_si/config.example.yaml"))
    parser.add_argument("--arm-dir", required=True)
    parser.add_argument("--site", action="append", required=True, metavar="FOLD:MGRS")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    index = raster_index(config["vrt_dir"])
    rows = [strip(args.arm_dir, int(s.split(":")[0]), s.split(":")[1], index, config, args.n)
            for s in args.site]
    w = max(r.shape[1] for r in rows)
    cv2.imwrite(args.out, np.vstack([np.pad(r, ((0, 6), (0, w - r.shape[1]), (0, 0)),
                                            constant_values=255) for r in rows]))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
