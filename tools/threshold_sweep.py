"""P(oil) histograms for a threshold sweep: GT-aware reconstruction vs test background tiles.

Asks whether an arm separates oil from look-alikes better, or only predicts oil less often:
at the same recall on the reconstruction, does it raise fewer false alarms on the oil-free
test background tiles? The argmax used everywhere else is the threshold P(oil) >= 0.5; this
tool keeps P(oil) so any threshold can be scored without running the model again.

Inference is the reconstruction's own (main/recon_gt_aware_module.py): same tiles from the
fold's test_txt, same normalization, same identity/hflip/vflip TTA, same stitching order
(later tiles overwrite earlier ones), same pixel_mapping; the background tiles are the
fold's test-scene rows of dataset.bg_coord_txt, as in its bg_tiles option.

Per fold it writes <out>/fold_<f>.npz:
  edges         NBINS + 1 bin edges on [0, 1]; 0.5 is an edge, so bin >= 0.5 == argmax oil
  recon_scenes  scene names (S)
  recon_oil     (S, NBINS) scored GT-oil pixels per P(oil) bin
  recon_other   (S, NBINS) scored non-oil pixels (unlabeled counts as background)
  bg_scenes     scene of each background tile (T)
  bg_hist       (T, NBINS) pixels per P(oil) bin on each background tile

  .venv/bin/python tools/threshold_sweep.py --out hnm_runs/threshold_sweep/A \\
      --run-dir 1=<fold-1 run> --run-dir 2=<fold-2 run> --run-dir 3=<fold-3 run>
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "main")]
from preprocess.hnm_si.probe import load_probe, probe_settings, read_run  # noqa: E402
from recon_gt_aware_module import (MASK_IGNORE, _apply_pixel_mapping, _find_vrt,  # noqa: E402
                                   _group_patches_by_scene, _patch_iter, _tta_predict_prob)

NBINS = 400
EDGES = np.linspace(0.0, 1.0, NBINS + 1)


def bin_counts(prob):
    """Pixels per bin; bin k holds EDGES[k] <= P < EDGES[k + 1] (P == 1 in the last bin)."""
    idx = np.clip(np.searchsorted(EDGES, prob.ravel(), side="right") - 1, 0, NBINS - 1)
    return np.bincount(idx, minlength=NBINS)


def run_settings(run_dir):
    args, _ = read_run(run_dir)
    ds = args["dataset"]
    recon = next(t["reconstruction"] for t in args.get("post_tests") or []
                 if (t.get("reconstruction") or {}).get("mode") == "gt_aware")
    oil = int(ds.get("target_class_index", 0))
    if oil != 0:
        raise ValueError("Expected the oil class at index 0")
    return dict(
        test_txt=REPO_ROOT / ds["test_txt"], bg_txt=REPO_ROOT / ds["bg_coord_txt"],
        vrt_dir=Path(recon.get("vrt_dir", ds.get("vrt_dir"))),
        mask_dir=Path(recon.get("mask_dir", ds.get("labels_dir"))),
        patch_size=int(recon.get("patch_size", ds.get("patch_size") or 256)),
        in_channels=int(args["architecture_cfg"].get("in_channels", 8)),
        band_indices=ds.get("band_indices"), spectral_index=ds.get("spectral_index"),
        pixel_mapping=ds.get("pixel_mapping"), mean_std=probe_settings(args))


def tiles(s, vrt_path, coords, device):
    mean, std = s["mean_std"]
    yield from _patch_iter(vrt_path, coords, s["patch_size"], mean, std, device,
                           s["in_channels"], band_indices=s["band_indices"],
                           spectral_index=s["spectral_index"])


def recon_probs(model, s, device, progress):
    """(scene, mapped GT, P(oil), tested) per test scene, stitched as the reconstruction does."""
    items = _group_patches_by_scene(s["test_txt"]).items()
    if progress:
        from tqdm import tqdm
        items = tqdm(items, desc="recon")
    for scene, coords in items:
        vrt = _find_vrt(s["vrt_dir"], scene)
        mask_path = s["mask_dir"] / f"{scene}.tif"
        if vrt is None or not mask_path.exists():
            raise FileNotFoundError(f"Missing VRT or mask for {scene}")
        with rasterio.open(mask_path) as m:
            gt = _apply_pixel_mapping(m.read(1), s["pixel_mapping"])
        H, W = gt.shape
        prob = np.zeros((H, W), np.float32)
        tested = np.zeros((H, W), bool)
        for x, y, tensor in tiles(s, vrt, coords, device):
            _, p = _tta_predict_prob(model, tensor)
            x1, y1 = min(W, x + s["patch_size"]), min(H, y + s["patch_size"])
            prob[y:y1, x:x1] = p[0, :y1 - y, :x1 - x]
            tested[y:y1, x:x1] = True
        yield scene, gt, prob, tested


def recon_hist(model, s, device, progress):
    names, oil_h, other_h = [], [], []
    for scene, gt, prob, tested in recon_probs(model, s, device, progress):
        oil = gt == 0
        names.append(scene)
        oil_h.append(bin_counts(prob[tested & oil]))
        other_h.append(bin_counts(prob[tested & ~oil]))
    return names, np.array(oil_h), np.array(other_h)


def bg_hist(model, s, scenes, device, progress):
    by_scene = {}
    with open(s["bg_txt"], newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["scene"] in scenes:
                by_scene.setdefault(r["scene"], []).append((int(r["x"]), int(r["y"])))
    items = sorted(by_scene.items())
    if progress:
        from tqdm import tqdm
        items = tqdm(items, desc="bg tiles")
    names, hists = [], []
    for scene, coords in items:
        vrt = _find_vrt(s["vrt_dir"], scene)
        with rasterio.open(s["mask_dir"] / f"{scene}.tif") as m:
            for x, y, tensor in tiles(s, vrt, coords, device):
                gt = m.read(1, window=Window(x, y, s["patch_size"], s["patch_size"]),
                            boundless=True, fill_value=MASK_IGNORE)
                if (gt == 0).any():
                    raise ValueError(f"Background tile {scene} ({x}, {y}) contains GT oil")
                _, p = _tta_predict_prob(model, tensor)
                names.append(scene)
                hists.append(bin_counts(p[0]))
    return names, np.array(hists)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-dir", action="append", required=True, metavar="FOLD=DIR")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    import torch
    device = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for item in args.run_dir:
        fold, run_dir = item.split("=", 1)
        target = out / f"fold_{int(fold)}.npz"
        if target.exists():
            print(f"fold {fold}: {target} exists, skipped", flush=True)
            continue
        s = run_settings(run_dir)
        if f"test_fold{int(fold)}" not in s["test_txt"].name:
            raise ValueError(f"{run_dir} is not a fold {fold} run ({s['test_txt'].name})")
        model = load_probe(run_dir, device, amp=False).model
        names, oil_h, other_h = recon_hist(model, s, device, not args.quiet)
        bg_names, bg_h = bg_hist(model, s, set(names), device, not args.quiet)
        tmp = target.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, edges=EDGES, recon_scenes=np.array(names),
                            recon_oil=oil_h, recon_other=other_h,
                            bg_scenes=np.array(bg_names), bg_hist=bg_h,
                            run_dir=np.array(str(run_dir)))
        tmp.replace(target)
        half = NBINS // 2
        tp, fn = int(oil_h[:, half:].sum()), int(oil_h[:, :half].sum())
        fp = int(other_h[:, half:].sum())
        print(f"fold {fold}: {len(names)} scenes, {len(bg_names)} bg tiles; at P>=0.5 "
              f"TP {tp} FN {fn} FP {fp} bg_fp {int(bg_h[:, half:].sum())}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
