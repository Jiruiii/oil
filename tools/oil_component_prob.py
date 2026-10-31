"""P(oil) of every GT oil component in the GT-aware reconstruction, for one arm.

The threshold sweep (tools/threshold_sweep.py) showed HNM arms lose some oil that no
threshold recovers. This tool keeps P(oil) per GT oil component so tools/analyze_lost_oil.py
can ask which oil that is. Inference is the sweep's own (recon_probs: the reconstruction's
tiles, TTA, stitching order and pixel_mapping).

Components: 8-connected GT oil (mapped mask == 0) inside the tested coverage, >= MIN_PX px.
They depend only on the GT and the fold's test tiles, so every arm gets the same list.
Per fold it writes <out>/fold_<f>.npz:
  scene, label, area, bbox (K, 4 = x, y, w, h)   component identity
  hist          (K, NBINS) component pixels per P(oil) bin (threshold_sweep.EDGES)
  crop_origin   (K, 2 = x0, y0), crop_shape (K, 2 = h, w), crop_offset (K + 1,)
  crop_prob     uint8 round(255 * P(oil)) of each bbox padded by CROP_PAD, flattened
  crop_mask     uint8 on the same crops: bit 0 this component, bit 1 any GT oil, bit 2 tested

  .venv/bin/python tools/oil_component_prob.py --out hnm_runs/lost_oil/A \\
      --run-dir 1=<fold-1 run> --run-dir 2=<fold-2 run> --run-dir 3=<fold-3 run>
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from threshold_sweep import EDGES, NBINS, bin_counts, recon_probs, run_settings  # noqa: E402

MIN_PX = 20        # as reports/2026-10-22/analysis/contrast_recall.py
CROP_PAD = 24


def components(gt, tested):
    """Label image and (label, area, x, y, w, h) rows of GT oil components >= MIN_PX."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(((gt == 0) & tested).astype(np.uint8),
                                                        connectivity=8)
    rows = [(k, int(stats[k, 4]), *(int(v) for v in stats[k, :4]))
            for k in range(1, n) if stats[k, 4] >= MIN_PX]
    return lab, rows


def scene_records(scene, gt, prob, tested):
    lab, rows = components(gt, tested)
    H, W = gt.shape
    for k, area, x, y, w, h in rows:
        x0, y0 = max(0, x - CROP_PAD), max(0, y - CROP_PAD)
        x1, y1 = min(W, x + w + CROP_PAD), min(H, y + h + CROP_PAD)
        sel = lab[y:y + h, x:x + w] == k
        t = tested[y0:y1, x0:x1]
        crop = np.where(t, prob[y0:y1, x0:x1], 0)
        bits = ((lab[y0:y1, x0:x1] == k) | 2 * (gt[y0:y1, x0:x1] == 0) | 4 * t).astype(np.uint8)
        yield dict(scene=scene, label=k, area=area, bbox=(x, y, w, h),
                   hist=bin_counts(prob[y:y + h, x:x + w][sel]), origin=(x0, y0),
                   crop=np.round(255 * crop).astype(np.uint8), bits=bits)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--run-dir", action="append", required=True, metavar="FOLD=DIR")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    import torch
    from preprocess.hnm_si.probe import load_probe
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
        recs = [r for scene, gt, prob, tested in recon_probs(model, s, device, not args.quiet)
                for r in scene_records(scene, gt, prob, tested)]
        crops = [r["crop"] for r in recs]
        tmp = target.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp, edges=EDGES, scene=np.array([r["scene"] for r in recs]),
            label=np.array([r["label"] for r in recs]), area=np.array([r["area"] for r in recs]),
            bbox=np.array([r["bbox"] for r in recs]), hist=np.array([r["hist"] for r in recs]),
            crop_origin=np.array([r["origin"] for r in recs]),
            crop_shape=np.array([c.shape for c in crops]),
            crop_offset=np.concatenate([[0], np.cumsum([c.size for c in crops])]),
            crop_prob=np.concatenate([c.ravel() for c in crops]),
            crop_mask=np.concatenate([r["bits"].ravel() for r in recs]), run_dir=np.array(str(run_dir)))
        tmp.replace(target)
        hist = np.array([r["hist"] for r in recs])
        print(f"fold {fold}: {len(recs)} components, {int(hist.sum())} oil px, "
              f"recall at P>=0.5 {hist[:, NBINS // 2:].sum() / hist.sum():.4f}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
