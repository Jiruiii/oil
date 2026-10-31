"""Are the oil slicks HNM loses close to its hard negatives in the baseline's feature space?

tools/analyze_lost_oil.py found that HNM arms drop faint, thin, weak-contrast slicks whole,
and that their hard-negative tiles look like such slicks. This tool checks that in the
baseline (A) model's own features, the space the hard negatives were mined in:
  pen  DeepLabHead penultimate activations (256-d, after ReLU; the classifier is linear in them)
  enc  Prithvi encoder output fed to the head (1024-d)
Both are 16 x 16 cells per 256 px tile. A region's vector is the pixel-weighted mean of the
cells it covers. Single forward pass (no TTA). Vectors per fold:
  oil      every GT oil component of the test scenes (same components as oil_component_prob)
  train    each train tile's GT oil pixels (what A learned as oil)
  hard_<a> each train background tile arm <a> selected that has probe FP: its FP pixels
           (A's P(oil) >= 0.5 on mask == 1), i.e. the look-alike A confused with oil
  random   each random-arm train background tile: its mask == 1 pixels
Per test component, similarity = mean cosine of the TOPK nearest vectors of a set (features
z-scored with the fold's pooled reference vectors). margin_<a> = sim(hard_<a>) - sim(train);
knn_hard_<a> = share of hard_<a> among the KNN nearest vectors of train oil + hard_<a>.
Then: do the components arm <a> lost (analyze_lost_oil class "lost") sit closer to hard_<a>
than the components both A and <a> find?  AUC with a scene bootstrap.

  .venv/bin/python tools/feature_space.py --out hnm_runs/feature_space \\
      --components hnm_runs/lost_oil/components.csv --run-dir 1=<A fold 1> ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from oil_component_prob import components  # noqa: E402
from threshold_sweep import REPO_ROOT, run_settings, tiles  # noqa: E402
from recon_gt_aware_module import (_apply_pixel_mapping, _find_vrt,  # noqa: E402
                                   _group_patches_by_scene)

FOLDS = (1, 2, 3)
TOPK = 5
KNN = 10
HARD = {"C": "hnm", "D": "hnm_si"}
BOOT = 1000


class Extractor:
    """Single forward; returns P(oil) (H, W) and per-cell features {pen, enc} (D, h, w)."""

    def __init__(self, model):
        import torch
        self.torch, self.model, self.store = torch, model, {}
        model.head.register_forward_pre_hook(lambda m, a: self.store.__setitem__("enc", a[0]))
        model.head[3].register_forward_hook(lambda m, a, o: self.store.__setitem__("pen", o))

    def __call__(self, tensor):
        with self.torch.no_grad():
            out = self.model(tensor)["out"]
        p = self.torch.softmax(out, dim=1)[0, 0].cpu().numpy()
        return p, {k: v[0].float().cpu().numpy() for k, v in self.store.items()}


def cell_weights(mask, h, w):
    """(h, w) pixel counts of a (S, S) boolean mask per feature cell."""
    s = mask.shape[0]
    return mask.reshape(h, s // h, w, s // w).sum(axis=(1, 3)).astype(np.float64)


def pooled(feats, weights):
    tot = weights.sum()
    return {k: (v * weights).sum(axis=(1, 2)) / tot for k, v in feats.items()}, tot


def window(arr, x, y, size, fill):
    out = np.full((size, size), fill, arr.dtype)
    part = arr[y:y + size, x:x + size]
    out[:part.shape[0], :part.shape[1]] = part
    return out


def test_oil(ext, s, device, keys):
    """Pixel-weighted mean features of every GT oil component of the fold's test scenes."""
    sums, wts = {}, {}
    for scene, coords in _group_patches_by_scene(s["test_txt"]).items():
        with rasterio.open(s["mask_dir"] / f"{scene}.tif") as m:
            gt = _apply_pixel_mapping(m.read(1), s["pixel_mapping"])
        H, W = gt.shape
        size = s["patch_size"]
        tested = np.zeros((H, W), bool)
        for x, y in coords:
            tested[y:y + size, x:x + size] = True
        lab, rows = components(gt, tested)
        want = {k for k, *_ in rows}
        for x, y, tensor in tiles(s, _find_vrt(s["vrt_dir"], scene), coords, device):
            lt = window(lab, x, y, size, 0)
            present = [k for k in np.unique(lt) if k in want]
            if not present:
                continue
            _, feats = ext(tensor)
            h, w = feats["pen"].shape[-2:]
            for k in present:
                cw = cell_weights(lt == k, h, w)
                for name, f in feats.items():
                    key = (scene, int(k), name)
                    sums[key] = sums.get(key, 0) + (f * cw).sum(axis=(1, 2))
                wts[(scene, int(k))] = wts.get((scene, int(k)), 0) + cw.sum()
    rows = keys[["scene", "label"]].itertuples(index=False)
    out = {name: [] for name in ("pen", "enc")}
    for scene, k in rows:
        if (scene, k) not in wts:
            raise ValueError(f"component {scene} #{k} not covered by any test tile")
        for name in out:
            out[name].append(sums[(scene, k, name)] / wts[(scene, k)])
    return {k: np.array(v) for k, v in out.items()}


def tile_vectors(ext, s, device, items, select):
    """items: (scene, x, y) rows; select(p, mask) -> pixel mask to pool, or None to skip."""
    out, meta = {"pen": [], "enc": []}, []
    size = s["patch_size"]
    for scene, g in items.groupby("scene"):
        coords = list(zip(g.x.astype(int), g.y.astype(int)))
        with rasterio.open(s["mask_dir"] / f"{scene}.tif") as m:
            mask = m.read(1)
        for x, y, tensor in tiles(s, _find_vrt(s["vrt_dir"], scene), coords, device):
            p, feats = ext(tensor)
            sel = select(p, window(mask, x, y, size, 255))
            if sel is None or not sel.any():
                continue
            v, n = pooled(feats, cell_weights(sel, *feats["pen"].shape[-2:]))
            for name in out:
                out[name].append(v[name])
            meta.append((scene, x, y, int(n)))
    return {k: np.array(v) for k, v in out.items()}, pd.DataFrame(meta, columns=["scene", "x", "y", "px"])


def patch_rows(txt):
    by = _group_patches_by_scene(txt)
    return pd.DataFrame([(s, x, y) for s, c in by.items() for x, y in c], columns=["scene", "x", "y"])


def extract(run_dir, fold, comps, sel_dir, device, out):
    import torch
    from preprocess.hnm_si.probe import load_probe
    s = run_settings(run_dir)
    if f"test_fold{fold}" not in s["test_txt"].name:
        raise ValueError(f"{run_dir} is not a fold {fold} run")
    ext = Extractor(load_probe(run_dir, device, amp=False).model)
    keys = comps[comps.fold == fold][["scene", "label"]].reset_index(drop=True)
    data = {"oil_" + k: v for k, v in test_oil(ext, s, device, keys).items()}
    train_txt = REPO_ROOT / str(s["test_txt"]).replace("test_fold", "train_fold")
    groups = {"train": (patch_rows(Path(train_txt)), lambda p, m: m == 0)}
    for a, arm in HARD.items():
        sel = pd.read_csv(Path(sel_dir) / arm / f"fold_{fold}" / "selected_train_bg.tsv", sep="\t")
        groups[f"hard_{a}"] = (sel, lambda p, m: (p >= 0.5) & (m == 1))
    rnd = pd.read_csv(Path(sel_dir) / "random" / f"fold_{fold}" / "selected_train_bg.tsv", sep="\t")
    groups["random"] = (rnd, lambda p, m: m == 1)
    for g, (items, select) in groups.items():
        vec, meta = tile_vectors(ext, s, device, items, select)
        print(f"fold {fold}: {g} {len(meta)} of {len(items)} tiles", flush=True)
        data.update({f"{g}_{k}": v for k, v in vec.items()})
        meta.to_csv(out / f"fold_{fold}_{g}.csv", index=False)
    np.savez_compressed(out / f"fold_{fold}.npz", **data)
    keys.to_csv(out / f"fold_{fold}_oil.csv", index=False)
    del ext
    torch.cuda.empty_cache()


def topk_sim(q, ref, k=TOPK):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    r = ref / np.linalg.norm(ref, axis=1, keepdims=True)
    s = q @ r.T
    return np.sort(s, axis=1)[:, -k:].mean(axis=1)


def knn_share(q, ref, other, k=KNN):
    """Share of ``ref`` rows among the k nearest (cosine) of ``ref`` + ``other``."""
    pool = np.vstack([ref, other])
    pool = pool / np.linalg.norm(pool, axis=1, keepdims=True)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    nn = np.argsort(q @ pool.T, axis=1)[:, -k:]
    return (nn < len(ref)).mean(axis=1)


def auc(score, y):
    """P(score of a positive > score of a negative), ties half."""
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().values[:len(pos)]
    return (r.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def scene_boot_auc(df, col, ycol, n=BOOT, seed=0):
    rng = np.random.default_rng(seed)
    scenes = df.scene.unique()
    by = {s: g for s, g in df.groupby("scene")}
    vals = []
    for _ in range(n):
        g = pd.concat([by[s] for s in rng.choice(scenes, len(scenes))])
        vals.append(auc(g[col].values, g[ycol].values))
    return np.nanpercentile(vals, [2.5, 97.5])


def analyze(comps, out):
    rows = []
    for f in FOLDS:
        z = np.load(out / f"fold_{f}.npz")
        keys = pd.read_csv(out / f"fold_{f}_oil.csv")
        for space in ("pen", "enc"):
            ref = np.vstack([z[f"{g}_{space}"] for g in ("train", "random", "hard_C", "hard_D")])
            mu, sd = ref.mean(axis=0), ref.std(axis=0) + 1e-6
            norm = {g: (z[f"{g}_{space}"] - mu) / sd
                    for g in ("oil", "train", "random", "hard_C", "hard_D")}
            sims = {g: topk_sim(norm["oil"], norm[g]) for g in ("train", "random", "hard_C", "hard_D")}
            sims.update({f"knn_{a}": knn_share(norm["oil"], norm[f"hard_{a}"], norm["train"])
                         for a in HARD})
            df = keys.assign(fold=f, space=space, **{f"sim_{g}": v for g, v in sims.items()})
            rows.append(df)
    sim = pd.concat(rows, ignore_index=True)
    df = sim.merge(comps, on=["fold", "scene", "label"], how="left", validate="many_to_one")
    md = ["# Lost oil vs hard negatives in the baseline (A) feature space", "",
          f"Similarity = mean cosine to the {TOPK} nearest vectors of a set (z-scored per fold). "
          "lost / kept from tools/analyze_lost_oil.py (kept = A and the arm both >= 0.5 recall). "
          f"AUC = P(a lost component scores higher than a kept one); 95% CI from {BOOT} scene "
          "bootstrap draws (folds pooled). margin = sim(hard negatives of the arm) - sim(train oil).", ""]
    res = []
    for space in ("pen", "enc"):
        d = df[df.space == space].copy()
        for a in HARD:
            d[f"margin_{a}"] = d[f"sim_hard_{a}"] - d.sim_train
            ra, rx = d.tp_A / d.area, d[f"tp_{a}"] / d.area
            sub = d[(d[f"class_{a}"] == "lost") | ((ra >= 0.5) & (rx >= 0.5))].copy()
            sub["y"] = sub[f"class_{a}"] == "lost"
            sub["neg_contrast_abs"] = -sub.contrast.abs()
            sub["neg_log_area"] = -np.log(sub.area)
            for col in (f"sim_hard_{a}", "sim_train", "sim_random", f"margin_{a}", f"sim_knn_{a}",
                        "neg_contrast_abs", "neg_log_area"):
                s = sub.dropna(subset=[col])
                lo, hi = scene_boot_auc(s, col, "y")
                res.append(dict(space=space, arm=a, score=col, n_lost=int(s.y.sum()),
                                n_kept=int((~s.y).sum()), auc=auc(s[col].values, s.y.values),
                                ci_lo=lo, ci_hi=hi, lost_median=s[s.y][col].median(),
                                kept_median=s[~s.y][col].median()))
    r = pd.DataFrame(res)
    r.to_csv(out / "auc.csv", index=False)
    lines = ["| space | arm | score | n lost / kept | AUC [95% CI] | median lost | median kept |",
             "|---|---|---|---|---|---|---|"]
    lines += [f"| {x.space} | {x.arm} | {x.score} | {x.n_lost} / {x.n_kept} | {x.auc:.3f} "
              f"[{x.ci_lo:.3f}, {x.ci_hi:.3f}] | {x.lost_median:.3f} | {x.kept_median:.3f} |"
              for x in r.itertuples()]
    md += lines + ["", "neg_contrast_abs, neg_log_area: attribute baselines (weak contrast, small size).", ""]
    df.to_csv(out / "similarities.csv", index=False)
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {out / 'summary.md'}", flush=True)
    for space in ("pen", "enc"):
        plot(df, out, space)


def plot(df, out, space):
    """PCA (fit on the reference sets) of one feature space: train oil, hard negatives, test oil."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, f in zip(axes, FOLDS):
        z = np.load(out / f"fold_{f}.npz")
        ref = np.vstack([z[f"{g}_{space}"] for g in ("train", "random", "hard_C")])
        mu, sd = ref.mean(axis=0), ref.std(axis=0) + 1e-6
        n = (ref - mu) / sd
        _, _, vt = np.linalg.svd(n - n.mean(axis=0), full_matrices=False)
        proj = lambda x: ((x - mu) / sd - n.mean(axis=0)) @ vt[:2].T  # noqa: E731
        for g, c, lab in (("random", "#bbbbbb", "random bg tiles"),
                          ("hard_C", "#e07b39", "hard negatives (C)"),
                          ("train", "#4c78a8", "train oil")):
            p = proj(z[f"{g}_{space}"])
            ax.scatter(p[:, 0], p[:, 1], s=4, c=c, alpha=.35, label=lab, linewidths=0)
        d = df[(df.fold == f) & (df.space == space)].reset_index(drop=True)
        keys = pd.read_csv(out / f"fold_{f}_oil.csv")
        d = keys.merge(d, on=["scene", "label"], how="left")
        p = proj(z[f"oil_{space}"])
        lost = (d.class_C == "lost").values
        kept = ((d.tp_A / d.area >= .5) & (d.tp_C / d.area >= .5)).values
        ax.scatter(p[kept, 0], p[kept, 1], s=14, facecolors="none", edgecolors="#2a7f3f",
                   linewidths=.8, label="test oil found by A and C")
        ax.scatter(p[lost, 0], p[lost, 1], s=30, c="#d62728", marker="x", linewidths=1.2,
                   label="test oil lost by C")
        ax.set_title(f"fold {f}")
        ax.set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[-1].legend(fontsize=8, markerscale=1.5)
    name = {"pen": "head penultimate", "enc": "Prithvi encoder"}[space]
    fig.suptitle(f"Baseline (A) {name} features, PCA on train oil + random bg + hard negatives")
    fig.tight_layout()
    fig.savefig(out / f"pca_{space}.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {out / f'pca_{space}.png'}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--components", required=True, help="analyze_lost_oil components.csv")
    parser.add_argument("--run-dir", action="append", default=[], metavar="FOLD=DIR",
                        help="baseline A runs; folds whose fold_<f>.npz exists are skipped")
    parser.add_argument("--selection-dir", default=str(REPO_ROOT / "hnm_runs/hnm_si_v1/arms"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    comps = pd.read_csv(args.components)
    for item in args.run_dir:
        fold, run_dir = item.split("=", 1)
        if not (out / f"fold_{int(fold)}.npz").exists():
            import torch
            extract(run_dir, int(fold), comps, args.selection_dir, torch.device(args.device), out)
    analyze(comps, out)


if __name__ == "__main__":
    main()
