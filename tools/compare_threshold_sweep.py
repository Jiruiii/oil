"""Compare arms at matched operating points from tools/threshold_sweep.py histograms.

The question: does an arm separate oil from look-alikes better, or only predict oil less
often? Lowering the P(oil) threshold raises recall and false alarms together, so arms are
compared at the same recall on the GT-aware reconstruction (threshold chosen per arm):
  bg_fp_per_M   false-positive pixels per million on the oil-free test background tiles
  bg_tile_rate  share of background tiles with at least one oil pixel (a look-alike alarm)
  precision     on the reconstruction pixels (boundary errors dominate here)
and, the other way round, recall at the same background-tile alarm rate.

The target is the reference arm's value at P(oil) >= 0.5 (its argmax) in each fold. Every
statistic is linearly interpolated between the two thresholds that bracket the target.
Intervals: the paired scene bootstrap of tools/compare_hnm_arms.py -- each fold's test
scenes are resampled with replacement, the same draw is applied to every arm, and the
threshold is re-chosen inside each draw; the three-fold mean resamples folds independently.

  .venv/bin/python tools/compare_threshold_sweep.py --out hnm_runs/threshold_sweep \\
      --arm A=hnm_runs/threshold_sweep/A --arm B=hnm_runs/threshold_sweep/B ... --ref A
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FOLDS = (1, 2, 3)
PAIRS = (("B", "A"), ("C", "A"), ("D", "A"), ("C", "B"), ("D", "B"), ("D", "C"),
         ("E", "B"), ("F", "B"), ("F", "E"), ("R", "A"),
         ("G", "C"), ("G", "G0"), ("G", "B"), ("G0", "C"), ("G", "A"), ("G0", "A"))


def tail(h):
    """Counts at or above each bin: column k == pixels with P(oil) >= EDGES[k]."""
    return np.cumsum(h[..., ::-1], axis=-1)[..., ::-1]


def load_fold(path, scenes=None):
    """Per-scene tail matrices over the union of reconstruction and background scenes."""
    z = np.load(path)
    rs, bs = list(z["recon_scenes"]), list(z["bg_scenes"])
    if scenes is None:
        scenes = sorted(set(rs) | set(bs))
    pos = {s: i for i, s in enumerate(scenes)}
    if set(rs) | set(bs) != set(scenes):
        raise ValueError(f"{path}: test scenes differ from the reference arm")
    nb = z["recon_oil"].shape[1]
    oil, other = np.zeros((len(scenes), nb)), np.zeros((len(scenes), nb))
    bg_px, bg_tiles = np.zeros((len(scenes), nb)), np.zeros((len(scenes), nb))
    idx = [pos[s] for s in rs]
    oil[idx], other[idx] = z["recon_oil"], z["recon_other"]
    # A tile raises an alarm at threshold k when its highest occupied bin is >= k.
    top = nb - 1 - np.argmax(z["bg_hist"][:, ::-1] > 0, axis=1)
    for s, h, t in zip(bs, z["bg_hist"], top):
        bg_px[pos[s]] += h
        bg_tiles[pos[s], t] += 1
    return scenes, dict(oil=tail(oil), other=tail(other), bg=tail(bg_px), tiles=tail(bg_tiles),
                        edges=z["edges"])


def curves(m, w):
    """Statistics per threshold bin for scene weights w (D, S) -> each (D, NBINS)."""
    tp, fp = w @ m["oil"], w @ m["other"]
    gt = (w @ m["oil"][:, :1])
    bg_fp, bg_px = w @ m["bg"], w @ m["bg"][:, :1]
    alarms, n_tiles = w @ m["tiles"], w @ m["tiles"][:, :1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(recall=tp / gt, precision=tp / (tp + fp), oil_iou=tp / (gt + fp),
                    bg_fp_per_M=1e6 * bg_fp / bg_px, bg_tile_rate=alarms / n_tiles)


def at_target(c, key, target, read):
    """``read`` where the non-increasing ``key`` crosses ``target`` (one value per row).

    NaN when only the lowest bin (P(oil) >= 0, every pixel called oil) reaches the target:
    the arm cannot reach that operating point at any real threshold.
    """
    x, y = c[key], c[read]
    n = (x >= target[:, None]).sum(axis=1)
    k = np.clip(n - 1, 0, x.shape[1] - 2)
    r = np.arange(len(k))
    x0, x1, y0, y1 = x[r, k], x[r, k + 1], y[r, k], y[r, k + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(x0 > x1, (x0 - target) / (x0 - x1), 0.0)
    return np.where(n >= 2, y0 + np.clip(f, 0, 1) * (y1 - y0), np.nan)


MATCHES = (("recall", "bg_tile_rate"), ("recall", "bg_fp_per_M"), ("recall", "precision"),
           ("bg_tile_rate", "recall"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--ref", default="A", help="arm whose argmax operating point is matched")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    arms = dict(a.split("=", 1) for a in args.arm)
    if args.ref not in arms:
        raise ValueError(f"--ref {args.ref} is not an arm")
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    curve_rows, point_rows, draws = [], [], {}
    for f in FOLDS:
        scenes, ref = load_fold(Path(arms[args.ref]) / f"fold_{f}.npz")
        data = {args.ref: ref}
        data.update({a: load_fold(Path(d) / f"fold_{f}.npz", scenes)[1]
                     for a, d in arms.items() if a != args.ref})
        edges = ref["edges"][:-1]
        half = int(np.searchsorted(edges, 0.5))
        w = np.vstack([np.ones(len(scenes)),
                       rng.multinomial(len(scenes), np.full(len(scenes), 1 / len(scenes)),
                                       size=args.draws)])
        c = {a: curves(m, w) for a, m in data.items()}
        for a in arms:
            for i, t in enumerate(edges):
                curve_rows.append(dict(arm=a, fold=f, threshold=t,
                                       **{k: v[0, i] for k, v in c[a].items()}))
        for key, read in MATCHES:
            target = c[args.ref][key][:, half]            # ref argmax point, per draw
            for a in arms:
                v = at_target(c[a], key, target, read)
                draws[(key, read, a, f)] = v
                point_rows.append(dict(match=key, target=target[0], read=read, arm=a, fold=f,
                                       value=v[0], at_argmax=c[a][read][0, half]))
    for key, read in MATCHES:
        for a in arms:
            v = np.mean([draws[(key, read, a, f)] for f in FOLDS], axis=0)
            draws[(key, read, a, "mean")] = v
            point_rows.append(dict(match=key, target=np.nan, read=read, arm=a, fold="mean",
                                   value=v[0], at_argmax=np.mean(
                                       [p["at_argmax"] for p in point_rows if p["match"] == key
                                        and p["read"] == read and p["arm"] == a
                                        and p["fold"] != "mean"])))
    diff_rows = []
    for key, read in MATCHES:
        for x, y in PAIRS:
            if x not in arms or y not in arms:
                continue
            for f in (*FOLDS, "mean"):
                d = draws[(key, read, x, f)] - draws[(key, read, y, f)]
                ok = np.isfinite(d[1:])
                lo, hi = np.percentile(d[1:][ok], [2.5, 97.5]) if ok.any() else (np.nan, np.nan)
                diff_rows.append(dict(match=key, read=read, pair=f"{x}-{y}", fold=f,
                                      diff=d[0], ci_lo=lo, ci_hi=hi, draws_reached=ok.mean()))
    pd.DataFrame(curve_rows).to_csv(out / "sweep_curves.csv", index=False)
    pts, diffs = pd.DataFrame(point_rows), pd.DataFrame(diff_rows)
    pts.to_csv(out / "matched_points.csv", index=False)
    diffs.to_csv(out / "matched_differences.csv", index=False)
    write_md(out, args, arms, pts, diffs)
    plot(out, pd.DataFrame(curve_rows), pts, args.ref)


def fmt(v, read):
    if not np.isfinite(v):
        return "unreachable"
    return f"{v:,.0f}" if read == "bg_fp_per_M" else f"{v:.4f}"


def write_md(out, args, arms, pts, diffs):
    lines = ["# Threshold sweep: matched operating points", "",
             f"Reference arm {args.ref}: its P(oil) >= 0.5 value in each fold is the target; every "
             "other arm's threshold is chosen to reach it. Single seed; 95% intervals from "
             f"{args.draws} paired scene-bootstrap draws (threshold re-chosen per draw). "
             "'unreachable': the arm reaches the target only by calling every pixel oil "
             "(P(oil) >= 0); the three-fold mean is then unreachable too. Intervals use the "
             "draws where both arms reach the target (share in matched_differences.csv).", ""]
    for key, read in MATCHES:
        lines += [f"## {read} at matched {key}", "",
                  "| arm | fold 1 | fold 2 | fold 3 | mean | at argmax (mean) |",
                  "|---|---|---|---|---|---|"]
        sub = pts[(pts.match == key) & (pts.read == read)]
        for a in arms:
            r = sub[sub.arm == a].set_index("fold")
            lines.append(f"| {a} | " + " | ".join(fmt(r.loc[f, "value"], read)
                                                   for f in (1, 2, 3, "mean"))
                         + f" | {fmt(r.loc['mean', 'at_argmax'], read)} |")
        tg = sub[(sub.arm == args.ref) & (sub.fold != "mean")]
        lines += ["", "targets (" + key + "): " + ", ".join(
            f"fold {r.fold} {r.target:.4f}" for r in tg.itertuples()), "",
                  "| pair | fold 1 | fold 2 | fold 3 | mean |", "|---|---|---|---|---|"]
        sd = diffs[(diffs.match == key) & (diffs.read == read)]
        for pair in sd.pair.unique():
            r = sd[sd.pair == pair].set_index("fold")
            lines.append(f"| {pair} | " + " | ".join(
                f"{fmt(r.loc[f, 'diff'], read)} [{fmt(r.loc[f, 'ci_lo'], read)}, "
                f"{fmt(r.loc[f, 'ci_hi'], read)}]" for f in (1, 2, 3, "mean")) + " |")
        lines.append("")
    (out / "matched_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out / 'matched_comparison.md'}")


def plot(out, cur, pts, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, f in zip(axes, (1, 2, 3)):
        for a, g in cur[cur.fold == f].groupby("arm"):
            g = g[g.threshold > 0]          # P(oil) >= 0 calls every pixel oil
            line, = ax.plot(g.recall, 100 * g.bg_tile_rate, label=a, lw=1.4)
            p = g.iloc[(g.threshold - 0.5).abs().argmin()]
            ax.plot(p.recall, 100 * p.bg_tile_rate, "o", color=line.get_color(), ms=5)
        t = pts[(pts.match == "recall") & (pts.arm == ref) & (pts.fold == f)].target.iloc[0]
        ax.axvline(t, color="grey", lw=0.8, ls="--")
        ax.set_title(f"fold {f}")
        ax.set_xlabel("recall (GT-aware reconstruction)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("background tiles with any oil pixel (%)")
    axes[-1].legend(title="arm", fontsize=8)
    fig.suptitle("Threshold 0.0025-0.9975; dot = argmax (0.5); dashed = reference recall")
    fig.tight_layout()
    fig.savefig(out / "sweep_recall_vs_bg_tiles.png", dpi=150)
    print(f"Wrote {out / 'sweep_recall_vs_bg_tiles.png'}")


if __name__ == "__main__":
    main()
