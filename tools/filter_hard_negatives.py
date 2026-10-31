"""G hnm_knn / G0 hnm_swap: replace C's hard negatives that sit too close to train oil.

tools/feature_space.py found that the slicks C loses sit among C's hard negatives in the
baseline (A) encoder space. This tool swaps out the hard negatives that are close to oil,
using train data only, and builds a matched random-swap control:
  score    share of train oil among the KNN nearest (cosine, z-scored as in feature_space)
           of train oil + C's selected hard negatives, the candidate itself excluded.
           A candidate's vector = A's encoder features pooled over its FP pixels
           (P(oil) >= 0.5 on mask == 1), single forward, like feature_space's hard_<a>.
  hnm_knn  C's selection minus score >= TAU, each replaced by an unselected candidate of the
           same scene with score < TAU (seeded); a scene that runs out is filled from the
           rest by the original scene round robin. Same count as C.
  hnm_swap the same number dropped per scene but chosen at random, refilled with the same
           number per scene drawn at random from the scene's unselected candidates.
Candidates without FP pixels in the single forward have no score: never dropped, never added.

  .venv/bin/python tools/filter_hard_negatives.py --stage simulate        # (4), saved features only
  .venv/bin/python tools/filter_hard_negatives.py --stage extract         # GPU, pool vectors
  .venv/bin/python tools/filter_hard_negatives.py --stage select          # write the two arms
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_space import FOLDS, Extractor, tile_vectors  # noqa: E402
from threshold_sweep import REPO_ROOT, run_settings  # noqa: E402

from preprocess.hnm.data import Patch, sha256, write_background  # noqa: E402
from preprocess.hnm_si.pipeline import (_check_selection, _implementation_hashes,  # noqa: E402
                                        _write_json, fold_inputs)
from preprocess.hnm_si.stats import stratified_select  # noqa: E402

KNN = 10
TAU = 0.3
ARMS_DIR = REPO_ROOT / "hnm_runs/hnm_si_v1/arms"
FEATURES = REPO_ROOT / "hnm_runs/feature_space"
OUT = REPO_ROOT / "hnm_runs/knn_filter"
POOL_STATUS = ("selected", "not_selected_quota")
METHODS = {"hnm_knn": "hnm_knn_encoder_filter_train_only_v1",
           "hnm_swap": "hnm_random_swap_control_train_only_v1"}


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def reference(fold, features=FEATURES):
    """z-score stats, unit train-oil and C hard vectors, and C's hard keys (feature_space (3))."""
    z = np.load(features / f"fold_{fold}.npz")
    ref = np.vstack([z[f"{g}_enc"] for g in ("train", "random", "hard_C", "hard_D")])
    mu, sd = ref.mean(axis=0), ref.std(axis=0) + 1e-6
    keys = [(s, int(x), int(y)) for s, x, y in
            pd.read_csv(features / f"fold_{fold}_hard_C.csv")[["scene", "x", "y"]].itertuples(index=False)]
    return mu, sd, unit((z["train_enc"] - mu) / sd), unit((z["hard_C_enc"] - mu) / sd), keys, z


def train_share(cand, train, hard, self_idx, k=KNN):
    """Share of ``train`` rows among the k nearest of train + hard; self_idx[i] >= 0 excludes hard[self_idx[i]]."""
    s = cand @ np.vstack([train, hard]).T
    rows = np.flatnonzero(self_idx >= 0)
    s[rows, len(train) + self_idx[rows]] = -np.inf
    nn = np.argpartition(-s, k - 1, axis=1)[:, :k]
    return (nn < len(train)).mean(axis=1)


def _rng(fold, scene, tag):
    return np.random.default_rng([42, fold, zlib.crc32(scene.encode("utf-8")), tag])


def build_arms(selected, unselected, score, fold, tau=TAU):
    """selected / unselected: {scene: [key]}; score: {key: float or nan}.

    Returns (knn, swap) lists of keys with identical per-scene counts, plus per-scene stats.
    """
    close = lambda k: score.get(k, np.nan) >= tau          # NaN -> False: never dropped
    ok = lambda k: score.get(k, np.nan) < tau               # NaN -> False: never added
    knn, swap, stats, spare = [], [], {}, {}
    adds = {}
    for s in sorted(set(selected) | set(unselected)):
        sel = selected.get(s, [])
        drop = [k for k in sel if close(k)]
        cand = [k for k in unselected.get(s, []) if ok(k)]
        cand = [cand[i] for i in _rng(fold, s, 1).permutation(len(cand))]
        add = cand[:len(drop)]
        spare[s] = cand[len(drop):]
        knn += [k for k in sel if not close(k)] + add
        adds[s] = len(add)
        stats[s] = {"selected": len(sel), "dropped": len(drop), "added_same_scene": len(add)}
    short = sum(v["dropped"] - v["added_same_scene"] for v in stats.values())
    extra = stratified_select({s: v for s, v in spare.items() if v}, short, 42 + 1000 * fold + 7)
    if len(extra) < short:
        raise ValueError(f"fold {fold}: only {len(extra)} filtered candidates for {short} slots")
    knn += extra
    for k in extra:
        adds[k[0]] += 1
        stats[k[0]]["added_other_scene"] = stats[k[0]].get("added_other_scene", 0) + 1
    for s in sorted(stats):
        sel, n_drop = selected.get(s, []), stats[s]["dropped"]
        rng = _rng(fold, s, 2)
        gone = {sel[i] for i in rng.permutation(len(sel))[:n_drop]}
        pool = unselected.get(s, [])
        add = [pool[i] for i in rng.permutation(len(pool))[:adds[s]]]
        if len(add) < adds[s]:
            raise ValueError(f"fold {fold} {s}: too few candidates for the swap control")
        swap += [k for k in sel if k not in gone] + add
    return knn, swap, stats


# --------------------------------------------------------------------------- stages
def simulate(features=FEATURES):
    """(4): drop rates on C's selected hard negatives, and around the slicks C lost (test labels)."""
    sim = pd.read_csv(features / "similarities.csv")
    sim = sim[sim.space == "enc"]
    rows = []
    for f in FOLDS:
        mu, sd, tr, hc, _, z = reference(f, features)
        nt = len(tr)
        share = train_share(hc, tr, hc, np.arange(len(hc)))
        keys = pd.read_csv(features / f"fold_{f}_oil.csv")
        d = keys.merge(sim[sim.fold == f], on=["scene", "label"], how="left", validate="one_to_one")
        lost = (d.class_C == "lost").to_numpy()
        oil = unit((z["oil_enc"] - mu) / sd)
        nn = np.argsort(oil @ np.vstack([tr, hc]).T, axis=1)[:, -KNN:]
        near = np.unique(nn[lost][nn[lost] >= nt] - nt)
        before = (nn >= nt).mean(axis=1)[lost].mean()
        for tau in (0.1, 0.2, 0.3, 0.5):
            keep = share < tau
            nn2 = np.argsort(oil @ np.vstack([tr, hc[keep]]).T, axis=1)[:, -KNN:]
            rows.append({"fold": f, "tau": tau, "hard": len(hc), "dropped": (~keep).mean(),
                         "near_lost": len(near), "near_lost_dropped": (~keep[near]).mean(),
                         "lost_knn_hard_before": before,
                         "lost_knn_hard_after": (nn2 >= nt).mean(axis=1)[lost].mean()})
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))
    return df


def _source(fold):
    d = ARMS_DIR / "hnm" / f"fold_{fold}"
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    df = pd.read_csv(d / "candidates_scored.tsv", sep="\t")
    return d, m, df


def extract(fold, device):
    from preprocess.hnm_si.probe import load_probe
    d, m, df = _source(fold)
    run_dir = m["probe"]["run_dir"]
    s = run_settings(run_dir)
    ext = Extractor(load_probe(run_dir, device, amp=False).model)
    items = df[df.status.isin(POOL_STATUS)][["scene", "x", "y"]]
    vec, meta = tile_vectors(ext, s, device, items, lambda p, msk: (p >= 0.5) & (msk == 1))
    print(f"fold {fold}: {len(meta)} of {len(items)} pool tiles have FP in a single forward", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"fold_{fold}_pool.npz", enc=vec["enc"].astype(np.float32))
    meta.to_csv(OUT / f"fold_{fold}_pool.csv", index=False)


def score_pool(fold):
    mu, sd, tr, hc, hkeys, z = reference(fold)
    meta = pd.read_csv(OUT / f"fold_{fold}_pool.csv")
    keys = [(s, int(x), int(y)) for s, x, y in meta[["scene", "x", "y"]].itertuples(index=False)]
    raw = np.load(OUT / f"fold_{fold}_pool.npz")["enc"].astype(np.float64)
    pos = {k: i for i, k in enumerate(hkeys)}
    self_idx = np.array([pos.get(k, -1) for k in keys])
    # The pool re-extraction must reproduce feature_space's vectors of C's selected tiles.
    hit = self_idx >= 0
    diff = np.abs(raw[hit] - z["hard_C_enc"][self_idx[hit]]).max() / np.abs(z["hard_C_enc"]).max()
    if hit.sum() != len(hkeys) or diff > 1e-2:
        raise ValueError(f"fold {fold}: pool vectors do not match feature_space "
                         f"({hit.sum()}/{len(hkeys)} keys, rel diff {diff:.2e})")
    share = train_share(unit((raw - mu) / sd), tr, hc, self_idx)
    return dict(zip(keys, share)), float(diff)


def write_arm(arm, fold, keys, src_dir, src_manifest, df, score, report):
    config = src_manifest["config"]
    ctx = fold_inputs(config, fold)
    selected = sorted((Patch(s, x, y, 0) for s, x, y in set(keys)), key=lambda r: r.key)
    if len(selected) != len(keys) or len(selected) != src_manifest["selected_count"]:
        raise ValueError(f"{arm} fold {fold}: {len(keys)} keys, {len(selected)} unique")
    _check_selection(config, ctx, selected)
    out = ARMS_DIR / arm / f"fold_{fold}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"{out}/manifest.json exists")
    chosen = set(keys)
    df = df.copy()
    df["knn_train_share"] = [score.get((s, int(x), int(y)), np.nan)
                             for s, x, y in df[["scene", "x", "y"]].itertuples(index=False)]
    in_pool = df.status.isin(POOL_STATUS)
    is_sel = pd.Series([(s, int(x), int(y)) in chosen
                        for s, x, y in df[["scene", "x", "y"]].itertuples(index=False)], index=df.index)
    df.loc[in_pool, "status"] = "not_selected_quota"
    df.loc[is_sel, "status"] = "selected"
    if (is_sel & ~in_pool).any():
        raise ValueError("Selected a window outside C's candidate pool")
    df.to_csv(out / "candidates_scored.tsv", sep="\t", index=False)
    write_background(out / "selected_train_bg.tsv", selected)
    train_scenes = ctx["scenes"]["train"]
    write_background(out / "bg_coords.tsv",
                     [r for r in ctx["baseline"] if r.scene not in train_scenes] + selected)
    _write_json(out / "selection_report.json", report)
    manifest = {k: v for k, v in src_manifest.items()
                if k not in ("bg_coords_sha256", "selected_train_bg_sha256", "candidates_sha256")}
    manifest.update({
        "method": METHODS[arm], "arm": arm,
        "bg_coords_sha256": sha256(out / "bg_coords.tsv"),
        "selected_train_bg_sha256": sha256(out / "selected_train_bg.tsv"),
        "candidates_sha256": sha256(out / "candidates_scored.tsv"),
        "source_arm": {"arm": "hnm", "manifest_sha256": sha256(src_dir / "manifest.json"),
                       "candidates_sha256": src_manifest["candidates_sha256"]},
        "knn_filter": {"k": KNN, "tau": TAU, "space": "A encoder, z-scored, cosine",
                       "reference": "train oil + hnm selected hard negatives (feature_space)",
                       "features_npz_sha256": sha256(FEATURES / f"fold_{fold}.npz"),
                       "pool_npz_sha256": sha256(OUT / f"fold_{fold}_pool.npz")},
        "implementation_sha256": _implementation_hashes() | {
            "filter_hard_negatives.py": sha256(Path(__file__))}})
    _write_json(out / "manifest.json", manifest)
    print(f"fold {fold} [{arm}]: {len(selected)} backgrounds -> {out}", flush=True)


def select(fold):
    src_dir, m, df = _source(fold)
    score, diff = score_pool(fold)
    pool = df[df.status.isin(POOL_STATUS)]
    selected, unselected = {}, {}
    for r in pool.itertuples(index=False):
        k = (r.scene, int(r.x), int(r.y))
        (selected if r.status == "selected" else unselected).setdefault(r.scene, []).append(k)
    knn, swap, stats = build_arms(selected, unselected, score, fold)
    vals = np.array([score.get(k, np.nan) for v in selected.values() for k in v])
    report = {"fold": fold, "k": KNN, "tau": TAU, "pool": int(len(pool)),
              "pool_scored": int(len(score)), "rel_diff_vs_feature_space": diff,
              "selected": int(sum(map(len, selected.values()))),
              "selected_unscored": int(np.isnan(vals).sum()),
              "dropped": int(sum(v["dropped"] for v in stats.values())),
              "added_other_scene": int(sum(v.get("added_other_scene", 0) for v in stats.values())),
              "scenes_with_drops": int(sum(v["dropped"] > 0 for v in stats.values())),
              "knn_overlap_with_hnm": len(set(knn) & {k for v in selected.values() for k in v}),
              "swap_overlap_with_hnm": len(set(swap) & {k for v in selected.values() for k in v}),
              "knn_swap_overlap": len(set(knn) & set(swap)),
              "share_selected_knn": float(np.nanmean([score.get(k, np.nan) for k in knn])),
              "share_selected_swap": float(np.nanmean([score.get(k, np.nan) for k in swap])),
              "share_selected_hnm": float(np.nanmean(vals)),
              "by_scene": stats}
    print(json.dumps({k: v for k, v in report.items() if k != "by_scene"}), flush=True)
    for arm, keys in (("hnm_knn", knn), ("hnm_swap", swap)):
        write_arm(arm, fold, keys, src_dir, m, df, score, report | {"arm": arm})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("simulate", "extract", "select"), required=True)
    parser.add_argument("--fold", type=int, action="append")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    folds = args.fold or list(FOLDS)
    if args.stage == "simulate":
        simulate()
    for f in folds:
        if args.stage == "extract":
            if (OUT / f"fold_{f}_pool.npz").exists():
                print(f"fold {f}: pool vectors exist, skipped", flush=True)
                continue
            import torch
            extract(f, torch.device(args.device))
        elif args.stage == "select":
            select(f)


if __name__ == "__main__":
    main()
