"""CLI: preflight, mine (probe FP + SI statistics), then select per frozen outer fold.

Mining scans only the fold's own train scenes. Selection only rewrites the train
background rows of the baseline manifest; GT-centric lists and all val/test
background rows are preserved, and manifest.json is written last.
"""
import argparse
import csv
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess.hnm.data import (Patch, inspect_rasters, raster_index, read_background,
                                 read_patches, sha256, split_manifest, validate_labels,
                                 write_background)

from . import mining
from .config import ARMS, load_config
from .indices import SI_NAMES, band_index, sensor_of
from .stats import (ledoit_wolf_reference, lower_cutoff, mahalanobis, outlier_scenes,
                    stratified_select)

SCHEMA_VERSION = 1
METHODS = {"hnm_si": "hnm_si_probe_fp_within_fold_train_only_v1",
           "hnm": "hnm_probe_fp_within_fold_train_only_v1",
           "random": "random_eligible_control_train_only_v1",
           "oof": "oof_classifier_scene_oof_train_only_v1",
           "oof_si": "oof_classifier_si_scene_oof_train_only_v1"}
# Spectral-index columns the SI filter compares with the oil reference, per arm.
SI_COLUMNS = {"hnm_si": ["p_mndwi", "p_ndi", "p_sr"],    # median over the probe's FP pixels
              "oof_si": ["w_mndwi", "w_ndi", "w_sr"]}    # median over the window's mask==1 pixels
MINING_FILES = ("config.py", "indices.py", "mining.py", "probe.py")


def _write_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _implementation_hashes(names=None):
    here = Path(__file__).parent
    paths = [here / n for n in names] if names else sorted(here.glob("*.py"))
    return {p.name: sha256(p) for p in paths}


def fold_inputs(config, fold):
    """Frozen split facts shared by every stage (train scenes, GT coordinates, baseline)."""
    splits, scenes = split_manifest(config["split_dir"], fold)
    train_scenes = sorted(scenes["train"])
    gt_xy = {}
    for p in splits["train"]:
        gt_xy.setdefault(p.scene, set()).add((p.x, p.y))
    baseline = read_background(config["baseline_bg_tsv"])
    hashes = {s: sha256(Path(config["split_dir"]) / f"{s}_fold{fold}.txt")
              for s in ("train", "val", "test")}
    return {"splits": splits, "scenes": scenes, "train_scenes": train_scenes,
            "gt_xy": gt_xy, "baseline": baseline, "split_sha256": hashes,
            "baseline_sha256": sha256(config["baseline_bg_tsv"])}


def probe_description(config, fold):
    from .probe import probe_settings, read_run
    run_dir = config["probe_run_dirs"][fold]
    args, weights = read_run(run_dir)
    mean, std = probe_settings(args)
    arch = args["architecture_cfg"]
    names = arch.get("input_band_names") or ["B01", "B02", "B03", "B04", "B08", "B8A", "B11", "B12"]
    if len(names) != len(mean):
        raise ValueError("Probe band names do not match its mean/std")
    return {"run_dir": run_dir, "checkpoint": str(weights), "checkpoint_sha256": sha256(weights),
            "args_sha256": sha256(Path(run_dir) / "args.yaml"),
            "model_family": arch.get("model_family"), "mean": mean, "std": std,
            "band_names": list(names)}


# Selection-only SI settings (cutoff, covariance pool size) may change without remining.
SELECT_ONLY_SI = ("top_q", "min_oil_pool")


def _scan_settings(config):
    return {"seed": config["seed"], "patch_size": config["patch_size"],
            "stride": config["stride"], "scale": config["scale"], "rules": config["rules"],
            "si": {k: v for k, v in config["si"].items() if k not in SELECT_ONLY_SI}}


def _mining_dir(config, fold):
    return Path(config["output_dir"]) / "mining" / f"fold_{fold}"


def _arm_dir(config, arm, fold):
    return Path(config["output_dir"]) / "arms" / arm / f"fold_{fold}"


def _table_inputs(config, scenes):
    index = raster_index(config["vrt_dir"])
    missing = [s for s in scenes if s not in index or not (Path(config["mask_dir"]) / f"{s}.tif").exists()]
    if missing:
        raise FileNotFoundError(f"Missing raster or mask for {len(missing)} scenes, e.g. {missing[:3]}")
    return index


# --------------------------------------------------------------------------- preflight
def preflight_fold(config, fold, device, probe_factory=None):
    from .indices import compute_si, reflectance
    from .probe import load_probe
    from rasterio.windows import Window
    import rasterio
    ctx = fold_inputs(config, fold)
    index = _table_inputs(config, ctx["train_scenes"])
    inspect_rasters([Patch(s, 0, 0, 0) for s in ctx["train_scenes"]], index,
                    config["mask_dir"], 8)
    desc = probe_description(config, fold)
    bands = band_index(desc["band_names"])
    counts, per_sensor = {}, {}
    first_window = {}
    for s in ctx["train_scenes"]:
        with rasterio.open(Path(config["mask_dir"]) / f"{s}.tif") as src:
            mask = src.read(1)
        mining.check_mask(mask, s)
        w = mining.eligible_windows(mask, config["patch_size"], config["stride"],
                                    config["rules"], ctx["gt_xy"].get(s, frozenset()))
        sensor = sensor_of(s)
        counts[s] = len(w)
        per_sensor.setdefault(sensor, {"scenes": 0, "mask_eligible_windows": 0})
        per_sensor[sensor]["scenes"] += 1
        per_sensor[sensor]["mask_eligible_windows"] += len(w)
        if w and sensor not in first_window:
            first_window[sensor] = (s, w[len(w) // 2])
    probe = (probe_factory or load_probe)(desc["run_dir"], device, config["amp"])
    scale_check = {}
    size = config["patch_size"]
    for sensor, (scene, (y, x, *_)) in first_window.items():
        with rasterio.open(index[scene]) as src:
            raw = src.read(window=Window(x, y, size, size))
        from .indices import model_input, model_valid_mask
        p = probe(model_input(raw, config["scale"], desc["mean"], desc["std"])[None])
        offset = config["si"]["reflectance_offset"]
        si, ok = compute_si(reflectance(raw, config["scale"], offset), bands,
                            config["si"]["den_floor"], config["si"]["clip"])
        scale_check[sensor] = {
            "scene": scene, "window": [x, y],
            "raw_dn_p1": {b: float(np.percentile(raw[i][model_valid_mask(raw)], 1))
                          for b, i in bands.items()},
            "reflectance_median": {b: float((np.median(raw[i][model_valid_mask(raw)]) - offset)
                                            / config["scale"]) for b, i in bands.items()},
            "si_valid_fraction": float(ok.mean()),
            "si_median": {n: float(np.nanmedian(si[k])) if ok.any() else None
                          for k, n in enumerate(SI_NAMES)},
            "probe_prob_range": [float(p.min()), float(p.max())]}
    report = {"fold": fold, "train_scenes": len(ctx["train_scenes"]),
              "gt_train_patches": len(ctx["splits"]["train"]),
              "baseline_train_backgrounds": sum(r.scene in ctx["scenes"]["train"] for r in ctx["baseline"]),
              "mask_eligible_windows_total": int(sum(counts.values())),
              "scenes_without_eligible_window": sum(v == 0 for v in counts.values()),
              "by_sensor": per_sensor, "probe": {k: v for k, v in desc.items() if k not in ("mean", "std")},
              "scale_check": scale_check}
    out = Path(config["output_dir"]) / "preflight"
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / f"fold_{fold}.json", report)
    return report


# --------------------------------------------------------------------------- mining
def _scene_rng(config, fold, scene):
    return np.random.default_rng([config["seed"], fold, zlib.crc32(scene.encode("utf-8"))])


def mine_fold(config, fold, device, max_scenes=None, probe_factory=None, progress=True):
    from .probe import load_probe
    ctx = fold_inputs(config, fold)
    scenes = ctx["train_scenes"][:max_scenes] if max_scenes else ctx["train_scenes"]
    index = _table_inputs(config, scenes)
    desc = probe_description(config, fold)
    bands = band_index(desc["band_names"])
    out = _mining_dir(config, fold)
    (out / "scenes").mkdir(parents=True, exist_ok=True)
    plan = {"schema_version": SCHEMA_VERSION, "outer_fold": fold, "scenes": scenes,
            "max_scenes": max_scenes, "split_sha256": ctx["split_sha256"],
            "baseline_bg_sha256": ctx["baseline_sha256"],
            "probe": {k: v for k, v in desc.items()},
            "scan": _scan_settings(config),
            "vrt_dir": config["vrt_dir"], "mask_dir": config["mask_dir"],
            "implementation_sha256": _implementation_hashes(MINING_FILES)}
    plan_path = out / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != json.loads(json.dumps(plan)):
            raise ValueError(f"{plan_path} differs from this run; mine into a new output_dir "
                             "instead of mixing artifacts from different settings")
    else:
        _write_json(plan_path, plan)
    probe = (probe_factory or load_probe)(desc["run_dir"], device, config["amp"])
    iterator = scenes
    if progress:
        from tqdm import tqdm
        iterator = tqdm(scenes, desc=f"fold {fold} scan")
    for scene in iterator:
        done = out / "scenes" / f"{scene}.json"
        if done.exists():
            continue
        rows, summary, ref = mining.scan_scene(
            scene, index[scene], Path(config["mask_dir"]) / f"{scene}.tif", config, probe,
            desc["mean"], desc["std"], bands, ctx["gt_xy"].get(scene, frozenset()),
            _scene_rng(config, fold, scene))
        tsv = out / "scenes" / f"{scene}.tsv"
        with open(tsv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(mining.COLUMNS)
            for r in rows:
                w.writerow(["" if isinstance(v, float) and np.isnan(v) else v for v in r])
        pool = ref["pool"] if ref["pool"] is not None else np.zeros((0, 3), np.float32)
        np.save(out / "scenes" / f"{scene}.pool.npy", pool)
        summary.update({"n_oil_px": ref["n_oil_px"], "n_valid_oil_px": ref["n_valid_oil_px"],
                        "fingerprint": None if ref["fingerprint"] is None
                        else [float(v) for v in ref["fingerprint"]]})
        _write_json(done, summary)
    return _finalize_mining(config, fold, out, scenes, plan, partial=bool(max_scenes))


def _finalize_mining(config, fold, out, scenes, plan, partial):
    frames, refs, pools = [], [], []
    for scene in scenes:
        summary = json.loads((out / "scenes" / f"{scene}.json").read_text(encoding="utf-8"))
        frames.append(pd.read_csv(out / "scenes" / f"{scene}.tsv", sep="\t", keep_default_na=True))
        fp = summary["fingerprint"] or [np.nan] * 3
        refs.append({**{k: summary[k] for k in ("scene", "sensor", "n_windows_total",
                                                 "n_mask_eligible", "n_nodata_rejected",
                                                 "n_scored", "n_with_fp", "n_oil_px",
                                                 "n_valid_oil_px")},
                     "reference_valid": summary["fingerprint"] is not None,
                     **dict(zip(("oil_mndwi", "oil_ndi", "oil_sr"), fp))})
        pools.append(np.load(out / "scenes" / f"{scene}.pool.npy"))
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=mining.COLUMNS)
    table.to_csv(out / "candidates.tsv", sep="\t", index=False)
    pd.DataFrame(refs).to_csv(out / "scene_reference.csv", index=False)
    pool = np.concatenate(pools) if pools else np.zeros((0, 3), np.float32)
    np.save(out / "oil_pool.npy", pool)
    manifest = {"schema_version": SCHEMA_VERSION, "outer_fold": fold,
                "status": "partial" if partial else "complete",
                "scene_count": len(scenes), "candidate_rows": int(len(table)),
                "plan_sha256": sha256(out / "plan.json"),
                "candidates_sha256": sha256(out / "candidates.tsv"),
                "scene_reference_sha256": sha256(out / "scene_reference.csv"),
                "oil_pool_sha256": sha256(out / "oil_pool.npy")}
    _write_json(out / "mining_manifest.json", manifest)
    print(f"Fold {fold} mining {manifest['status']}: {len(table)} scored windows, "
          f"{int((table['n_fp'] > 0).sum()) if len(table) else 0} with FP", flush=True)
    return manifest


# --------------------------------------------------------------------------- selection
def _load_mining(config, fold, ctx, allow_partial):
    out = _mining_dir(config, fold)
    mm_path = out / "mining_manifest.json"
    if not mm_path.exists():
        raise FileNotFoundError(f"Run --stage mine for fold {fold} first: {mm_path}")
    mm = json.loads(mm_path.read_text(encoding="utf-8"))
    if mm["status"] != "complete" and not (allow_partial and mm["status"] == "partial"):
        raise ValueError(f"Mining for fold {fold} is {mm['status']}; "
                         "rerun without --max-scenes (or pass --allow-partial for a smoke test)")
    for name, key in (("plan.json", "plan_sha256"), ("candidates.tsv", "candidates_sha256"),
                      ("scene_reference.csv", "scene_reference_sha256"),
                      ("oil_pool.npy", "oil_pool_sha256")):
        if sha256(out / name) != mm[key]:
            raise ValueError(f"Mining artifact changed after completion: {name}")
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    if plan["split_sha256"] != ctx["split_sha256"] or plan["baseline_bg_sha256"] != ctx["baseline_sha256"]:
        raise ValueError("Frozen splits or baseline background changed since mining")
    if plan["scenes"] != ctx["train_scenes"] and not allow_partial:
        raise ValueError("Mined scenes differ from the fold's training scenes")
    if plan["scan"] != json.loads(json.dumps(_scan_settings(config))):
        raise ValueError("Scan settings differ from the mined settings; remine into a new output_dir")
    return out, mm, plan


def _sensor_table(df, status_col="status"):
    return {s: {k: int(v) for k, v in g[status_col].value_counts().items()}
            for s, g in df.groupby("sensor")}


def select_fold(config, fold, arm=None, allow_partial=False):
    arm = arm or config["arm"]
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    ctx = fold_inputs(config, fold)
    mdir, mm, plan = _load_mining(config, fold, ctx, allow_partial)
    out = _arm_dir(config, arm, fold)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"{out}/manifest.json exists; use a new output_dir to reselect")
    rules, si_cfg, sel_cfg = config["rules"], config["si"], config["selection"]
    df = pd.read_csv(mdir / "candidates.tsv", sep="\t")
    df["status"] = "qualified"
    df["d_si"] = np.nan
    key_cols = ["scene", "x", "y"]
    if df.duplicated(key_cols).any():
        raise ValueError("Mined candidate coordinates are not unique")
    oof_info = None
    if arm in ("hnm", "hnm_si"):
        df.loc[df["n_fp"] < rules["min_fp_pixels"], "status"] = "no_fp"
    elif arm in ("oof", "oof_si"):
        from .oof import load_scores
        scores, oof_manifest, oof_manifest_sha = load_scores(config, fold, mm)
        df = df.merge(scores.drop(columns=[c for c in scores if c in df and c not in key_cols]),
                      on=key_cols, how="left", validate="one_to_one")
        if df["p_oil"].isna().any() or len(scores) != len(df):
            raise ValueError("OOF scores do not cover the mined candidates")
        # Hard = the scene-OOF classifier calls the oil-free window oil.
        df.loc[df["p_oil"] < rules["prob_threshold"], "status"] = "oof_not_oil"
        oof_info = {"oof_manifest_sha256": oof_manifest_sha,
                    "miner_sha256": oof_manifest["miner_sha256"],
                    "scores_sha256": oof_manifest["scores_sha256"]}
    report = {"fold": fold, "arm": arm, "mining_status": mm["status"],
              "scored_windows": int(len(df))}
    reference_info = None
    if arm in SI_COLUMNS:
        si_cols = SI_COLUMNS[arm]
        refs = pd.read_csv(mdir / "scene_reference.csv").set_index("scene")
        pool = np.load(mdir / "oil_pool.npy")
        if len(pool) < si_cfg["min_oil_pool"]:
            raise ValueError(f"Oil pool too small for covariance: {len(pool)}")
        ref = ledoit_wolf_reference(pool)
        live = df["status"] == "qualified"
        has_p = df[si_cols].notna().all(axis=1)
        df.loc[live & ~has_p, "status"] = "fp_region_below_si_min"
        ref_ok = df["scene"].map(refs["reference_valid"]).fillna(False).astype(bool)
        live = df["status"] == "qualified"
        df.loc[live & ~ref_ok, "status"] = "no_oil_reference"
        live = df["status"] == "qualified"
        centers = refs.loc[df.loc[live, "scene"], ["oil_mndwi", "oil_ndi", "oil_sr"]].to_numpy()
        points = df.loc[live, si_cols].to_numpy()
        df.loc[live, "d_si"] = mahalanobis(points, centers, ref["precision"])
        cutoff = lower_cutoff(df.loc[live, "d_si"], si_cfg["top_q"])
        if cutoff is not None:
            df.loc[live & (df["d_si"] <= cutoff), "status"] = "too_close_to_oil"
        reference_info = {
            "names": list(SI_NAMES), "candidate_columns": si_cols, "pool_pixels": ref["n"],
            "reference_scenes": int(refs["reference_valid"].sum()),
            "pool_median": ref["median"].tolist(), "pool_mean": ref["mean"].tolist(),
            "covariance": ref["cov"].tolist(), "shrinkage": ref["shrinkage"],
            "condition_number": ref["cond"], "top_q": si_cfg["top_q"], "cutoff": cutoff,
            "d_si_quantiles": {str(q): float(np.nanquantile(df.loc[live, "d_si"], q))
                               for q in (.01, .05, .25, .5, .75, .95)} if live.any() else {}}
        _write_json(out / "si_reference.json", reference_info)
    live = df["status"] == "qualified"
    dropped = outlier_scenes(df.loc[live, "scene"], sel_cfg["scene_outlier_sigma"])
    if dropped:
        df.loc[live & df["scene"].isin(dropped), "status"] = "scene_outlier"
    pool_df = df[df["status"] == "qualified"]
    train_baseline = [r for r in ctx["baseline"] if r.scene in ctx["scenes"]["train"]]
    target = (len(train_baseline) if sel_cfg["target_count"] == "baseline"
              else int(sel_cfg["target_count"]))
    report.update({"target_count": target, "pool_count": int(len(pool_df)),
                   "outlier_scenes": sorted(dropped)})
    status_view = lambda: {"status_counts": {k: int(v) for k, v in df["status"].value_counts().items()},
                           "by_sensor": _sensor_table(df)}
    shortfall = max(0, target - len(pool_df))
    report["shortfall"] = shortfall
    if shortfall and not sel_cfg["allow_shortfall"]:
        _write_json(out / "selection_report.json",
                    report | status_view() | {"outcome": "insufficient_candidates"})
        raise ValueError(f"Fold {fold} arm {arm}: {len(pool_df)} qualified candidates for a target "
                         f"of {target}; see {out / 'selection_report.json'}. Relax rules or set "
                         "selection.allow_shortfall to accept a smaller, recorded quota.")
    keys_by_scene = {}
    for r in pool_df.itertuples(index=False):
        keys_by_scene.setdefault(r.scene, []).append((r.scene, int(r.x), int(r.y)))
    chosen = stratified_select(keys_by_scene, target, config["seed"] + 1000 * fold)
    chosen_set = set(chosen)
    selected = sorted((Patch(s, x, y, 0) for s, x, y in chosen_set), key=lambda r: r.key)
    keys = df[key_cols].itertuples(index=False, name=None)
    is_sel = pd.Series([(s, int(x), int(y)) in chosen_set for s, x, y in keys], index=df.index)
    df.loc[is_sel, "status"] = "selected"
    df.loc[df["status"] == "qualified", "status"] = "not_selected_quota"
    _check_selection(config, ctx, selected)
    df.to_csv(out / "candidates_scored.tsv", sep="\t", index=False)
    write_background(out / "selected_train_bg.tsv", selected)
    train_scenes = ctx["scenes"]["train"]
    combined = [r for r in ctx["baseline"] if r.scene not in train_scenes] + selected
    write_background(out / "bg_coords.tsv", combined)
    sel_df = df[df["status"] == "selected"]
    report.update(status_view())
    report.update({"outcome": "complete", "selected_count": len(selected),
                   "selected_by_sensor": {k: int(v) for k, v in sel_df["sensor"].value_counts().items()},
                   "baseline_train_by_sensor": {k: int(v) for k, v in pd.Series(
                       [sensor_of(r.scene) for r in train_baseline]).value_counts().items()},
                   "selected_scene_count": int(sel_df["scene"].nunique()),
                   "mean_selected_n_fp": float(sel_df["n_fp"].mean()) if len(sel_df) else None,
                   "si_reference": reference_info})
    _write_json(out / "selection_report.json", report)
    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "partial" if mm["status"] != "complete" else "complete",
        "outer_fold": fold, "method": METHODS[arm], "arm": arm,
        "split_sha256": ctx["split_sha256"], "baseline_bg_sha256": ctx["baseline_sha256"],
        "bg_coords_sha256": sha256(out / "bg_coords.tsv"),
        "selected_train_bg_sha256": sha256(out / "selected_train_bg.tsv"),
        "candidates_sha256": sha256(out / "candidates_scored.tsv"),
        "mining_manifest_sha256": sha256(mdir / "mining_manifest.json"),
        "selected_count": len(selected), "target_count": target, "shortfall": shortfall,
        "probe": (plan["probe"] | {"scope": "within_fold_train_only"}) if oof_info is None else {
            "kind": "efficientnet_oof_classifiers", "scope": "scene_oof_within_fold_train",
            "checkpoint_sha256": oof_info["oof_manifest_sha256"], **oof_info},
        "config": config, "implementation_sha256": _implementation_hashes(),
        "sensor_policy": "pooled_oil_covariance_all_sensors_per_sensor_reporting_only",
        "label_policy": {"false_positive_background": "mask==1", "others_mask2": "recorded_not_fp",
                         "unlabeled_mask255": "never_fp"},
        "isolation": "oil reference, covariance and cutoff use this fold's train scenes only",
        "validation_note": ("probe best.pt was chosen with the fold's val set; mining is "
                            "within-fold on train scenes, not scene-level out-of-fold")
        if oof_info is None else ("each window was scored by a classifier that never trained "
                                  "on its scene; the pool and SI reference come from HNM_SI mining")}
    _write_json(out / "manifest.json", manifest)
    print(f"Fold {fold} [{arm}]: {len(selected)}/{target} backgrounds -> {out}", flush=True)
    return manifest


def _check_selection(config, ctx, selected):
    if not selected:
        raise ValueError("Empty selection")
    keys = [r.key for r in selected]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate selected coordinates")
    train_scenes = ctx["scenes"]["train"]
    if any(r.scene not in train_scenes for r in selected):
        raise ValueError("Selected a background outside the training scenes")
    if any((r.x, r.y) in ctx["gt_xy"].get(r.scene, ()) for r in selected):
        raise ValueError("Selected background duplicates a GT-centric tile")
    validate_labels(selected, config["mask_dir"], config["patch_size"])


# --------------------------------------------------------------------------- CLI
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("preflight", "mine", "oof", "select"), required=True,
                        help="oof: train scene-OOF classifiers and score the mined pool "
                             "(needed by --arm oof / oof_si)")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--max-scenes", type=int,
                        help="Smoke test: mine only the first N training scenes (result is partial)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Allow select on partial mining; manifest stays 'partial' and "
                             "training will refuse it")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    folds = [args.fold] if args.fold is not None else config["outer_folds"]
    if any(f not in config["outer_folds"] for f in folds):
        parser.error("--fold is not in configured outer_folds")
    device = None
    if args.stage in ("preflight", "mine", "oof"):
        import torch
        device = torch.device(config["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use a GPU host or explicitly set device: cpu")
    for fold in folds:
        if args.stage == "preflight":
            print(json.dumps(preflight_fold(config, fold, device), ensure_ascii=False, indent=2),
                  flush=True)
        elif args.stage == "mine":
            mine_fold(config, fold, device, args.max_scenes)
        elif args.stage == "oof":
            from .oof import mine_oof
            mine_oof(config, fold, device)
        else:
            select_fold(config, fold, args.arm, args.allow_partial)


if __name__ == "__main__":
    main()
