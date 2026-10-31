"""Load and validate HNM_SI settings; probe settings come from each fold's own run."""
import math
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ARMS = ("hnm_si", "hnm", "random", "oof", "oof_si")

DEFAULTS = {
    "outer_folds": [1, 2, 3], "seed": 42, "patch_size": 256, "stride": 192,
    "scale": 10000.0, "batch_size": 16, "device": "cuda", "amp": True,
    "arm": "hnm_si",
    # Original Step 1 parameters. Starting points, not validated for this dataset.
    "rules": {"min_mask1_ratio": .5, "max_mask255_ratio": .3, "max_nodata_ratio": .3,
              "prob_threshold": .5, "min_fp_pixels": 1},
    # Original Step 2/3 parameters.
    "si": {"top_q": .05, "min_fp_pixels": 100, "min_oil_pixels": 100,
           "oil_pool_samples_per_scene": 1000, "min_oil_pool": 1000,
           "den_floor": 1e-6, "clip": 10.0,
           # Delivered rasters carry +1000 DN additive offset (verified: sea floors at 1001).
           "reflectance_offset": 1000.0},
    "selection": {"target_count": "baseline", "allow_shortfall": False,
                  "scene_outlier_sigma": None},
    # oof / oof_si arms: EfficientNetV2-S classifiers scored scene-out-of-fold (see oof.py).
    # Classifier settings follow preprocess/hnm; neg_pool is the fixed cached negative sample.
    "oof": {"inner_folds": 5, "model_name": "tf_efficientnetv2_s", "pretrained": True,
            "epochs": 70, "batch_size": 8, "workers": 8, "lr": 1e-4, "weight_decay": .01,
            "classifier_neg_ratio": 5, "neg_pool": 15000, "infer_batch_size": 64},
}
PATH_KEYS = ("split_dir", "baseline_bg_tsv", "vrt_dir", "mask_dir", "output_dir")
NESTED = ("rules", "si", "selection", "oof")


def _resolve(base, value):
    p = Path(value).expanduser()
    return str((base / p).resolve() if not p.is_absolute() else p)


def load_config(path):
    path = Path(path).resolve()
    supplied = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(supplied, dict):
        raise ValueError("HNM_SI config must be a mapping")
    allowed = set(DEFAULTS) | set(PATH_KEYS) | {"probe_run_dirs"}
    if set(supplied) - allowed:
        raise ValueError(f"Unknown HNM_SI settings: {sorted(set(supplied) - allowed)}")
    config = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    for k, v in supplied.items():
        if k in NESTED:
            if not isinstance(v, dict) or set(v) - set(DEFAULTS[k]):
                raise ValueError(f"Invalid {k} settings: {sorted(set(v or {}) - set(DEFAULTS[k]))}")
            config[k].update(v)
        else:
            config[k] = v
    for k in PATH_KEYS:
        if not config.get(k):
            raise ValueError(f"Missing setting: {k}")
        config[k] = _resolve(path.parent, config[k])
    runs = config.get("probe_run_dirs")
    if not isinstance(runs, dict) or not runs:
        raise ValueError("probe_run_dirs must map each outer fold to its baseline run directory")
    config["probe_run_dirs"] = {int(k): _resolve(path.parent, v) for k, v in runs.items()}
    _validate(config)
    return config


def _validate(c):
    folds = c["outer_folds"]
    if not folds or len(set(folds)) != len(folds) or any(not isinstance(f, int) or f < 1 for f in folds):
        raise ValueError("outer_folds must contain unique positive integers")
    missing = [f for f in folds if f not in c["probe_run_dirs"]]
    if missing:
        raise ValueError(f"No probe run directory for folds {missing}")
    if c["arm"] not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    for k in ("patch_size", "stride", "batch_size"):
        if not isinstance(c[k], int) or c[k] < 1:
            raise ValueError(f"Invalid {k}")
    if not math.isfinite(c["scale"]) or c["scale"] <= 0:
        raise ValueError("Invalid scale")
    if c["device"] not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    r, s, sel = c["rules"], c["si"], c["selection"]
    for k in ("min_mask1_ratio", "max_mask255_ratio", "max_nodata_ratio", "prob_threshold"):
        if not 0 <= r[k] <= 1:
            raise ValueError(f"rules.{k} must be in [0, 1]")
    if r["min_fp_pixels"] < 1:
        raise ValueError("rules.min_fp_pixels must be >= 1")
    if not 0 <= s["top_q"] < 1:
        raise ValueError("si.top_q must be in [0, 1)")
    for k in ("min_fp_pixels", "min_oil_pixels", "oil_pool_samples_per_scene", "min_oil_pool"):
        if not isinstance(s[k], int) or s[k] < 1:
            raise ValueError(f"si.{k} must be a positive integer")
    if s["den_floor"] <= 0 or s["clip"] <= 0:
        raise ValueError("si.den_floor and si.clip must be positive")
    if not math.isfinite(s["reflectance_offset"]) or s["reflectance_offset"] < 0:
        raise ValueError("si.reflectance_offset must be a finite non-negative DN offset")
    t = sel["target_count"]
    if not (t == "baseline" or (isinstance(t, int) and not isinstance(t, bool) and t > 0)):
        raise ValueError("selection.target_count must be 'baseline' or a positive integer")
    o = c["oof"]
    for k in ("inner_folds", "epochs", "batch_size", "classifier_neg_ratio", "neg_pool",
              "infer_batch_size"):
        if not isinstance(o[k], int) or o[k] < (2 if k == "inner_folds" else 1):
            raise ValueError(f"oof.{k} must be a positive integer")
    if not isinstance(o["workers"], int) or o["workers"] < 0 or o["lr"] <= 0:
        raise ValueError("Invalid oof.workers or oof.lr")
    sig = sel["scene_outlier_sigma"]
    if sig is not None and (not isinstance(sig, (int, float)) or sig <= 0):
        raise ValueError("selection.scene_outlier_sigma must be null or positive")
