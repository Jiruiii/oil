"""CLI: plan, validate, then mine within each frozen outer training split."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml

from .data import (inspect_rasters, label_gt_patches, oof_splits, raster_index, read_background,
                   read_groups, sha256, split_manifest, validate_labels,
                   write_background)
from .sampling import select_negatives


DEFAULTS = {
    "outer_folds": [1, 2, 3], "inner_folds": 5, "seed": 42,
    "patch_size": 256, "scale": 10000.0,
    "model_name": "tf_efficientnetv2_s", "pretrained": True,
    "epochs": 70, "batch_size": 8, "workers": 4,
    "lr": .0001, "weight_decay": .01, "classifier_neg_ratio": 5,
    "keep_fraction": .5, "hard_fraction": .5,
    "device": "cuda", "group_tsv": None,
}
PATH_KEYS = ("split_dir", "baseline_bg_tsv", "candidate_bg_tsv", "vrt_dir",
             "mask_dir", "output_dir", "group_tsv")


def load_config(path):
    path = Path(path).resolve()
    supplied = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(supplied, dict):
        raise ValueError("HNM config must be a mapping")
    allowed = set(DEFAULTS) | set(PATH_KEYS) | {"mean", "std"}
    if set(supplied) - allowed:
        raise ValueError(f"Unknown HNM settings: {sorted(set(supplied) - allowed)}")
    config = DEFAULTS | supplied
    config.setdefault("candidate_bg_tsv", config.get("baseline_bg_tsv"))
    for k in PATH_KEYS:
        if config.get(k):
            p = Path(config[k]).expanduser()
            config[k] = str((path.parent / p).resolve() if not p.is_absolute() else p)
        elif k != "group_tsv":
            raise ValueError(f"Missing setting: {k}")
    mean, std = config.get("mean", []), config.get("std", [])
    if not mean or len(mean) != len(std) or not np.isfinite(mean + std).all() or min(std) <= 0:
        raise ValueError("Provide finite per-band mean/std, with std > 0")
    for k in ("inner_folds", "patch_size", "epochs", "batch_size", "classifier_neg_ratio"):
        if not isinstance(config[k], int) or config[k] < (2 if k == "inner_folds" else 1):
            raise ValueError(f"Invalid {k}")
    for k in ("scale", "lr"):
        if not math.isfinite(config[k]) or config[k] <= 0:
            raise ValueError(f"Invalid {k}")
    if not 0 < config["keep_fraction"] <= 1 or not 0 <= config["hard_fraction"] <= 1:
        raise ValueError("keep_fraction must be (0,1], hard_fraction [0,1]")
    if config["workers"] < 0 or config["device"] not in ("cpu", "cuda"):
        raise ValueError("Invalid worker count or device")
    folds = config["outer_folds"]
    if not folds or len(set(folds)) != len(folds) or any(not isinstance(f, int) or f < 1 for f in folds):
        raise ValueError("outer_folds must contain unique positive integers")
    return config


def prepare_fold(config, fold):
    groups = read_groups(config["group_tsv"])
    splits, scenes = split_manifest(config["split_dir"], fold, groups)
    candidates = read_background(config["candidate_bg_tsv"])
    background = [p for p in candidates if p.scene in scenes["train"]]
    if not background:
        raise ValueError(f"No training background candidates for fold {fold}")
    gt_records = label_gt_patches(splits["train"], config["mask_dir"], config["patch_size"])
    records = gt_records + background
    if len({r.key for r in records}) != len(records):
        raise ValueError("A training coordinate appears as both positive and background")
    oof = oof_splits(records, config["inner_folds"], config["seed"] + fold, groups)
    index = raster_index(config["vrt_dir"])
    inspect_rasters(records, index, config["mask_dir"], len(config["mean"]))
    hashes = {s: sha256(Path(config["split_dir"]) / f"{s}_fold{fold}.txt")
              for s in ("train", "val", "test")}
    plan = {
        "schema_version": 1, "outer_fold": fold, "config": config,
        "implementation_sha256": {p.name: sha256(p) for p in
                                   sorted(Path(__file__).parent.glob("*.py"))},
        "split_sha256": hashes,
        "baseline_bg_sha256": sha256(config["baseline_bg_tsv"]),
        "candidate_bg_sha256": sha256(config["candidate_bg_tsv"]),
        "group_sha256": sha256(config["group_tsv"]) if config["group_tsv"] else None,
        "positive_count": sum(r.label for r in gt_records),
        "fixed_gt_background_count": sum(not r.label for r in gt_records),
        "candidate_count": len(background),
        "selected_count": max(1, int(len(background) * config["keep_fraction"])),
        "grouping": "explicit" if groups else "scene",
        "inner_folds": [{"fold": i + 1, "train_count": len(tr), "heldout_count": len(ho),
                         "train_scenes": sorted({records[j].scene for j in tr}),
                         "heldout_scenes": sorted({records[j].scene for j in ho})}
                        for i, (tr, ho) in enumerate(oof)],
    }
    return plan, records, background, oof, index, scenes


def mine_fold(config, fold, device):
    from .models import infer, train_miner
    plan, records, background, oof, index, scenes = prepare_fold(config, fold)
    out = Path(config["output_dir"]) / f"fold_{fold}"
    # Never silently reuse checkpoints after data or parameter changes.
    out.mkdir(parents=True, exist_ok=False)
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    validate_labels(records, config["mask_dir"], config["patch_size"], {r.key for r in background})
    scores = np.full(len(background), np.nan, dtype=np.float32)
    score_fold = np.zeros(len(background), dtype=int)
    bg_offset = len(records) - len(background)
    histories = {}
    for inner, (train, heldout) in enumerate(oof, 1):
        checkpoint = out / f"miner_oof_{inner}.pt"
        histories[f"oof_{inner}"] = train_miner(
            [records[i] for i in train], index, config, checkpoint,
            config["seed"] + 1000 * fold + inner, device)
        heldout_neg = [i for i in heldout if i >= bg_offset]
        if not heldout_neg:
            continue
        values = infer([records[i] for i in heldout_neg], index, config, checkpoint, device)
        bg_idx = np.array(heldout_neg) - bg_offset
        scores[bg_idx] = values
        score_fold[bg_idx] = inner
    if not np.isfinite(scores).all() or np.any(score_fold == 0):
        raise ValueError("Incomplete OOF background coverage")
    full_checkpoint = out / "miner_embeddings.pt"
    histories["embeddings"] = train_miner(records, index, config, full_checkpoint,
                                           config["seed"] + 1000 * fold, device)
    features = infer(background, index, config, full_checkpoint, device, embeddings=True)
    hard, diverse = select_negatives(scores, features, plan["selected_count"], config["hard_fraction"])
    selected_ids = set(hard + diverse)
    selected = [r for i, r in enumerate(background) if i in selected_ids]
    write_background(out / "selected_train_bg.tsv", selected)
    # Only train backgrounds change. Preserve all non-training baseline rows and their order.
    baseline = read_background(config["baseline_bg_tsv"])
    combined = [r for r in baseline if r.scene not in scenes["train"]] + selected
    write_background(out / "bg_coords.tsv", combined)
    np.save(out / "embeddings.npy", features)
    import csv
    with (out / "scores.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["scene", "x", "y", "oof_fold", "el2n", "bucket"])
        hard_set = set(hard)
        for i, r in enumerate(background):
            bucket = "hard" if i in hard_set else "diverse" if i in selected_ids else "unselected"
            writer.writerow([*r.key, score_fold[i], float(scores[i]), bucket])
    (out / "training_history.json").write_text(json.dumps(histories, indent=2))
    manifest = plan | {
        "status": "complete", "selected_count": len(selected),
        "bg_coords_sha256": sha256(out / "bg_coords.tsv"),
        "selected_train_bg_sha256": sha256(out / "selected_train_bg.tsv"),
        "hard_count": len(hard), "diverse_count": len(diverse),
        "method": "train_only_group_oof_el2n_and_common_embedding",
        "land_policy": "not_applied_no_land_mask",
    }
    # Written last: partial runs cannot be used for segmentation training.
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Fold {fold} complete: {len(selected)}/{len(background)} backgrounds -> {out}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("plan", "validate", "mine"), default="plan")
    parser.add_argument("--fold", type=int, help="Only process one configured outer fold")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    folds = [args.fold] if args.fold is not None else config["outer_folds"]
    if any(f not in config["outer_folds"] for f in folds):
        parser.error("--fold is not in configured outer_folds")
    device = None
    if args.stage == "mine":
        import torch
        device = torch.device(config["device"])
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use a GPU host or explicitly set device: cpu")
    for fold in folds:
        if args.stage == "mine":
            mine_fold(config, fold, device)
        else:
            plan, records, background, _, _, _ = prepare_fold(config, fold)
            if args.stage == "validate":
                validate_labels(records, config["mask_dir"], config["patch_size"],
                                {r.key for r in background})
            print(json.dumps({k: v for k, v in plan.items() if k != "config"},
                             ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
