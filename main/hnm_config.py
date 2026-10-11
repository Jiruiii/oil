"""Validate completed HNM artifacts before applying them to a frozen CV fold."""
import json
from pathlib import Path

from preprocess.hnm.data import read_background, sha256, split_manifest


def apply_hnm_to_fold(config, fold):
    hnm = config.get("hnm", {})
    if not hnm.get("enabled", False):
        return
    dataset = config["dataset"]
    if (dataset.get("bg_pool_cfg") or {}).get("draws_freeze_csv"):
        raise ValueError("HNM cannot be combined with the rotating background sampler")
    artifact = Path(hnm["artifacts_dir"]) / f"fold_{fold}"
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ValueError("HNM artifact is incomplete or has an unsupported schema")
    if manifest["outer_fold"] != fold:
        raise ValueError("HNM outer fold mismatch")
    for s in ("train", "val", "test"):
        if sha256(dataset[f"{s}_txt"]) != manifest["split_sha256"][s]:
            raise ValueError(f"HNM {s} split differs from the frozen mining split")
    source = dataset["bg_coord_txt"]
    if sha256(source) != manifest["baseline_bg_sha256"]:
        raise ValueError("HNM baseline background manifest changed")
    mined = artifact / "bg_coords.tsv"
    selected_path = artifact / "selected_train_bg.tsv"
    if (sha256(mined) != manifest["bg_coords_sha256"] or
            sha256(selected_path) != manifest["selected_train_bg_sha256"]):
        raise ValueError("HNM output checksum mismatch")
    mining_config = manifest["config"]
    for k, source_key in (("vrt_dir", "vrt_dir"), ("labels_dir", "mask_dir")):
        if Path(dataset[k]).resolve() != Path(mining_config[source_key]).resolve():
            raise ValueError(f"HNM data location differs: {k}")
    if int(dataset.get("patch_size", config.get("imgsz", 256))) != mining_config["patch_size"]:
        raise ValueError("HNM patch size differs from segmentation patch size")
    _, scenes = split_manifest(config["cross_validation"]["split_dir"], fold)
    original = read_background(source)
    combined = read_background(mined)
    selected = read_background(selected_path)
    if not selected or any(r.scene not in scenes["train"] for r in selected):
        raise ValueError("Selected HNM backgrounds must all belong to this training split")
    if len(selected) != manifest["selected_count"]:
        raise ValueError("HNM selected count mismatch")
    if [r for r in combined if r.scene not in scenes["train"]] != [
            r for r in original if r.scene not in scenes["train"]]:
        raise ValueError("HNM changed non-training backgrounds")
    if [r for r in combined if r.scene in scenes["train"]] != selected:
        raise ValueError("Combined HNM training backgrounds differ from selected manifest")
    dataset["bg_coord_txt"] = str(mined.resolve())
    dataset["hnm_manifest"] = str((artifact / "manifest.json").resolve())
    print(f"[HNM] Fold {fold}: {len(selected)} training backgrounds; "
          "validation/test backgrounds unchanged")
