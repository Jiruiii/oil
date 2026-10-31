"""Write main/local_ring<N>.yaml: the baseline with oil_ignore_ring = N (train only).

Derived from main/local_gt_expand.yaml with only these changes: result paths, names, the
added ``dataset.oil_ignore_ring: N`` (unlabeled pixels within N px of GT oil are ignored by
the training loss; val/test scoring is unchanged) and the reconstruction's optional extra
metrics (background-tile FP and Boundary IoU). Refuses to write if anything else differs.

  .venv/bin/python tools/make_ring_config.py --ring 8
"""
import argparse
import difflib
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = REPO_ROOT / "main" / "local_gt_expand.yaml"
ALLOWED = re.compile(r"^\s*(results_base_dir|excel_log_path|- experiment_name|- test_name):")
ADDED = re.compile(r"^\s*(oil_ignore_ring|bg_tiles|boundary_iou_px):")


def ring_config_text(base_text, ring, boundary_px):
    name = f"ring{ring}"
    text = base_text
    for pattern, repl in (
            (r"^(results_base_dir:\s*)'[^']*'", rf"\1'{REPO_ROOT}/result-seg/CV_gt_expand_{name}'"),
            (r"^(excel_log_path:\s*)'[^']*'",
             rf"\1'{REPO_ROOT}/result_excel/result_CV_gt_expand_{name}_log.xlsx'"),
            (r"^(\s*- experiment_name:\s*)'[^']*'", rf"\1'local_{name}_run'"),
            (r"^(\s*- test_name:\s*)'[^']*'", rf"\1'local_{name}_run_reconstruction'"),
            (r"^(\s*)(bg_coord_txt:.*)$", rf"\1\2\n\1oil_ignore_ring: {ring}"),
            (r"^(\s*)(patch_size: 256)\s*$", rf"\1\2\n\1bg_tiles: true\n\1boundary_iou_px: {boundary_px}")):
        text, n = re.subn(pattern, repl, text, count=1, flags=re.M)
        if n != 1:
            raise ValueError(f"Baseline yaml lacks the line for {pattern!r}")
    return text


def check_only_allowed_changes(base_text, text):
    for line in difflib.unified_diff(base_text.splitlines(), text.splitlines(), lineterm="", n=0):
        if line.startswith(("---", "+++", "@@")):
            continue
        body = line[1:]
        if line[0] == "-" and not ALLOWED.match(body):
            raise ValueError(f"Unexpected removed line: {line}")
        if line[0] == "+" and body.strip() and not (ALLOWED.match(body) or ADDED.match(body)):
            raise ValueError(f"Unexpected added line: {line}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ring", type=int, required=True)
    parser.add_argument("--boundary-px", type=int, default=8)
    args = parser.parse_args(argv)
    base_text = BASE.read_text(encoding="utf-8")
    text = ring_config_text(base_text, args.ring, args.boundary_px)
    check_only_allowed_changes(base_text, text)
    exp = yaml.safe_load(text)["experiments"][0]
    recon = exp["post_tests"][0]["reconstruction"]
    assert exp["dataset"]["oil_ignore_ring"] == args.ring
    assert recon["bg_tiles"] is True and recon["boundary_iou_px"] == args.boundary_px
    out = REPO_ROOT / "main" / f"local_ring{args.ring}.yaml"
    out.write_text(text, encoding="utf-8")
    print(f"{out.relative_to(REPO_ROOT)}: {exp['experiment_name']}")


if __name__ == "__main__":
    main()
