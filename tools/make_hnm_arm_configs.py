"""Write main/local_hnm_<arm>.yaml for step 2 of HNM_EXPERIMENT_PLAN.md.

Each file is the baseline main/local_gt_expand.yaml with only these lines changed:
results_base_dir, excel_log_path, experiment_name, the post-test name, and an added
``hnm`` block pointing at hnm_runs/hnm_si_v1/arms/<arm>. Everything else (seed, optimizer,
loss, splits, val/test backgrounds, evaluation) stays line-for-line identical; the script
checks that by diffing against the baseline and refuses to write otherwise.

  .venv/bin/python tools/make_hnm_arm_configs.py            # every arm with complete artifacts
  .venv/bin/python tools/make_hnm_arm_configs.py --arm hnm_si --seed 43
"""
import argparse
import difflib
import json
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE = REPO_ROOT / "main" / "local_gt_expand.yaml"
ARMS = ("random", "hnm", "hnm_si", "oof", "oof_si", "hnm_knn", "hnm_swap")
ALLOWED = re.compile(r"^\s*(results_base_dir|excel_log_path|- experiment_name|- test_name|seed):")


def arm_config_text(base_text, arm, artifacts_root, seed=None):
    suffix = arm if seed is None else f"{arm}_seed{seed}"
    replacements = [
        (r"^(results_base_dir:\s*)'[^']*'",
         rf"\1'{REPO_ROOT}/result-seg/CV_gt_expand_hnm_si_v1_{suffix}'"),
        (r"^(excel_log_path:\s*)'[^']*'",
         rf"\1'{REPO_ROOT}/result_excel/result_CV_gt_expand_hnm_si_v1_{suffix}_log.xlsx'"),
        (r"^(\s*- experiment_name:\s*)'[^']*'", rf"\1'local_hnm_{suffix}_run'"),
        (r"^(\s*- test_name:\s*)'[^']*'", rf"\1'local_hnm_{suffix}_run_reconstruction'"),
    ]
    if seed is not None:
        replacements.append((r"^(\s*seed:\s*)\d+", rf"\g<1>{seed}"))
    text = base_text
    for pattern, repl in replacements:
        text, n = re.subn(pattern, repl, text, count=1, flags=re.M)
        if n != 1:
            raise ValueError(f"Baseline yaml lacks the line for {pattern!r}")
    block = (f"    hnm:\n      enabled: true\n"
             f"      artifacts_dir: '{Path(artifacts_root) / arm}'\n\n")
    anchor = "    cross_validation:\n"
    if text.count(anchor) != 1:
        raise ValueError("Cannot find the cross_validation block to anchor the hnm block")
    return text.replace(anchor, block + anchor)


def check_only_allowed_changes(base_text, text):
    added = []
    for line in difflib.unified_diff(base_text.splitlines(), text.splitlines(), lineterm="", n=0):
        if line.startswith(("---", "+++", "@@")):
            continue
        if line[0] == "-" and not ALLOWED.match(line[1:]):
            raise ValueError(f"Unexpected removed line: {line}")
        if line[0] == "+":
            added.append(line[1:])
    others = [l for l in added if l.strip() and not ALLOWED.match(l)]
    if len(others) != 3 or others[:2] != ["    hnm:", "      enabled: true"] or \
            not others[2].startswith("      artifacts_dir:"):
        raise ValueError(f"Unexpected added lines: {others}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", choices=ARMS, action="append")
    parser.add_argument("--seed", type=int, help="override the seed (step 4); default keeps 42")
    parser.add_argument("--artifacts-root", default=str(REPO_ROOT / "hnm_runs" / "hnm_si_v1" / "arms"))
    args = parser.parse_args(argv)
    base_text = BASE.read_text(encoding="utf-8")
    root = Path(args.artifacts_root)
    arms = args.arm or [a for a in ARMS if all(
        (root / a / f"fold_{f}" / "manifest.json").exists() for f in (1, 2, 3))]
    for arm in arms:
        for fold in (1, 2, 3):
            manifest = root / arm / f"fold_{fold}" / "manifest.json"
            if json.loads(manifest.read_text(encoding="utf-8")).get("status") != "complete":
                raise ValueError(f"{manifest} is not complete")
        text = arm_config_text(base_text, arm, root, args.seed)
        check_only_allowed_changes(base_text, text)
        cfg = yaml.safe_load(text)
        exp = cfg["experiments"][0]
        assert exp["hnm"] == {"enabled": True, "artifacts_dir": str(root / arm)}
        name = arm if args.seed is None else f"{arm}_seed{args.seed}"
        out = REPO_ROOT / "main" / f"local_hnm_{name}.yaml"
        out.write_text(text, encoding="utf-8")
        print(f"{out.relative_to(REPO_ROOT)}: {exp['experiment_name']} -> {cfg['results_base_dir']}")


if __name__ == "__main__":
    main()
