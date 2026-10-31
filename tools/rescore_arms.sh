#!/usr/bin/env bash
# Re-score arms A-F with tools/analyze_baseline_fp.py (adds Boundary IoU atoms; all other
# numbers are unchanged). Two arms at a time at low priority, so it can share the machine
# with a training run. Output: hnm_runs/diagnostics_v2/<arm>/.
# Usage: nohup tools/rescore_arms.sh > hnm_runs/diagnostics_v2.log 2>&1 &
cd "$(dirname "$0")/.."
CFG=preprocess/hnm_si/config.example.yaml
OUT=hnm_runs/diagnostics_v2

run_arm() {
  local name=$1; shift
  mkdir -p "$OUT/$name"
  nice -n 10 .venv/bin/python tools/analyze_baseline_fp.py --config $CFG --quiet \
    --skip-backgrounds "$@" --out "$OUT/$name" > "$OUT/$name/run.log" 2>&1
  echo "[$(date -Is)] $name exit $?"
}

run_dirs() {
  local f
  for f in 1 2 3; do
    echo "--run-dir" "$f=$(ls -d result-seg/CV_gt_expand_hnm_si_v1_$1/*_run_fold$f)"
  done
}

run_arm A &
run_arm B $(run_dirs random) &
wait
run_arm C $(run_dirs hnm) &
run_arm D $(run_dirs hnm_si) &
wait
run_arm E $(run_dirs oof) &
run_arm F $(run_dirs oof_si) &
wait
echo "[$(date -Is)] ALL DONE"
