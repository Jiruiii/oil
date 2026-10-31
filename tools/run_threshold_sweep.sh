#!/usr/bin/env bash
# P(oil) histograms for every arm (tools/threshold_sweep.py), then the matched-operating-point
# comparison (tools/compare_threshold_sweep.py). One arm at a time on the GPU; finished folds
# are skipped, so the script can be re-run after an interruption.
# Usage: setsid nohup tools/run_threshold_sweep.sh > hnm_runs/threshold_sweep.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
OUT=hnm_runs/threshold_sweep
PY=.venv/bin/python

run_dirs() {   # $1 = result-seg subfolder
  local f
  for f in 1 2 3; do
    echo "--run-dir" "$f=$(ls -d result-seg/$1/*_run_fold$f)"
  done
}
baseline_dirs() {
  $PY -c "import sys; sys.path.insert(0, '.')
from preprocess.hnm_si.config import load_config
d = load_config('preprocess/hnm_si/config.example.yaml')['probe_run_dirs']
print(' '.join(f'--run-dir {f}={d[f]}' for f in (1, 2, 3)))"
}
sweep() {   # $1 = arm, rest = --run-dir arguments
  local arm=$1; shift
  echo "[$(date -Is)] $arm start"
  $PY tools/threshold_sweep.py --quiet --out "$OUT/$arm" "$@"
  echo "[$(date -Is)] $arm done"
}

sweep A $(baseline_dirs)
sweep B $(run_dirs CV_gt_expand_hnm_si_v1_random)
sweep C $(run_dirs CV_gt_expand_hnm_si_v1_hnm)
sweep D $(run_dirs CV_gt_expand_hnm_si_v1_hnm_si)
sweep E $(run_dirs CV_gt_expand_hnm_si_v1_oof)
sweep F $(run_dirs CV_gt_expand_hnm_si_v1_oof_si)
sweep R $(run_dirs CV_gt_expand_ring8)
$PY tools/compare_threshold_sweep.py --out $OUT --ref A \
  $(for a in A B C D E F R; do echo --arm $a=$OUT/$a; done)
echo "[$(date -Is)] ALL DONE"
