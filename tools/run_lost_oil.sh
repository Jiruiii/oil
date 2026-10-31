#!/usr/bin/env bash
# P(oil) per GT oil component (tools/oil_component_prob.py) for A-D, then which oil the HNM
# arms lose (tools/analyze_lost_oil.py). Finished folds are skipped on a re-run.
# Usage: setsid nohup tools/run_lost_oil.sh > hnm_runs/lost_oil.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
OUT=hnm_runs/lost_oil
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
infer() {   # $1 = arm, rest = --run-dir arguments
  local arm=$1; shift
  echo "[$(date -Is)] $arm start"
  $PY tools/oil_component_prob.py --quiet --out "$OUT/$arm" "$@"
  echo "[$(date -Is)] $arm done"
}

infer A $(baseline_dirs)
infer B $(run_dirs CV_gt_expand_hnm_si_v1_random)
infer C $(run_dirs CV_gt_expand_hnm_si_v1_hnm)
infer D $(run_dirs CV_gt_expand_hnm_si_v1_hnm_si)
echo "[$(date -Is)] INFERENCE DONE"
$PY tools/analyze_lost_oil.py --out $OUT --ref A $(for a in A B C D; do echo --arm $a=$OUT/$a; done)
echo "[$(date -Is)] ALL DONE"
