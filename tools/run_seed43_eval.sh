#!/usr/bin/env bash
# Evaluate the seed-43 reruns of C hnm / G hnm_knn / G0 hnm_swap (RUN_LOG 2026-10-29 (2)) with the
# same steps as tools/run_knn_eval.sh, then compare them with the seed-42 A and B outputs.
# In the comparisons C, G, G0 are the seed-43 runs; A, B are unchanged (seed 42).
# Output: hnm_runs/seed43/{threshold_sweep,lost_oil,diagnostics_v2,diagnostics}/; finished arms are not skipped.
# Usage: setsid nohup tools/run_seed43_eval.sh > hnm_runs/seed43_eval.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
OUT=hnm_runs/seed43
PY=.venv/bin/python

run_dirs() {   # $1 = result-seg subfolder
  local f
  for f in 1 2 3; do
    echo "--run-dir" "$f=$(ls -d result-seg/$1/*_run_fold$f)"
  done
}

for arm in G0 G C; do
  case $arm in
    C) dirs=$(run_dirs CV_gt_expand_hnm_si_v1_hnm_seed43) ;;
    G) dirs=$(run_dirs CV_gt_expand_hnm_si_v1_hnm_knn_seed43) ;;
    G0) dirs=$(run_dirs CV_gt_expand_hnm_si_v1_hnm_swap_seed43) ;;
  esac
  echo "[$(date -Is)] $arm threshold sweep"
  $PY tools/threshold_sweep.py --quiet --out "$OUT/threshold_sweep/$arm" $dirs
  echo "[$(date -Is)] $arm component P(oil)"
  $PY tools/oil_component_prob.py --quiet --out "$OUT/lost_oil/$arm" $dirs
  echo "[$(date -Is)] $arm diagnostics"
  mkdir -p $OUT/diagnostics_v2/$arm
  $PY tools/analyze_baseline_fp.py --config preprocess/hnm_si/config.example.yaml --quiet \
    --skip-backgrounds $dirs --out $OUT/diagnostics_v2/$arm > $OUT/diagnostics_v2/$arm/run.log 2>&1
done

echo "[$(date -Is)] comparisons"
$PY tools/compare_threshold_sweep.py --out $OUT/threshold_sweep --ref A \
  --arm A=hnm_runs/threshold_sweep/A --arm B=hnm_runs/threshold_sweep/B \
  $(for a in C G G0; do echo --arm $a=$OUT/threshold_sweep/$a; done)
$PY tools/analyze_lost_oil.py --out $OUT/lost_oil --ref A \
  --arm A=hnm_runs/lost_oil/A --arm B=hnm_runs/lost_oil/B \
  $(for a in C G G0; do echo --arm $a=$OUT/lost_oil/$a; done)
$PY tools/compare_hnm_arms.py --out $OUT/diagnostics \
  --arm A=hnm_runs/diagnostics_v2/A --arm B=hnm_runs/diagnostics_v2/B \
  $(for a in C G G0; do echo --arm $a=$OUT/diagnostics_v2/$a; done)
echo "[$(date -Is)] ALL DONE"
