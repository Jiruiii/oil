#!/usr/bin/env bash
# Evaluate G hnm_knn / G0 hnm_swap like the other arms: P(oil) histograms + matched operating
# points, P(oil) per GT oil component + lost-oil analysis, and the diagnostics comparison.
# Arms A-F, R reuse their existing outputs; finished G/G0 folds are skipped on a re-run.
# Output: hnm_runs/knn_filter/{threshold_sweep,lost_oil,diagnostics}/ (+ diagnostics_v2/{G,G0}).
# Usage: setsid nohup tools/run_knn_eval.sh > hnm_runs/knn_eval.log 2>&1 &
set -e
cd "$(dirname "$0")/.."
OUT=hnm_runs/knn_filter
PY=.venv/bin/python

run_dirs() {   # $1 = result-seg subfolder
  local f
  for f in 1 2 3; do
    echo "--run-dir" "$f=$(ls -d result-seg/$1/*_run_fold$f)"
  done
}
G_DIRS=$(run_dirs CV_gt_expand_hnm_si_v1_hnm_knn)
G0_DIRS=$(run_dirs CV_gt_expand_hnm_si_v1_hnm_swap)

for arm in G G0; do
  dirs=$G_DIRS; [ $arm = G0 ] && dirs=$G0_DIRS
  echo "[$(date -Is)] $arm threshold sweep"
  $PY tools/threshold_sweep.py --quiet --out "$OUT/threshold_sweep/$arm" $dirs
  echo "[$(date -Is)] $arm component P(oil)"
  $PY tools/oil_component_prob.py --quiet --out "$OUT/lost_oil/$arm" $dirs
  echo "[$(date -Is)] $arm diagnostics"
  mkdir -p hnm_runs/diagnostics_v2/$arm
  $PY tools/analyze_baseline_fp.py --config preprocess/hnm_si/config.example.yaml --quiet \
    --skip-backgrounds $dirs --out hnm_runs/diagnostics_v2/$arm > hnm_runs/diagnostics_v2/$arm/run.log 2>&1
done

echo "[$(date -Is)] comparisons"
$PY tools/compare_threshold_sweep.py --out $OUT/threshold_sweep --ref A \
  $(for a in A B C D E F R; do echo --arm $a=hnm_runs/threshold_sweep/$a; done) \
  --arm G=$OUT/threshold_sweep/G --arm G0=$OUT/threshold_sweep/G0
$PY tools/analyze_lost_oil.py --out $OUT/lost_oil --ref A \
  $(for a in A B C D; do echo --arm $a=hnm_runs/lost_oil/$a; done) \
  --arm G=$OUT/lost_oil/G --arm G0=$OUT/lost_oil/G0
$PY tools/compare_hnm_arms.py --out $OUT/diagnostics \
  $(for a in A B C D E F R G G0; do echo --arm $a=hnm_runs/diagnostics_v2/$a; done)
echo "[$(date -Is)] ALL DONE"
