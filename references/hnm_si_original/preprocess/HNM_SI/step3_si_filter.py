"""
HNM_SI Step 3 — 套 SI distance cutoff，輸出 filtered candidate pool。

讀 Step 2 (step2_si_distance.py) 輸出的 si_distance_per_candidate.csv，
按 per-fold quantile 套兩道閾值剔除：
  - Top safety cutoff (預設 5%)：剔除距離最小（最像油的）candidates → 防漏標油汙染 HN pool
  - Bottom too-easy cutoff (預設關閉)：剔除距離最大（最不像油的）candidates → 集中 hard cases
    ⚠ FP-LocalBG ≈ 0 證據顯示多數 FP 是空間紋理錯，SI 上跟 BG 同質。
       啟用 bottom cutoff 會誤砍 spatial-pattern FP（DeepLab 真正需要學的）→ 風險高。

輸出（per fold）：
  cfhnm_within_si_filtered_foldN.txt
    每行：<scene>_patch_x<x>_y<y>\\t<role>\\t<n_fp_px>\\t<n_mask255_px>\\t<mean_oil_prob>\\t<max_oil_prob>\\t<d_si>
  cfhnm_within_si_filter_summary.csv  (per fold filter 統計)

Usage:
  conda run -n yolo11 python step3_si_filter.py \\
      --phase0_csv <si_distance_per_candidate.csv> \\
      --out_dir   <filtered output dir> \\
      [--top_q 0.05]    # safety: 剔最像油的 top X%, 預設 5%; 設 0 即關閉
      [--bot_q 0]       # too-easy: 剔最不像油的 bottom X%, 預設 0 (關閉); 設 0.5 即剔 50%
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase0_csv', required=True, type=Path,
                    help='Phase 0 si_distance_per_candidate.csv path')
    ap.add_argument('--out_dir', required=True, type=Path,
                    help='output dir for filtered candidate txt + summary')
    ap.add_argument('--top_q', type=float, default=0.05,
                    help='safety cutoff: remove top X%% closest to oil (default 0.05; set 0 to disable)')
    ap.add_argument('--bot_q', type=float, default=0.0,
                    help='too-easy cutoff: remove bottom X%% farthest from oil (default 0=disabled)')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.phase0_csv)
    print(f'[load] {len(df):,} candidates from {args.phase0_csv}')

    # 只保留 fingerprint_valid 的 candidates
    df = df[df['fingerprint_valid'] == True].reset_index(drop=True)
    print(f'[load] {len(df):,} with valid fingerprint')

    summary_rows = []
    for fold in sorted(df['fold'].unique()):
        sub = df[df['fold'] == fold].copy()
        n_before = len(sub)

        # Per-fold quantiles
        d_top_cutoff = sub['d_si'].quantile(args.top_q) if args.top_q > 0 else -np.inf
        d_bot_cutoff = sub['d_si'].quantile(1.0 - args.bot_q) if args.bot_q > 0 else np.inf

        # Filter: keep candidates with top_cutoff < d_si < bot_cutoff
        keep_mask = (sub['d_si'] > d_top_cutoff) & (sub['d_si'] < d_bot_cutoff)
        kept = sub[keep_mask].reset_index(drop=True)
        n_after = len(kept)

        n_dropped_top = int((sub['d_si'] <= d_top_cutoff).sum()) if args.top_q > 0 else 0
        n_dropped_bot = int((sub['d_si'] >= d_bot_cutoff).sum()) if args.bot_q > 0 else 0

        print(f'\n=== Fold {fold} ===')
        print(f'  before: {n_before:,}')
        if args.top_q > 0:
            print(f'  top {args.top_q*100:.0f}% safety cutoff (d<{d_top_cutoff:.4f}): dropped {n_dropped_top:,}')
        if args.bot_q > 0:
            print(f'  bottom {args.bot_q*100:.0f}% too-easy cutoff (d>{d_bot_cutoff:.4f}): dropped {n_dropped_bot:,}')
        print(f'  after: {n_after:,} ({n_after/n_before*100:.1f}%)')

        # Write filtered pool
        out_txt = args.out_dir / f'cfhnm_within_si_filtered_fold{fold}.txt'
        with open(out_txt, 'w') as f:
            for _, r in kept.iterrows():
                f.write(f"{r['patch_id']}\t{r['role']}\t{int(r['n_fp_px'])}\t"
                        f"{int(r['n_mask255_px'])}\t{r['mean_oil_prob']:.4f}\t"
                        f"{r['max_oil_prob']:.4f}\t{r['d_si']:.6f}\n")
        print(f'  wrote {out_txt.name}')

        summary_rows.append({
            'fold': fold,
            'n_before': n_before,
            'top_q': args.top_q,
            'top_cutoff': d_top_cutoff if args.top_q > 0 else None,
            'n_dropped_top': n_dropped_top,
            'bot_q': args.bot_q,
            'bot_cutoff': d_bot_cutoff if args.bot_q > 0 else None,
            'n_dropped_bot': n_dropped_bot,
            'n_after': n_after,
            'keep_ratio': round(n_after / n_before, 4) if n_before > 0 else 0,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.out_dir / 'cfhnm_within_si_filter_summary.csv', index=False)
    print('\n=== Summary ===')
    print(summary_df.to_string(index=False))
    print(f'\n[Done] outputs at {args.out_dir}')


if __name__ == '__main__':
    main()
