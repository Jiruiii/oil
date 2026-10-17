#!/usr/bin/env python3
"""
run_HNM_SI.py — HNM_SI 四步 pipeline 的一鍵 driver。

把 step1 → step2 → step3 → step4 串成單一入口:
  - ★ 集中設定所有路徑與權重(見下方 CONFIG 區),透過 CLI 傳給各 step
    → 換環境只改本檔頂端 CONFIG,不用動 step1/step2/step4 內的常數
  - 集中設定超參數(top_q / bot_q / ratio / folds / device)
  - 由單一 RUN_ROOT 自動推導四步的輸入/輸出路徑接線(不用手動填四次路徑)
  - Preflight: 跑前檢查所有輸入路徑 + 5 個 fold 權重是否存在
  - 冪等 skip: 某步輸出已齊全就跳過(--force 強制重跑)
  - --steps 子集: 只重跑後段(例如調 top_q/ratio 時 --steps 3 4)

執行:
  # 常用: 自動命名,run-root 與 final-split 皆自動落到 CONFIG 的 ROOT 底下
  #       {YYMMDD}_{NN}[_{tag}] 目錄(NN 當日流水,避免覆蓋前次結果)
  conda run -n yolo11 python run_HNM_SI.py --ratio 0.5 --top-q 0.05
  # → run-root      : <RUN_ROOT_BASE>/260729_01_r050
  # → final-split   : <FINAL_SPLIT_ROOT>/260729_01_r050

  # 想要別的 tag:
  conda run -n yolo11 python run_HNM_SI.py --ratio 0.5 --tag ablation_topq10 --top-q 0.10

  # 只重跑 step3、step4(改了 top_q / ratio),明確指定既存 run-root:
  conda run -n yolo11 python run_HNM_SI.py --run-root <既存 dir> --steps 3 4 --top-q 0.05

執行目錄佈局(單一 RUN_ROOT):
  <RUN_ROOT>/
  ├── step1_mined/       step1 --out_dir
  ├── step2_si/          step2 --out_dir
  ├── step3_filtered/    step3 --out_dir
  ├── final_split/       step4 最終產物  ← 最終訓練用 train/val/test_foldN.txt
  └── run.log            orchestration 事件記錄(各 step 的 tqdm 進度仍直接串到終端機)

  最終 split 預設落到 <FINAL_SPLIT_ROOT>/<自動命名>(供訓練 yaml 直接讀)。
  也可用 --final-split-dir 或 CONFIG 的 FINAL_SPLIT_DIR 手動指定。

注意: --folds 子集只作用於 step1/step2(sliding-window 推論)。step4 內部硬編碼 [1..5],
      完整 pipeline 仍需五個 fold 都齊全。
"""
import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ==========================================
# ★ 資料來源路徑(單一設定點 — 換環境只改這裡,會由 CLI 傳給各 step)
# ==========================================
STACKED_TIF_DIR = '/home2/sun/OIL_PROJECT_dataset/full_band/processed/stack_tif'      # 11-band 影像(input)
MASK_DIR        = '/home2/sun/OIL_PROJECT_dataset/full_band/mask'                     # class-index mask(label)
SCENE_LEVEL_DIR = '/mnt/backups/sun/OIL_PROJECT_dataset/full_band/data_split_MS5/5_fold/scene_level'
PATCH_LEVEL_DIR = '/mnt/backups/sun/OIL_PROJECT_dataset/full_band/data_split_MS5/5_fold/patch_level_GB1.0'

# 每個 fold 的 DeepLab baseline probe(探勘模型;由 main/main_runner.py 事先訓練產出)
FOLD_BEST_PT = {
    1: '/mnt/backups/sun/OIL_PROJECT_MutiBand_0422_VRT_training/result-seg/CV_MS5/20260512-231439_MS5_CV_deeplabv3+_GB1.0_baseline_fold1/weights/best.pt',
    2: '/mnt/backups/sun/OIL_PROJECT_MutiBand_0422_VRT_training/result-seg/CV_MS5/20260513-173838_MS5_CV_deeplabv3+_GB1.0_baseline_fold2/weights/best.pt',
    3: '/mnt/backups/sun/OIL_PROJECT_MutiBand_0422_VRT_training/result-seg/CV_MS5/20260514-013955_MS5_CV_deeplabv3+_GB1.0_baseline_fold3/weights/best.pt',
    4: '/mnt/backups/sun/OIL_PROJECT_MutiBand_0422_VRT_training/result-seg/CV_MS5_rerun24/20260516-171513_MS5_CV_deeplabv3+_GB1.0_baseline_rerun_fold4/weights/best.pt',
    5: '/mnt/backups/sun/OIL_PROJECT_MutiBand_0422_VRT_training/result-seg/CV_MS5/20260514-034340_MS5_CV_deeplabv3+_GB1.0_baseline_fold5/weights/best.pt',
}

# 最終 split 輸出位置(step4 產物)。若明確設路徑會 override 下方自動命名邏輯。
# None = 走「自動命名」: <FINAL_SPLIT_ROOT>/<自動名稱>(見下方)。
FINAL_SPLIT_DIR = None

# ==========================================
# 自動命名輸出目錄的根位置
#   實際目錄 = <ROOT>/{YYMMDD}_{NN}[_{tag}]
#   NN     = 當日 2 碼流水號(01,02,…),跨兩個 ROOT 取最大 + 1
#   tag    = --tag 指定的短描述;省略時自動用 r{ratio*100:03d}
#   → 每次執行自動遞增,不會覆蓋前次結果
# ==========================================
RUN_ROOT_BASE    = Path('/mnt/backups/sun/result-HNM_SI')
FINAL_SPLIT_ROOT = Path('/mnt/backups/sun/OIL_PROJECT_dataset/full_band/data_split_MS5/5_fold')

# ==========================================
# 預設超參數(可被 argparse 覆蓋)
# ==========================================
DEFAULT_FOLDS  = [1, 2, 3, 4, 5]
DEFAULT_DEVICE = 'cuda'
DEFAULT_TOP_Q  = 0.05   # step3 safety cutoff(剔最像油的 top X%)
DEFAULT_BOT_Q  = 0.0    # step3 too-easy cutoff(剔最不像油的 top X%, 預設關)
DEFAULT_RATIO  = 0.5    # step4 HN:oil 採樣比例

STEP_FILES = {
    1: HERE / 'step1_mine_fp.py',
    2: HERE / 'step2_si_distance.py',
    3: HERE / 'step3_si_filter.py',
    4: HERE / 'step4_assemble.py',
}


# ==========================================
# 工具
# ==========================================
def _auto_run_name(tag: str) -> str:
    """產生 {YYMMDD}_{NN}[_{tag}]。NN 從 RUN_ROOT_BASE 與 FINAL_SPLIT_ROOT 中
    今日已存在的目錄名找最大 2 碼流水 + 1(從 01 起),避免覆蓋前次結果。"""
    date_str = datetime.now().strftime('%y%m%d')
    pattern = re.compile(rf'^{date_str}_(\d{{2}})')
    used = []
    for base in (RUN_ROOT_BASE, FINAL_SPLIT_ROOT):
        if base.is_dir():
            for d in base.iterdir():
                m = pattern.match(d.name)
                if m:
                    used.append(int(m.group(1)))
    nn = max(used) + 1 if used else 1
    if nn > 99:
        raise RuntimeError(f'今日流水號 {nn} 超過 99, 請整理 {RUN_ROOT_BASE} 或 {FINAL_SPLIT_ROOT}')
    name = f'{date_str}_{nn:02d}'
    if tag:
        name = f'{name}_{tag}'
    return name


class Logger:
    """同時寫終端機與 run.log 的 orchestration 事件記錄(不攔截各 step 的 stdout)。"""
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.fh = open(log_path, 'a', encoding='utf-8')

    def __call__(self, msg=''):
        print(msg)
        self.fh.write(msg + '\n')
        self.fh.flush()

    def close(self):
        self.fh.close()


# ==========================================
# Preflight
# ==========================================
def preflight(steps, folds, log):
    """跑前快速失敗檢查(全部對本檔 CONFIG,單一來源)。回傳 True=通過,False=有缺。"""
    log('\n' + '=' * 70)
    log('  Preflight 檢查')
    log('=' * 70)

    ok = True

    # 0) step4 內部硬編碼 [1..5],fold 子集會在 load_filtered 找不到檔案而 crash → 先擋
    if 4 in steps and folds != [1, 2, 3, 4, 5]:
        log(f'\n  ❌ step4 內部固定跑 fold 1..5,但 --folds={folds} 為子集。')
        log(f'       請完整跑五個 fold,或把 step4 從 --steps 移除。')
        ok = False

    # 1) 5 個 fold 權重存在(只在會跑 step1/step2 時檢查)
    if 1 in steps or 2 in steps:
        log('\n[1] FOLD_BEST_PT 權重存在')
        for f in folds:
            p = FOLD_BEST_PT.get(f)
            if not p:
                log(f'  ❌ fold{f}: CONFIG 的 FOLD_BEST_PT 未定義')
                ok = False
            elif not Path(p).exists():
                log(f'  ❌ fold{f}: best.pt 不存在 → {p}')
                ok = False
            else:
                log(f'  ✓ fold{f}: {p}')

    # 2) 外部輸入目錄存在
    log('\n[2] 外部輸入目錄(由 HNM_SI 以外的前置程式產出)')
    for name, d in [('mask/', MASK_DIR), ('stack_tif/', STACKED_TIF_DIR),
                    ('scene_level/', SCENE_LEVEL_DIR), ('patch_level/', PATCH_LEVEL_DIR)]:
        if Path(d).is_dir():
            log(f'  ✓ {name:14s} {d}')
        else:
            log(f'  ❌ {name:14s} 不存在 → {d}')
            ok = False

    # 3) step1/step2 讀 scene_level train/val txt + patch_level train txt(算 mean/std)
    if 1 in steps or 2 in steps:
        for f in folds:
            for role in ('train', 'val'):
                p = Path(SCENE_LEVEL_DIR) / f'{role}_fold{f}.txt'
                if not p.exists():
                    log(f'  ❌ 缺 scene_level: {p.name}')
                    ok = False
            p = Path(PATCH_LEVEL_DIR) / f'train_fold{f}.txt'
            if not p.exists():
                log(f'  ❌ 缺 patch_level: {p.name}(mean/std 取樣用)')
                ok = False

    # 4) step4 需要 list_oil.txt(oil 基數計算用)
    if 4 in steps:
        p = Path(PATCH_LEVEL_DIR) / 'list_oil.txt'
        if not p.exists():
            log(f'  ❌ 缺 {p}(step4 oil 基數計算用)')
            ok = False

    log('\n' + ('  Preflight 通過。' if ok else '  ⚠ Preflight 失敗,請先補齊上列缺項。'))
    return ok


# ==========================================
# Step 執行
# ==========================================
def _expected_outputs(step, dirs, folds):
    """回傳該 step 預期產出的關鍵檔案清單(用於冪等 skip 判斷)。"""
    if step == 1:
        return [dirs['step1'] / f'cfhnm_within_mined_fold{f}.txt' for f in folds]
    if step == 2:
        return [dirs['step2'] / 'si_distance_per_candidate.csv']
    if step == 3:
        return [dirs['step3'] / f'cfhnm_within_si_filtered_fold{f}.txt' for f in folds]
    if step == 4:
        return ([dirs['final'] / f'train_fold{f}.txt' for f in folds]
                + [dirs['final'] / f'val_fold{f}.txt' for f in folds])
    return []


def _build_cmd(step, dirs, args):
    """組各 step 的命令列(用當前 python 直接跑,tqdm 進度即時串到終端機)。
    所有資料路徑/權重由本檔 CONFIG 集中傳入,step 端不再各自 hardcode。"""
    py = sys.executable
    folds_arg = [str(f) for f in args.folds]
    # 選定 fold 的權重覆蓋參數: "N=/path/to/best.pt"
    fold_pt_arg = [f'{f}={FOLD_BEST_PT[f]}' for f in args.folds]
    if step == 1:
        return [py, str(STEP_FILES[1]),
                '--out_dir', str(dirs['step1']),
                '--folds', *folds_arg,
                '--device', args.device,
                '--mask_dir', MASK_DIR,
                '--stacked_tif_dir', STACKED_TIF_DIR,
                '--scene_level_dir', SCENE_LEVEL_DIR,
                '--patch_level_dir', PATCH_LEVEL_DIR,
                '--fold_best_pt', *fold_pt_arg]
    if step == 2:
        return [py, str(STEP_FILES[2]),
                '--step1_dir', str(dirs['step1']),
                '--out_dir', str(dirs['step2']),
                '--folds', *folds_arg,
                '--device', args.device,
                '--mask_dir', MASK_DIR,
                '--stacked_tif_dir', STACKED_TIF_DIR,
                '--patch_level_dir', PATCH_LEVEL_DIR,
                '--fold_best_pt', *fold_pt_arg]
    if step == 3:
        return [py, str(STEP_FILES[3]),
                '--phase0_csv', str(dirs['step2'] / 'si_distance_per_candidate.csv'),
                '--out_dir', str(dirs['step3']),
                '--top_q', str(args.top_q),
                '--bot_q', str(args.bot_q)]
    if step == 4:
        return [py, str(STEP_FILES[4]),
                '--filtered_dir', str(dirs['step3']),
                '--out_split_dir', str(dirs['final']),
                '--ratio', str(args.ratio),
                '--patch_level_src', PATCH_LEVEL_DIR,
                '--scene_level', SCENE_LEVEL_DIR]
    raise ValueError(step)


def run_step(step, dirs, args, log):
    """跑單一 step。回傳 True=成功/跳過,False=失敗(呼叫端中止)。"""
    outputs = _expected_outputs(step, dirs, args.folds)
    if not args.force and outputs and all(p.exists() for p in outputs):
        log(f'\n[step{step}] 輸出已齊全 → 跳過(--force 可強制重跑)')
        return True

    cmd = _build_cmd(step, dirs, args)
    log('\n' + '=' * 70)
    log(f'  STEP {step}  |  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    log(f'  $ {" ".join(cmd)}')
    log('=' * 70)

    t0 = time.time()
    # 不攔截 stdout/stderr → 各 step 的 tqdm 進度條即時顯示
    proc = subprocess.run(cmd)
    dt = time.time() - t0

    if proc.returncode != 0:
        log(f'\n[step{step}] ❌ 失敗(returncode={proc.returncode}),耗時 {dt:.1f}s。中止 pipeline。')
        return False
    log(f'\n[step{step}] ✓ 完成,耗時 {dt/60:.1f} 分。')
    return True


# ==========================================
# 主程式
# ==========================================
def main():
    ap = argparse.ArgumentParser(
        description='HNM_SI 四步 pipeline driver(step1→step2→step3→step4)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--run-root', type=Path, default=None,
                    help='本次 run 的根目錄。不給則自動命名為 '
                         '<RUN_ROOT_BASE>/{YYMMDD}_{NN}[_{tag}](當日流水,不覆蓋前次)')
    ap.add_argument('--final-split-dir', type=Path, default=None, dest='final_split_dir',
                    help='最終 split(step4 產物)輸出位置。不給則自動命名為 '
                         '<FINAL_SPLIT_ROOT>/{YYMMDD}_{NN}[_{tag}],與 run-root 同名'
                         '(讓中間產物與最終 split 一眼對得起來)')
    ap.add_argument('--tag', default=None,
                    help='自動命名的 tag 尾綴(可省;省略時自動用 r{ratio*100:03d},'
                         '例如 ratio=0.5 → r050)')
    ap.add_argument('--steps', nargs='+', type=int, default=[1, 2, 3, 4],
                    choices=[1, 2, 3, 4], help='要執行的步驟子集')
    ap.add_argument('--folds', nargs='+', type=int, default=DEFAULT_FOLDS,
                    help='step1/step2 的 fold 子集(step4 內部固定 1..5)')
    ap.add_argument('--device', default=DEFAULT_DEVICE)
    ap.add_argument('--top-q', type=float, default=DEFAULT_TOP_Q, dest='top_q',
                    help='step3 safety cutoff(剔最像油的 top X%%)')
    ap.add_argument('--bot-q', type=float, default=DEFAULT_BOT_Q, dest='bot_q',
                    help='step3 too-easy cutoff(預設 0=關)')
    ap.add_argument('--ratio', type=float, default=DEFAULT_RATIO,
                    help='step4 HN:oil 採樣比例')
    ap.add_argument('--force', action='store_true',
                    help='忽略已存在輸出,強制重跑選定步驟')
    ap.add_argument('--skip-preflight', action='store_true',
                    help='略過 preflight 檢查(不建議)')
    args = ap.parse_args()

    steps = sorted(set(args.steps))
    args.folds = sorted(set(args.folds))

    # 自動命名(當 --run-root / --final-split-dir 皆未給)
    if args.run_root is None or args.final_split_dir is None:
        tag = args.tag if args.tag else f'r{int(round(args.ratio * 100)):03d}'
        auto_name = _auto_run_name(tag)
        if args.run_root is None:
            args.run_root = RUN_ROOT_BASE / auto_name
        if args.final_split_dir is None:
            args.final_split_dir = FINAL_SPLIT_ROOT / auto_name

    run_root = args.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    # 最終 split 位置優先序: --final-split-dir > CONFIG FINAL_SPLIT_DIR > <RUN_ROOT>/final_split
    final_dir = args.final_split_dir or (Path(FINAL_SPLIT_DIR) if FINAL_SPLIT_DIR else run_root / 'final_split')
    dirs = {
        'step1': run_root / 'step1_mined',
        'step2': run_root / 'step2_si',
        'step3': run_root / 'step3_filtered',
        'final': final_dir,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    log = Logger(run_root / 'run.log')
    log('\n' + '#' * 70)
    log(f'#  HNM_SI pipeline  |  {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    log(f'#  run_root : {run_root}')
    log(f'#  final    : {final_dir}')
    log(f'#  steps    : {steps}')
    log(f'#  folds    : {args.folds}')
    log(f'#  top_q={args.top_q}  bot_q={args.bot_q}  ratio={args.ratio}  device={args.device}')
    log(f'#  force    : {args.force}')
    log('#' * 70)

    try:
        if not args.skip_preflight:
            if not preflight(steps, args.folds, log):
                log('\n中止(preflight 未通過)。')
                sys.exit(1)

        for step in steps:
            if not run_step(step, dirs, args, log):
                sys.exit(1)

        log('\n' + '#' * 70)
        log(f'#  全部完成。最終 split 於: {dirs["final"]}')
        log('#' * 70)
    finally:
        log.close()


if __name__ == '__main__':
    main()
