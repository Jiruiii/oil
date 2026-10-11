# ======================================================================================
# ===       main/main_runner.py (基於您原始碼，加入記憶體管理的最終修正版)         ===
# ======================================================================================
import sys
from pathlib import Path
# ★ 將 main/ 加入 sys.path 尾端，讓本地 deeplab_mados/ 模組優先被 import
_MAIN_DIR = Path(__file__).resolve().parent.parent
if str(_MAIN_DIR) not in sys.path:
    sys.path.append(str(_MAIN_DIR))

import yaml
import argparse
import datetime
from collections import OrderedDict
import copy
import pandas as pd
# ⭐⭐⭐ 新增：匯入 torch 和 gc 模組 ⭐⭐⭐
import torch
import gc

class _FoldTee:
    """Tee stdout to a fold-specific log file while keeping the original stdout live."""
    def __init__(self, filepath, orig):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(filepath, 'w', buffering=1, encoding='utf-8')
        self.orig = orig
    def write(self, data):
        self.orig.write(data)
        self.file.write(data)
    def flush(self):
        self.orig.flush()
        self.file.flush()
    def fileno(self):
        return self.orig.fileno()
    def close(self):
        self.file.close()

# (您原有的模組匯入保持不變)
from training_module import train_model
from evaluation_module import evaluate_and_visualize
from reconstruct_module import run_reconstruction
from tracking_module import log_to_excel, log_run_json
from utils import get_image_counts, create_temp_data_yaml

import re as _re
import random as _random
import numpy as _np
from collections import defaultdict as _defaultdict
_PATCH_RE_MR = _re.compile(r'^(.+)_patch_x(\d+)_y(\d+)$')


def set_global_seed(seed: int):
    """[C1] 統一設 torch / numpy / random / cudnn，讓 fold 結果可重現。
    DataLoader worker 的 RNG 另外在 deeplab_adapter.py 內以 worker_init_fn 處理。
    """
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # [2026-07-21 可重現性修正] cudnn.deterministic 只鎖卷積；DeepLabV3+ 解碼器的
    # bilinear upsample 等 atomicAdd 反向仍非決定性，需 use_deterministic_algorithms。
    # warn_only=True：有決定性實作就用、沒有的只警告不中斷（雙跑仍完成可比對）。
    # CUBLAS_WORKSPACE_CONFIG 由啟動指令的環境變數設定（cublas 決定性前置）。
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as _e:
        print(f"  [Seed][warn] use_deterministic_algorithms 設定失敗: {_e}")
    print(f"  [Seed] global seed = {seed}（torch/numpy/random/cudnn/det-algos 已鎖）")


def _count_patches_pos_neg(txt_path, mask_dir, patch_size=256, bg_coord_txt=None,
                           bg_draws_freeze_csv=None):
    """掃 txt 內每個 patch 的 mask window，回傳 (pos, neg, oil_px, bg_px).
    pos = patch 內至少 1 個 oil 像素 (mask==0); neg = 沒有 oil。
    oil_px / bg_px 為跨所有 patch 的 pixel-level 累加 (ignore 255 不計)。
    [2026-07-06] bg_coord_txt 提供時，把該 split 場景（= txt 內出現的場景，
    比照訓練 SegmentationDataset 的 bg_scene_filter=本 split 場景）對應的
    bg_coords 背景 patch 也一併計入，讓 pos/neg 與 oil_ratio 反映真實訓練組成
    （txt 油汙中心 patch + bg_coords 背景 patch），而非只算 txt。
    """
    if not txt_path:
        return 0, 0, 0, 0
    txt_path = Path(txt_path)
    if not txt_path.exists():
        return 0, 0, 0, 0
    import rasterio
    from rasterio.windows import Window

    patches_by_scene = _defaultdict(list)
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = _PATCH_RE_MR.match(line)
        if not m:
            continue
        patches_by_scene[m.group(1)].append((int(m.group(2)), int(m.group(3))))

    # [2026-07-06] 併入本 split 場景的背景 patch；scene_set 於併入前定版，
    # 只收場景已在 txt 出現的 bg 座標（= 訓練用的 bg_scene_filter），避免跨 split 洩漏。
    # [P8][codex R5-5] bg_draws_freeze_csv 有給時，bg_coord_txt 是「池」而非「每 epoch 曝光量」：
    # 逐景只取前 base_A 筆（append-only 構造下即 Arm A 座標＝每 epoch draw 數），
    # 否則 7,720 行的池會冒充訓練曝光。池大小另行記錄，不混入 pos/neg。
    _draws = None
    if bg_draws_freeze_csv and Path(bg_draws_freeze_csv).exists():
        import csv as _csv
        with open(bg_draws_freeze_csv, newline='') as fh:
            _draws = {r['scene']: int(r['base_A']) for r in _csv.DictReader(fh)}
    if bg_coord_txt and Path(bg_coord_txt).exists():
        import csv
        scene_set = set(patches_by_scene.keys())
        _bg_by_scene = _defaultdict(list)
        with open(bg_coord_txt, 'r') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                s = row['scene']
                if s not in scene_set:
                    continue
                _bg_by_scene[s].append((int(row['x']), int(row['y'])))
        for s, pool in _bg_by_scene.items():
            if _draws is None:
                patches_by_scene[s].extend(pool)
                continue
            d = int(_draws.get(s, 0))
            if d and not pool:
                raise RuntimeError(f'BG draw 稽核：{s} 要抽 {d} 筆但 pool 為空')
            # sensitivity upper arm 允許從 180 張 unique BG pool 有放回抽樣；
            # P8 的 d<=q 路徑在此仍退化為原本的逐景前綴。
            patches_by_scene[s].extend(pool[i % len(pool)] for i in range(d))

    pos = neg = oil_px = bg_px = 0
    mask_dir = Path(mask_dir)
    for scene, coords in patches_by_scene.items():
        mp = mask_dir / f'{scene}.tif'
        if not mp.exists():
            neg += len(coords)
            continue
        with rasterio.open(mp) as src:
            for (x, y) in coords:
                win = Window(x, y, patch_size, patch_size)
                arr = src.read(1, window=win, boundless=True, fill_value=255)
                o = int((arr == 0).sum())
                b = int(((arr == 1) | (arr == 2)).sum())
                if o > 0:
                    pos += 1
                else:
                    neg += 1
                oil_px += o
                bg_px += b
    return pos, neg, oil_px, bg_px

#
# run_evaluation_job 函式完全保持不變，因為它的職責是單次評估，不涉及跨實驗的記憶體管理。
#
def run_evaluation_job(exp_config, model_to_evaluate, results_path, excel_path, desired_order, run_timestamp, training_metrics=None):
    """
    一個獨立的評估工作函式，負責執行單次評估並記錄結果。
    (此函式內容與您提供的版本完全相同，一字不改)
    """
    print(f"\n--- 開始評估任務: {exp_config.get('test_name', exp_config.get('experiment_name'))} ---")
    print(f"--- 使用模型: {model_to_evaluate} ---")

    eval_dataset_config = exp_config.get('dataset', {})
    if not eval_dataset_config:
        print("   [錯誤] 評估任務未定義 'dataset'。")
        return

    eval_dataset_config.setdefault('train', 'images/train')
    eval_dataset_config.setdefault('val', 'images/val')
    eval_dataset_config.setdefault('test', 'images/test')
    eval_dataset_config.setdefault('nc', 1)
    eval_dataset_config.setdefault('names', ['oil'])
    
    eval_results_path = results_path
    eval_results_path.mkdir(exist_ok=True, parents=True)
    
    temp_yaml_path = create_temp_data_yaml(eval_dataset_config, eval_results_path)

    log_data = {k: v for k, v in exp_config.items() if not isinstance(v, (dict, list))}
    log_data['run_timestamp'] = run_timestamp
    log_data['Experiment_name'] = log_data.pop('experiment_name', 'N/A')
    if 'test_name' in log_data:
        log_data['Test name'] = log_data.pop('test_name')

    # [C7] 評估協議標註：GT_expand 的 test patch 全部對齊 oil bbox，
    # 數字不能直接跟 0422 sliding-window per-patch 或 reconstruction 結果比較。
    log_data['eval_protocol'] = 'patch_oil_bbox_centered'
    split_dir = eval_dataset_config.get('test_txt') or eval_dataset_config.get('val_txt') or ''
    log_data['eval_data_source'] = str(Path(split_dir).parent.name) if split_dir else 'unknown'

    if 'imgsz' in log_data: log_data['Image size'] = log_data.pop('imgsz')
    if 'epochs' in log_data: log_data['Epochs'] = log_data.pop('epochs', None)
    if 'batch_size' in log_data: log_data['Batch size'] = log_data.pop('batch_size', None)

    if training_metrics:
        log_data.update(training_metrics)

    log_data.update(get_image_counts(eval_dataset_config, eval_results_path))

    # 正負樣本（patch-level） + 像素層級 oil/bg 比例（per split）
    mask_dir_for_count = eval_dataset_config.get('labels_dir')
    patch_size_for_count = int(eval_dataset_config.get('patch_size', exp_config.get('imgsz', 256)))
    if mask_dir_for_count:
        for split in ('train', 'val', 'test'):
            txt_path = eval_dataset_config.get(f'{split}_txt')
            pos, neg, oil_px, bg_px = _count_patches_pos_neg(
                txt_path, mask_dir_for_count, patch_size_for_count,
                bg_coord_txt=eval_dataset_config.get('bg_coord_txt'),
                bg_draws_freeze_csv=(eval_dataset_config.get('bg_pool_cfg') or {}).get('draws_freeze_csv'))
            tot_px = oil_px + bg_px
            log_data[f'{split}_pos_patch'] = pos
            log_data[f'{split}_neg_patch'] = neg
            log_data[f'{split}_oil_px'] = oil_px
            log_data[f'{split}_bg_px'] = bg_px
            log_data[f'{split}_oil_ratio'] = f'{oil_px/tot_px:.4f}' if tot_px else 'N/A'

    # [P8][codex R5-5 / R6-3] 池大小與每 epoch 曝光量必須分欄，不得互相冒充；
    # 且必須逐 fold 記錄（corpus 值 7720/3510 不是任何一折的曝光量）。
    _bp = (eval_dataset_config.get('bg_pool_cfg') or {}).get('draws_freeze_csv')
    if _bp and Path(_bp).exists():
        import csv as _csv2
        with open(_bp, newline='') as _fh:
            _rows = list(_csv2.DictReader(_fh))
        _base = {r['scene']: int(r['base_A']) for r in _rows}
        log_data['bg_pool_size_corpus'] = sum(int(r['realized_q']) for r in _rows)
        log_data['bg_draws_per_epoch_corpus'] = sum(_base.values())
        log_data['bg_pool_freeze_csv'] = str(_bp)
        # 逐 fold：train 場景取自本折 train_txt，與訓練用的 bg_scene_filter 同一集合
        _tr_txt = eval_dataset_config.get('train_txt')
        _bg_txt = eval_dataset_config.get('bg_coord_txt')
        if _tr_txt and Path(_tr_txt).exists():
            _tr_scenes, _n_gt = set(), 0
            for _l in Path(_tr_txt).read_text().splitlines():
                _l = _l.strip()
                if not _l:
                    continue
                _n_gt += 1
                _m = _PATCH_RE_MR.match(_l)
                if _m:
                    _tr_scenes.add(_m.group(1))
            _draws = sum(_base.get(s, 0) for s in _tr_scenes)
            _pool = 0
            if _bg_txt and Path(_bg_txt).exists():
                with open(_bg_txt) as _fh:
                    _pool = sum(1 for r in _csv2.DictReader(_fh, delimiter='\t')
                                if r['scene'] in _tr_scenes)
            _ds_len = _n_gt + _draws
            _bs = int(exp_config.get('batch_size', 16))
            log_data['train_bg_pool_size'] = _pool
            log_data['train_bg_draws_per_epoch'] = _draws
            log_data['train_dataset_len'] = _ds_len
            log_data['steps_per_epoch'] = _ds_len // _bs          # drop_last=True
            log_data['bg_reuse_cycle_epochs'] = round(_pool / _draws, 4) if _draws else None
            # fail-fast 模式下 runtime validity failure 會 raise，故能走到這裡即為 0
            log_data['runtime_bg_validity_failures'] = 0

    log_data['best_model_path'] = str(model_to_evaluate)
    log_data['Results_folder'] = str(eval_results_path)

    eval_metrics = evaluate_and_visualize(exp_config, temp_yaml_path, model_to_evaluate, eval_results_path)
    if eval_metrics:
        log_data.update(eval_metrics)

    ordered_log_data = OrderedDict()
    for key in desired_order:
        if key in log_data: ordered_log_data[key] = log_data[key]
    for key, value in log_data.items():
        if key not in ordered_log_data: ordered_log_data[key] = value

    # [新版] 直接把此 fold 的 patch 指標寫成 fold 資料夾內的 run_metrics.json（source of truth）
    log_run_json(eval_results_path, ordered_log_data, kind='patch')
    return eval_metrics


def _log_cv_summary(base_name, all_patch_metrics, all_recon_metrics, excel_path, desired_order, timestamp):
    """計算所有 fold 的指標平均值，寫一列 CV Summary 到 Excel。"""
    valid_patch = [m for m in all_patch_metrics if m]
    valid_recon = [m for m in all_recon_metrics if m]

    if not valid_patch and not valid_recon:
        print("[CV Summary] 沒有有效的 fold 結果，跳過 Summary。")
        return

    summary = OrderedDict()
    summary['Experiment_name'] = f"{base_name}_CV_Summary"
    summary['run_timestamp'] = timestamp
    summary['cv_num_folds'] = len([m for m in all_patch_metrics if m is not None])

    def _avg_metrics(metrics_list, prefix='avg_'):
        all_keys = set()
        for m in metrics_list: all_keys.update(m.keys())
        result = {}
        for key in sorted(all_keys):
            vals = []
            for m in metrics_list:
                try: vals.append(float(m.get(key, 'nan')))
                except (ValueError, TypeError): pass
            if vals:
                result[f'{prefix}{key}'] = f"{sum(vals) / len(vals):.4f}"
        return result

    if valid_patch:
        summary.update(_avg_metrics(valid_patch))
    if valid_recon:
        summary.update(_avg_metrics(valid_recon))

    # [新版] CV Summary 不再寫檔；restructure_excel.py 會從各 fold 的 run_metrics.json
    # 自動重算跨 fold 平均（含 std），這裡僅印出供 console 檢視。
    print(f"\n{'='*60}")
    print(f"📊 CV Summary ({summary['cv_num_folds']} folds):")
    for k, v in summary.items():
        if k.startswith('avg_'):
            print(f"   {k}: {v}")
    print(f"{'='*60}")


def _refresh_restructured_excel(result_base, excel_log_path):
    """每跑完一個 fold（或全部任務結束）就全掃重建該實驗的 restructured Excel。
    idempotent + 原子替換；失敗只 warning，絕不影響訓練結果。"""
    try:
        import subprocess as _sub
        _script = Path(__file__).parent / 'restructure_excel.py'
        if not _script.exists():
            return
        _cmd = [sys.executable, str(_script)]
        if result_base:
            _cmd += ['--result-base', str(result_base)]
        if excel_log_path:
            _dst = Path(excel_log_path).with_name(Path(excel_log_path).stem + '_restructured.xlsx')
            _cmd += ['--dst', str(_dst)]
        _sub.run(_cmd, check=False)
    except Exception as _err:
        print(f"[Excel] ⚠ restructured Excel 刷新失敗（不影響訓練結果）: {_err}")


def run_cv_experiment(exp_config, results_base_dir, excel_path, desired_order, run_timestamp, fold_log_dir=None):
    """執行完整的 Cross Validation：N folds 訓練 + patch-test + reconstruction，最後輸出 CV Summary。"""
    cv_cfg = exp_config.get('cross_validation', {})
    num_folds = cv_cfg.get('num_folds', 3)
    start_fold = cv_cfg.get('start_fold', 1)
    split_dir = Path(cv_cfg.get('split_dir'))
    base_name = exp_config['experiment_name']

    print(f"\n{'='*80}")
    print(f"🔄 啟動 Cross Validation: {base_name} ({num_folds} folds, 從 Fold {start_fold} 開始)")
    print(f"   Split dir: {split_dir}")
    print(f"{'='*80}\n")

    all_patch_metrics = []
    all_recon_metrics = []
    mode = exp_config.get('mode', 'train')

    base_seed = int(exp_config.get('seed', 42))
    for fold_idx in range(start_fold, num_folds + 1):
        print(f"\n{'='*60}")
        print(f"==> Fold {fold_idx}/{num_folds}  [mode={mode}]")
        print(f"{'='*60}")
        # [C1] 每個 fold 用 (base_seed + fold_idx) 鎖種子，folds 之間獨立但可重現
        set_global_seed(base_seed + fold_idx)

        fold_ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        fold_name = f"{base_name}_fold{fold_idx}"

        # Per-fold stdout tee
        _tee = None
        _orig_stdout = sys.stdout
        if fold_log_dir:
            log_path = Path(fold_log_dir) / f"{fold_ts}_{fold_name}.log"
            _tee = _FoldTee(log_path, _orig_stdout)
            sys.stdout = _tee
            print(f"[FoldTee] 訓練 log 同步寫入: {log_path}")

        fold_config = copy.deepcopy(exp_config)
        fold_config['experiment_name'] = fold_name
        fold_config['experiment_type'] = 'cross_validation'
        fold_config['dataset']['train_txt'] = str(split_dir / f'train_fold{fold_idx}.txt')
        fold_config['dataset']['val_txt'] = str(split_dir / f'val_fold{fold_idx}.txt')
        fold_config['dataset']['test_txt'] = str(split_dir / f'test_fold{fold_idx}.txt')
        fold_config['dataset']['current_fold'] = fold_idx

        fold_results_path = results_base_dir / f"{fold_ts}_{fold_name}"
        fold_results_path.mkdir(exist_ok=True, parents=True)

        # --- 訓練 or 直接使用 base_model ---
        training_results = {}
        if mode == 'train':
            training_results = train_model(fold_config, fold_results_path) or {}
            if not (training_results.get('selected_model_path') or
                    training_results.get('best_model_path')):
                print(f"[Error] Fold {fold_idx} 訓練失敗，跳過。")
                all_patch_metrics.append(None)
                all_recon_metrics.append(None)
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                if _tee is not None:
                    sys.stdout = _orig_stdout; _tee.close(); _tee = None
                continue
            model_path = (training_results.get('selected_model_path') or
                          training_results['best_model_path'])
            print(f"  [Fold {fold_idx}] Best Val mIoU: {training_results.get('best_val_miou', 'N/A')}")
        else:  # mode == 'test'
            model_path = str(fold_config.get('base_model', ''))
            if not model_path or not Path(model_path).exists():
                print(f"[Error] Fold {fold_idx} mode=test 但 base_model 不存在: {model_path}，跳過。")
                all_patch_metrics.append(None)
                all_recon_metrics.append(None)
                if _tee is not None:
                    sys.stdout = _orig_stdout; _tee.close(); _tee = None
                continue
            print(f"  [Fold {fold_idx}] 使用既有模型: {model_path}")

        # --- Patch-level test (GT-expand: test_fold{N}.txt per-patch eval) ---
        patch_metrics = run_evaluation_job(
            fold_config, model_path, fold_results_path,
            excel_path, desired_order, fold_ts, training_results
        )
        all_patch_metrics.append(patch_metrics or {})

        # 訓練結束後清理記憶體，再進 reconstruction
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # --- Reconstruction test ---
        fold_recon_metrics = {}
        for test_job in fold_config.get('post_tests', []):
            if not test_job.get('run', True): continue
            recon_cfg = copy.deepcopy(test_job.get('reconstruction', {}))
            if not recon_cfg.get('enabled', False): continue
            scene_list_val = recon_cfg.get('scene_list_txt', '')
            if scene_list_val and not scene_list_val.endswith('.txt'):
                recon_cfg['scene_list_txt'] = str(Path(scene_list_val) / f'test_fold{fold_idx}.txt')

            test_ts = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            test_results_path = fold_results_path / f"post_test_{test_ts}_{test_job['test_name']}"
            test_results_path.mkdir(exist_ok=True, parents=True)

            test_run_config = copy.deepcopy(fold_config)
            for key, value in test_job.items():
                if isinstance(value, dict) and isinstance(test_run_config.get(key), dict):
                    test_run_config[key].update(value)
                else:
                    test_run_config[key] = value

            # 根據 yaml 的 reconstruction.mode 派發：
            #   'gt_aware' / 'bbox_only' → 只在 test_fold{N}.txt 油汙周邊 patch 跑 inference，貼回原座標
            #   其他（含未指定）         → 原 sliding-window 全圖重組（0422 行為）
            recon_mode = str(recon_cfg.get('mode', 'sliding_window')).lower()
            try:
                if recon_mode in ('gt_aware', 'bbox_only', 'gt-aware'):
                    from recon_gt_aware_module import run_reconstruction_gt_aware
                    fold_recon_metrics = run_reconstruction_gt_aware(
                        recon_cfg, test_run_config, model_path, test_results_path) or {}
                else:
                    fold_recon_metrics = run_reconstruction(
                        recon_cfg, test_run_config, model_path, test_results_path) or {}
            except Exception as recon_err:
                print(f"[警告] Fold {fold_idx} Reconstruction ({recon_mode}) 發生錯誤，跳過：{recon_err}")
                fold_recon_metrics = {}

            if fold_recon_metrics:
                recon_log = OrderedDict()
                recon_log['Experiment_name'] = f"{fold_name}_recon"
                recon_log['run_timestamp'] = fold_ts
                recon_log['Results_folder'] = str(test_results_path)
                recon_log['best_model_path'] = str(model_path)
                recon_log.update({k: v for k, v in training_results.items() if not isinstance(v, (dict, list))})
                recon_log.update(fold_recon_metrics)
                # [新版] recon 指標寫進 post_test 資料夾的 run_metrics.json
                log_run_json(test_results_path, recon_log, kind='recon')

        all_recon_metrics.append(fold_recon_metrics)

        # Restore stdout and flush/close the fold log file
        if _tee is not None:
            sys.stdout = _orig_stdout
            _tee.close()
            _tee = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("  - CUDA cache 已清空。")

        # 每跑完一 fold 就刷新 restructured Excel（可看進度、中途 crash 也有記錄）
        _refresh_restructured_excel(results_base_dir, excel_path)

    # --- CV Summary ---
    _log_cv_summary(base_name, all_patch_metrics, all_recon_metrics, excel_path, desired_order, run_timestamp)


def main():
    parser = argparse.ArgumentParser(description="執行多波段油汙偵測實驗排程")
    parser.add_argument(
        '--config', 
        type=str, 
        default='OIL_PROJECT_MutiBand_0422_VRT_training/main/experiments.yaml', 
        help='指定要執行的 YAML 設定檔路徑'
    )
    args = parser.parse_args()

    config_path = args.config
    print(f"\n{'='*60}")
    print(f"🚀 啟動任務，讀取設定檔: {config_path}")
    print(f"{'='*60}\n")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            master_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"錯誤：找不到設定檔 {config_path}！請檢查路徑。")
        return

    results_base_dir = Path(master_config['results_base_dir'])
    excel_path = master_config['excel_log_path']
    # [2026-07-13] 停用 per-fold FoldTee 檔案 log：訓練 stdout 已由 nohup 重導成一份，
    # 這裡再寫 train_log/ 會重複佔空間。硬性關掉，忽略各 yaml 殘留的 fold_log_dir key。
    fold_log_dir = None
    completed_experiments_paths = {}

    desired_order = [
        # --- 基本資訊 ---
        'Experiment_name', 'Test name', 'run_timestamp', 'training_time_minutes',
        'Ram_GB', 'gpu_name', 'mode', 'experiment_type', 'Results_folder', 'best_model_path',
        'base_model','architecture', 'Epochs', 'Batch size','nominal_batch_size', 'Image size', 'patience',

        # --- 資料集數量與設定 ---
        'eval_conf', 'eval_iou', 'train_count', 'val_count', 'test_count',

        # --- MADOS 語意分割核心指標 ---
        'mIoU', 'Accuracy', 'F1-macro',
        'IoU(Oil)', 'Precision(Oil)', 'Recall(Oil)', 'F1(Oil)',
        'IoU(Others)', 'Precision(Others)', 'Recall(Others)', 'F1(Others)',
        'IoU(Background)', 'Precision(Background)', 'Recall(Background)', 'F1(Background)'
    ]

    for exp_config in master_config.get('experiments', []):
        
        # ⭐⭐⭐ 核心修改：將每個實驗的完整執行邏輯包裹在 try...finally 中 ⭐⭐⭐
        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            current_exp_config = copy.deepcopy(exp_config)

            current_exp_config.setdefault('run', True)
            current_exp_config.setdefault('eval_conf', 0.25)
            current_exp_config.setdefault('eval_iou', 0.6)
            
            if not current_exp_config['run']:
                print(f"\n--- 跳過實驗: {current_exp_config.get('experiment_name')} (run 設為 False) ---")
                continue

            # ⭐ Cross Validation 模式：偵測到 cross_validation.enabled 就走 CV 流程
            cv_cfg = current_exp_config.get('cross_validation', {})
            if cv_cfg.get('enabled', False):
                run_cv_experiment(current_exp_config, results_base_dir, excel_path, desired_order, timestamp, fold_log_dir=fold_log_dir)
                continue

            results_path = None
            dependency_failed = False
            is_finetune = False
            base_model_path_str = str(current_exp_config.get('base_model', ''))
            
            # (您的偵錯日誌和依賴解析邏輯保持完全不變)
            print("\n" + "="*80)
            print(f"==> 開始處理實驗: {current_exp_config.get('experiment_name')}")
            print(f"    - 當前已完成的依賴項: {list(completed_experiments_paths.keys())}")
            print(f"    - 此實驗的 base_model: {base_model_path_str}")
            print("="*80)

            if '{{' in base_model_path_str:
                is_finetune = True
                match_found = False
                for name, completed_path in completed_experiments_paths.items():
                    placeholder = f'{{{{{name}}}}}'
                    if placeholder in base_model_path_str:
                        match_found = True
                        if completed_path is None:
                            print(f"==> [錯誤] 跳過實驗 '{current_exp_config['experiment_name']}'，因其依賴的實驗 '{name}' 執行失敗。")
                            dependency_failed = True
                            break
                        
                        new_base_model = base_model_path_str.replace(placeholder, str(completed_path))
                        print(f"==> [成功] 找到依賴 '{name}'，將 base_model 解析為: {new_base_model}")
                        current_exp_config['base_model'] = new_base_model
                        
                        if current_exp_config['mode'] == 'train':
                            results_path = completed_path.parent / f"{timestamp}_{current_exp_config['experiment_name']}"
                        break
                
                if not match_found:
                     print(f"==> [警告] 在 base_model 中找到佔位符，但在已完成的實驗中找不到匹配的依賴項。請檢查 yaml 中的 experiment_name 是否有錯字。")

            if dependency_failed:
                continue
                
            if results_path is None:
                results_path = results_base_dir / f"{timestamp}_{current_exp_config['experiment_name']}"
            
            results_path.mkdir(exist_ok=True, parents=True)

            inherited_training_metrics = {}
            model_to_evaluate = None
            training_success = False

            if current_exp_config['mode'] == 'test':
                model_to_evaluate = current_exp_config['base_model']
                # (繼承指標的邏輯不變)
            elif current_exp_config['mode'] == 'train':
                current_exp_config['experiment_type'] = 'finetune' if is_finetune else 'train'
                training_results = train_model(current_exp_config, results_path) or {}
                if training_results and ('selected_model_path' in training_results or
                                         'best_model_path' in training_results):
                    model_to_evaluate = (training_results.get('selected_model_path') or
                                         training_results.get('best_model_path'))
                    training_success = True
                    inherited_training_metrics = training_results

            if model_to_evaluate and Path(model_to_evaluate).exists():
                # (後續評估邏輯不變)
                post_tests = current_exp_config.get('post_tests', [])
                if not post_tests:
                    if current_exp_config['mode'] == 'train':
                        print(f"\n沒有 post_tests, 對主任務 '{current_exp_config['experiment_name']}' 進行評估。")
                        run_evaluation_job(current_exp_config, model_to_evaluate, results_path, excel_path, desired_order, timestamp, inherited_training_metrics)

                        # ⭐ 檢查主任務是否有重組設定
                        recon_cfg = current_exp_config.get('reconstruction', {})
                        if recon_cfg.get('enabled', False):
                            run_reconstruction(recon_cfg, current_exp_config, model_to_evaluate, results_path)

                else:
                    print(f"\n偵測到 post_tests, 開始執行後續測試任務...")
                    for test_job in post_tests:
                        if not test_job.get('run', True): continue

                        test_timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                        test_results_path = results_path / f"post_test_{test_timestamp}_{test_job['test_name']}"

                        test_run_config = copy.deepcopy(current_exp_config)
                        # ⭐️ 更新字典：保留原本的 mean, std, names，只更新 path 等新設定
                        for key, value in test_job.items():
                            if isinstance(value, dict) and isinstance(test_run_config.get(key), dict):
                                test_run_config[key].update(value)
                            else:
                                test_run_config[key] = value
                        
                        test_run_config.setdefault('eval_conf', current_exp_config['eval_conf'])
                        test_run_config.setdefault('eval_iou', current_exp_config['eval_iou'])

                        # --- Patch-level eval（暫時跳過：目前只跑 reconstruction）---
                        # run_evaluation_job(test_run_config, model_to_evaluate, test_results_path, excel_path, desired_order, test_timestamp, inherited_training_metrics)

                        # ⭐ 檢查 post_tests 內部的重組設定 (對應您的 YAML 結構)
                        recon_cfg = test_job.get('reconstruction', {})
                        if recon_cfg.get('enabled', False):
                            run_reconstruction(recon_cfg, test_run_config, model_to_evaluate, test_results_path)

            if current_exp_config['mode'] == 'train':
                completed_experiments_paths[current_exp_config['experiment_name']] = results_path if training_success else None

        except Exception as e:
            # 如果實驗中發生任何未捕獲的錯誤，打印出來，但確保 finally 區塊會執行
            print(f"[嚴重錯誤] 實驗 '{exp_config.get('experiment_name')}' 執行期間發生未預期的錯誤: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # ⭐⭐⭐ 核心清理區塊 ⭐⭐⭐
            # 無論 try 區塊是否成功，這個區塊都保證在每個頂層實驗結束後執行
            print(f"\n--- 完成實驗: {exp_config.get('experiment_name')}，正在進行記憶體清理... ---")
            
            # 1. 呼叫 Python 的垃圾回收器，清理可能存在的循環引用
            gc.collect()
            
            # 2. 清空 PyTorch 的 CUDA 快取，這是最關鍵的一步
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("  - CUDA cache 已清空。")
            
            print("--- 記憶體清理完成 ---")


    print("\n所有已設定的實驗執行完畢！")

    reconstruction_configs = master_config.get('reconstruction', {})
    if reconstruction_configs:
        for recon_name, recon_cfg in reconstruction_configs.items():
            if recon_cfg.get('enabled', False):
                print(f"\n\n🚀 偵測到全局重組任務 [{recon_name}] 已啟用 (enabled: true)！")
                
                model_path_str = str(recon_cfg.get('model_path', ''))
                if '{{' in model_path_str:
                    for exp_name, completed_path in completed_experiments_paths.items():
                        placeholder = f'{{{{{exp_name}}}}}'
                        if placeholder in model_path_str and completed_path is not None:
                            # 自動把 {{實驗名稱}} 替換成剛剛跑完帶有時間戳記的真實資料夾路徑
                            recon_cfg['model_path'] = model_path_str.replace(placeholder, str(completed_path))
                            print(f"🔗 已自動綁定最新訓練模型: {recon_cfg['model_path']}")
                            break
            
                # 為了拿到架構設定 (in_channels, pixel_mapping 等)
                base_exp_config = master_config.get('experiments', [{}])[0]
                
                try:
                    # 改用 run_reconstruction_v2
                    run_reconstruction(recon_cfg, base_exp_config, recon_cfg['model_path'], Path(recon_cfg['model_path']).parent)
                except Exception as e:
                    print(f"❌ 重組任務發生錯誤: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

    print("\n所有任務執行完畢！")

    # ---- 全部任務結束後最終刷新一次 restructured Excel（per-fold 已各刷一次，這是收尾）----
    print("\n[Excel] 最終刷新 restructured Excel")
    _refresh_restructured_excel(master_config.get('results_base_dir'), master_config.get('excel_log_path'))


if __name__ == '__main__':
    main()
