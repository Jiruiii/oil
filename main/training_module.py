import time
import torch
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from utils import create_temp_data_yaml
from deeplab_adapter import DeepLabAdapter
# osdmamba/segformer 為可選後端，import 延後到實際使用時，避免在無 mamba-ssm 環境中啟動就掛掉
# from osdmamba_adapter import OSDMambaAdapter
# from segformer_adapter import SegformerAdapter
import yaml

def calculate_accumulation_steps(physical_batch_size, nominal_batch_size):
    """
    計算梯度累積步數。
    如果顯存只能跑 batch=4 (physical)，但我們希望模擬 batch=64 (nominal) 的效果，
    則 accumulation_steps = 64 / 4 = 16。
    """
    if physical_batch_size >= nominal_batch_size:
        return 1
    return max(round(nominal_batch_size / physical_batch_size), 1)

def plot_custom_metrics(results_path, exp_name):
    """
    為非 YOLO 模型繪製訓練過程圖表 (Loss / mIoU / Per-class IoU 三欄)。
    讀取 Adapter 產生的 results.csv，per-class 欄位動態偵測。
    """
    csv_path = results_path / 'results.csv'
    if not csv_path.exists():
        return

    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]

        # 動態找出所有 per-class IoU 欄位
        per_class_cols = [c for c in df.columns if c.startswith('val_iou_')]

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(exp_name, fontsize=13)

        # --- 子圖 1：Loss ---
        ax = axes[0]
        if 'train_loss' in df.columns:
            ax.plot(df['epoch'], df['train_loss'], label='Train Loss', linewidth=2)
        if 'val_loss' in df.columns:
            ax.plot(df['epoch'], df['val_loss'], label='Val Loss', linewidth=2)
        ax.set_title('Loss Curve')
        ax.set_xlabel('Epochs'); ax.set_ylabel('Loss')
        ax.legend(); ax.grid(True, alpha=0.3)

        # --- 子圖 2：mIoU ---
        ax = axes[1]
        if 'train_miou' in df.columns:
            ax.plot(df['epoch'], df['train_miou'], label='Train mIoU', linestyle='--', alpha=0.7)
        if 'val_miou' in df.columns:
            ax.plot(df['epoch'], df['val_miou'], label='Val mIoU', linewidth=2)
        ax.set_title('mIoU Curve')
        ax.set_xlabel('Epochs'); ax.set_ylabel('mIoU')
        ax.legend(); ax.grid(True, alpha=0.3)

        # --- 子圖 3：Per-class Val IoU（動態）---
        ax = axes[2]
        for col in per_class_cols:
            class_label = col.replace('val_iou_', '')
            ax.plot(df['epoch'], df[col], label=class_label, linewidth=2)
        ax.set_title('Per-class Val IoU')
        ax.set_xlabel('Epochs'); ax.set_ylabel('IoU')
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = results_path / 'custom_metrics_plot.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"  [資訊] 自訂訓練圖表已儲存至: {save_path}")

    except Exception as e:
        print(f"  [警告] 繪製圖表時發生錯誤: {e}")

def train_model(config, results_path):
    exp_name = config['experiment_name']
    print(f"\n{'='*20} 開始訓練: {exp_name} {'='*20}")
    print(f"結果將儲存於: {results_path}")

    try:
        # --- 1. 準備資料設定 ---
        dataset_config = config.get('dataset', {})
        if not dataset_config:
            print("  [錯誤] 訓練任務未定義 'dataset'。")
            return None

        # 設定預設值
        for key, val in [('train', 'images/train'), ('val', 'images/val'), ('test', 'images/test'), ('nc', 1), ('names', ['oil'])]:
            dataset_config.setdefault(key, val)

        temp_yaml_path = create_temp_data_yaml(dataset_config, results_path)

        # --- 2. 初始化模型 ---
        arch = config.get('architecture', 'yolo') # 預設為 yolo
        architecture_cfg = config.get('architecture_cfg', {})
        
        # ⭐⭐⭐ 唯一修改處：確保 in_channels 被正確寫入 dataset_config 交給 Adapter ⭐⭐⭐
        if 'in_channels' in architecture_cfg:
            dataset_config['in_channels'] = architecture_cfg['in_channels']
        elif 'in_channels' in config:
            dataset_config['in_channels'] = config['in_channels']
        dataset_config.setdefault('in_channels', 3)
        
        if arch in ('deeplabv3+', 'deeplabv3plus'):   # 'deeplabv3+'=歷史名(實為V3)；'deeplabv3plus'=新yaml真名,family由 architecture_cfg.model_family 決定
            print(f"  [模型] 載入 DeepLab Adapter (Input Channels: {dataset_config['in_channels']})")
            model = DeepLabAdapter(config.get('base_model'), dataset_cfg=dataset_config, architecture_cfg=architecture_cfg)
            is_custom_model = True

        elif arch == 'osdmamba':
            print(f"  [模型] 載入 OSDMamba (SwinUMambaD) Adapter (Input Channels: {dataset_config['in_channels']})")
            from osdmamba_adapter import OSDMambaAdapter
            model = OSDMambaAdapter(config.get('base_model'), dataset_cfg=dataset_config, architecture_cfg=architecture_cfg)
            is_custom_model = True

        elif arch == 'segformer':
            print(f"  [模型] 載入 SegFormer Adapter")
            model = SegformerAdapter(config) 
            is_custom_model = True

        elif arch == 'yolo':
            print(f"  [模型] 載入 YOLOv8")
            from ultralytics import YOLO
            model = YOLO(config['base_model'])
            is_custom_model = False

        else:
            print(f"  [錯誤] 不支援的模型架構: {arch}")
            return None

        # --- 3. 準備訓練參數 (參數解耦) ---
        batch_size = config.get('batch_size', 16)
        imgsz = config.get('imgsz', 640)
        epochs = config.get('epochs', 100)
        
        train_args = {
            'data': str(temp_yaml_path),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'patience': config.get('patience', 100),
            'project': str(results_path.parent),
            'name': results_path.name,
            'exist_ok': True,
            'lr0': config.get('lr0', 1e-4),
            'workers': config.get('workers', 4),
            'conf': config.get('eval_conf', 0.25),
            'iou': config.get('eval_iou', 0.6),
            'cls': config.get('cls_weight', 0.5),
            'mixup': config.get('mixup', 0.0),
            # Custom adapters consume this flag directly; do not let YAML say
            # AMP=true while the runtime silently stays FP32.
            'use_amp': bool(config.get('use_amp', False)),
            # Geographic sensitivity A/B：不同 arm 的 epoch 長度會因 rho/BG draws
            # 不同，必須以 optimizer step（並以 step-wise LR）鎖定計算預算。
            'max_optimizer_steps': config.get('max_optimizer_steps'),
            # val=train 時不得以 train-like val 選模；可明確指定 last。
            'checkpoint_policy': config.get('checkpoint_policy', 'best'),
        }

        # --- 4. 處理梯度累積 (Gradient Accumulation) ---
        if is_custom_model:
            target_batch_size = config.get('nominal_batch_size', batch_size) 
            acc_steps = calculate_accumulation_steps(batch_size, target_batch_size)
            train_args['accumulation_steps'] = acc_steps
            
            if acc_steps > 1:
                print(f"  [策略] 啟動梯度累積: Physical Batch={batch_size}, Steps={acc_steps}, Equivalent Batch={batch_size * acc_steps}")
            else:
                print(f"  [策略] 無梯度累積: Batch={batch_size}")

        # ⭐⭐⭐ 完全保留：儲存 args.yaml (確保實驗可追溯) ⭐⭐⭐ 
        args_save_path = results_path / 'args.yaml'
        config_to_save = {}
        for k, v in config.items():
            if isinstance(v, Path):
                config_to_save[k] = str(v)
            elif isinstance(v, list):
                config_to_save[k] = [str(item) if isinstance(item, Path) else item for item in v]
            else:
                config_to_save[k] = v
                
        if is_custom_model and 'accumulation_steps' in train_args:
            config_to_save['calculated_accumulation_steps'] = train_args['accumulation_steps']

        with open(args_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_to_save, f, allow_unicode=True, sort_keys=False)
            
        # 完全保留的提示文字
        print(f"  [紀錄] 訓練配置檔已儲存至: {args_save_path}")

        # --- 5. 開始訓練 (統一呼叫點) ---
        start_time = time.time()
        process = psutil.Process()
        initial_ram = process.memory_info().rss
        
        training_output = model.train(**train_args)

        training_time = time.time() - start_time
        final_ram = process.memory_info().rss

        # [W4] 訓練過程中若 _auto_calculate_stats_vrt 算出 mean/std，
        # 這個值此時已寫進 model.dataset_cfg；把 args.yaml 重存一次讓 runtime mean/std 落地
        try:
            if hasattr(model, 'dataset_cfg') and model.dataset_cfg.get('mean') and model.dataset_cfg.get('std'):
                cfg_now = yaml.safe_load(args_save_path.read_text()) if args_save_path.exists() else config_to_save
                cfg_now.setdefault('dataset', {})
                cfg_now['dataset']['mean'] = model.dataset_cfg['mean']
                cfg_now['dataset']['std']  = model.dataset_cfg['std']
                with open(args_save_path, 'w', encoding='utf-8') as f:
                    yaml.dump(cfg_now, f, allow_unicode=True, sort_keys=False)
                print(f"  [W4] mean/std 已寫回 args.yaml")
        except Exception as _w4err:
            print(f"  ⚠ [W4] args.yaml mean/std 寫回失敗（不影響訓練）: {_w4err}")

        # 訓練完成後立即釋放模型，避免與後續 val/recon 疊加記憶體
        del model
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # --- 6. 處理結果與繪圖 ---
        if isinstance(training_output, dict):
            best_model_path_str = training_output.get('best_model_path')
            selected_model_path_str = training_output.get('selected_model_path') \
                or best_model_path_str
            best_val_miou = training_output.get('best_val_miou')
        else:
            best_model_path_str = str(results_path / 'weights' / 'best.pt')
            selected_model_path_str = best_model_path_str
            best_val_miou = None

        best_model_path = Path(best_model_path_str) if best_model_path_str else None
        selected_model_path = Path(selected_model_path_str) if selected_model_path_str else None

        if is_custom_model:
            plot_custom_metrics(results_path, exp_name)

        # ⭐⭐⭐ 完全保留：錯誤捕捉與檔案檢查 ⭐⭐⭐
        if not selected_model_path or not selected_model_path.exists():
            print(f"  [錯誤] 訓練結束，但找不到選定權重: {selected_model_path}")
            return None

        training_results = {
            'training_time_minutes': f"{training_time / 60:.2f}",
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'Ram_GB': f"{(final_ram - initial_ram) / (1024 ** 3):.2f}",
            'best_model_path': str(best_model_path),
            'selected_model_path': str(selected_model_path),
            'checkpoint_policy': (training_output.get('checkpoint_policy', 'best')
                                  if isinstance(training_output, dict) else 'best'),
        }
        if isinstance(training_output, dict):
            if training_output.get('last_model_path'):
                training_results['last_model_path'] = training_output['last_model_path']
            if training_output.get('global_optimizer_step') is not None:
                training_results['global_optimizer_step'] = int(
                    training_output['global_optimizer_step'])
        if best_val_miou is not None:
            training_results['best_val_miou'] = f"{best_val_miou:.4f}"
        
        # 完全保留的提示文字
        print(f"訓練完成！耗時: {training_results['training_time_minutes']} 分鐘")
        return training_results
        
    except Exception as e:
        print(f"訓練過程中發生嚴重錯誤: {e}")
        import traceback; traceback.print_exc()
        return None
