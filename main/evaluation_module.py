# ==============================================================================
# ===       完整 evaluation_module.py 檔案 (加入 architecture_cfg 傳遞)       ===
# ==============================================================================
from deeplab_adapter import DeepLabAdapter

def evaluate_and_visualize(exp_config, data_yaml_path, model_path, results_path):
    exp_name = exp_config.get('test_name') or exp_config['experiment_name']
    print(f"\n{'='*100}")
    print(f"=== 開始執行語意分割標準評估: {exp_name} ===")
    print(f"{'='*100}")
    
    arch = exp_config.get('architecture', 'yolo')
    dataset_cfg = exp_config.get('dataset', {})
    
    # ⭐ 核心新增：提取 architecture_cfg，裡面包含最重要的 in_channels: 11
    arch_cfg = exp_config.get('architecture_cfg', {})
    
    # [C6] 顯式分發，未支援架構直接 raise，避免 OSDMamba ckpt 被靜默載成 YOLO
    if arch in ('deeplabv3+', 'deeplabv3plus'):
        # ⭐ 核心修正：將 dataset_cfg 與 architecture_cfg 同時傳入供動態解析
        model = DeepLabAdapter(model_path, dataset_cfg=dataset_cfg, architecture_cfg=arch_cfg)
    elif arch == 'osdmamba':
        from osdmamba_adapter import OSDMambaAdapter
        model = OSDMambaAdapter(model_path, dataset_cfg=dataset_cfg, architecture_cfg=arch_cfg)
    elif arch == 'yolo':
        from ultralytics import YOLO
        model = YOLO(model_path)
    else:
        raise ValueError(f"[evaluation_module] 不支援的 architecture: {arch!r}")

    imgsz = exp_config.get('imgsz', 240)
    workers = exp_config.get('workers', 8)
    train_batch = exp_config.get('batch_size', 8)
    val_batch = exp_config.get('val_batch_size', train_batch * 2)
    
    print(f"  [評估設定] Image Size: {imgsz}, Workers: {workers}, Batch Size: {val_batch}")
    
    # 將 batch 參數傳給 Adapter
    eval_results = model.val(
        data=str(data_yaml_path),
        split='test',
        imgsz=imgsz,
        workers=workers,
        batch=val_batch # 👈 這裡傳遞了
    )

    # 評估完成後立即釋放模型，避免與 reconstruction 疊加記憶體
    import gc
    del model
    gc.collect()
    if __import__('torch').cuda.is_available():
        __import__('torch').cuda.empty_cache()

    return dict(eval_results)