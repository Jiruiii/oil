# ==============================================================================
# ===                 utils.py (修復路徑解析問題的完整版)                    ===
# ==============================================================================
import yaml
from pathlib import Path
import os

def create_temp_data_yaml(dataset_config, target_dir):
    """
    在目標資料夾中建立一個暫時的 data.yaml 檔案。
    ⭐ 修正：強制將 'path' 轉換為絕對路徑，確保 YOLO/DeepLab 不會迷路。
    ⭐ 支援 flat-dir 模式：若無 'path' 但有 images_dir/labels_dir/txt，直接 dump 不做路徑解析。
    """
    os.makedirs(target_dir, exist_ok=True)
    temp_yaml_path = target_dir / 'data.yaml'

    # 複製一份設定，以免修改到原始字典
    config_copy = dataset_config.copy()

    # --- 路徑解析邏輯（只有 path 有值才做 resolve）---
    raw_path = config_copy.get('path', '')
    if raw_path:
        path_obj = Path(raw_path)
        if not path_obj.is_absolute():
            path_obj = (Path.cwd() / raw_path).resolve()
        config_copy['path'] = str(path_obj)

    with open(temp_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_copy, f, allow_unicode=True, sort_keys=False)

    return temp_yaml_path

def get_image_counts(dataset_config, yaml_parent_dir=None):
    """
    根據 dataset 字典計算 train/val/test 圖片數量。
    ⭐ 支援 txt-based 計數：若有 train_txt/val_txt/test_txt，直接數行數。
    ⭐ 修正：增強路徑解析能力，支援絕對路徑，解決 'Path not found' 問題。
    """
    # txt-based 計數（flat-dir + CV 模式）
    train_txt = dataset_config.get('train_txt')
    val_txt = dataset_config.get('val_txt')
    test_txt = dataset_config.get('test_txt')
    if any([train_txt, val_txt, test_txt]):
        def _count_txt(txt_path):
            if not txt_path: return 0
            p = Path(txt_path)
            if not p.exists(): return "File not found"
            return len([l for l in p.read_text().splitlines() if l.strip()])
        return {
            'train_count': _count_txt(train_txt),
            'val_count': _count_txt(val_txt),
            'test_count': _count_txt(test_txt),
        }

    counts = {'train_count': 0, 'val_count': 0, 'test_count': 0}
    
    try:
        # 1. 解析 Base Path (與 evaluation_module 邏輯一致)
        raw_path = dataset_config.get('path', '')
        dataset_path = Path(raw_path)
        
        # 判斷是否為絕對路徑
        if not dataset_path.is_absolute():
            dataset_path = (Path.cwd() / raw_path).resolve()
            
        # 如果資料集根目錄不存在，直接回傳錯誤
        if not dataset_path.exists():
            return {k: "Path not found" for k in counts}

        # 2. 定義子資料夾 (YOLO 格式通常是 images/train, images/val 等)
        # 使用 .get() 提供預設值，防止 KeyError
        subsets = {
            'train': dataset_config.get('train', 'images/train'),
            'val': dataset_config.get('val', 'images/val'),
            'test': dataset_config.get('test', 'images/test')
        }

        # 3. 計算數量
        for key, sub_path in subsets.items():
            if sub_path is None:
                counts[f'{key}_count'] = "Not specified"
                continue
                
            full_path = dataset_path / sub_path
            
            if full_path.is_dir():
                # 計算 .jpg, .jpeg, .png 檔案的總數
                count = len(list(full_path.glob('*.jpg'))) + \
                        len(list(full_path.glob('*.png'))) + \
                        len(list(full_path.glob('*.jpeg')))
                counts[f'{key}_count'] = count
            else:
                # 嘗試不加 base path 直接找 (有些 yaml 設定檔可能寫了完整路徑)
                alt_path = Path(sub_path)
                if alt_path.is_absolute() and alt_path.is_dir():
                    count = len(list(alt_path.glob('*.jpg'))) + \
                            len(list(alt_path.glob('*.png'))) + \
                            len(list(alt_path.glob('*.jpeg')))
                    counts[f'{key}_count'] = count
                else:
                    counts[f'{key}_count'] = "Dir not found"
                
    except Exception as e:
        print(f"[Warning] 無法計算圖片數量: {e}")
        return {k: "Error" for k in counts}
        
    return counts

# --- ⭐ 專為分類任務設計的函式 (保留並優化路徑處理) ⭐ ---

def get_cls_image_counts(dataset_config: dict) -> dict:
    """
    計算分類資料集 train, val, test 中的圖片數量。
    分類資料集結構應為: path/train/ClassNameA/, path/train/ClassNameB/ ...
    """
    counts = {}
    try:
        # 同樣套用強固的路徑解析
        raw_path = dataset_config.get('path', '')
        data_root = Path(raw_path)
        if not data_root.is_absolute():
            data_root = (Path.cwd() / raw_path).resolve()
        
        if not data_root.exists():
             return {'train_count': 'Path not found', 'val_count': 'Path not found', 'test_count': 'Path not found'}

        for split in ['train', 'val', 'test']:
            count = 0
            split_path = data_root / split
            
            if split_path.is_dir():
                # 遍歷所有子目錄（如 Oil, No_Oil）
                for class_dir in split_path.iterdir():
                    if class_dir.is_dir():
                        count += len(list(class_dir.glob('*.jpg')))
                        count += len(list(class_dir.glob('*.png')))
                        count += len(list(class_dir.glob('*.jpeg')))
                counts[f'{split}_count'] = count
            else:
                counts[f'{split}_count'] = "Dir not found"
            
        return counts
    except Exception as e:
        print(f" 計算分類圖片數量時出錯: {e}")
        return {'train_count': -1, 'val_count': -1, 'test_count': -1}