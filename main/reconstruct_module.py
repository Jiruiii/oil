import torch
import torch.nn.functional as F
import numpy as np
import cv2
import json
import re
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from xml.etree import ElementTree as ET
import albumentations as A
from albumentations.pytorch import ToTensorV2
import gc


def _read_vrt_parallel(vrt_path, target_h, target_w, max_workers=8):
    """Parse VRT XML, read each band's TIF in parallel, resize with cv2.
    Returns float32 array of shape (bands, H, W) — 4× faster than rasterio VRT read."""
    import rasterio
    tree = ET.parse(str(vrt_path))
    band_paths = [el.find('.//SourceFilename').text.strip()
                  for el in tree.findall('VRTRasterBand')]

    def _read_one(path):
        with rasterio.open(path) as src:
            data = src.read(1)
        if data.shape != (target_h, target_w):
            data = cv2.resize(data, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        return data.astype(np.float32)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        bands = list(ex.map(_read_one, band_paths))
    return np.stack(bands, axis=0)  # (C, H, W)

from deeplab_adapter import DeepLabAdapter


def _load_model_for_recon(arch, model_path, dataset_cfg, arch_cfg):
    """根據 architecture 載入對應 adapter，回傳 (adapter, forward_fn)。"""
    if arch == 'osdmamba':
        from osdmamba_adapter import OSDMambaAdapter
        adapter = OSDMambaAdapter(model_path=model_path, dataset_cfg=dataset_cfg, architecture_cfg=arch_cfg)
    else:
        adapter = DeepLabAdapter(model_path=model_path, dataset_cfg=dataset_cfg, architecture_cfg=arch_cfg)

    def _fwd(tensors):
        out = adapter.model(tensors)
        return out['out'] if isinstance(out, dict) else out

    return adapter, _fwd

# 動態類別調色盤
COLOR_PALETTE = {
    1: [0, 255, 255], 2: [0, 0, 255], 3: [0, 255, 0], 4: [255, 0, 0],
    5: [255, 255, 0], 6: [255, 0, 255], 7: [0, 165, 255], 8: [128, 0, 128],
    9: [0, 128, 128], 10: [128, 128, 0], 11: [0, 0, 128], 12: [130, 0, 75],
    13: [147, 20, 255], 14: [203, 192, 255], 15: [42, 42, 165]
}

def generate_gt_mask_from_json(json_path, target_shape, class_names, bg_idx):
    """從 JSON 繪製 Ground Truth Mask"""
    h, w = target_shape
    default_val = bg_idx if bg_idx is not None else 255
    gt_mask = np.full((h, w), default_val, dtype=np.uint8)

    if not json_path or not json_path.exists():
        return gt_mask

    name_to_id = {str(v).lower(): k for k, v in class_names.items()}

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for shape in data.get('shapes', []):
        label = shape.get('label', '').lower()
        points = np.int32(np.array(shape['points'], dtype=np.float32))

        cls_id = name_to_id.get(label)
        if cls_id is not None:
            cv2.fillPoly(gt_mask, [points], color=cls_id)

    return gt_mask

def calculate_miou(pred_mask, gt_mask, num_classes, bg_idx=None):
    """計算 mIoU（包含所有類別，只排除忽略標籤 255）"""
    ious = []
    valid_mask = (gt_mask != 255)
    class_ious = {}

    for cls in range(num_classes):
        valid_pred, valid_gt = pred_mask[valid_mask], gt_mask[valid_mask]

        inter = np.logical_and(valid_pred == cls, valid_gt == cls).sum()
        union = np.logical_or(valid_pred == cls, valid_gt == cls).sum()

        score = inter / union if union > 0 else 0
        class_ious[cls] = score
        ious.append(score)  # ⭐ 包含所有類別，不再排除背景

    return np.mean(ious) if ious else 0, class_ious

def fuzzy_find_file(directory, scene_name, extension):
    """先試 exact stem 比對；找不到才回退到 keyword 模糊比對。

    Bug fix 2026-06-23: 原本只做 keyword 模糊比對，會在「目標檔不存在」時挑
    keyword 命中數最多的別張檔（同樣前綴的不同 scene）回來，導致 overlay
    用到錯的衛星底圖。改成優先 exact match，命中就立刻回傳；exact 失敗
    才走 fuzzy（保留向後相容，不影響舊資料）。
    """
    # 1. Fast path: exact stem match
    exact = directory / f"{scene_name}{extension}"
    if exact.exists():
        return exact

    # 2. Fallback: keyword fuzzy match (legacy)
    keywords = scene_name.replace('__', '_').split('_')
    keywords = [k for k in keywords if len(k) > 2]

    best_match = None
    max_hits = 0
    min_keywords_required = max(1, len(keywords) // 2 + 1)  # 至少要命中一半 keyword 才算 match

    for f in directory.glob(f"*{extension}"):
        hits = sum(1 for k in keywords if k.lower() in f.name.lower())
        if hits >= min_keywords_required and hits > max_hits:
            max_hits = hits
            best_match = f

    return best_match


def _read_bg_png(bg_path, img_h, img_w):
    """Read background PNG from disk (may be on NAS). Returns uint8 BGR array."""
    if bg_path and Path(str(bg_path)).exists():
        img = cv2.imread(str(bg_path))
        if img is not None:
            if img.shape[:2] != (img_h, img_w):
                bh, bw = img.shape[:2]
                pad_h, pad_w = max(0, img_h - bh), max(0, img_w - bw)
                if pad_h > 0 or pad_w > 0:
                    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                img = img[:img_h, :img_w]
            return img
    return np.zeros((img_h, img_w, 3), dtype=np.uint8)


def _save_visualization(overlay_path, overlay_data, mask_path, mask_data):
    """Write overlay JPG and mask-comparison PNG. Designed to run in a background thread."""
    cv2.imwrite(str(overlay_path), overlay_data, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(mask_path), mask_data)

def compute_conf_matrix(pred_mask, gt_mask, num_classes):
    """計算混淆矩陣，過濾忽略標籤 255"""
    valid_mask = (gt_mask != 255)
    pred_v = pred_mask[valid_mask]
    gt_v = gt_mask[valid_mask]

    k = (gt_v >= 0) & (gt_v < num_classes)
    hist = np.bincount(
        num_classes * gt_v[k].astype(int) + pred_v[k].astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist

def extract_patches_sliding_window(image, patch_size, stride):
    """
    從大圖用滑動窗格切出所有 patches（純記憶體操作）。
    為確保邊緣被完整覆蓋，最後一欄/列固定包含靠右/靠下的完整 patch。
    回傳: List of (patch_array, x, y)
    """
    h, w = image.shape[:2]
    # 產生起始座標，確保最後一格貼齊邊緣（用 dict 去除重複）
    ys = list(dict.fromkeys(list(range(0, h - patch_size, stride)) + [h - patch_size]))
    xs = list(dict.fromkeys(list(range(0, w - patch_size, stride)) + [w - patch_size]))
    patches = []
    for y in ys:
        for x in xs:
            patches.append((image[y:y+patch_size, x:x+patch_size], x, y))
    return patches

def run_reconstruction(recon_cfg, exp_config, model_path, output_base_dir):
    """以 Sliding Window 讀取完整大圖進行推論與重組"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"🚀 啟動大圖重組 (Sliding Window 移動窗格版)")

    arch_cfg = exp_config.get('architecture_cfg', {})
    dataset_cfg = exp_config.get('dataset', {})

    arch = exp_config.get('architecture', 'deeplabv3+')
    model_adapter, _fwd = _load_model_for_recon(arch, model_path, dataset_cfg, arch_cfg)
    model = model_adapter  # kept for attribute access (in_channels, class_names…)
    model.model.eval()

    num_classes = arch_cfg.get('num_classes', dataset_cfg.get('num_classes', 2))
    class_names = dataset_cfg.get('names', {0: 'Oil'})
    bg_idx = next((k for k, v in class_names.items() if 'background' in str(v).lower()), None)
    target_idx = arch_cfg.get('target_class_index', 0)

    patch_size = recon_cfg.get('image_size', 256)
    stride = recon_cfg.get('stride', patch_size // 2)
    infer_batch_size = recon_cfg.get('infer_batch_size', 32)
    num_workers = recon_cfg.get('num_workers', 4)

    # 建立輸出資料夾
    recon_out_dir = Path(output_base_dir) / "reconstruction"
    overlay_dir = recon_out_dir / "overlays"
    visual_dir = recon_out_dir / "mask_compare"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)

    # ⭐ 決定讀取模式：VRT 或 original_data_root
    use_vrt = bool(recon_cfg.get('vrt_dir'))

    if use_vrt:
        # === VRT 模式：動態從 VRT 檔讀取各場景 ===
        vrt_dir = Path(recon_cfg.get('vrt_dir'))
        scene_list_txt = Path(recon_cfg.get('scene_list_txt', ''))

        if not vrt_dir.exists():
            print(f"❌ [錯誤] vrt_dir 不存在: {vrt_dir}")
            return

        if not scene_list_txt.exists():
            print(f"❌ [錯誤] scene_list_txt 不存在: {scene_list_txt}")
            return

        # 讀取場景名稱清單
        scene_names = []
        with open(scene_list_txt, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    scene_names.append(line)

        if not scene_names:
            print(f"❌ [錯誤] scene_list_txt 內沒有任何場景")
            return

        print(f"✅ VRT 模式：找到 {len(scene_names)} 個場景")
        print(f"   vrt_dir={vrt_dir}")
        print(f"   patch_size={patch_size}, stride={stride}, infer_batch_size={infer_batch_size}")

        # 建立 scene_stem → 檔案路徑索引（支援 .tif / .vrt 及分群子目錄）
        _vrt_file_map = {
            p.stem: p for p in vrt_dir.rglob('*')
            if p.suffix.lower() in ('.tif', '.tiff', '.vrt')
        }
        scene_vrt_map = {}
        for scene_path_str in scene_names:
            scene_name = scene_path_str.split('/')[-1]
            scene_file = _vrt_file_map.get(scene_name)
            if scene_file is None:
                print(f"  ⚠️ 找不到場景檔案，跳過: {scene_name}")
                continue
            scene_vrt_map[scene_name] = scene_file

        if not scene_vrt_map:
            print(f"❌ [錯誤] 找不到任何可用的 TIF 檔，重組終止！")
            return

        print(f"✅ 可用 TIF 檔數: {len(scene_vrt_map)}")

    else:
        # === 傳統模式：直接讀取完整大圖（已堆疊的 TIF） ===
        full_img_dir = Path(recon_cfg.get('original_data_root', ''))
        if not full_img_dir.exists():
            print(f"❌ [錯誤] original_data_root 不存在: {full_img_dir}")
            return

        full_img_files = sorted(list(full_img_dir.glob('*.tif')) + list(full_img_dir.glob('*.tiff')))
        if not full_img_files:
            print(f"❌ [錯誤] 在 {full_img_dir} 找不到任何大圖，重組終止！")
            return

        print(f"✅ 傳統模式：找到 {len(full_img_files)} 張大圖")
        print(f"   patch_size={patch_size}, stride={stride}, infer_batch_size={infer_batch_size}")

    # 讀取 Normalization
    mean = dataset_cfg.get('mean', [0.5] * model.in_channels)
    std = dataset_cfg.get('std', [0.5] * model.in_channels)
    transforms = A.Compose([A.Normalize(mean=mean, std=std, max_pixel_value=1.0), ToTensorV2()])

    vis_dir = Path(recon_cfg.get('vis_image_dir', ''))
    json_dir_path = Path(recon_cfg.get('json_dir', str(vis_dir.parent / 'JSON')))

    # ⭐ 如果提供了 fold_test_scenes，只篩選這些場景對應的檔案
    fold_test_scenes = recon_cfg.get('fold_test_scenes')
    if fold_test_scenes:
        print(f"  🔍 篩選模式：只處理指定 fold 的 {len(fold_test_scenes)} 個場景")
        fold_test_scenes_set = set(fold_test_scenes)
    else:
        fold_test_scenes_set = None

    try:
        import tifffile
    except ImportError:
        print("請安裝 tifffile: pip install tifffile")
        return

    # 初始化全域混淆矩陣
    global_conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    # ==================== ANNOT-ONLY METRICS START ====================
    # 在 recon 階段同時計算「只在標記區」(annot-only) 的混淆矩陣，避免事後重跑。
    # 跟既有 raw recon 的差異：raw 用 JSON 生 GT (未標註區當 bg，FP 在 unannot 算入)，
    # annot-only 用 mask TIF (未標註區 255 排除)，FP 在 unannot 不算。
    annot_only_enabled = recon_cfg.get('compute_annot_only', True)
    annot_mask_dir = recon_cfg.get('annot_mask_dir', '/home/alanyh/oil_dataset/new/full_band/mask')
    global_annot_conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64) if annot_only_enabled else None
    if annot_only_enabled:
        print(f"  [Annot-Only] 啟用 annot-only metric 計算 (mask_dir={annot_mask_dir})")
    # ==================== ANNOT-ONLY METRICS END ====================

    # 決定要遍歷的場景
    if use_vrt:
        scenes_to_process = [(name, path) for name, path in scene_vrt_map.items()]
    else:
        scenes_to_process = [(f.stem, f) for f in full_img_files]

    # Async save pool: imwrite runs in background while next scene is being processed
    _save_pool = ThreadPoolExecutor(max_workers=2)
    _pending_save = None  # Future for previous scene's visualization saves

    for scene_name, scene_source in scenes_to_process:
        # ⭐ 如果指定了 fold_test_scenes，跳過不在清單中的場景
        if fold_test_scenes_set is not None and scene_name not in fold_test_scenes_set:
            continue

        print(f"\n正在重組: {scene_name} ...")

        # 讀取影像尺寸或完整大圖（VRT 模式只讀 metadata，避免全圖佔用 RAM）
        if use_vrt:
            try:
                import rasterio
            except ImportError:
                print("❌ 請安裝 rasterio: pip install rasterio")
                return
            with rasterio.open(str(scene_source)) as src:
                img_h, img_w = src.height, src.width
        else:
            # 從已堆疊的 TIF 讀取
            full_img = tifffile.imread(str(scene_source))
            np.nan_to_num(full_img, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            full_img = np.clip(full_img, 0.0, 10.0)
            if full_img.ndim == 3 and full_img.shape[0] < full_img.shape[1] and full_img.shape[0] < full_img.shape[2]:
                full_img = np.ascontiguousarray(full_img.transpose(1, 2, 0), dtype=np.float32)
            else:
                if full_img.ndim == 2:
                    full_img = np.expand_dims(full_img, axis=-1)
                if full_img.dtype != np.float32:
                    full_img = full_img.astype(np.float32)
            img_h, img_w = full_img.shape[:2]

        # 影像若小於 patch_size 則跳過
        if img_h < patch_size or img_w < patch_size:
            print(f"  ⚠️ 影像尺寸 ({img_h}x{img_w}) 小於 patch_size ({patch_size})，跳過。")
            continue

        prob_map = np.zeros((img_h, img_w, num_classes), dtype=np.float32)
        weight_map = np.zeros((img_h, img_w, 1), dtype=np.float32)

        # --- 載入完整場景到記憶體 (VRT 或 TIF 都統一走這條路) ---
        if use_vrt:
            # 並行讀取 8 個 band TIF + cv2 resize（比 VRT src.read() 快 4×）
            print(f"  並行讀取場景 ({img_h}×{img_w}×8 bands)...")
            raw = _read_vrt_parallel(scene_source, img_h, img_w, max_workers=num_workers)
            full_img = (raw / 10000.0).transpose(1, 2, 0)  # (H, W, C)
            del raw
            np.nan_to_num(full_img, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            full_img = np.clip(full_img, 0.0, 10.0)
        # else: full_img 已在上方 TIF 路徑中載入

        # --- 統一推論路徑 (VRT / TIF 共用) ---
        all_patches = extract_patches_sliding_window(full_img, patch_size, stride)
        print(f"  切出 {len(all_patches)} 個 patches")

        # 在 inference 期間並行讀取 bg PNG（NAS 讀取 ~40s，inference ~120s，完全藏進去）
        bg_png_path = fuzzy_find_file(vis_dir, scene_name, '.jpg') or fuzzy_find_file(vis_dir, scene_name, '.png')
        _bg_future = _save_pool.submit(_read_bg_png, bg_png_path, img_h, img_w)

        def apply_transform(patch):
            return transforms(image=patch)['image']

        with torch.no_grad(), torch.autocast('cuda'), ThreadPoolExecutor(max_workers=num_workers) as preproc_ex:
            for batch_start in tqdm(range(0, len(all_patches), infer_batch_size), desc="Batch Inference (TTA)"):
                batch = all_patches[batch_start:batch_start + infer_batch_size]

                tensor_list = list(preproc_ex.map(apply_transform, [p[0] for p in batch]))
                tensors = torch.stack(tensor_list).to(device)

                # TTA: 原圖 + 水平翻轉 + 垂直翻轉
                logits_orig  = _fwd(tensors).float().cpu()
                logits_hflip = torch.flip(_fwd(torch.flip(tensors, dims=[3])).float(), dims=[3]).cpu()
                logits_vflip = torch.flip(_fwd(torch.flip(tensors, dims=[2])).float(), dims=[2]).cpu()
                fused_logits = (logits_orig + logits_hflip + logits_vflip) / 3.0
                probs = F.softmax(fused_logits, dim=1).numpy().transpose(0, 2, 3, 1)

                for i, (_, x, y) in enumerate(batch):
                    prob_map[y:y+patch_size, x:x+patch_size, :] += probs[i]
                    weight_map[y:y+patch_size, x:x+patch_size, 0] += 1.0
                del tensors, tensor_list, logits_orig, logits_hflip, logits_vflip, fused_logits, probs

        # in-place 除法，避免 prob_map/weight_map 產生額外 ~1GB 暫存陣列
        np.divide(prob_map, np.maximum(weight_map, 1.0), out=prob_map)
        final_mask = np.argmax(prob_map, axis=-1).astype(np.uint8)
        del prob_map, weight_map
        gc.collect()

        # ⭐ Cloud mask 後處理（SWIR1+blue dual-threshold，跟 HNM Step 1 一致）
        # 啟用方式: recon_cfg['cloud_mask']['enabled'] = True
        cloud_cfg = recon_cfg.get('cloud_mask', {}) or {}
        if cloud_cfg.get('enabled', False) and bg_idx is not None:
            blue_band_idx  = cloud_cfg.get('blue_band_idx', 1)   # 0-indexed: B02 (492nm)
            swir1_band_idx = cloud_cfg.get('swir1_band_idx', 6)  # 0-indexed: B11 (1614nm)
            blue_thresh    = cloud_cfg.get('blue_thresh', 0.20)
            swir1_thresh   = cloud_cfg.get('swir1_thresh', 0.05)
            try:
                # 使用已載入記憶體的 full_img (reflectance 格式)
                blue_patch  = full_img[:, :, blue_band_idx]
                swir1_patch = full_img[:, :, swir1_band_idx]
                cloud_mask_raw = (swir1_patch > swir1_thresh) & (blue_patch > blue_thresh)
                cloud_px = int(cloud_mask_raw.sum())
                if cloud_px > 0:
                    final_mask[cloud_mask_raw] = int(bg_idx)
                    print(f"  ☁ Cloud mask: {cloud_px:>10,d} px → bg "
                          f"(SWIR1>{swir1_thresh} AND blue>{blue_thresh})")
                else:
                    print(f"  ☁ Cloud mask: 0 px 觸發（場景無雲或門檻太嚴）")
                del blue_patch, swir1_patch, cloud_mask_raw
            except Exception as e:
                print(f"  ⚠ Cloud mask 失敗: {e}")

        del full_img, all_patches
        gc.collect()

        # 4. GT 比對
        json_path = fuzzy_find_file(json_dir_path, scene_name, '.json')
        gt_mask = generate_gt_mask_from_json(json_path, (img_h, img_w), class_names, bg_idx)
        miou, class_ious = calculate_miou(final_mask, gt_mask, num_classes, bg_idx=bg_idx)
        oil_iou = class_ious.get(target_idx, 0)

        if json_path:
            global_conf_matrix += compute_conf_matrix(final_mask, gt_mask, num_classes)

        # ==================== ANNOT-ONLY METRICS START ====================
        if annot_only_enabled:
            try:
                import rasterio as _rio
                mask_tif_path = Path(annot_mask_dir) / f'{scene_name}.tif'
                if mask_tif_path.exists():
                    with _rio.open(str(mask_tif_path)) as _msrc:
                        mask_raw = _msrc.read(1)
                    if mask_raw.shape != (img_h, img_w):
                        mask_raw = cv2.resize(mask_raw, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                    gt_annot = np.full_like(mask_raw, 255, dtype=np.uint8)
                    gt_annot[mask_raw == 0] = 0   # oil
                    gt_annot[mask_raw == 1] = bg_idx if bg_idx is not None else 1  # annotated bg
                    gt_annot[mask_raw == 2] = bg_idx if bg_idx is not None else 1  # others → bg
                    global_annot_conf_matrix += compute_conf_matrix(final_mask, gt_annot, num_classes)
                    del mask_raw, gt_annot
                else:
                    print(f"  ⚠ [Annot-Only] 找不到 mask TIF: {mask_tif_path}, 跳過此 scene 的 annot-only 計算")
            except Exception as _e:
                print(f"  ⚠ [Annot-Only] 計算失敗: {_e}")
        # ==================== ANNOT-ONLY METRICS END ====================

        # ⭐ 分別報告各類別的 IoU
        iou_report = " | ".join([f"{class_names.get(i, f'C{i}')}: {class_ious.get(i, 0):.4f}" for i in range(num_classes)])
        print(f"  👉 mIoU: {miou:.4f} | {iou_report}")
        if json_path: print(f"  👉 成功載入標註檔: {json_path.name}")
        else: print(f"  ⚠️ 找不到 {scene_name} 相關的 JSON")

        # 5. 視覺化輸出
        # bg PNG 在 inference 期間已並行讀取完畢
        bg_img = _bg_future.result()

        overlay = bg_img.copy()
        for cls_idx, cls_name in class_names.items():
            if cls_idx == bg_idx: continue
            color = [0, 255, 255] if 'oil' in str(cls_name).lower() else ([0, 0, 255] if 'other' in str(cls_name).lower() else COLOR_PALETTE.get((cls_idx % 15) + 1, [255, 255, 255]))
            overlay[final_mask == cls_idx] = color

        final_overlay = cv2.addWeighted(overlay, 0.4, bg_img, 0.6, 0)

        COMPARE_PALETTE = {'oil': [0, 255, 255], 'others': [0, 255, 0], 'background': [139, 0, 0]}
        pred_vis = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        pred_vis[:] = COMPARE_PALETTE['background']
        gt_gray = (gt_mask * (255 // max(1, num_classes - 1))).astype(np.uint8)
        gt_vis = cv2.cvtColor(gt_gray, cv2.COLOR_GRAY2BGR)
        for cls_idx, cls_name in class_names.items():
            if cls_idx == bg_idx: continue
            c_name_lower = str(cls_name).lower()
            color = COMPARE_PALETTE['oil'] if 'oil' in c_name_lower else (COMPARE_PALETTE['others'] if 'other' in c_name_lower else [128, 128, 128])
            pred_vis[final_mask == cls_idx] = color
        comparison_img = cv2.hconcat([gt_vis, pred_vis])

        # 等前一個 scene 的存檔完成（避免記憶體同時持有過多大圖）
        if _pending_save is not None:
            _pending_save.result()

        # 異步存檔：imwrite 在背景執行，主迴圈直接進入下一個 scene
        _overlay_path = overlay_dir / f"{scene_name}_iou_{oil_iou:.4f}.jpg"
        _mask_path    = visual_dir  / f"{scene_name}_iou_{oil_iou:.4f}_masks.png"
        _pending_save = _save_pool.submit(_save_visualization,
                                          _overlay_path, final_overlay,
                                          _mask_path, comparison_img)
        # final_overlay / comparison_img 由 thread 持有，不在這裡 del
        del bg_img, overlay, pred_vis, gt_gray, gt_vis
        del final_mask, gt_mask
        gc.collect()
        try:
            import ctypes; ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 等待最後一個 scene 的異步存檔完成
    if _pending_save is not None:
        _pending_save.result()
    _save_pool.shutdown(wait=False)

    # =========================================================
    # ===            Reconstruction Overall Metrics         ===
    # =========================================================
    print(f"\n{'='*60}")
    print(f"🏆 Reconstruction Overall Metrics (全域大圖真實統計，無重複計算)")
    print(f"{'='*60}")

    recon_metrics = {}

    if np.sum(global_conf_matrix) == 0:
        print("沒有計算到任何有效的 GT Mask！")
    else:
        intersection = np.diag(global_conf_matrix)
        gt_totals = global_conf_matrix.sum(axis=1)
        pred_totals = global_conf_matrix.sum(axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            ious = intersection / (gt_totals + pred_totals - intersection)
            recalls = intersection / gt_totals
            precisions = intersection / pred_totals
            f1s = 2 * (precisions * recalls) / (precisions + recalls)

        ious = np.nan_to_num(ious, nan=0.0)
        recalls = np.nan_to_num(recalls, nan=0.0)
        precisions = np.nan_to_num(precisions, nan=0.0)
        f1s = np.nan_to_num(f1s, nan=0.0)

        oa = intersection.sum() / (global_conf_matrix.sum() + 1e-9)

        # ⭐ 計算包含所有類別的 mIoU
        global_miou = np.mean(ious)  # 包含所有類別
        global_f1_macro = np.mean(f1s)  # 包含所有類別

        # 同時計算前景類別的 mIoU（用於參考）
        valid_indices = [i for i in range(num_classes) if i != bg_idx]
        foreground_iou = ious[valid_indices]
        foreground_f1 = f1s[valid_indices]
        foreground_miou = np.mean(foreground_iou) if len(foreground_iou) > 0 else 0
        foreground_f1_macro = np.mean(foreground_f1) if len(foreground_f1) > 0 else 0

        # Pandas 表格渲染（對齊 val() Table 5 格式）
        CLASS_NAMES = [class_names.get(c, f'C{c}') for c in range(num_classes)]

        conf_matrix_pct = np.zeros_like(global_conf_matrix, dtype=float)
        np.divide(global_conf_matrix * 100, gt_totals[:, None],
                  out=conf_matrix_pct, where=gt_totals[:, None] != 0)

        df = pd.DataFrame(conf_matrix_pct, index=CLASS_NAMES, columns=CLASS_NAMES)
        df = df.map(lambda x: f"{x:.1f}")
        df['# of pixels (test set)'] = gt_totals.astype(str)
        df['Recall (%)'] = [f"{x:.1f}" for x in (recalls * 100)]
        df.loc['IoU (%)']       = [f"{x:.1f}" for x in (ious * 100)]       + ['mIoU:',     f"{global_miou * 100:.1f}"]
        df.loc['Precision (%)'] = [f"{x:.1f}" for x in (precisions * 100)] + ['OA:',       f"{oa * 100:.1f}"]
        df.loc['F1-score (%)']  = [f"{x:.1f}" for x in (f1s * 100)]        + ['F1-macro:', f"{global_f1_macro * 100:.1f}"]

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\n" + "=" * 145)
        print("Table 5 (Reconstruction): Confusion Matrix on full images. Columns = predicted, Rows = reference (%).")
        print("=" * 145)
        print(df.to_string())
        print("=" * 145 + "\n")

        # 建立回傳指標字典
        recon_metrics = {
            'recon_mIoU': f"{global_miou:.4f}",  # ⭐ 包含所有類別的 mIoU
            'recon_Accuracy': f"{oa:.4f}",
            'recon_F1-macro': f"{global_f1_macro:.4f}",
        }
        # 如果有背景類別，額外報告前景類別的 mIoU 以供參考
        if bg_idx is not None and len(valid_indices) < num_classes:
            recon_metrics['recon_mIoU_foreground'] = f"{foreground_miou:.4f}"
            recon_metrics['recon_F1-macro_foreground'] = f"{foreground_f1_macro:.4f}"

        # 報告所有類別的指標（不只是前景）
        for c_idx in range(num_classes):
            c_name = class_names.get(c_idx, f'Class_{c_idx}')
            recon_metrics[f'recon_IoU({c_name})'] = f"{ious[c_idx]:.4f}"
            recon_metrics[f'recon_Precision({c_name})'] = f"{precisions[c_idx]:.4f}"
            recon_metrics[f'recon_Recall({c_name})'] = f"{recalls[c_idx]:.4f}"
            recon_metrics[f'recon_F1({c_name})'] = f"{f1s[c_idx]:.4f}"

    # ==================== ANNOT-ONLY METRICS SUMMARY ====================
    if annot_only_enabled and global_annot_conf_matrix is not None and np.sum(global_annot_conf_matrix) > 0:
        annot_intersection = np.diag(global_annot_conf_matrix)
        annot_gt_totals = global_annot_conf_matrix.sum(axis=1)
        annot_pred_totals = global_annot_conf_matrix.sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            annot_ious = np.nan_to_num(annot_intersection / (annot_gt_totals + annot_pred_totals - annot_intersection), nan=0.0)
            annot_recalls = np.nan_to_num(annot_intersection / annot_gt_totals, nan=0.0)
            annot_precisions = np.nan_to_num(annot_intersection / annot_pred_totals, nan=0.0)
            annot_f1s = np.nan_to_num(2 * (annot_precisions * annot_recalls) / (annot_precisions + annot_recalls), nan=0.0)
        annot_miou = float(np.mean(annot_ious))
        annot_f1_macro = float(np.mean(annot_f1s))
        annot_oa = float(annot_intersection.sum() / (np.sum(global_annot_conf_matrix) + 1e-9))

        print(f"\n{'='*60}")
        print(f"🏆 Annot-Only Metrics (排除 mask==255 unannotated 區，用於論文數據指標)")
        print(f"{'='*60}")
        for c_idx in range(num_classes):
            c_name = class_names.get(c_idx, f'Class_{c_idx}')
            print(f"  {c_name}: IoU={annot_ious[c_idx]:.4f}  Prec={annot_precisions[c_idx]:.4f}  "
                  f"Recall={annot_recalls[c_idx]:.4f}  F1={annot_f1s[c_idx]:.4f}")
        print(f"  mIoU={annot_miou:.4f}  F1-macro={annot_f1_macro:.4f}  OA={annot_oa:.4f}")

        recon_metrics['annot_recon_mIoU'] = f"{annot_miou:.4f}"
        recon_metrics['annot_recon_F1-macro'] = f"{annot_f1_macro:.4f}"
        recon_metrics['annot_recon_Accuracy'] = f"{annot_oa:.4f}"
        for c_idx in range(num_classes):
            c_name = class_names.get(c_idx, f'Class_{c_idx}')
            recon_metrics[f'annot_recon_IoU({c_name})'] = f"{annot_ious[c_idx]:.4f}"
            recon_metrics[f'annot_recon_Precision({c_name})'] = f"{annot_precisions[c_idx]:.4f}"
            recon_metrics[f'annot_recon_Recall({c_name})'] = f"{annot_recalls[c_idx]:.4f}"
            recon_metrics[f'annot_recon_F1({c_name})'] = f"{annot_f1s[c_idx]:.4f}"

    print(f"\n=== 重組任務完成！ ===")
    return recon_metrics
