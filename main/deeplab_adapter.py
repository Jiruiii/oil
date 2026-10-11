# ======================================================================================
# ===           deeplab_adapter.py (YAML 驅動動態版：EMA + TTA + 動態映射)             ===
# ======================================================================================
import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import yaml

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Per-process rasterio dataset cache: avoids re-opening VRT/TIF on every __getitem__ call.
# Each DataLoader worker process gets its own copy (fork semantics), so no cross-process sharing.
_ds_cache: dict = {}

def _cached_rasterio_open(path: str):
    if path not in _ds_cache:
        import rasterio
        _ds_cache[path] = rasterio.open(path)
    return _ds_cache[path]

def _read_txt_list(txt_path):
    """讀取 txt 檔案，回傳非空行列表（patch 檔名）"""
    return [l.strip() for l in Path(txt_path).read_text().splitlines() if l.strip()]

class FocalLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        valid_mask = (targets != self.ce.ignore_index)
        return focal_loss[valid_mask].mean() if valid_mask.sum() > 0 else focal_loss.sum()


class TverskyLoss(nn.Module):
    """[recall 導向] 前景(oil)類的 Tversky loss。beta>alpha → FN 懲罰重於 FP → 拉 recall。
    ignore_index 像素排除；oil_class = 前景 index。batch-level soft 計數，smooth 防除零。
    focal_gamma = 直接指數 q（非 Abraham 倒數 1/γ）；q>1 → Focal Tversky：(1-TI)^q，聚焦難 batch。
    focal_gamma==1.0 時走原路徑，逐 bit 等同純 Tversky（保 baseline/既有 run 可比）。"""
    def __init__(self, alpha=0.3, beta=0.7, oil_class=0, ignore_index=255, smooth=1.0, focal_gamma=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha; self.beta = beta
        self.oil_class = oil_class; self.ignore_index = ignore_index; self.smooth = smooth
        self.focal_gamma = focal_gamma

    def forward(self, inputs, targets):
        p = torch.softmax(inputs, dim=1)[:, self.oil_class]   # (B,H,W) P(oil)
        vm = (targets != self.ignore_index).float()
        g = (targets == self.oil_class).float()
        TP = (p * g * vm).sum()
        FP = (p * (1 - g) * vm).sum()
        FN = ((1 - p) * g * vm).sum()
        ti = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        base = 1 - ti
        if self.focal_gamma == 1.0:   # 逐 bit 保留原 Tversky 路徑
            return base
        return base.pow(self.focal_gamma)

# --- 波段消融 組2/組3：Zakzouk et al. (2024) 光譜指數 ---
# 輸入：清理後的 8-band patch (8, H, W) float32（已 /10000、nodata 歸零）
# 8-band 0-based index: B01=0, B02=1, B03=2, B04=3, B08=4, B8A=5, B11=6, B12=7
_INDEX_EPS = 1e-6

def compute_spectral_index(patch, name):
    """依 name 計算 3 通道光譜指數，回傳 (3, H, W) float32。
    index_c: R=(B01-B11)/(B01+B11), G=(B01+B02)/(B03+B11), B=(B01+B02)/(B03+B12)
    index_d: 同 index_c，但第三通道分母 B12 改 B8A
    分母加 epsilon 防除零（nodata 已歸零的像素三通道皆為 0，不影響 zero-fill 判斷）；
    輸出 clip 到 [-20, 20]，防極暗像素比值爆表污染 mean/std 統計。"""
    b01, b02, b03 = patch[0], patch[1], patch[2]
    b8a, b11, b12 = patch[5], patch[6], patch[7]
    if name == 'index_c':
        chans = [(b01 - b11) / (b01 + b11 + _INDEX_EPS),
                 (b01 + b02) / (b03 + b11 + _INDEX_EPS),
                 (b01 + b02) / (b03 + b12 + _INDEX_EPS)]
    elif name == 'index_d':
        chans = [(b01 - b11) / (b01 + b11 + _INDEX_EPS),
                 (b01 + b02) / (b03 + b11 + _INDEX_EPS),
                 (b01 + b02) / (b03 + b8a + _INDEX_EPS)]
    else:
        raise ValueError(f"未知 spectral_index: {name}（支援 index_c / index_d）")
    return np.clip(np.stack(chans), -20.0, 20.0).astype(np.float32)

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None, dataset_cfg=None, file_list=None,
                 bg_fold_only=None, bg_fold_exclude=None, bg_scene_filter=None,
                 apply_oil_ignore_ring=False):
        self.image_dir = Path(image_dir) if image_dir else None
        self.mask_dir = Path(mask_dir)
        self.transforms = transforms
        dataset_cfg = dataset_cfg or {}

        # 從 YAML 讀取動態設定
        self.num_classes = dataset_cfg.get('num_classes', 2)
        self.pixel_mapping = dataset_cfg.get('pixel_mapping', None)
        # [ignore-ring] 油汙 GT 外圈 N px 內的「未標註(raw 255)」像素改回 ignore,
        # 只在訓練集生效(apply_oil_ignore_ring=True 才啟用;val/eval 語意不變)。
        # 動機:標註為保守框,油邊緣未框到的薄油被 255→bg 映射教成背景,
        # 系統性推向 under-segmentation(status ⑤-7 recall 0.47 診斷)。
        self.oil_ignore_ring = int(dataset_cfg.get('oil_ignore_ring', 0) or 0) \
            if apply_oil_ignore_ring else 0
        self.target_class_index = int(dataset_cfg.get('target_class_index', 0))
        self.patch_size = int(dataset_cfg.get('patch_size', 256))
        self.vrt_dir = dataset_cfg.get('vrt_dir', None)
        # band_indices: list of 0-based indices to select from the stacked TIF after reading.
        # None (default) = use all bands. Example: [0,1,2,5,6,7] for 6-band subset.
        band_indices = dataset_cfg.get('band_indices', None)
        self.band_indices = np.array(band_indices, dtype=int) if band_indices is not None else None
        # spectral_index: 'index_c'/'index_d'（Zakzouk 2024 組2/組3），從 8-band 算 3 通道指數。
        # 與 band_indices 互斥；有設時優先於 band_indices。
        self.spectral_index = dataset_cfg.get('spectral_index', None)

        # ★ VRT 正樣本模式：從 fold TXT 檔名解析座標，從 scene VRT 讀取影像
        # 設 read_images_from_vrt: true 後不需要 images_dir
        self.read_images_from_vrt = dataset_cfg.get('read_images_from_vrt', False)
        self.pos_coords = []

        # coord_vrt 模式：動態從 VRT 讀取 bg 樣本（不需預切 bg patch）
        # 不填 bg_coord_txt 時維持現有 flat-dir 模式，行為完全不變
        # [W2 修法] bg_scene_filter 是 set of scene names；提供就只收這些 scene 的 BG patch，
        #         比 fold_only/fold_exclude 嚴謹（避免 val/test 互相滲漏）
        self.bg_coords = []
        # [P8][codex R6-2] 擴池模式下 BG 的 runtime validity failure 改為 fail-fast（見 _get_coord_vrt_item）
        self.bg_fail_fast = bool((dataset_cfg.get('bg_pool_cfg') or {}).get('draws_freeze_csv'))
        bg_coord_txt = dataset_cfg.get('bg_coord_txt', None)
        if bg_coord_txt and self.vrt_dir:
            self._load_bg_coords(bg_coord_txt,
                                 fold_only=bg_fold_only,
                                 fold_exclude=bg_fold_exclude,
                                 scene_filter=bg_scene_filter)

        # 預先建立 scene→path 索引（支援 .tif / .vrt 及分群子目錄）
        self._vrt_map: dict = {}
        if self.vrt_dir:
            self._vrt_map = {
                p.stem: p for p in Path(self.vrt_dir).rglob('*')
                if p.suffix.lower() in ('.tif', '.tiff', '.vrt')
            }

        if self.read_images_from_vrt and self.vrt_dir and file_list is not None:
            import re as _re
            _patch_re = _re.compile(r'^(.+)_patch_x(\d+)_y(\d+)$')
            vrt_map = self._vrt_map
            missing_vrt = 0
            for fname in file_list:
                stem = Path(fname).stem
                m = _patch_re.match(stem)
                if not m:
                    continue
                scene, x, y = m.group(1), int(m.group(2)), int(m.group(3))
                vrt_path = vrt_map.get(scene)
                if vrt_path:
                    self.pos_coords.append({'stem': stem, 'scene': scene, 'vrt_path': vrt_path, 'x': x, 'y': y})
                else:
                    missing_vrt += 1
            self.image_files = []
            # [F4] fail-fast：找不到 VRT 的 patch 原本只計數跳過，dataset 長度靜默縮小，
            # 訓練分布被無聲改變。除非明確設 OIL_ALLOW_MISSING=1（debug 用），一律 raise。
            if missing_vrt:
                if os.environ.get('OIL_ALLOW_MISSING') == '1':
                    print(f"  ⚠ [fold_vrt] {missing_vrt} 筆找不到對應 VRT，已跳過（OIL_ALLOW_MISSING=1）")
                else:
                    raise RuntimeError(
                        f"[F4] {missing_vrt} 筆 fold patch 找不到對應 VRT（vrt_dir={self.vrt_dir}）。"
                        f"資料不完整不可靜默縮小 dataset；確認 VRT 目錄或設 OIL_ALLOW_MISSING=1 除錯。")
            print(f"  [fold_vrt] 載入 {len(self.pos_coords)} 筆 fold patch 座標（oil + others，VRT 模式）")
        elif file_list is not None:
            found = []
            for fname in file_list:
                p = self.image_dir / fname
                if p.exists():
                    found.append(p)
                else:
                    stem = Path(fname).stem
                    for ext in ['.tif', '.tiff', '.png', '.npy', '.jpg']:
                        alt = self.image_dir / (stem + ext)
                        if alt.exists():
                            found.append(alt)
                            break
            self.image_files = sorted(found)
        else:
            self.image_files = sorted([
                f for f in self.image_dir.iterdir()
                if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.npy']
            ])

    def _load_bg_coords(self, txt_path: str, fold_only=None, fold_exclude=None, scene_filter=None):
        """讀取 bg 座標 TSV，解析成 list of dict {scene, x, y}。
        - scene_filter: 一個 set/iterable，只收 scene 名在裡面的（推薦做法，[W2 修法]）
        - fold_only / fold_exclude: 舊版兼容，若 tsv 內無 'fold' 欄則自動忽略
        """
        import csv
        if scene_filter is not None:
            scene_filter = set(scene_filter)
        filt_n = 0
        with open(txt_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                # scene filter（優先）
                if scene_filter is not None and row['scene'] not in scene_filter:
                    filt_n += 1
                    continue
                # fold 欄位向下相容（新 tsv 可以不放 fold 欄）
                if 'fold' in row:
                    fold_val = int(row['fold']) if row['fold'] else None
                    if fold_only is not None and fold_val != fold_only:
                        continue
                    if fold_exclude is not None and fold_val == fold_exclude:
                        continue
                self.bg_coords.append({
                    'scene': row['scene'],
                    'x': int(row['x']),
                    'y': int(row['y']),
                })
        if scene_filter is not None:
            info = f"scene_filter={len(scene_filter)} scenes, 載入 {len(self.bg_coords)} 筆，過濾掉 {filt_n} 筆"
        elif fold_only:
            info = f"fold={fold_only}, 載入 {len(self.bg_coords)} 筆"
        elif fold_exclude:
            info = f"exclude fold={fold_exclude}, 載入 {len(self.bg_coords)} 筆"
        else:
            info = f"all folds, 載入 {len(self.bg_coords)} 筆"
        print(f"  [coord_vrt] {info}")

    def set_augmentation_seed(self, seed):
        """[2026-07-22 可重現性] 重設 Albumentations Compose 的內部 RNG。
        Albumentations 2.0.x 的 Compose 有獨立 RNG，torch/np/random seed 皆無效，
        必須顯式 set_random_seed 才能跨 run 重現。worker_init_fn 對每個 worker 以
        distinct seed 呼叫；workers=0 由 Compose 建構時的 seed= 覆蓋。"""
        if self.transforms is not None and hasattr(self.transforms, 'set_random_seed'):
            self.transforms.set_random_seed(int(seed) & 0x7FFFFFFF)

    def __len__(self):
        return len(self.pos_coords) + len(self.image_files) + len(self.bg_coords)

    def __getitem__(self, idx):
        n_pos   = len(self.pos_coords)
        n_files = len(self.image_files)

        # VRT 正樣本分支
        if idx < n_pos:
            return self._get_pos_vrt_item(idx)

        # coord_vrt 負樣本分支
        if idx >= n_pos + n_files:
            return self._get_coord_vrt_item(idx - n_pos - n_files)

        img_path = self.image_files[idx - n_pos]
        
        # 尋找對應的 Mask
        if img_path.suffix.lower() in ['.tif', '.tiff']:
            mask_path = self.mask_dir / (img_path.stem + '.tif')
            if not mask_path.exists(): mask_path = self.mask_dir / (img_path.stem + '.tiff')
        else:
            mask_path = self.mask_dir / (img_path.stem + '.png')
            
        if not mask_path.exists():
            for ext in ['.png', '.tif']:
                if (self.mask_dir / (img_path.stem + ext)).exists():
                    mask_path = self.mask_dir / (img_path.stem + ext); break
        
        # 讀取影像
        if img_path.suffix.lower() == '.npy':
            image = np.load(str(img_path))
            if image.ndim == 3 and image.shape[0] < image.shape[2]: image = image.transpose(1, 2, 0)
            image = image.astype(np.float32)
        elif img_path.suffix.lower() in ['.tif', '.tiff']:
            try: import tifffile; image = tifffile.imread(str(img_path))
            except: image = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if image is not None:
                if image.ndim == 3 and image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
                     image = image.transpose(1, 2, 0)
                if image.ndim == 2: image = np.expand_dims(image, axis=-1)
                image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            image = cv2.imread(str(img_path))
            if image is not None: image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 讀取 Mask
        if mask_path.suffix.lower() in ['.tif', '.tiff']:
            try: import tifffile; mask = tifffile.imread(str(mask_path))
            except: mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        else:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            
        # [F4] fail-fast：mask 讀不到原本合成全 0，等於捏造標籤
        if mask is None:
            raise RuntimeError(f"[F4] mask 讀取失敗（cv2/tifffile 皆回 None）: {mask_path}")

        # ⭐ 核心魔法：根據 YAML 定義的 mapping 動態替換像素，完全不寫死 15 或 2
        mask = mask.astype(np.int64)
        if self.pixel_mapping:
            new_mask = np.full_like(mask, 255) # 預設全部設為 ignore_index
            for orig_val, new_val in self.pixel_mapping.items():
                new_mask[mask == int(orig_val)] = int(new_val)
            mask = new_mask
        
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']
            
        if mask.ndim == 3: mask = mask.squeeze(-1)
        return image, mask.long()

    def _get_pos_vrt_item(self, idx: int):
        """從 stacked TIF 讀取正樣本影像，mask 從全場景 mask TIF 以 Window 讀取。"""
        from rasterio.windows import Window

        coord = self.pos_coords[idx]
        win = Window(coord['x'], coord['y'], self.patch_size, self.patch_size)
        try:
            src = _cached_rasterio_open(str(coord['vrt_path']))
            patch = src.read(window=win, boundless=True, fill_value=0).astype(np.float32) / 10000.0
        except Exception as e:
            # [F4] fail-fast：讀取失敗原本回 None 由 collate 靜默過濾，樣本無聲消失
            raise RuntimeError(f"[F4][pos_vrt] 影像讀取失敗 {coord['vrt_path']}: {e}") from e

        np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        patch[patch > 100.0] = 0.0   # zero out float32 nodata fill (e.g., 9.97e36)
        patch[patch < -100.0] = 0.0
        if self.spectral_index:
            patch = compute_spectral_index(patch, self.spectral_index)
        elif self.band_indices is not None:
            patch = patch[self.band_indices]
        image = np.ascontiguousarray(patch.transpose(1, 2, 0), dtype=np.float32)

        mask_tif = self.mask_dir / f"{coord['scene']}.tif"
        try:
            msrc = _cached_rasterio_open(str(mask_tif))
            mask = msrc.read(1, window=win, boundless=True, fill_value=255).astype(np.int64)
        except Exception as e:
            # [F4] fail-fast：mask 讀取失敗原本合成全 255，經 pixel_mapping 255→bg
            # 會把「讀取失敗的正樣本」靜默變成純背景監督
            raise RuntimeError(f"[F4][mask] mask 讀取失敗 {mask_tif}: {e}") from e
        mask_raw = mask
        if self.pixel_mapping:
            new_mask = np.full_like(mask, 255)
            for orig_val, new_val in self.pixel_mapping.items():
                new_mask[mask == int(orig_val)] = int(new_val)
            mask = new_mask
        # [ignore-ring] 油汙外圈 ring px 內、且原始值為 255(未標註)的像素 → ignore。
        # 手標背景(raw 1/2)不受影響;遠場 255→bg 監督全保留。
        if self.oil_ignore_ring > 0:
            oil = (mask == self.target_class_index)
            if oil.any():
                k = 2 * self.oil_ignore_ring + 1
                dilated = cv2.dilate(oil.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
                ring = dilated & ~oil & (mask_raw == 255)
                mask[ring] = 255

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(np.ascontiguousarray(mask))
        if mask.ndim == 3:
            mask = mask.squeeze(-1)
        return image, mask.long()

    def _get_coord_vrt_item(self, coord_idx: int):
        """從 VRT 動態讀取 bg patch，回傳 (image_tensor, full-bg mask)。
        nodata 比例 > 10% 時回傳 None，由 oil_collate_fn 過濾。
        rasterio.open 必須在此處開啟，不能在 __init__（多 worker 無法共用 file handle）。
        """
        import rasterio
        from rasterio.windows import Window

        coord = self.bg_coords[coord_idx]
        vrt_path = self._vrt_map.get(coord['scene'],
                       Path(self.vrt_dir) / f"{coord['scene']}.tif")

        try:
            with rasterio.open(str(vrt_path)) as src:
                win = Window(coord['x'], coord['y'], self.patch_size, self.patch_size)
                patch = src.read(window=win, boundless=True, fill_value=0).astype(np.float32) / 10000.0
        except Exception as e:
            # [F4] fail-fast：讀取「失敗」必須炸；nodata>10% 回 None（下方）是合法設計、保留
            raise RuntimeError(f"[F4][coord_vrt] bg 影像讀取失敗 {vrt_path}: {e}") from e

        # nodata 過濾：NaN 比例 > 10% 視為無效 patch
        if np.isnan(patch).mean() > 0.1:
            # [P8][codex R6-2] 擴池模式下改為 fail-fast：回傳 None 會被 collate 濾掉，
            # 形成縮小 batch 而 n_skipped_small 未必攔得住，等於「名目曝光相同但靜默少樣本」，
            # 破壞與 Arm A 的實際曝光 parity。P8 的池已離線預檢全過，此處若仍觸發即為
            # 資料或預檢準則不一致，必須整折 abort 而非默默降級。非 P8 模式行為不變。
            if getattr(self, 'bg_fail_fast', False):
                raise RuntimeError(
                    f'[P8] BG runtime validity failure：{coord["scene"]} '
                    f'(x={coord["x"]}, y={coord["y"]}) NaN 比例 > 10%，但該座標已通過離線預檢。'
                    '曝光 parity 已被破壞，整折中止。')
            return None
        np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        patch[patch > 100.0] = 0.0   # zero out float32 nodata fill (e.g., 9.97e36)
        patch[patch < -100.0] = 0.0
        if self.spectral_index:
            patch = compute_spectral_index(patch, self.spectral_index)
        elif self.band_indices is not None:
            patch = patch[self.band_indices]

        image = np.ascontiguousarray(patch.transpose(1, 2, 0), dtype=np.float32)  # CHW→HWC

        if self.transforms:
            transformed = self.transforms(image=image, mask=np.zeros(image.shape[:2], dtype=np.int64))
            image = transformed['image']

        # bg patch 的 per-pixel label 全填 Background class（1），不走 pixel_mapping
        bg_class = next(
            (k for k, v in (self.pixel_mapping or {}).items() if int(v) == 1),
            1  # 找不到 mapping 時預設 class 1
        )
        # 取 pixel_mapping 的 value（class id），不是 key（pixel value）
        # Background class id 固定為 1（由 pixel_mapping 中 value=1 確認）
        mask = torch.ones((self.patch_size, self.patch_size), dtype=torch.long)
        return image, mask


def oil_collate_fn(batch):
    """過濾 coord_vrt 模式中因 nodata 回傳的 None 樣本。"""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


class EpochRotatingBGSampler(torch.utils.data.Sampler):
    """[P8] 每 epoch 從擴充 BG 池中逐景取 d_s 個座標，跨 epoch 無狀態循環輪替。

    設計約束（codex R5-1/R5-3，2026-08-27 共識）：
    - dataset 內容 immutable：整池座標一次載入，輪替只發生在 index 層，
      故 persistent_workers 不需重啟、worker 端 dataset 不被改寫。
    - 逐景 draw 數 d_s 鎖定為 Arm A 的既有座標數 → dataset 長度、
      optimizer steps、LR-vs-step 曲線與 Arm A 完全相同（唯一變因＝看到哪些座標）。
    - 選取是 epoch 的無狀態函數 start_{s,e} = (e·d_s) mod q_s，
      不需把 cursor 存進 checkpoint（epoch 本身已在 checkpoint 內）。
    - 退化保證：當所有 q_s == d_s（即未擴充的 k=10 模式）時，選取結果必然等於
      list(range(n))，此時本 sampler 與 DataLoader(shuffle=True) 的 RandomSampler
      逐位元相同（同一 generator、同一次 randperm、同一 n）。__init__ 會實地 assert。
    """

    def __init__(self, dataset, generator, draws_per_scene, allow_replacement=False):
        self.n_fixed = len(dataset.pos_coords) + len(dataset.image_files)
        self.generator = generator
        self.epoch = 0
        self.allow_replacement = bool(allow_replacement)
        pools: dict = {}
        for j, coord in enumerate(dataset.bg_coords):
            pools.setdefault(coord['scene'], []).append(self.n_fixed + j)
        self.pools = pools
        self.draws = {}
        for scene, idxs in pools.items():
            d = draws_per_scene.get(scene)
            if d is None:
                raise RuntimeError(f'[P8] 場景 {scene} 缺少凍結 draw 數 d_s（檢查 pool_freeze.csv 覆蓋率）')
            if d > len(idxs) and not self.allow_replacement:
                raise RuntimeError(f'[P8] 場景 {scene} d_s={d} > 池大小 {len(idxs)}；池必須是 Arm A 的超集')
            if d < 0:
                raise RuntimeError(f'[BG sampler] 場景 {scene} d_s={d} < 0')
            self.draws[scene] = int(d)
        self.n = self.n_fixed + sum(self.draws.values())
        self.rotating = any(len(self.pools[s]) != self.draws[s] for s in self.pools)
        if not self.rotating:
            # k=10 退化模式：必須逐位元還原原本的 index 空間，否則 parity 論證不成立
            sel = self._selected()
            if sel != list(range(self.n)):
                raise RuntimeError('[P8] 退化模式下 index 序列與原始 dataset 順序不符；'
                                   'bg_coords.tsv 的同場景列必須連續且順序不變')

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _selected(self):
        sel = list(range(self.n_fixed))
        for scene, pool in self.pools.items():   # dict 保序＝TSV 首次出現順序
            d, q = self.draws[scene], len(pool)
            start = (self.epoch * d) % q
            sel.extend(pool[(start + i) % q] for i in range(d))
        return sel

    def __len__(self):
        return self.n

    def __iter__(self):
        sel = self._selected()
        if len(sel) != self.n:
            raise RuntimeError(f'[P8] 曝光量漂移：本 epoch 選出 {len(sel)} 筆，應為 {self.n}')
        perm = torch.randperm(len(sel), generator=self.generator).tolist()
        # ⚠ 必須複製 RandomSampler 的 generator 消耗次數，否則第 2 個 epoch 起就與歷史 run 分岔。
        # torch 2.7.1 的 RandomSampler.__iter__（replacement=False, num_samples==n）為
        #     for _ in range(num_samples // n): yield from randperm(n, generator)
        #     yield from randperm(n, generator)[: num_samples % n]
        # 尾段的 `[:0]` 不產出任何 index，但 randperm 已先被求值 → 每 epoch 共消耗 2 次。
        # 這是版本相依行為；S2a index parity gate 就是用來在 torch 升版後攔住這件事。
        torch.randperm(len(sel), generator=self.generator)
        return iter([sel[i] for i in perm])


class _IndexRecordingSampler(torch.utils.data.Sampler):
    """[P8][S2b] 只在 parity dump 模式下使用：包住實際 sampler 並記錄本 epoch 送出的 index 序列。
    對 DataLoader 而言行為與被包住的 sampler 完全相同（同一 generator、同一產出順序）。"""

    def __init__(self, inner):
        self.inner = inner
        self.last = []

    def set_epoch(self, epoch):
        if hasattr(self.inner, 'set_epoch'):
            self.inner.set_epoch(epoch)

    def __len__(self):
        return len(self.inner)

    def __iter__(self):
        self.last = list(iter(self.inner))
        return iter(self.last)


def _load_bg_draws_freeze(csv_path):
    """讀 pool_freeze.csv 的逐景 base_A 欄＝每 epoch draw 數 d_s（曝光鎖定量）。"""
    import csv as _csv
    out = {}
    with open(csv_path, newline='') as fh:
        for row in _csv.DictReader(fh):
            out[row['scene']] = int(row['base_A'])
    return out


class DeepLabPredictionResult:
    # 保持不變，將目標標記繪製出來
    def __init__(self, original_image, pred_mask, target_class_index=0):
        self.target_class_index = target_class_index
        self.original_image = original_image
        self.pred_mask_np = pred_mask
        target_mask = (pred_mask == target_class_index)
        self.masks = torch.from_numpy(target_mask).unsqueeze(0) if target_mask.sum() > 0 else None

    def plot(self):
        img = self.original_image.copy()
        if img.ndim == 3 and img.shape[2] > 3: img = img[:, :, :3]
        if img.dtype != np.uint8:
            img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255.0).astype(np.uint8)
        if self.masks is not None:
            overlay = np.zeros_like(img)
            overlay[self.pred_mask_np == self.target_class_index] = (0, 255, 255)
            img = cv2.addWeighted(img, 1.0, overlay, 0.5, 0)
        return img

class _SMPDeepLabV3PlusWrapper(nn.Module):
    """[V3+] 把 smp DeepLabV3Plus 的 tensor 輸出包成 torchvision 的 {'out': logits} 契約，
    讓訓練/驗證/TTA/recon 六處呼叫點不感知模型種類。輸出尺寸不等於輸入時以 bilinear 對齊。"""
    def __init__(self, body):
        super().__init__()
        self.body = body

    def forward(self, x):
        out = self.body(x)
        if out.shape[-2:] != x.shape[-2:]:
            out = nn.functional.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return {'out': out}


class DeepLabAdapter(nn.Module):
    # ⭐️ 新增 architecture_cfg 參數
    def __init__(self, model_path=None, dataset_cfg=None, architecture_cfg=None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_cfg = dataset_cfg or {}
        self.architecture_cfg = architecture_cfg or {}
        
        # ⭐️ 優先從 architecture_cfg 讀取，防呆預設為 3 通道 / 2 類別
        self.in_channels = self.architecture_cfg.get('in_channels', self.dataset_cfg.get('in_channels', 3))
        # band_indices：與 OilDataset 一致，供 _auto_calculate_stats_vrt 切片用；None=全波段
        _bi = self.dataset_cfg.get('band_indices', None)
        self.band_indices = np.array(_bi, dtype=int) if _bi is not None else None
        # spectral_index：與 SegmentationDataset 一致，供 _auto_calculate_stats_vrt 算指數通道用
        self.spectral_index = self.dataset_cfg.get('spectral_index', None)
        self.num_classes = self.architecture_cfg.get('num_classes', self.dataset_cfg.get('num_classes', 2))
        self.target_class_index = self.dataset_cfg.get('target_class_index', 0)
        
        # 安全防呆：避免索引越界
        if self.target_class_index >= self.num_classes:
            print(f"⚠️ [警告] target_class_index ({self.target_class_index}) 超過類別數 ({self.num_classes})，重置為 0。")
            self.target_class_index = 0

        self.class_names = self.dataset_cfg.get('names', {i: f'Class {i}' for i in range(self.num_classes)})

        # [V3+] model_family 切換（2026-07-11 Claude×Codex 共識）：
        #   'torchvision_deeplabv3'（預設）= 原本的 torchvision DeepLabV3；歷史 yaml 無此鍵，行為與 checkpoint 完全不變
        #   'smp_deeplabv3plus'           = segmentation_models_pytorch 的 DeepLabV3Plus（真 V3+，含低階特徵 decoder）
        self.model_family = self.architecture_cfg.get('model_family', 'torchvision_deeplabv3')
        if self.model_family == 'torchvision_deeplabv3':
            self._build_torchvision_deeplabv3()
        elif self.model_family == 'smp_deeplabv3plus':
            self._build_smp_deeplabv3plus()
        elif self.model_family == 'prithvi_eo_v2':
            self._build_prithvi_eo_v2()
        elif self.model_family == 'prithvi_eo_v2_dlv3':
            self._build_prithvi_eo_v2_dlv3()
        else:
            raise RuntimeError(f"[V3+] 未知 model_family: {self.model_family}"
                               f"（可用: torchvision_deeplabv3 / smp_deeplabv3plus / "
                               f"prithvi_eo_v2 / prithvi_eo_v2_dlv3）")
        self._finalize_checkpoint(model_path)

    def _build_torchvision_deeplabv3(self):
        """原本的 torchvision DeepLabV3 建構路徑（A1 conv1 改造 / 11-P1 SSL 載入 / Reg 頭部 Dropout），逐字未動。"""
        # 完全不傳入預訓練權重
        # from torchvision.models.segmentation import deeplabv3_resnet50
        # self.model = deeplabv3_resnet50(weights=None)

        from torchvision.models.segmentation import deeplabv3_resnet50
        from torchvision.models import ResNet50_Weights
        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

        # [A1] 預設改成 ImageNet 預訓 backbone (DEFAULT = COCO_WITH_VOC_LABELS_V1，內含 ImageNet ResNet50 encoder)
        # YAML 可覆寫：'None' = 從零訓、'DEFAULT' = ImageNet/COCO、或其他 enum 字串
        weight_str = self.architecture_cfg.get('torchvision_weights', 'DEFAULT')
        if weight_str == 'None':
            actual_weights = None
        else:
            try:
                actual_weights = getattr(DeepLabV3_ResNet50_Weights, weight_str)
            except AttributeError:
                actual_weights = eval(f"torchvision.models.segmentation.{weight_str}")
        self.model = deeplabv3_resnet50(weights=actual_weights)

        # [A1] in_channels != 3 時改造 conv1：保留 ImageNet 權重，B/G/R 對應到 8-band 的 B02/B03/B04
        # 8-band 順序 (依 yaml mean/std 為 [B01, B02, B03, B04, B05, B08, B11, B12]):
        #   idx  0   1   2   3   4   5   6   7
        #   band B01 B02 B03 B04 B05 B08 B11 B12
        #                ↑   ↑   ↑
        #                G   ?   R  (Sentinel-2 RGB = B04(R), B03(G), B02(B))
        if self.in_channels != 3:
            old_conv = self.model.backbone.conv1               # (64, 3, 7, 7)
            new_conv = nn.Conv2d(
                self.in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            if actual_weights is not None:
                with torch.no_grad():
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
                    new_conv.weight.zero_()
                    if self.in_channels == 8:
                        # 8-band 標準配置：B02/B03/B04 對應 ImageNet B/G/R，其餘用平均
                        # torchvision ImageNet conv1 通道順序: 0=R, 1=G, 2=B
                        new_conv.weight[:, 1] = old_conv.weight[:, 2]   # B02 ← ImageNet B
                        new_conv.weight[:, 2] = old_conv.weight[:, 1]   # B03 ← ImageNet G
                        new_conv.weight[:, 3] = old_conv.weight[:, 0]   # B04 ← ImageNet R
                        for ch in [0, 4, 5, 6, 7]:
                            new_conv.weight[:, ch:ch+1] = mean_w
                        print(f"  [A1] conv1 改造完成（8ch）：B02/B03/B04 ← ImageNet B/G/R，其餘 5 band 用平均初始化")
                    else:
                        # 非標準波段數（band_indices 子集或指數通道）：無明確 RGB 語意對應，全部用平均初始化
                        for ch in range(self.in_channels):
                            new_conv.weight[:, ch:ch+1] = mean_w
                        print(f"  [A1] conv1 改造完成（{self.in_channels}ch）：全通道用 ImageNet 三通道平均初始化")
            else:
                # 沒有預訓權重 → 維持原本 from-scratch 行為（Kaiming 初始化由 nn.Conv2d 預設處理）
                print(f"  [A1] torchvision_weights=None，conv1 從零初始化（沿用舊行為）")
            self.model.backbone.conv1 = new_conv

        # [11-P1] 遙感 SSL backbone：用外部 ResNet50 state_dict（如 SSL4EO-S12）覆蓋整個 backbone。
        # ckpt conv1 為 13ch（S2 L1C 順序 B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12），
        # backbone_ckpt_band_map 指定要抽哪幾個輸入通道對應本 pipeline 的波段。
        backbone_ckpt = self.architecture_cfg.get('backbone_ckpt', None)
        if backbone_ckpt:
            if not Path(backbone_ckpt).exists():
                raise RuntimeError(f"[11-P1] backbone_ckpt 不存在: {backbone_ckpt}")
            ckpt_sd = torch.load(backbone_ckpt, map_location='cpu', weights_only=True)
            band_map = self.architecture_cfg.get('backbone_ckpt_band_map', None)
            if band_map is not None:
                if len(band_map) != self.in_channels:
                    raise RuntimeError(f"[11-P1] backbone_ckpt_band_map 長度 {len(band_map)} != in_channels {self.in_channels}")
                ckpt_sd['conv1.weight'] = ckpt_sd['conv1.weight'][:, list(band_map)].clone()
            bb_sd = self.model.backbone.state_dict()
            # 比照 [C6]：shape 不符或缺 key 一律報明細，不靜默丟棄
            missing  = [k for k in bb_sd if k not in ckpt_sd]
            mismatch = [(k, tuple(ckpt_sd[k].shape), tuple(bb_sd[k].shape))
                        for k in bb_sd if k in ckpt_sd and ckpt_sd[k].shape != bb_sd[k].shape]
            if mismatch or missing:
                head = ', '.join(f'{k}: {a}->{b}' for k, a, b in mismatch[:5]) or str(missing[:5])
                raise RuntimeError(f"[11-P1] backbone_ckpt 與 backbone 不合（缺 {len(missing)}、shape 不符 {len(mismatch)}）: {head}")
            self.model.backbone.load_state_dict(ckpt_sd, strict=False)
            print(f"  [11-P1] backbone 已載入 SSL ckpt：{Path(backbone_ckpt).name}"
                  f"（{len(bb_sd)} key 全對上，conv1 band_map={band_map}）")

        # [11-P1/B] linear-probe：凍結整個 backbone（含 band_map 後的 conv1，權重皆為 SSL 預訓），
        # 只訓 decoder/classifier，用來診斷 geoFM 特徵本身是否帶油汙訊號。
        # 註：BatchNorm 的 running_mean/var 仍在 train 模式更新（隨本域分佈微調），僅 affine 權重被凍。
        if self.architecture_cfg.get('freeze_backbone', False):
            n_frozen = sum(1 for _ in self.model.backbone.parameters())
            for p in self.model.backbone.parameters():
                p.requires_grad = False
            print(f"  [11-P1/B] freeze_backbone=True：backbone {n_frozen} 個張量已凍結（linear-probe，只訓 decoder+classifier）")

        # 加 Dropout2d 在最終 1×1 conv 前，緩解 overconfidence（上輪 train_loss 0.01 vs val_loss 0.4 的根因）
        dropout_p = float(self.architecture_cfg.get('classifier_dropout', 0.3))
        _cls_in = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Sequential(
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(_cls_in, self.num_classes, kernel_size=1),
        )
        if self.model.aux_classifier:
            _aux_in = self.model.aux_classifier[4].in_channels
            self.model.aux_classifier[4] = nn.Sequential(
                nn.Dropout2d(p=dropout_p),
                nn.Conv2d(_aux_in, self.num_classes, kernel_size=1),
            )
        print(f"  [Reg] classifier head 加 Dropout2d(p={dropout_p})")

    def _finalize_checkpoint(self, model_path):
        """兩個 model_family 共用的收尾：搬到 device、載入 model_path checkpoint（C6 嚴格檢查）。"""
        self.model.to(self.device)
        if model_path and Path(model_path).exists():
            payload = torch.load(model_path, map_location=self.device, weights_only=False)
            # 固定-step 實驗直接以 last.pt（resume checkpoint）評估。舊路徑的 best.pt
            # 是 plain state_dict；兩種格式都顯式支援，避免把 checkpoint 外層 key
            # 靜默當成模型 key 而載入隨機權重。
            state_dict = payload['model_state'] if (
                isinstance(payload, dict) and isinstance(payload.get('model_state'), dict)
            ) else payload
            model_dict = self.model.state_dict()
            # [C6] 過去做 shape-相符過濾、不符靜默丟棄 → 容易把 OSDMamba/YOLO ckpt 灌成隨機初始化的 DeepLab。
            # 改成：找出 shape 不符或缺失的 key，全部報出來，明確 raise。
            missing  = [k for k in model_dict if k not in state_dict]
            mismatch = [(k, tuple(state_dict[k].shape), tuple(model_dict[k].shape))
                        for k in model_dict if k in state_dict and state_dict[k].shape != model_dict[k].shape]
            if mismatch:
                head = ', '.join(f'{k}: {a}->{b}' for k, a, b in mismatch[:5])
                raise RuntimeError(f"[DeepLabAdapter] checkpoint shape mismatch（共 {len(mismatch)} key）: {head}")
            if missing:
                # 容許「decoder/輸出層」之類重新初始化，但要把缺哪些印出來
                print(f"  [DeepLabAdapter] checkpoint 缺 {len(missing)} key，沿用模型初始化（前 5 個）：{missing[:5]}")
            self.model.load_state_dict(state_dict, strict=False)

    def _build_smp_deeplabv3plus(self):
        """[V3+] smp DeepLabV3Plus 建構路徑（2026-07-11 Claude×Codex 共識設定，首輪凍結不調參）：
        ResNet50 encoder、OS=16、ASPP rates=(6,12,18)（論文標準值）、其餘 smp 0.5.0 出廠預設
        （decoder_channels=256、separable ASPP、aspp_dropout=0.5、upsampling=4）。
        A1 conv1 改造與 11-P1 SSL 載入語意與 V3 路徑一致；[Reg] classifier dropout 為 V3 專屬，不套用。"""
        import segmentation_models_pytorch as smp

        if 'torchvision_weights' in self.architecture_cfg:
            raise RuntimeError("[V3+] smp_deeplabv3plus 不吃 torchvision_weights，請改用 encoder_weights"
                               "（'imagenet' 或 None）；兩套欄位混用一律報錯不靜默轉換。")
        encoder_weights = self.architecture_cfg.get('encoder_weights', 'imagenet')
        if encoder_weights in ('None', 'null'):
            encoder_weights = None
        body = smp.DeepLabV3Plus(
            encoder_name='resnet50',
            encoder_weights=encoder_weights,
            encoder_output_stride=16,
            decoder_atrous_rates=(6, 12, 18),
            in_channels=3,               # 固定以 3ch 建模，8ch 由下方 A1 改造接手（不用 smp 隱含通道擴展）
            classes=self.num_classes,
        )
        print(f"  [V3+] smp DeepLabV3Plus 建構完成：resnet50 / OS16 / rates(6,12,18) / "
              f"encoder_weights={encoder_weights}")

        # [A1] conv1 8ch 改造：規則與 V3 路徑逐字同款（B02/B03/B04 ← ImageNet B/G/R，其餘平均）
        if self.in_channels != 3:
            old_conv = body.encoder.conv1                      # (64, 3, 7, 7)
            new_conv = nn.Conv2d(
                self.in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            if encoder_weights is not None:
                with torch.no_grad():
                    mean_w = old_conv.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
                    new_conv.weight.zero_()
                    if self.in_channels == 8:
                        new_conv.weight[:, 1] = old_conv.weight[:, 2]   # B02 ← ImageNet B
                        new_conv.weight[:, 2] = old_conv.weight[:, 1]   # B03 ← ImageNet G
                        new_conv.weight[:, 3] = old_conv.weight[:, 0]   # B04 ← ImageNet R
                        for ch in [0, 4, 5, 6, 7]:
                            new_conv.weight[:, ch:ch+1] = mean_w
                        print(f"  [A1] conv1 改造完成（8ch）：B02/B03/B04 ← ImageNet B/G/R，其餘 5 band 用平均初始化")
                    else:
                        for ch in range(self.in_channels):
                            new_conv.weight[:, ch:ch+1] = mean_w
                        print(f"  [A1] conv1 改造完成（{self.in_channels}ch）：全通道用 ImageNet 三通道平均初始化")
            else:
                print(f"  [A1] encoder_weights=None，conv1 從零初始化（沿用舊行為）")
            body.encoder.conv1 = new_conv

        # [11-P1] SSL backbone 載入：目標換成 smp encoder；fc.* 是唯一允許的不對稱（任一方向），其餘嚴格比對
        backbone_ckpt = self.architecture_cfg.get('backbone_ckpt', None)
        if backbone_ckpt:
            if not Path(backbone_ckpt).exists():
                raise RuntimeError(f"[11-P1] backbone_ckpt 不存在: {backbone_ckpt}")
            ckpt_sd = torch.load(backbone_ckpt, map_location='cpu', weights_only=True)
            band_map = self.architecture_cfg.get('backbone_ckpt_band_map', None)
            if band_map is not None:
                if len(band_map) != self.in_channels:
                    raise RuntimeError(f"[11-P1] backbone_ckpt_band_map 長度 {len(band_map)} != in_channels {self.in_channels}")
                ckpt_sd['conv1.weight'] = ckpt_sd['conv1.weight'][:, list(band_map)].clone()
            enc_sd = body.encoder.state_dict()
            missing  = [k for k in enc_sd if k not in ckpt_sd and not k.startswith('fc.')]
            mismatch = [(k, tuple(ckpt_sd[k].shape), tuple(enc_sd[k].shape))
                        for k in enc_sd if k in ckpt_sd and ckpt_sd[k].shape != enc_sd[k].shape]
            if mismatch or missing:
                head = ', '.join(f'{k}: {a}->{b}' for k, a, b in mismatch[:5]) or str(missing[:5])
                raise RuntimeError(f"[11-P1] backbone_ckpt 與 smp encoder 不合（缺 {len(missing)}、shape 不符 {len(mismatch)}）: {head}")
            body.encoder.load_state_dict({k: v for k, v in ckpt_sd.items() if not k.startswith('fc.')}, strict=False)
            print(f"  [11-P1] smp encoder 已載入 SSL ckpt：{Path(backbone_ckpt).name}"
                  f"（{len(enc_sd)} key 全對上，conv1 band_map={band_map}）")

        # [11-P1/B] linear-probe：凍結 encoder（語意同 V3 路徑的 freeze backbone）
        if self.architecture_cfg.get('freeze_backbone', False):
            n_frozen = sum(1 for _ in body.encoder.parameters())
            for p in body.encoder.parameters():
                p.requires_grad = False
            print(f"  [11-P1/B] freeze_backbone=True：encoder {n_frozen} 個張量已凍結（linear-probe，只訓 decoder+segmentation_head）")

        self.model = _SMPDeepLabV3PlusWrapper(body)

    # [15] Prithvi-EO-2.0-300M 官方規格常數（config.json @ pinned revision，
    # 溯源與 SHA256 見 prithvi_vendor_manifest.json；mean/std_ref = 官方 raw 統計 /10000）
    _PRITHVI_300M_SPEC = {
        'revision': '9eb1b1102806593963daa333bcc491b1c6f8562f',
        'patch_size': (1, 16, 16), 'embed_dim': 1024, 'depth': 24,
        'num_heads': 16, 'mlp_ratio': 4,
        'input_band_names': ['B01', 'B02', 'B03', 'B04', 'B08', 'B8A', 'B11', 'B12'],
        'backbone_band_names': ['B02', 'B03', 'B04', 'B8A', 'B11', 'B12'],
        'input_band_indices': [1, 2, 3, 5, 6, 7],
        'mean_ref': [0.1087, 0.1342, 0.1433, 0.2734, 0.1958, 0.1363],
        'std_ref':  [0.2248, 0.2179, 0.2178, 0.1850, 0.1242, 0.1049],
        'checkpoint_sha256': 'faab2c8b776f0723f93431747603a7ec586a54fc45473613b4322c29752486e7',
        'checkpoint_size_bytes': 1326658396,
    }

    def _prithvi_build_encoder(self):
        """[15] Prithvi encoder 共用建構（UPerNet 與 dlv3 兩 family 逐字同一條路）：
        spec 驗證 + ckpt/manifest pin 驗證 + encoder 建構 + C6 嚴格載入 + freeze。
        本體從 _build_prithvi_eo_v2 純搬移（2026-07-15 dlv3 協商案），行為不變。
        C6 規則：encoder 逐 key 嚴格比對，pos_embed（可重建 sin-cos buffer，預訓 T=4 grid 與
        本設定 T=1/256px 不同形）為唯一允許缺席的 key；MAE decoder keys 以 'decoder.'/'mask_token'
        prefix allowlist 排除；其餘缺 key/shape 不符一律 raise，禁泛用 strict=False。"""
        from prithvi_upernet import PrithviViTEncoder

        spec = self._PRITHVI_300M_SPEC
        cfg = self.architecture_cfg

        # ── spec 一致性驗證（共識條款：不可只信 index，band 名與 index 交叉核對）──
        if self.in_channels != len(spec['input_band_names']):
            raise RuntimeError(f"[15] in_channels {self.in_channels} != 8（8-band VRT 輸入）")
        for key in ('input_band_names', 'backbone_band_names', 'input_band_indices'):
            if list(cfg.get(key, [])) != list(spec[key]):
                raise RuntimeError(f"[15] architecture_cfg.{key}={cfg.get(key)} 與 300M spec "
                                   f"{spec[key]} 不符；band 對映錯位是硬錯誤，不得靜默修正")
        derived = [spec['input_band_names'][i] for i in spec['input_band_indices']]
        if derived != spec['backbone_band_names']:
            raise RuntimeError(f"[15] index→band 名交叉核對失敗：{derived}")
        mean = list(self.dataset_cfg.get('mean') or [])
        std = list(self.dataset_cfg.get('std') or [])
        if len(mean) != 8 or len(std) != 8:
            raise RuntimeError(f"[15] yaml mean/std 必須為 8 槽（收到 {len(mean)}/{len(std)}）")
        for j, idx in enumerate(spec['input_band_indices']):
            if abs(mean[idx] - spec['mean_ref'][j]) > 1e-6 or abs(std[idx] - spec['std_ref'][j]) > 1e-6:
                raise RuntimeError(f"[15] yaml mean/std 槽 {idx}（{spec['backbone_band_names'][j]}）="
                                   f"({mean[idx]},{std[idx]}) 與官方統計/10000 "
                                   f"({spec['mean_ref'][j]},{spec['std_ref'][j]}) 不符")
        for idx in (0, 4):   # B01/B08 被丟棄，槽位必須是 identity 以免誤導
            if mean[idx] != 0.0 or std[idx] != 1.0:
                raise RuntimeError(f"[15] 未使用槽 {idx}（B01/B08）mean/std 應為 identity(0,1)，"
                                   f"收到 ({mean[idx]},{std[idx]})")

        # ── checkpoint + manifest 驗證（pin revision、SHA256）──
        backbone_ckpt = cfg.get('backbone_ckpt', None)
        if not backbone_ckpt or not Path(backbone_ckpt).exists():
            raise RuntimeError(f"[15] backbone_ckpt 不存在: {backbone_ckpt}"
                               f"（先跑 main/fetch_prithvi_checkpoint.py）")
        manifest_path = Path(str(backbone_ckpt) + '.manifest.json')
        if not manifest_path.exists():
            raise RuntimeError(f"[15] 缺 manifest: {manifest_path}（fetch 腳本會生成，不得手放 ckpt）")
        import hashlib, json as _json
        manifest = _json.loads(manifest_path.read_text())
        if manifest.get('revision') != spec['revision']:
            raise RuntimeError(f"[15] manifest revision {manifest.get('revision')} != pin {spec['revision']}")
        if manifest.get('sha256') != spec['checkpoint_sha256']:
            raise RuntimeError(f"[15] manifest SHA256 {manifest.get('sha256')} != 官方 pin "
                               f"{spec['checkpoint_sha256']}")
        if int(manifest.get('size_bytes', -1)) != spec['checkpoint_size_bytes']:
            raise RuntimeError(f"[15] manifest size {manifest.get('size_bytes')} != 官方 pin "
                               f"{spec['checkpoint_size_bytes']}")
        if Path(backbone_ckpt).stat().st_size != spec['checkpoint_size_bytes']:
            raise RuntimeError(f"[15] ckpt size {Path(backbone_ckpt).stat().st_size} != 官方 pin "
                               f"{spec['checkpoint_size_bytes']}")
        h = hashlib.sha256()
        with open(backbone_ckpt, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 22), b''):
                h.update(chunk)
        if h.hexdigest() != spec['checkpoint_sha256']:
            raise RuntimeError(f"[15] ckpt SHA256 與官方 pin 不符（檔案損毀或被替換）")

        # ── encoder 建構（300M 規格寫死；輸入尺寸/T 由 cfg，256 時 pos_embed 內建 bicubic 插值）──
        input_size = int(cfg.get('input_size', 256))
        num_frames = int(cfg.get('num_frames', 1))
        encoder = PrithviViTEncoder(
            img_size=input_size, patch_size=spec['patch_size'], num_frames=num_frames,
            in_chans=len(spec['backbone_band_names']), embed_dim=spec['embed_dim'],
            depth=spec['depth'], num_heads=spec['num_heads'], mlp_ratio=spec['mlp_ratio'],
            coords_encoding=[], coords_scale_learn=False,
            grad_checkpointing=bool(cfg.get('grad_checkpointing', False)),
        )

        # ── C6 嚴格載入：MAE ckpt 的 encoder.* namespace ──
        full_sd = torch.load(backbone_ckpt, map_location='cpu', weights_only=True)
        enc_sd = {k[len('encoder.'):]: v for k, v in full_sd.items() if k.startswith('encoder.')}
        dropped = [k for k in full_sd if not k.startswith('encoder.')]
        bad_drop = [k for k in dropped if not (k.startswith('decoder.') or k == 'mask_token')]
        if bad_drop:
            raise RuntimeError(f"[15] ckpt 出現 allowlist 外的非 encoder key（前 5）: {bad_drop[:5]}")
        # pos_embed：唯一允許的特例（丟棄 ckpt 版、保留本地 sin-cos 重建，同官方 inference.py 做法）
        enc_sd.pop('pos_embed', None)
        target_sd = encoder.state_dict()
        missing  = [k for k in target_sd if k not in enc_sd and k != 'pos_embed']
        unexpected = [k for k in enc_sd if k not in target_sd]
        mismatch = [(k, tuple(enc_sd[k].shape), tuple(target_sd[k].shape))
                    for k in enc_sd if k in target_sd and enc_sd[k].shape != target_sd[k].shape]
        if missing or unexpected or mismatch:
            head = (', '.join(f'{k}: {a}->{b}' for k, a, b in mismatch[:5])
                    or str((missing + unexpected)[:5]))
            raise RuntimeError(f"[15] encoder ckpt 不合（缺 {len(missing)}、多 {len(unexpected)}、"
                               f"shape 不符 {len(mismatch)}）: {head}")
        encoder.load_state_dict(enc_sd, strict=False)   # 僅 pos_embed 缺席（上面已驗證）
        print(f"  [15] Prithvi encoder 已載入 {Path(backbone_ckpt).name}"
              f"（{len(enc_sd)}/{len(target_sd)} key 嚴格對上；pos_embed 用本地 sin-cos 重建，"
              f"img_size={input_size}, T={num_frames}）")

        if cfg.get('freeze_backbone', False):
            n_frozen = sum(1 for _ in encoder.parameters())
            for p in encoder.parameters():
                p.requires_grad = False
            print(f"  [15] freeze_backbone=True：encoder {n_frozen} 個張量已凍結")

        return encoder

    def _build_prithvi_eo_v2(self):
        """[15] Prithvi-EO-2.0-300M（非 TL）+ 官方式 UperNet（2026-07-13 Claude×Codex 共識第 4 案）。
        輸入 8-band 已由 dataset 端 /10000+yaml mean/std 正規化；本路徑選 6 band 餵 ViT。"""
        from prithvi_upernet import PrithviUperNetSegmenter

        spec = self._PRITHVI_300M_SPEC
        cfg = self.architecture_cfg
        encoder = self._prithvi_build_encoder()

        self.model = PrithviUperNetSegmenter(
            encoder=encoder,
            band_indices=spec['input_band_indices'],
            feature_indices=list(cfg.get('feature_indices', [5, 11, 17, 23])),
            num_classes=self.num_classes,
            decoder_channels=int(cfg.get('decoder_channels', 256)),
            head_dropout=float(cfg.get('head_dropout', 0.1)),
        )
        print(f"  [15] PrithviUperNetSegmenter 建構完成：300M/非TL, taps={self.model.feature_indices}, "
              f"decoder_ch={cfg.get('decoder_channels', 256)}, head_dropout={cfg.get('head_dropout', 0.1)}, "
              f"grad_ckpt={cfg.get('grad_checkpointing', False)}")

    def _build_prithvi_eo_v2_dlv3(self):
        """[15-dlv3] Prithvi-EO-2.0-300M encoder + torchvision DeepLabV3(ASPP) head
        （2026-07-15 Claude×Codex 協商定稿；decoder 消融 arm）。
        encoder 建構/載入與 UPerNet family 走同一條 _prithvi_build_encoder()；
        head=整包 torchvision DeepLabHead，rates 預註冊 (6,12,18)、tap=[depth-1]、無 aux。"""
        from prithvi_deeplab import PrithviDeepLabV3Segmenter, REGISTERED_ASPP_RATES

        spec = self._PRITHVI_300M_SPEC
        cfg = self.architecture_cfg
        encoder = self._prithvi_build_encoder()

        feature_indices = list(cfg.get('feature_indices', [spec['depth'] - 1]))
        if len(feature_indices) != 1:
            raise RuntimeError(f"[15-dlv3] DeepLabV3 head 只吃單一 tap，"
                               f"feature_indices 必須長度 1，收到 {feature_indices}")
        rates = tuple(cfg.get('aspp_atrous_rates', REGISTERED_ASPP_RATES))

        self.model = PrithviDeepLabV3Segmenter(
            encoder=encoder,
            band_indices=spec['input_band_indices'],
            feature_index=feature_indices[0],
            num_classes=self.num_classes,
            aspp_atrous_rates=rates,
        )
        print(f"  [15-dlv3] PrithviDeepLabV3Segmenter 建構完成：300M/非TL, tap=[{feature_indices[0]}], "
              f"ASPP rates={rates}, aux=無, grad_ckpt={cfg.get('grad_checkpointing', False)}")

    def _auto_calculate_stats_vrt(self, vrt_dir, train_files, sample_size=500):
        """VRT / stacked TIF 模式下，從 fold TXT 座標抽樣計算 mean/std。"""
        import re as _re, random, rasterio
        from rasterio.windows import Window
        print(f"\n⏳ [自動計算 VRT] 從 stacked TIF 抽樣計算 Mean/Std（最多 {sample_size} 個 patch）...")

        _patch_re = _re.compile(r'^(.+)_patch_x(\d+)_y(\d+)$')
        tif_map = {
            p.stem: p for p in Path(vrt_dir).rglob('*')
            if p.suffix.lower() in ('.tif', '.tiff', '.vrt')
        }

        coords = []
        for fname in (train_files or []):
            stem = Path(fname).stem
            m = _patch_re.match(stem)
            if m:
                scene, x, y = m.group(1), int(m.group(2)), int(m.group(3))
                if scene in tif_map:
                    coords.append({'tif_path': tif_map[scene], 'x': x, 'y': y})

        if not coords:
            print("  ⚠ 找不到任何有效座標，使用預設 mean/std=0.5")
            return [0.5] * self.in_channels, [0.5] * self.in_channels

        samples = random.sample(coords, min(len(coords), sample_size))
        sum_pixels = np.zeros(self.in_channels, dtype=np.float64)
        sum_sq    = np.zeros(self.in_channels, dtype=np.float64)
        total_px  = 0
        used = skipped_nan = skipped_zero = skipped_err = 0

        for c in tqdm(samples, desc="抽樣計算中"):
            try:
                with rasterio.open(str(c['tif_path'])) as src:
                    win = Window(c['x'], c['y'], self.patch_size, self.patch_size)
                    patch_raw = src.read(window=win, boundless=True, fill_value=0).astype(np.float32)
            except Exception:
                skipped_err += 1
                continue

            # [C2] 過濾污染樣本：先看 raw 的 nan 比例（fill 之前不算）
            nan_ratio = float(np.isnan(patch_raw).mean())
            if nan_ratio > 0.10:
                skipped_nan += 1
                continue

            patch = patch_raw / 10000.0
            np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            patch[patch > 100.0] = 0.0     # float32 nodata fill (e.g., 9.97e36)
            patch[patch < -100.0] = 0.0
            if self.spectral_index:
                patch = compute_spectral_index(patch, self.spectral_index)
            elif self.band_indices is not None:
                patch = patch[self.band_indices]

            # [C2] 再看 0 值比例（含 boundless 邊界外 fill 與 nodata 抹平）
            zero_ratio = float((patch == 0).all(axis=0).mean())  # 每像素皆為 0 才算 fill
            if zero_ratio > 0.30:
                skipped_zero += 1
                continue

            patch = patch.reshape(self.in_channels, -1)   # (C, H*W)
            sum_pixels += patch.sum(axis=1, dtype=np.float64)
            sum_sq     += (patch.astype(np.float64) ** 2).sum(axis=1)
            total_px   += patch.shape[1]
            used += 1

        print(f"  [VRT stats audit] 採用 {used} / 抽樣 {len(samples)}　跳過：nan>10% {skipped_nan}　zero>30% {skipped_zero}　讀失敗 {skipped_err}")
        if total_px == 0:
            print("  ⚠ 全部跳過，回傳預設 0.5")
            return [0.5] * self.in_channels, [0.5] * self.in_channels

        mean = (sum_pixels / total_px).tolist()
        std  = np.sqrt(np.maximum(sum_sq / total_px - np.array(mean) ** 2, 0)).tolist()
        print(f"✅ [VRT Mean/Std] 計算完成（採用 {used} patches）")
        print(f"    mean: {mean}")
        print(f"    std:  {std}")
        return mean, std

    def _auto_calculate_stats(self, image_dir, file_list=None):
        """如果 YAML 沒提供 mean/std，自動從訓練集抽樣計算"""
        print(f"\n⏳ [自動計算] YAML 未提供 Mean/Std，正在掃描影像 (最多抽樣 500 張)...")
        if file_list is not None:
            image_files = [Path(image_dir) / f for f in file_list if (Path(image_dir) / f).exists()]
        else:
            image_files = list(Path(image_dir).glob('*.*'))
            image_files = [f for f in image_files if f.suffix.lower() in ['.png', '.jpg', '.tif', '.tiff', '.npy']]
        
        if not image_files: return (0.5,) * self.in_channels, (0.5,) * self.in_channels
        
        import random
        sample_size = min(len(image_files), 500)
        samples = random.sample(image_files, sample_size)
        
        sum_pixels = np.zeros(self.in_channels)
        sum_sq_pixels = np.zeros(self.in_channels)
        total_pixels = 0
        
        for f in tqdm(samples, desc="計算中"):
            if f.suffix.lower() == '.npy': img = np.load(str(f))
            elif f.suffix.lower() in ['.tif', '.tiff']:
                try: import tifffile; img = tifffile.imread(str(f))
                except: img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            else: img = cv2.imread(str(f)); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
            
            if img is None: continue
            if img.ndim == 3 and img.shape[0] < img.shape[2]: img = img.transpose(1, 2, 0)
            if img.ndim == 2: img = np.expand_dims(img, axis=-1)
            
            img = img.astype(np.float32) / (img.max() + 1e-8) # 簡單歸一化計算
            img = img.reshape(-1, self.in_channels)
            sum_pixels += img.sum(axis=0)
            sum_sq_pixels += (img ** 2).sum(axis=0)
            total_pixels += img.shape[0]
            
        mean = (sum_pixels / total_pixels).tolist()
        std = (np.sqrt(sum_sq_pixels / total_pixels - np.array(mean) ** 2)).tolist()
        print(f"✅ 計算完成！請將以下數值貼回 YAML 檔案以節省下次啟動時間：\n  mean: {mean}\n  std: {std}\n")
        return mean, std

    def train(self, data, epochs, imgsz, batch, project, name, **kwargs):
        if self.model_family in ('prithvi_eo_v2', 'prithvi_eo_v2_dlv3'):
            min_batch = int(self.architecture_cfg.get('min_physical_batch_size', 2))
            if int(batch) < min_batch:
                raise RuntimeError(
                    f"[15] Prithvi head physical batch={batch} < {min_batch}；"
                    "PPM/ASPPPooling 的 global pool→BatchNorm 在 batch=1、1x1 pooled feature 下無法訓練。"
                    "若 batch=2 仍 OOM，依預註冊規則停止，不得靜默改 head/norm。"
                )
        with open(data, 'r') as f: data_config = yaml.safe_load(f)

        images_dir = data_config.get('images_dir')
        labels_dir = data_config.get('labels_dir')
        train_txt = data_config.get('train_txt')
        val_txt = data_config.get('val_txt')
        train_files = _read_txt_list(train_txt) if train_txt else None
        val_files = _read_txt_list(val_txt) if val_txt else None

        if images_dir:
            train_img_dir = Path(images_dir)
            train_mask_dir = Path(labels_dir) if labels_dir else train_img_dir.parent / 'labels'
            val_img_dir = train_img_dir
            val_mask_dir = train_mask_dir
        elif data_config.get('read_images_from_vrt', False):
            # VRT 模式：影像從 scene VRT 動態讀取，不需要 images_dir
            train_img_dir = None
            train_mask_dir = Path(labels_dir) if labels_dir else None
            val_img_dir = None
            val_mask_dir = train_mask_dir
        else:
            dataset_path = Path(data_config.get('path', ''))
            if not dataset_path.is_absolute(): dataset_path = Path.cwd() / str(dataset_path).lstrip('./')
            train_img_dir = dataset_path / data_config.get('train', 'images/train')
            train_mask_dir = train_img_dir.parent.parent / 'labels' / train_img_dir.name
            val_img_dir = dataset_path / data_config.get('val', 'images/val')
            val_mask_dir = val_img_dir.parent.parent / 'labels' / val_img_dir.name

        # 讀取或動態計算統計值
        mean = self.dataset_cfg.get('mean')
        std = self.dataset_cfg.get('std')
        if mean and std:
            print(f"  [Mean/Std] ✅ 從 YAML 讀取既有值（無需重新計算）")
            print(f"    mean: {mean}")
            print(f"    std:  {std}")
        elif train_img_dir is not None:
            mean, std = self._auto_calculate_stats(train_img_dir, file_list=train_files)
            self.dataset_cfg['mean'], self.dataset_cfg['std'] = mean, std
        else:
            # VRT / stacked TIF 模式：從 fold 訓練座標抽樣計算
            vrt_dir = self.dataset_cfg.get('vrt_dir', '')
            patch_size = int(self.dataset_cfg.get('patch_size', 256))
            self.patch_size = patch_size
            mean, std = self._auto_calculate_stats_vrt(vrt_dir, train_files)
            self.dataset_cfg['mean'], self.dataset_cfg['std'] = mean, std

        # [2026-07-22 可重現性] Albumentations 2.0.x Compose 有獨立 RNG（torch/np/random seed 無效），
        # 建構時傳 seed= 讓 workers=0 主進程路徑決定性；workers>0 由 worker_init_fn 各自 set_random_seed 覆蓋。
        _aug_seed = int(torch.initial_seed() & 0x7FFFFFFF)
        transforms_train = A.Compose([
            # [C5] GT_expand patch 緊貼油汙 bbox，scale 太小會把油汙裁出去；收緊到 0.9 + 降頻率
            A.RandomResizedCrop(size=(int(imgsz), int(imgsz)), scale=(0.9, 1.0), p=0.5),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.8),
            A.GaussNoise(std_range=(0.005, 0.02), p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2(),
        ], seed=_aug_seed)
        transforms_val = A.Compose([
            A.Resize(int(imgsz), int(imgsz)), A.Normalize(mean=mean, std=std, max_pixel_value=1.0), ToTensorV2(),
        ], seed=_aug_seed)

        current_fold = data_config.get('current_fold')

        # [W2 修法] 從 patch list 萃取 scene 名稱，只用「同 fold 該 split 的 scenes」對應的 BG patch，
        # 避免舊版 bg_fold_only=current_fold 把 val/test 場景 BG 灌進 train 的 leak。
        def _scenes_from_files(files):
            if not files: return None
            import re as _re
            pat = _re.compile(r'^(.+)_patch_x\d+_y\d+$')
            out = set()
            for f in files:
                stem = Path(f).stem
                m = pat.match(stem)
                if m: out.add(m.group(1))
            return out

        train_scenes = _scenes_from_files(train_files)
        val_scenes   = _scenes_from_files(val_files)

        train_dataset = SegmentationDataset(
            train_img_dir, train_mask_dir, transforms_train, self.dataset_cfg, file_list=train_files,
            bg_scene_filter=train_scenes,
            apply_oil_ignore_ring=True,   # oil_ignore_ring 只在 train 生效;yaml 未設時為 0=行為不變
        )
        val_dataset = SegmentationDataset(
            val_img_dir, val_mask_dir, transforms_val, self.dataset_cfg, file_list=val_files,
            bg_scene_filter=val_scenes,
        )

        workers = int(kwargs.get('workers', 4))
        # persistent_workers=True keeps worker processes alive between epochs,
        # so the per-process _ds_cache (rasterio file handles) stays warm.
        persist = workers > 0
        # [C1] seed-aware DataLoader：用呼叫端設定好的 RNG，使 shuffle / augmentation / 抽樣可重現
        loader_seed = int(torch.initial_seed() & 0xFFFFFFFF)
        loader_generator = torch.Generator()
        loader_generator.manual_seed(loader_seed)
        def _seed_worker(wid):
            wseed = loader_seed + wid
            import numpy as _np, random as _r
            _np.random.seed(wseed & 0xFFFFFFFF)
            _r.seed(wseed)
            # [2026-07-22 可重現性] 每個 worker 用 distinct seed 重設其 Albumentations Compose 內部 RNG，
            # 否則 augmentation 跨 run 不可重現（Albumentations 獨立 RNG，torch/np/random seed 無效）。
            _info = torch.utils.data.get_worker_info()
            if _info is not None and hasattr(_info.dataset, 'set_augmentation_seed'):
                _info.dataset.set_augmentation_seed(wseed)
        # [C3] oil_collate_fn 永遠掛上：原本只在 use_coord_vrt 為 True 時才用，
        # 但 _get_pos_vrt_item 在 nan>10% 或讀取失敗時也會回 None，default collate 會炸。
        # oil_collate_fn 對「沒有 None」的情況等同 default collate，無 overhead。
        # [P8] bg_pool_cfg.draws_freeze_csv 有給時改用 EpochRotatingBGSampler：
        # BG 池可大於每 epoch 曝光量，逐景 draw 數鎖定為 pool_freeze.csv 的 base_A。
        # 不給時 sampler 為 None、shuffle=True，行為與既有實驗逐位元相同。
        bg_pool_cfg = (self.dataset_cfg.get('bg_pool_cfg') or {})
        bg_sampler = None
        if bg_pool_cfg.get('draws_freeze_csv'):
            bg_sampler = EpochRotatingBGSampler(
                train_dataset, loader_generator,
                _load_bg_draws_freeze(bg_pool_cfg['draws_freeze_csv']),
                allow_replacement=bool(bg_pool_cfg.get('allow_replacement', False)))
            print(f"  [P8] BG 輪替 sampler：bg_pool_size={len(train_dataset.bg_coords)}, "
                  f"bg_draws_per_epoch={bg_sampler.n - bg_sampler.n_fixed}, "
                  f"dataset_len={len(bg_sampler)}, rotating={bg_sampler.rotating}, "
                  f"replacement={bg_sampler.allow_replacement}")
        # [P8][S2b] OIL_P8_PARITY_DUMP 設定時記錄逐 batch 指紋。為了同時取得 sample ID，
        # ref 路徑改為顯式 RandomSampler(dataset, generator=loader_generator) —— 這正是
        # DataLoader(shuffle=True) 內部所建的同一物件，故逐位元等價。僅在 dump 模式下生效。
        parity_dump = os.environ.get('OIL_P8_PARITY_DUMP')
        parity_rec = None
        if parity_dump:
            from torch.utils.data import RandomSampler
            base = bg_sampler if bg_sampler is not None else \
                RandomSampler(train_dataset, generator=loader_generator)
            parity_rec = _IndexRecordingSampler(base)
            bg_sampler = parity_rec
            print(f"  [P8][S2b] parity dump → {parity_dump}"
                  f"（sampler={type(base).__name__}, len={len(parity_rec)}）")
        # 迴圈內 `batch` 會被資料 batch 遮蔽，batch size 必須在此先捕獲
        _p_bsz = int(batch)
        train_loader = DataLoader(
            train_dataset, batch_size=int(batch), shuffle=(bg_sampler is None),
            sampler=bg_sampler,
            num_workers=workers, pin_memory=True, drop_last=True,
            collate_fn=oil_collate_fn,
            persistent_workers=persist,
            worker_init_fn=_seed_worker,
            generator=loader_generator,
        )
        # [F9] validation 不丟資料：drop_last=True 會丟掉尾 batch（fold2 丟 6、fold3 丟 4 筆，
        # 皆為尾端 bg patch），可能輕微高估 Oil IoU 並影響 best checkpoint 選擇。
        val_loader = DataLoader(val_dataset, batch_size=int(batch), shuffle=False,
                                num_workers=workers, pin_memory=True, drop_last=False,
                                collate_fn=oil_collate_fn,
                                persistent_workers=persist,
                                worker_init_fn=_seed_worker)

        # 損失函數設定：優先看 dataset.loss.type（tversky）；否則沿用 class_weights + FocalLoss（baseline 路徑不變）
        loss_cfg = self.dataset_cfg.get('loss', None)
        raw_cw = self.dataset_cfg.get('class_weights', None)
        _loss_type = str(loss_cfg.get('type', '')).lower() if loss_cfg else ''
        if _loss_type in ('tversky', 'focal_tversky'):
            _oil = int(self.dataset_cfg.get('target_class_index', 0))
            _a = float(loss_cfg.get('alpha', 0.3)); _b = float(loss_cfg.get('beta', 0.7))
            _g = float(loss_cfg.get('gamma', 1.0))   # 直接指數 q；focal_tversky 用 4/3，tversky 預設 1.0(逐bit同純Tversky)
            ce_loss_fn = TverskyLoss(alpha=_a, beta=_b, oil_class=_oil, ignore_index=255, focal_gamma=_g)
            _name = 'FocalTverskyLoss' if _g != 1.0 else 'TverskyLoss'
            print(f"  [Loss] 啟用 {_name}(oil={_oil}, alpha={_a}, beta={_b}, direct_q={_g}, reduction=batch)")
        elif raw_cw:
            cw_tensor = torch.tensor([float(w) for w in raw_cw], dtype=torch.float32).to(self.device)
            ce_loss_fn = FocalLoss(weight=cw_tensor, ignore_index=255, gamma=2.0)
            print(f"  [Loss] 啟用 FocalLoss + 權重: {raw_cw}")
        else:
            ce_loss_fn = FocalLoss(ignore_index=255, gamma=2.0)
            print(f"  [Loss] 啟用標準 FocalLoss")

        lr0 = float(kwargs.get('lr0', 1e-4))
        # [A1] differential lr：預訓 backbone 用較小 lr 避免破壞 ImageNet 特徵；
        # 改造過的 conv1 + decoder/classifier 需要全 lr 從頭學
        backbone_lr_mult = float(self.architecture_cfg.get('backbone_lr_mult', 1.0))
        if self.architecture_cfg.get('freeze_backbone', False):
            # [11-P1/B] backbone 已凍結，只把 requires_grad=True 的張量（decoder+classifier）交給 optimizer
            trainable = [p for p in self.model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=lr0, weight_decay=1e-3)
            print(f"  [11-P1/B] freeze_backbone → 只優化 {len(trainable)} 個可訓練張量（decoder+classifier），lr={lr0:.2e}")
        elif backbone_lr_mult != 1.0:
            if self.model_family == 'smp_deeplabv3plus':
                # [V3+] 分組對應：backbone→encoder（不含 conv1）、head→decoder+segmentation_head
                enc = self.model.body.encoder
                backbone_params = [p for n, p in enc.named_parameters() if not n.startswith('conv1.')]
                new_conv1_params = list(enc.conv1.parameters())
                head_params = list(self.model.body.decoder.parameters()) + \
                              list(self.model.body.segmentation_head.parameters())
            elif self.model_family == 'prithvi_eo_v2':
                # [15] 分組：backbone→ViT encoder；head→UperNet decoder+SegmentationHead；
                # 無 conv1 改造組（band 選擇無參數），空 list 不影響下方涵蓋性 assert
                backbone_params = list(self.model.encoder.parameters())
                new_conv1_params = []
                head_params = list(self.model.decoder.parameters()) + list(self.model.head.parameters())
            elif self.model_family == 'prithvi_eo_v2_dlv3':
                # [15-dlv3] 分組：backbone→ViT encoder（含 terminal norm）；head→DeepLabHead 整包
                backbone_params = list(self.model.encoder.parameters())
                new_conv1_params = []
                head_params = list(self.model.head.parameters())
            else:
                backbone_params = [p for n, p in self.model.backbone.named_parameters() if not n.startswith('conv1.')]
                new_conv1_params = list(self.model.backbone.conv1.parameters())
                head_params = list(self.model.classifier.parameters()) + (
                    list(self.model.aux_classifier.parameters()) if self.model.aux_classifier else []
                )
            # [V3+共識] 每個可訓練參數必須恰好落在一組，缺漏或重複皆 raise
            grouped = backbone_params + new_conv1_params + head_params
            assert len(grouped) == len({id(p) for p in grouped}), "[lr groups] 參數重複分組"
            n_model = sum(1 for p in self.model.parameters() if p.requires_grad)
            assert len(grouped) == n_model, f"[lr groups] 分組涵蓋 {len(grouped)} != 模型可訓練參數 {n_model}"
            # weight_decay 1e-4 → 1e-3 加強正則化、緩解上輪觀察到的 overconfidence overfit
            optimizer = torch.optim.AdamW([
                {'params': backbone_params,   'lr': lr0 * backbone_lr_mult},
                {'params': new_conv1_params,  'lr': lr0},   # 改造後的 conv1 視同 head
                {'params': head_params,       'lr': lr0},
            ], weight_decay=1e-3)
            print(f"  [A1] differential lr：backbone={lr0 * backbone_lr_mult:.2e}, conv1+head={lr0:.2e}, weight_decay=1e-3")
        else:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr0, weight_decay=1e-3)
        max_optimizer_steps_raw = kwargs.get('max_optimizer_steps', None)
        max_optimizer_steps = (int(max_optimizer_steps_raw)
                               if max_optimizer_steps_raw is not None else None)
        if max_optimizer_steps is not None and max_optimizer_steps <= 0:
            raise ValueError(f'max_optimizer_steps 必須 > 0，收到 {max_optimizer_steps}')
        step_budget_mode = max_optimizer_steps is not None
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_optimizer_steps if step_budget_mode else epochs,
            eta_min=lr0 * 0.01,
        )
        if step_budget_mode:
            print(f"  [StepBudget] 固定 optimizer steps={max_optimizer_steps:,}；"
                  "CosineAnnealingLR 改為每 optimizer step 更新")
        # [V3+共識] 移除 ModelEma：原本 decay=0.0 數學上等於直接複製當前模型，從未有平滑效果；
        # val/best.pt 改用 self.model，數值行為不變（歷史 best.pt 的 key namespace 本來就是 model 的）。
        use_amp = bool(kwargs.get('use_amp', False))
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_val_miou = -1.0
        weights_path = Path(project) / name / "weights"; weights_path.mkdir(parents=True, exist_ok=True)
        best_model_path = weights_path / "best.pt"
        last_checkpoint_path = weights_path / "last.pt"   # ★ 每 epoch 存的 resume checkpoint
        log_file = Path(project) / name / 'results.csv'

        class_col_headers = ','.join([f"val_iou_{self.class_names.get(c, f'C{c}')}" for c in range(self.num_classes)])

        accumulation_steps = int(kwargs.get('accumulation_steps', 1))
        patience = int(kwargs.get('patience', 100))
        patience_counter = 0
        start_epoch = 0   # 預設從頭開始
        global_optimizer_step = 0

        # [P8][codex R5-3] BG 輪替模式禁止 mid-fold resume：
        # checkpoint 存了 torch/cuda/numpy/random RNG，但沒有存 DataLoader 的 loader_generator
        # 狀態，也沒有存 persistent worker 內 Albumentations Compose 的內部 RNG。resume 後
        # loader_generator 會從 fold seed 重新建立、workers 從初始 seed 重生，訓練軌跡不可重現。
        # 正式 P8 run 一律 epoch 0 重跑於新的 run directory，不得接續 partial run。
        if bg_sampler is not None and last_checkpoint_path.exists():
            raise RuntimeError(
                f'[P8] 偵測到 {last_checkpoint_path} 但本 run 使用 BG 輪替 sampler，'
                '禁止 mid-fold resume（loader_generator 與 worker augmentation RNG 未存檔，'
                '接續會使訓練軌跡不可重現）。請改用全新 run directory 由 epoch 0 重跑。')

        # ────── 從 checkpoint 恢復 ──────
        if last_checkpoint_path.exists():
            print(f"  [Resume] 找到 last.pt，正在恢復...")
            try:
                ckpt = torch.load(last_checkpoint_path, map_location=self.device)
                self.model.load_state_dict(ckpt['model_state'])
                optimizer.load_state_dict(ckpt['optimizer_state'])
                scheduler.load_state_dict(ckpt['scheduler_state'])
                if use_amp and ckpt.get('scaler_state'):
                    scaler.load_state_dict(ckpt['scaler_state'])
                best_val_miou    = ckpt.get('best_val_miou', -1.0)
                patience_counter = ckpt.get('patience_counter', 0)
                start_epoch      = ckpt.get('epoch', 0)   # 下一個要跑的 epoch index
                global_optimizer_step = int(ckpt.get('global_optimizer_step', 0))
                # RNG 狀態恢復（讓資料順序可重現）
                rng = ckpt.get('rng_state', {})
                if rng.get('torch') is not None:
                    torch.set_rng_state(rng['torch'].cpu() if hasattr(rng['torch'], 'cpu') else rng['torch'])
                if rng.get('cuda') is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, 'cpu') else s for s in rng['cuda']])
                if rng.get('numpy') is not None:
                    np.random.set_state(rng['numpy'])
                if rng.get('random') is not None:
                    random.setstate(rng['random'])
                print(f"  [Resume] ✅ 從 epoch {start_epoch + 1} 繼續 | best_val_miou={best_val_miou:.4f} | patience={patience_counter}")
            except Exception as e:
                print(f"  [Resume] ⚠️ 載入 last.pt 失敗（{e}），改從頭訓練")
                start_epoch = 0
                best_val_miou = -1.0
                patience_counter = 0

        # log file：resume 時 append，全新訓練時寫 header
        if start_epoch == 0:
            with open(log_file, 'w') as f:
                f.write(f"epoch,lr,train_loss,val_loss,train_miou,val_miou,{class_col_headers}\n")
        else:
            print(f"  [Resume] log 檔案改為 append 模式")

        _parity_rows, _parity_epochs = [], []
        step_budget_reached = False
        for epoch in range(start_epoch, int(epochs)):
            if bg_sampler is not None:
                bg_sampler.set_epoch(epoch)   # [P8] 無狀態輪替：選取只依 epoch，不依累積狀態
            self.model.train()
            total_train_loss = 0
            train_int, train_uni = torch.zeros(self.num_classes, device=self.device), torch.zeros(self.num_classes, device=self.device)

            accum_count = 0
            n_skipped_small = 0
            _p_bi, _p_opt_steps = 0, 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
                if batch is None:  # coord_vrt nodata 過濾
                    continue
                images, masks = batch
                if parity_rec is not None:
                    import hashlib as _h
                    _bs = int(images.shape[0])
                    _ids = parity_rec.last[_p_bi * _p_bsz:(_p_bi + 1) * _p_bsz]
                    _parity_rows.append({
                        'epoch': epoch, 'batch': _p_bi, 'physical_bs': _bs,
                        'sample_ids': _ids,
                        'img_sha': _h.sha256(images.detach().cpu().numpy().tobytes()).hexdigest(),
                        'mask_sha': _h.sha256(masks.detach().cpu().numpy().tobytes()).hexdigest(),
                        'lr': float(scheduler.get_last_lr()[0]),
                    })
                    _p_bi += 1
                # [15] oil_collate_fn 過濾 None 後 batch 可能縮水（drop_last 擋不住）；
                # Prithvi UPerNet PPM 的 BatchNorm train mode 不吃 batch=1（[1,C,1,1]）。
                # 未達下限整個 microbatch 跳過；位置在 accum_count += 1 之前，
                # skipped microbatch 不推進 accumulation/optimizer step 計數（只訓練端，val 不丟）。
                if (self.model_family in ('prithvi_eo_v2', 'prithvi_eo_v2_dlv3')
                        and images.shape[0] < int(self.architecture_cfg.get('min_physical_batch_size', 2))):
                    n_skipped_small += 1
                    continue
                images, masks = images.to(self.device), masks.to(self.device)
                with torch.autocast('cuda', enabled=use_amp):
                    outputs = self.model(images)
                    loss = ce_loss_fn(outputs['out'], masks) / accumulation_steps
                scaler.scale(loss).backward()
                total_train_loss += loss.item() * accumulation_steps
                if parity_rec is not None:
                    _parity_rows[-1]['loss'] = repr(float(loss.item()))

                accum_count += 1
                if accum_count % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer); scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    _p_opt_steps += 1
                    global_optimizer_step += 1
                    if step_budget_mode:
                        scheduler.step()
                        if global_optimizer_step >= max_optimizer_steps:
                            step_budget_reached = True

                with torch.no_grad():
                    preds = torch.argmax(outputs['out'], dim=1)
                    # [修法 A] train IoU 也採同樣指標：ignore 視為 bg class 1，跟 val/recon 對齊
                    masks_iou = torch.where(masks == 255, torch.tensor(1, device=masks.device), masks)
                    for c in range(self.num_classes):
                        train_int[c] += ((preds == c) & (masks_iou == c)).sum()
                        train_uni[c] += ((preds == c) | (masks_iou == c)).sum()

                if step_budget_reached:
                    break

            # flush 剩餘未 step 的梯度（epoch 結尾非整除時）
            if not step_budget_reached and accum_count % accumulation_steps != 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                _p_opt_steps += 1
                global_optimizer_step += 1
                if step_budget_mode:
                    scheduler.step()
                    if global_optimizer_step >= max_optimizer_steps:
                        step_budget_reached = True

            # [P8][S2b] parity dump：跑滿指定 epoch 數即寫檔並中止（不進 validation、不存 checkpoint）
            if parity_rec is not None:
                _parity_epochs.append({
                    'epoch': epoch, 'n_batches': _p_bi, 'optimizer_steps': _p_opt_steps,
                    'n_skipped_small': n_skipped_small, 'lr_after_step': None,
                    'dataset_len': len(train_loader.dataset), 'loader_len': len(train_loader),
                    'index_seq_sha': __import__('hashlib').sha256(
                        repr(parity_rec.last).encode()).hexdigest(),
                })
                if len(_parity_epochs) >= int(os.environ.get('OIL_P8_PARITY_EPOCHS', 2)):
                    import json as _json
                    Path(parity_dump).parent.mkdir(parents=True, exist_ok=True)
                    Path(parity_dump).write_text(_json.dumps(
                        {'epochs': _parity_epochs, 'batches': _parity_rows,
                         'has_bg_pool_cfg': bool(bg_pool_cfg.get('draws_freeze_csv'))},
                        ensure_ascii=False, indent=1))
                    print(f"\n[P8][S2b] parity dump 已寫入 {parity_dump}"
                          f"（{len(_parity_epochs)} epoch, {len(_parity_rows)} batch）；中止訓練。")
                    import sys as _sys
                    _sys.exit(0)

            if not step_budget_mode:
                scheduler.step()
            if n_skipped_small:
                print(f"  [15] 本 epoch 跳過 {n_skipped_small} 個未達 min_physical_batch_size 的 "
                      f"microbatch（collate 縮水守門，不推進 accumulation 計數）")
            n_batches = max(accum_count, 1)
            train_miou = (train_int / (train_uni + 1e-6)).mean().item()

            self.model.eval()
            total_val_loss = 0
            val_int, val_uni = torch.zeros(self.num_classes, device=self.device), torch.zeros(self.num_classes, device=self.device)
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                    if batch is None:  # [C3] collate 過濾後可能整個 batch 為 None
                        continue
                    images, masks = batch
                    images, masks = images.to(self.device), masks.to(self.device)
                    with torch.autocast('cuda', enabled=use_amp):
                        outputs = self.model(images)
                    total_val_loss += ce_loss_fn(outputs['out'].float(), masks).item()
                    preds = torch.argmax(outputs['out'], dim=1)
                    # [修法 A] val IoU：ignore 視為 bg class 1，避免方框 cheat
                    masks_iou = torch.where(masks == 255, torch.tensor(1, device=masks.device), masks)
                    for c in range(self.num_classes):
                        val_int[c] += ((preds == c) & (masks_iou == c)).sum()
                        val_uni[c] += ((preds == c) | (masks_iou == c)).sum()

            val_ious = val_int / (val_uni + 1e-6)
            val_miou, val_oil_iou = val_ious.mean().item(), val_ious[self.target_class_index].item()

            avg_train_loss = total_train_loss / n_batches
            avg_val_loss = total_val_loss / len(val_loader)
            current_lr = scheduler.get_last_lr()[0]

            step_note = (f", OptStep {global_optimizer_step}/{max_optimizer_steps}"
                         if step_budget_mode else '')
            print(f"Epoch {epoch+1}/{int(epochs)}{step_note}: T-Loss {avg_train_loss:.4f}, V-Loss {avg_val_loss:.4f}, LR: {current_lr:.2e}")
            print(f"  T-mIoU: {train_miou:.4f} | V-mIoU: {val_miou:.4f} | Target({self.class_names.get(self.target_class_index)}): {val_oil_iou:.4f}")
            per_class_str = ' | '.join([f"{self.class_names.get(c, f'C{c}')}: {val_ious[c].item():.3f}" for c in range(self.num_classes)])
            print(f"  Per-class IoU → {per_class_str}")

            class_iou_vals = ','.join([f"{val_ious[c].item():.6f}" for c in range(self.num_classes)])
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{current_lr:.8f},{avg_train_loss:.6f},{avg_val_loss:.6f},{train_miou:.6f},{val_miou:.6f},{class_iou_vals}\n")

            if val_miou > best_val_miou:
                best_val_miou = val_miou
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"  [★ Best] 更新最佳模型 (Val mIoU: {best_val_miou:.4f})")
            else:
                patience_counter += 1
                print(f"  [Info]   未改善 {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"\n[EarlyStopping] 連續 {patience} 個 Epoch 未改善。")
                    break

            # ────── 每個 epoch 存 checkpoint ──────
            ckpt = {
                'epoch': epoch + 1,   # 下一個要跑的 epoch index
                'model_state':     self.model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state':    scaler.state_dict() if use_amp else None,
                'best_val_miou':   best_val_miou,
                'patience_counter': patience_counter,
                'global_optimizer_step': global_optimizer_step,
                'rng_state': {
                    'torch':  torch.get_rng_state(),
                    'cuda':   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    'numpy':  np.random.get_state(),
                    'random': random.getstate(),
                },
            }
            # 先存暫存檔再 rename，避免存到一半當機導致 checkpoint 壞掉
            tmp_path = last_checkpoint_path.with_suffix('.tmp')
            torch.save(ckpt, tmp_path)
            tmp_path.replace(last_checkpoint_path)

            if step_budget_reached:
                print(f"\n[StepBudget] 已精確完成 {global_optimizer_step:,} optimizer steps。")
                break

        if step_budget_mode and global_optimizer_step != max_optimizer_steps:
            raise RuntimeError(
                f'[StepBudget] 訓練在 {global_optimizer_step:,} steps 結束，'
                f'未達設定的 {max_optimizer_steps:,}；epochs/early-stopping 上限不得先觸發。')
        checkpoint_policy = str(kwargs.get('checkpoint_policy', 'best')).lower()
        if checkpoint_policy not in ('best', 'last'):
            raise ValueError(f'checkpoint_policy 僅支援 best/last，收到 {checkpoint_policy}')
        selected = last_checkpoint_path if checkpoint_policy == 'last' else best_model_path
        return {
            'best_model_path': str(best_model_path),
            'last_model_path': str(last_checkpoint_path),
            'selected_model_path': str(selected),
            'checkpoint_policy': checkpoint_policy,
            'global_optimizer_step': global_optimizer_step,
            'best_val_miou': float(best_val_miou),
        }

    def val(self, data, split='test', imgsz=240, **kwargs):
        import pandas as pd
        print(f"--- Running DeepLabV3+ Validation on {split} Set (TTA + Table 5) ---")

        with open(data, 'r') as f: data_config = yaml.safe_load(f)

        images_dir = data_config.get('images_dir')
        labels_dir = data_config.get('labels_dir')
        test_txt = data_config.get('test_txt')
        test_files = _read_txt_list(test_txt) if test_txt else None

        # [W2 修法,同 train/val] test dataset 只載入 test 場景自己的 BG patch;
        # 舊版漏傳 bg_scene_filter → 全資料集 3,510 個 BG patch 混入 test(含 train/val 場景,洩漏)
        test_scenes = None
        if test_files:
            import re as _re
            _pat = _re.compile(r'^(.+)_patch_x\d+_y\d+$')
            test_scenes = set()
            for f in test_files:
                m = _pat.match(Path(f).stem)
                if m: test_scenes.add(m.group(1))

        if images_dir:
            target_img_dir = Path(images_dir)
            target_mask_dir = Path(labels_dir) if labels_dir else target_img_dir.parent / 'labels'
        elif data_config.get('read_images_from_vrt', False):
            # VRT 模式：影像從 scene VRT 動態讀取，不需要 images_dir
            target_img_dir = None
            target_mask_dir = Path(labels_dir) if labels_dir else None
        else:
            dataset_path = Path(data_config.get('path', ''))
            if not dataset_path.is_absolute(): dataset_path = Path.cwd() / str(dataset_path).lstrip('./')
            target_img_dir = dataset_path / data_config.get(split, f'images/{split}')
            target_mask_dir = target_img_dir.parent.parent / 'labels' / target_img_dir.name

        # 動態吃 YAML 設定的 mean / std
        mean = self.dataset_cfg.get('mean', (0.5,) * self.in_channels)
        std = self.dataset_cfg.get('std', (0.5,) * self.in_channels)

        transforms = A.Compose([
            A.Resize(int(imgsz), int(imgsz)),
            A.Normalize(mean=mean, std=std, max_pixel_value=1.0),
            ToTensorV2(),
        ])

        test_dataset = SegmentationDataset(target_img_dir, target_mask_dir, transforms, self.dataset_cfg, file_list=test_files,
                                           bg_scene_filter=test_scenes)
        val_batch_size = int(kwargs.get('batch', 16))
        test_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=kwargs.get('workers', 8), pin_memory=True, collate_fn=oil_collate_fn)  # [C3]

        self.model.eval()
        
        # 初始化混淆矩陣
        num_c = self.num_classes
        conf_matrix = np.zeros((num_c, num_c), dtype=np.int64)

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Testing with TTA ({split})"):
                if batch is None:  # [C3]
                    continue
                images, masks = batch
                images, masks = images.to(self.device), masks.to(self.device)
                
                # TTA 預測 (原圖 + 垂直 + 水平)
                outputs = self.model(images)['out']
                outputs_hflip = torch.flip(self.model(torch.flip(images, dims=[3]))['out'], dims=[3])
                outputs_vflip = torch.flip(self.model(torch.flip(images, dims=[2]))['out'], dims=[2])
                
                fused_outputs = (outputs + outputs_hflip + outputs_vflip) / 3.0
                preds = torch.argmax(fused_outputs, dim=1)
                
                # [修法 A] ignore 視為 bg class 1（跟 train/val/recon 對齊）→ 整 patch 都納入計算
                masks_iou = torch.where(masks == 255, torch.tensor(1, device=masks.device), masks)
                preds_flat = preds.flatten().cpu().numpy()
                masks_flat = masks_iou.flatten().cpu().numpy()
                # 累加至混淆矩陣
                counts = np.bincount(num_c * masks_flat + preds_flat, minlength=num_c**2)
                conf_matrix += counts.reshape(num_c, num_c)

        # 釋放 DataLoader workers（帶 pin_memory 的 8 workers 會佔大量 CPU 記憶體）
        del test_loader
        import gc; gc.collect()

        # --- 計算矩陣指標 ---
        intersection = np.diag(conf_matrix)
        gt_totals = conf_matrix.sum(axis=1)
        pred_totals = conf_matrix.sum(axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            recall = np.where(gt_totals > 0, intersection / gt_totals, 0)
            precision = np.where(pred_totals > 0, intersection / pred_totals, 0)
            iou_denom = gt_totals + pred_totals - intersection
            iou = np.where(iou_denom > 0, intersection / iou_denom, 0)
            f1_denom = precision + recall
            f1 = np.where(f1_denom > 0, 2 * precision * recall / f1_denom, 0)

        # Macro metrics use every class. Oil-only metrics are reported separately
        # below, so excluding Background here would mislabel F1(Oil) as F1-macro.
        oa = intersection.sum() / (conf_matrix.sum() + 1e-9)
        miou = np.mean(iou) if len(iou) > 0 else 0
        f1_macro = np.mean(f1) if len(f1) > 0 else 0

        # --- 使用 Pandas 渲染 Table 5 ---
        CLASS_NAMES = [self.class_names.get(c, f'C{c}') for c in range(self.num_classes)]
        
        conf_matrix_pct = np.zeros_like(conf_matrix, dtype=float)
        np.divide(conf_matrix * 100, gt_totals[:, None], out=conf_matrix_pct, where=gt_totals[:, None]!=0)
        
        df = pd.DataFrame(conf_matrix_pct, index=CLASS_NAMES, columns=CLASS_NAMES)
        df = df.map(lambda x: f"{x:.1f}")
        df['# of pixels (test set)'] = gt_totals.astype(str)
        df['Recall (%)'] = [f"{x:.1f}" for x in (recall * 100)]
        df.loc['IoU (%)'] = [f"{x:.1f}" for x in (iou * 100)] + ['mIoU:', f"{miou * 100:.1f}"]
        df.loc['Precision (%)'] = [f"{x:.1f}" for x in (precision * 100)] + ['OA:', f"{oa * 100:.1f}"]
        df.loc['F1-score (%)'] = [f"{x:.1f}" for x in (f1 * 100)] + ['F1-macro:', f"{f1_macro * 100:.1f}"]

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print("\n" + "="*145)
        print("Table 5: The resulting Confusion Matrix on test set. Columns represent predicted labels, rows represent reference labels (%).")
        print("="*145)
        print(df.to_string())
        print("="*145 + "\n")

        # ⭐️ 回傳字典以供寫入 Excel (動態支援多類別)
        eval_results = {
            'mIoU': f"{miou:.4f}",
            'Accuracy': f"{oa:.4f}",
            'F1-macro': f"{f1_macro:.4f}",
            'metric_semantics': 'macro_all_classes_from_confusion_matrix_v2',
        }

        # Keep every per-class metric explicit, including Background.
        for c_idx in range(num_c):
            c_name = self.class_names.get(c_idx, f'Class_{c_idx}')
            eval_results[f'IoU({c_name})'] = f"{iou[c_idx]:.4f}"
            eval_results[f'Precision({c_name})'] = f"{precision[c_idx]:.4f}"
            eval_results[f'Recall({c_name})'] = f"{recall[c_idx]:.4f}"
            eval_results[f'F1({c_name})'] = f"{f1[c_idx]:.4f}"

        # 混淆矩陣 raw counts + 像素層級類別分布
        total_px = int(conf_matrix.sum())
        for gt_i in range(num_c):
            gt_name = self.class_names.get(gt_i, f'C{gt_i}')
            gt_px = int(gt_totals[gt_i])
            eval_results[f'TestPixels({gt_name})'] = gt_px
            eval_results[f'TestPixelRatio({gt_name})'] = f"{gt_px/total_px:.4f}" if total_px else 'N/A'
            for pr_i in range(num_c):
                pr_name = self.class_names.get(pr_i, f'C{pr_i}')
                eval_results[f'CM_GT({gt_name})_Pred({pr_name})'] = int(conf_matrix[gt_i, pr_i])
        
        # 將字典包裝為相容的物件回傳
        class OutputWrapper(dict):
            def __init__(self, d):
                super().__init__(d)
                self.box = type('B', (), {'map': 0, 'map50': 0, 'mp':0, 'mr':0})()
                self.seg = type('S', (), {'map': 0, 'map50': 0, 'mp':0, 'mr':0})()
        
        return OutputWrapper(eval_results)
