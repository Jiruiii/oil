"""GT-aware partial reconstruction for GT_expand.

訓練分佈：以油汙 connected-component bbox 為中心切 256×256 patch
測試行為：用同樣邏輯（吃 test_fold{N}.txt 的 patch list）只在「油汙周邊 patch」做 inference，
         把預測貼回原場景座標。其他區域留空（未測試），不會被當作 oil/bg 計入指標。

vs 0422 sliding-window 全圖重組：
  - 不會把純海/雲 patch 丟給模型（OOD）
  - per-scene IoU 只算「實際有 inference 覆蓋的像素」，數字更代表 GT_expand 設計的真實能力
  - 視覺化：bg 衛星 + pred 紅色填充（半透明）+ GT cyan 外框，未測試區域維持 bg 原樣

Public entry: `run_reconstruction_gt_aware(recon_cfg, exp_config, model_path, results_path)`
介面與 reconstruct_module.run_reconstruction 相容，可被 main_runner 用 mode flag 派發。
"""
from pathlib import Path
import re
from collections import defaultdict
import numpy as np
import cv2
import torch
import rasterio
from rasterio.windows import Window
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# 8-band VRT 順序 (rasterio 1-indexed): [B01, B02, B03, B04, B08, B8A, B11, B12]
# True color RGB: R=B04, G=B03, B=B02
TRUE_COLOR_BANDS = (4, 3, 2)

# Mask 編碼：0=oil, 1=annot_bg, 2=other, 255=ignore
# pixel_mapping 後：oil → 0, BG → 1, ignore → 255
MASK_IGNORE = 255

PATCH_RE = re.compile(r'^(.+)_patch_x(\d+)_y(\d+)$')

# 視覺化：跟 client_demo 一致，pred=黃外框、GT=cyan 外框
PRED_BGR = (0, 255, 255)   # yellow (BGR)
GT_BGR   = (255, 255, 0)   # cyan
CAPTION_BG = (32, 32, 32)
CAPTION_FG = (240, 240, 240)
PAD_FRAC   = 0.15
MIN_CROP   = 1024
TARGET_LONG = 1920


def _prediction_artifact_policy(recon_cfg):
    """Resolve the post-2026-08-05 GT-expand output policy."""
    config = {'enabled': True, 'required': True}
    config.update(dict(recon_cfg.get('prediction_artifact', {}) or {}))
    enabled = bool(config.get('enabled', True))
    required = bool(config.get('required', True))
    retain_legacy_images = bool(
        config.get('retain_legacy_images', (not enabled) or (not required))
    )
    return config, enabled, retain_legacy_images


def _group_patches_by_scene(txt_path: Path):
    """讀 test_fold{N}.txt → {scene: [(x, y), ...]}."""
    by_scene = defaultdict(list)
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        m = PATCH_RE.match(line)
        if m:
            by_scene[m.group(1)].append((int(m.group(2)), int(m.group(3))))
    return by_scene


def _find_vrt(vrt_dir: Path, scene: str):
    cands = list(vrt_dir.rglob(f'{scene}.vrt')) + list(vrt_dir.rglob(f'{scene}.tif'))
    return cands[0] if cands else None


def _stretch_band(arr, p_lo=2, p_hi=98):
    valid_mask = np.isfinite(arr) & (arr > 0)
    if not valid_mask.any():
        return np.zeros_like(arr, dtype=np.uint8)
    lo, hi = np.percentile(arr[valid_mask], [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    out = np.clip((arr - lo) / (hi - lo), 0, 1)
    out = (out * 255).astype(np.uint8)
    out[~valid_mask] = 0
    return out


def _read_truecolor_window(vrt_path: Path, y0, x0, y1, x1):
    """只讀指定 window 的 True color (3 bands)，回傳 (H, W, 3) BGR uint8。"""
    win = Window(x0, y0, x1 - x0, y1 - y0)
    with rasterio.open(str(vrt_path)) as src:
        bands = [src.read(b, window=win) for b in TRUE_COLOR_BANDS]
    rgb = np.stack([_stretch_band(b) for b in bands], axis=-1)
    return rgb[..., ::-1].copy()   # RGB → BGR


def _find_oil_bbox(*bool_masks, H, W):
    """回傳所有 oil 區域聯集 bbox 加 padding；shape 不對齊時 clip 到 (H, W)."""
    combo = np.zeros((H, W), dtype=bool)
    for m in bool_masks:
        if m is not None and m.shape == (H, W):
            combo |= m
    ys, xs = np.where(combo)
    if len(ys) == 0:
        cy, cx = H // 2, W // 2
        side = MIN_CROP
        return max(0, cy - side // 2), max(0, cx - side // 2), \
               min(H, cy + side // 2), min(W, cx + side // 2)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    bh, bw = y1 - y0, x1 - x0
    py = int(bh * PAD_FRAC) + 32
    px = int(bw * PAD_FRAC) + 32
    y0 = max(0, y0 - py); y1 = min(H, y1 + py)
    x0 = max(0, x0 - px); x1 = min(W, x1 + px)
    if (y1 - y0) < MIN_CROP:
        cy = (y0 + y1) // 2
        y0 = max(0, cy - MIN_CROP // 2); y1 = min(H, y0 + MIN_CROP)
    if (x1 - x0) < MIN_CROP:
        cx = (x0 + x1) // 2
        x0 = max(0, cx - MIN_CROP // 2); x1 = min(W, x0 + MIN_CROP)
    return y0, x0, y1, x1


def _ascii_scene_name(name: str) -> str:
    """cv2.putText 的 Hershey 字型不支援 CJK，中文會畫成 ?????。
    移除非 ASCII 字元（中文），只保留 Latin 識別資訊
    （如 …Agia_Zoni_II_20170913_S2_34SGH）；overlay 專用，不動檔名/CSV 的原名。"""
    s = ''.join(ch if ord(ch) < 128 else '' for ch in name)
    s = re.sub(r'[_\s]+', '_', s).strip('_ ')
    return s or name


def _wrap_scene_name(name: str, max_chars: int):
    if len(name) <= max_chars:
        return [name]
    pivot = name.rfind('_', 0, max_chars)
    if pivot < max_chars // 3:
        pivot = max_chars
    return [name[:pivot], name[pivot:]]


def _attach_caption(img_bgr, scene_name, iou):
    scene_name = _ascii_scene_name(scene_name)   # 去中文避免 ????? 亂碼（僅影響圖上文字）
    """寬度 >= NARROW_THRESHOLD 用「文字左 / legend 右」並排版面；
    窄圖切換為「legend 疊在文字下方」垂直版面，避免兩欄壓在一起。"""
    W = img_bgr.shape[1]
    NARROW_THRESHOLD = 700
    narrow = W < NARROW_THRESHOLD

    if narrow:
        text_col_w = W - 48
    else:
        legend_col_w = 280
        text_col_w = max(W - legend_col_w - 60, W // 2)
    max_chars = max(20, int(text_col_w / 13))
    name_lines = _wrap_scene_name(scene_name, max_chars)
    pad_top = 24; pad_bot = 24; line_h = 36

    # 共通行：場景名 + IoU + bg note，加上 legend 兩行（narrow 才疊在下方）
    text_rows = len(name_lines) + 2     # name(可能多行) + IoU + bg note
    legend_rows_below = 2 if narrow else 0
    rows = text_rows + legend_rows_below
    cap_h = pad_top + line_h * rows + pad_bot
    bar = np.full((cap_h, W, 3), CAPTION_BG, dtype=np.uint8)

    # 文字欄
    y = pad_top + 26
    for ln in name_lines:
        cv2.putText(bar, ln, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, CAPTION_FG, 2, cv2.LINE_AA)
        y += line_h
    cv2.putText(bar, f"Oil IoU: {iou:.4f}", (24, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.92, (0, 255, 255), 2, cv2.LINE_AA)
    y += line_h
    cv2.putText(bar, "Background: True color (B04 / B03 / B02)",
                (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # legend
    if narrow:
        # 疊在文字下方
        y += line_h
        cv2.line(bar, (24, y - 8), (24 + 56, y - 8), PRED_BGR, 5)
        cv2.putText(bar, "Pred (oil)", (24 + 72, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, CAPTION_FG, 2, cv2.LINE_AA)
        y += line_h
        cv2.line(bar, (24, y - 8), (24 + 56, y - 8), GT_BGR, 5)
        cv2.putText(bar, "GT (oil)", (24 + 72, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, CAPTION_FG, 2, cv2.LINE_AA)
    else:
        lx = W - legend_col_w + 10
        sx0, sx1 = lx, lx + 56
        sy1 = pad_top + 30; sy2 = sy1 + line_h
        cv2.line(bar, (sx0, sy1), (sx1, sy1), PRED_BGR, 5)
        cv2.putText(bar, "Pred (oil)", (sx1 + 16, sy1 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, CAPTION_FG, 2, cv2.LINE_AA)
        cv2.line(bar, (sx0, sy2), (sx1, sy2), GT_BGR, 5)
        cv2.putText(bar, "GT (oil)", (sx1 + 16, sy2 + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, CAPTION_FG, 2, cv2.LINE_AA)

    divider = np.full((2, W, 3), (80, 80, 80), dtype=np.uint8)
    return np.vstack([img_bgr, divider, bar])


def _patch_iter(vrt_path: Path, coords, patch_size, mean, std, device, in_channels, band_indices=None, spectral_index=None):
    """yield (x, y, normalized tensor (1, C, H, W))."""
    _bi = np.array(band_indices, dtype=int) if band_indices is not None else None
    if spectral_index:
        from deeplab_adapter import compute_spectral_index  # 與訓練同一份指數計算，避免重複實作
    with rasterio.open(str(vrt_path)) as src:
        mean_t = torch.tensor(mean, dtype=torch.float32).view(1, in_channels, 1, 1).to(device)
        std_t  = torch.tensor(std,  dtype=torch.float32).view(1, in_channels, 1, 1).to(device)
        for (x, y) in coords:
            win = Window(x, y, patch_size, patch_size)
            patch = src.read(window=win, boundless=True, fill_value=0).astype(np.float32) / 10000.0
            np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            patch[patch > 100.0] = 0.0
            patch[patch < -100.0] = 0.0
            if spectral_index:
                patch = compute_spectral_index(patch, spectral_index)
            elif _bi is not None:
                patch = patch[_bi]
            t = torch.from_numpy(patch).unsqueeze(0).to(device)   # (1, C, H, W)
            t = (t - mean_t) / std_t
            yield x, y, t


@torch.no_grad()
def _tta_predict_prob(model, tensor):
    """原圖 + hflip + vflip 三組 logits 平均 → softmax → argmax；回傳 pred 與 prob."""
    out0 = model(tensor)['out']
    outH = torch.flip(model(torch.flip(tensor, dims=[3]))['out'], dims=[3])
    outV = torch.flip(model(torch.flip(tensor, dims=[2]))['out'], dims=[2])
    fused = (out0 + outH + outV) / 3.0
    prob  = torch.softmax(fused, dim=1)
    pred  = torch.argmax(prob, dim=1)
    return pred[0].cpu().numpy(), prob[0].cpu().numpy()    # (H, W) int, (C, H, W) float


@torch.no_grad()
def _tta_predict(model, tensor):
    """相容介面（analysis/large_scene_pr 腳本 import 用）：只回傳 argmax pred."""
    pred, _ = _tta_predict_prob(model, tensor)
    return pred


def _compute_scene_iou(pred_canvas, tested_mask, gt_mask, num_classes, ignore_idx=MASK_IGNORE, bg_class=1):
    """[修法 A] 在 tested_mask 全域算 IoU；ignore (mask=255) 像素視為 bg。
    這樣模型在『沒被人類標註的區域』塗 oil 也會被罰，避免方框 cheat 拿虛胖 IoU。"""
    valid = tested_mask
    if not valid.any():
        return {f'iou_C{c}': 0.0 for c in range(num_classes)}
    p = pred_canvas[valid].astype(np.int32)
    g = gt_mask[valid].astype(np.int32)
    g = np.where(g == ignore_idx, bg_class, g)   # ignore → bg
    ious = {}
    for c in range(num_classes):
        intersection = ((p == c) & (g == c)).sum()
        union        = ((p == c) | (g == c)).sum()
        ious[f'iou_C{c}'] = float(intersection / union) if union > 0 else 0.0
    return ious


def _draw_overlay_cropped(bg_bgr_crop, pred_oil_crop, gt_oil_crop, scene_name, iou):
    """純外框 + caption，bg/masks 都已是裁切後小圖。Pred=黃外框、GT=cyan 外框。"""
    out = bg_bgr_crop.copy()
    if gt_oil_crop is not None and gt_oil_crop.any():
        ctrs, _ = cv2.findContours((gt_oil_crop.astype(np.uint8) * 255),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, ctrs, -1, GT_BGR, 4)
    if pred_oil_crop is not None and pred_oil_crop.any():
        ctrs, _ = cv2.findContours((pred_oil_crop.astype(np.uint8) * 255),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, ctrs, -1, PRED_BGR, 3)
    h, w = out.shape[:2]
    if max(h, w) > TARGET_LONG:
        scale = TARGET_LONG / max(h, w)
        out = cv2.resize(out, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return _attach_caption(out, scene_name, iou)


def _apply_pixel_mapping(raw_mask: np.ndarray, mapping: dict):
    """raw mask（場景原值） → 模型 class id；未列在 mapping 的值保持 255 (ignore)."""
    if not mapping:
        return raw_mask.astype(np.int64)
    out = np.full_like(raw_mask, MASK_IGNORE, dtype=np.int64)
    for orig, new in mapping.items():
        out[raw_mask == int(orig)] = int(new)
    return out


def run_reconstruction_gt_aware(recon_cfg, exp_config, model_path, results_path):
    """主 entry。
    recon_cfg 期望欄位:
      vrt_dir, mask_dir (可從 dataset_cfg.labels_dir 推), test_txt (從 dataset.test_txt),
      vis_image_dir (optional; 不指定就用 true color from VRT),
      patch_size (default 256), oil_class (default 0)
    """
    ds_cfg   = exp_config.get('dataset', {})
    arch_cfg = exp_config.get('architecture_cfg', {})
    test_txt = ds_cfg.get('test_txt')
    if not test_txt:
        print('[recon_gt_aware] 沒有 test_txt，跳過')
        return {}
    test_txt = Path(test_txt)
    if not test_txt.exists():
        print(f'[recon_gt_aware] test_txt 不存在: {test_txt}')
        return {}

    vrt_dir   = Path(recon_cfg.get('vrt_dir', ds_cfg.get('vrt_dir')))
    mask_dir  = Path(recon_cfg.get('mask_dir', ds_cfg.get('labels_dir')))
    patch_size = int(recon_cfg.get('patch_size', ds_cfg.get('patch_size', 256)))
    oil_class  = int(ds_cfg.get('target_class_index', 0))
    num_classes = int(arch_cfg.get('num_classes', ds_cfg.get('num_classes', 2)))
    in_channels = int(arch_cfg.get('in_channels', ds_cfg.get('in_channels', 8)))
    band_indices = ds_cfg.get('band_indices', None)
    spectral_index = ds_cfg.get('spectral_index', None)
    mean = ds_cfg.get('mean'); std = ds_cfg.get('std')
    if not mean or not std:
        raise ValueError('[recon_gt_aware] dataset.mean / std 必填（從 args.yaml 取）')
    pixel_mapping = ds_cfg.get('pixel_mapping', None)
    # 選用的額外指標（預設關閉，未設定時輸出與舊版完全相同）：
    #   bg_tiles: true        → 另推論本折 test 背景小圖，回報含背景誤報的 pooled Oil IoU
    #   boundary_iou_px: N    → 回報 Boundary IoU（Cheng et al. 2021），邊界帶寬 N px
    _bg_tiles = bool(recon_cfg.get('bg_tiles', False))
    _boundary_px = int(recon_cfg.get('boundary_iou_px', 0) or 0)
    if _bg_tiles or _boundary_px:
        from metrics_extra import bg_tile_counts, boundary_counts

    # 載入模型（用 DeepLabAdapter，跟訓練同一路徑）
    from deeplab_adapter import DeepLabAdapter
    adapter = DeepLabAdapter(model_path=model_path, dataset_cfg=ds_cfg, architecture_cfg=arch_cfg)
    adapter.model.eval()
    device = adapter.device

    out_dir = Path(results_path) / 'gt_aware_recon'
    out_dir.mkdir(parents=True, exist_ok=True)

    # From 2026-08-05 onward, GT-expand reconstruction keeps the compact JSON
    # hard mask as its default authoritative prediction output.  Explicitly
    # disabling the artifact restores the historical overlay default.
    _artifact_cfg, _artifact_enabled, _retain_legacy_images = \
        _prediction_artifact_policy(recon_cfg)
    overlay_dir = out_dir / 'overlays'
    if _retain_legacy_images:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    # recon_cfg['save_raw_dir'] 未設時完全不影響既有行為
    _raw_dir = recon_cfg.get('save_raw_dir')
    if _raw_dir:
        _raw_dir = Path(_raw_dir); _raw_dir.mkdir(parents=True, exist_ok=True)

    by_scene = _group_patches_by_scene(test_txt)
    print(f'[recon_gt_aware] {len(by_scene)} 個 test 場景，總 patch {sum(len(v) for v in by_scene.values())}')

    # Pixel-exact prediction artifact.  Its default location remains inside the
    # run's result-seg tree: <results_path>/gt_aware_recon/prediction_artifacts.
    _artifact_writer = None
    _artifact_probability_dir = None
    if _artifact_enabled:
        try:
            from predmask_codec import PredictionRunWriter, RunBudgetExceeded, build_run_metadata
            _fold_match = re.search(r'fold(\d+)', test_txt.name)
            _fold = int(_fold_match.group(1)) if _fold_match else None
            _artifact_out_value = _artifact_cfg.get('output_dir')
            if _artifact_out_value:
                _artifact_out = Path(_artifact_out_value)
                if not _artifact_out.is_absolute():
                    _artifact_out = Path(results_path) / _artifact_out
            else:
                _artifact_out = out_dir / 'prediction_artifacts'
            _run_meta = build_run_metadata(
                repo_name='OIL_PROJECT_MutiBand_GT_expand',
                repo_root=Path(__file__).resolve().parents[1],
                arm=exp_config.get('experiment_name', Path(results_path).name),
                fold=_fold,
                model_path=model_path,
                resolved_config={'dataset': ds_cfg, 'architecture_cfg': arch_cfg,
                                 'reconstruction': recon_cfg},
                entrypoint='main/main_runner.py',
                recon_callable='recon_gt_aware_module.run_reconstruction_gt_aware',
                expected_scenes=len(by_scene),
                inputs={'test_txt': str(test_txt), 'vrt_dir': str(vrt_dir),
                        'mask_dir': str(mask_dir)},
                code_files=['main/predmask_codec.py', 'main/recon_gt_aware_module.py',
                            'main/main_runner.py'],
                unsupported_harnesses=['branch_a_oilonly', 'stage0_panorama'],
            )
            _artifact_writer = PredictionRunWriter(
                _artifact_out,
                _run_meta,
                required=bool(_artifact_cfg.get('required', False)),
                max_runs=int(_artifact_cfg.get('max_runs', 10_000_000)),
                block_rows=int(_artifact_cfg.get('block_rows', 128)),
            )
            _prob_cfg = _artifact_cfg.get('probability', {}) or {}
            if bool(_prob_cfg.get('enabled', False)) and _raw_dir is None:
                _probability_value = _prob_cfg.get('output_dir')
                if _probability_value:
                    _artifact_probability_dir = Path(_probability_value)
                    if not _artifact_probability_dir.is_absolute():
                        _artifact_probability_dir = _artifact_out / _artifact_probability_dir
                else:
                    _artifact_probability_dir = _artifact_out / 'probability'
                _artifact_probability_dir.mkdir(parents=True, exist_ok=True)
        except Exception as _artifact_exc:
            if bool(_artifact_cfg.get('required', False)):
                raise
            _artifact_writer = None
            _artifact_probability_dir = None
            print(f'  ⚠ prediction artifact 初始化失敗，已停用可選輸出: {_artifact_exc}')

    per_scene_iou = []
    cls_oil_inter = cls_oil_union = 0
    cls_bg_inter  = cls_bg_union  = 0
    b_inter = b_union = 0

    for scene, coords in tqdm(by_scene.items(), desc='gt_aware_recon'):
        vrt_path = _find_vrt(vrt_dir, scene)
        if vrt_path is None:
            if _artifact_writer:
                _artifact_writer.record_skip(scene, 'skip_vrt_missing', scene)
            print(f'  ⚠ VRT 找不到: {scene}'); continue
        mask_path = mask_dir / f'{scene}.tif'
        if not mask_path.exists():
            if _artifact_writer:
                _artifact_writer.record_skip(scene, 'skip_mask_missing', str(mask_path))
            print(f'  ⚠ Mask 找不到: {mask_path}'); continue

        with rasterio.open(str(mask_path)) as msrc:
            H, W = msrc.height, msrc.width
            raw_mask = msrc.read(1)
        gt_mask = _apply_pixel_mapping(raw_mask, pixel_mapping)

        pred_canvas  = np.full((H, W), MASK_IGNORE, dtype=np.uint8)
        tested_mask  = np.zeros((H, W), dtype=bool)
        prob_canvas  = (np.full((H, W), np.nan, dtype=np.float16)
                        if (_raw_dir or _artifact_probability_dir) else None)

        for x, y, tensor in _patch_iter(vrt_path, coords, patch_size, mean, std, device, in_channels, band_indices=band_indices, spectral_index=spectral_index):
            pred, prob = _tta_predict_prob(adapter.model, tensor)   # (patch, patch), (C, patch, patch)
            x1, y1 = min(W, x + patch_size), min(H, y + patch_size)
            ph, pw = y1 - y, x1 - x
            pred_canvas[y:y1, x:x1] = pred[:ph, :pw]
            tested_mask[y:y1, x:x1] = True
            if prob_canvas is not None:
                prob_canvas[y:y1, x:x1] = prob[oil_class, :ph, :pw].astype(np.float16)

        _artifact_raw = None
        _artifact_probability_failed = False
        if _artifact_writer:
            try:
                _artifact_raw = _artifact_writer.encode_raw(pred_canvas)
            except RunBudgetExceeded as _artifact_exc:
                _artifact_writer.record_failure(scene, 'exceeds_run_budget', str(_artifact_exc))
                print(f'  ⚠ prediction artifact run budget exceeded for {scene}: {_artifact_exc}')
            except Exception as _artifact_exc:
                _artifact_writer.record_failure(scene, 'raw_encode_failed', repr(_artifact_exc))
                print(f'  ⚠ prediction artifact raw encode failed for {scene}: {_artifact_exc}')

        if _raw_dir:
            # stitch 後/argmax 前 P(oil)；footprint 外=NaN（tested_mask 可由此還原）
            np.savez_compressed(_raw_dir / f'{scene}_prob_oil.npz', prob_oil=prob_canvas)
            cv2.imwrite(str(_raw_dir / f'{scene}_pred_mask.png'), pred_canvas)
        elif _artifact_probability_dir and prob_canvas is not None:
            from predmask_codec import atomic_save_npz
            try:
                atomic_save_npz(_artifact_probability_dir / f'{scene}_prob_oil.npz',
                                prob_oil=prob_canvas)
            except Exception as _artifact_exc:
                _artifact_probability_failed = True
                _artifact_writer.record_failure(
                    scene, 'probability_write_failed', repr(_artifact_exc)
                )
                print(f'  ⚠ prediction probability write failed for {scene}: {_artifact_exc}')

        # IoU
        iou = _compute_scene_iou(pred_canvas, tested_mask, gt_mask, num_classes)
        per_scene_iou.append({'scene': scene, **iou})

        # [修法 A] 累積 pooled IoU：用整個 tested_mask；ignore 視為 bg
        v = tested_mask
        _artifact_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        if v.any():
            p, g = pred_canvas[v], gt_mask[v]
            g = np.where(g == MASK_IGNORE, 1, g)
            if _artifact_writer:
                from predmask_codec import compute_confusion_matrix
                _artifact_cm = compute_confusion_matrix(p, g, num_classes)
            cls_oil_inter += int(((p == 0) & (g == 0)).sum())
            cls_oil_union += int(((p == 0) | (g == 0)).sum())
            cls_bg_inter  += int(((p == 1) & (g == 1)).sum())
            cls_bg_union  += int(((p == 1) | (g == 1)).sum())
        if _boundary_px:
            _bi, _bu = boundary_counts(gt_mask == oil_class, (pred_canvas == oil_class) & v,
                                       v, _boundary_px)
            b_inter += _bi
            b_union += _bu
            per_scene_iou[-1].update({'boundary_inter': _bi, 'boundary_union': _bu})

        if _artifact_writer and _artifact_raw is not None:
            from predmask_codec import probability_reference, sha256_file
            _prob_path = None
            if _raw_dir:
                _prob_path = _raw_dir / f'{scene}_prob_oil.npz'
            elif _artifact_probability_dir:
                _prob_path = _artifact_probability_dir / f'{scene}_prob_oil.npz'
            _probability = (
                probability_reference(
                    status='persisted', path=_prob_path, key='prob_oil',
                    dtype='float16', shape=(H, W),
                    semantics='P(oil) after mean-logit TTA and last-write-wins stitch, pre-argmax',
                    uncovered_sentinel='nan', coverage_ref='#/coverage',
                ) if _prob_path and _prob_path.exists()
                else {'status': 'unknown' if _artifact_probability_failed else 'not_persisted'}
            )
            _mask_provenance = {'path': str(mask_path)}
            _vrt_provenance = {'path': str(vrt_path)}
            try:
                _mask_provenance.update({
                    'size': mask_path.stat().st_size,
                    'sha256': None,
                    'hash_status': 'not_computed_large_nas_raster',
                })
            except OSError as _artifact_exc:
                _mask_provenance['metadata_error'] = repr(_artifact_exc)
            try:
                _vrt_size = vrt_path.stat().st_size
                _vrt_provenance['size'] = _vrt_size
                if _vrt_size <= 64 * 1024 * 1024:
                    _vrt_provenance['sha256'] = sha256_file(vrt_path)
                else:
                    _vrt_provenance.update({
                        'sha256': None,
                        'hash_status': 'not_computed_size_exceeds_67108864',
                    })
            except OSError as _artifact_exc:
                _vrt_provenance['metadata_error'] = repr(_artifact_exc)
            _artifact_writer.write_scene_safe(
                scene_id=scene,
                final_mask=pred_canvas,
                raw_payload=_artifact_raw,
                coverage={'mode': 'derived_from_untested', 'untested_value': MASK_IGNORE},
                labels={
                    'scheme': 'class_index', 'num_classes': num_classes,
                    'names': {str(k): str(v) for k, v in dict(ds_cfg.get('names', {})).items()},
                    'target_class_index': oil_class, 'background_class_index': 1,
                    'untested_value': MASK_IGNORE,
                    'pixel_mapping': ({str(k): int(v) for k, v in pixel_mapping.items()}
                                      if pixel_mapping else {}),
                },
                reconstruction={
                    'mode': 'gt_aware', 'patch_size': patch_size,
                    'stitch': 'last_write_wins',
                    'patch_coordinates': [[int(x), int(y)] for x, y in coords],
                    'tta_fusion': 'mean_logits_then_softmax_argmax',
                },
                postprocess=[],
                probability=_probability,
                metric_semantics={
                    'gt_source': 'canonical_mask_tif', 'valid': 'coverage',
                    'gt_ignore_policy': '255_to_class_1', 'absent_union_iou': 0.0,
                },
                metric_atoms={'primary': {'confusion_matrix': _artifact_cm}},
                provenance={
                    'canonical_mask_tif': _mask_provenance,
                    'vrt': _vrt_provenance,
                },
            )

        if _retain_legacy_images:
            # 舊視覺化只在明確要求時保留；日常可由 prediction JSON 隨時復現。
            pred_oil_full = (pred_canvas == oil_class) & tested_mask
            gt_oil_full   = (gt_mask == oil_class)
            y0, x0, y1, x1 = _find_oil_bbox(pred_oil_full, gt_oil_full, H=H, W=W)
            bg_crop  = _read_truecolor_window(vrt_path, y0, x0, y1, x1)
            pred_crop = pred_oil_full[y0:y1, x0:x1]
            gt_crop   = gt_oil_full[y0:y1, x0:x1]
            oil_iou   = iou.get(f'iou_C{oil_class}', 0.0)
            overlay   = _draw_overlay_cropped(bg_crop, pred_crop, gt_crop, scene, oil_iou)
            out_jpg = overlay_dir / f'{scene}_iou_{oil_iou:.4f}.jpg'
            cv2.imwrite(str(out_jpg), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # 寫 per-scene CSV
    import csv
    csv_path = out_dir / 'per_scene_iou.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scene'] + [f'iou_C{c}' for c in range(num_classes)]
                                + (['boundary_inter', 'boundary_union'] if _boundary_px else []))
        writer.writeheader()
        writer.writerows(per_scene_iou)

    pooled_oil = cls_oil_inter / cls_oil_union if cls_oil_union > 0 else 0.0
    extra = {}
    if _boundary_px:
        extra['recon_gtaware_boundary_iou_px'] = _boundary_px
        extra['recon_gtaware_pooled_boundary_iou'] = f'{(b_inter / b_union if b_union else 0.0):.4f}'
    if _bg_tiles:
        extra.update(_score_bg_tiles(ds_cfg, by_scene, vrt_dir, mask_dir, patch_size, mean, std,
                                     device, in_channels, band_indices, spectral_index,
                                     adapter.model, oil_class, out_dir, bg_tile_counts))
        union_with_bg = cls_oil_union + extra['recon_gtaware_bg_tiles_fp_px']
        extra['recon_gtaware_pooled_oil_iou_with_bg'] = (
            f'{(cls_oil_inter / union_with_bg if union_with_bg else 0.0):.4f}')
    pooled_bg  = cls_bg_inter  / cls_bg_union  if cls_bg_union  > 0 else 0.0
    mean_oil = float(np.mean([s[f'iou_C{oil_class}'] for s in per_scene_iou])) if per_scene_iou else 0.0
    mean_miou = float(np.mean([np.mean([s[f'iou_C{c}'] for c in range(num_classes)]) for s in per_scene_iou])) if per_scene_iou else 0.0

    print(f'[recon_gt_aware] 完成 {len(per_scene_iou)} 場景')
    print(f'  pooled Oil IoU = {pooled_oil:.4f}')
    print(f'  pooled BG  IoU = {pooled_bg:.4f}')
    print(f'  per-scene mean Oil IoU = {mean_oil:.4f}')
    print(f'  per-scene mean mIoU    = {mean_miou:.4f}')
    print(f'  per-scene CSV: {csv_path}')
    for _k, _v in extra.items():
        print(f'  {_k} = {_v}')
    if _retain_legacy_images:
        print(f'  overlays   : {overlay_dir} ({len(per_scene_iou)} 張)')
    else:
        print('  overlays   : disabled (use visualize_prediction_artifact.py on demand)')
    if _artifact_writer:
        _artifact_manifest = _artifact_writer.finalize_safe()
        if _artifact_manifest is not None:
            print(f'  prediction artifacts: {_artifact_manifest}')

    return {
        'recon_gtaware_pooled_oil_iou': f'{pooled_oil:.4f}',
        'recon_gtaware_pooled_bg_iou':  f'{pooled_bg:.4f}',
        'recon_gtaware_mean_oil_iou':   f'{mean_oil:.4f}',
        'recon_gtaware_mean_miou':      f'{mean_miou:.4f}',
        'recon_gtaware_num_scenes':     len(per_scene_iou),
        **extra,
    }


def _score_bg_tiles(ds_cfg, by_scene, vrt_dir, mask_dir, patch_size, mean, std, device,
                    in_channels, band_indices, spectral_index, model, oil_class, out_dir,
                    bg_tile_counts):
    """推論本折 test 場景的凍結背景小圖（bg_coord_txt 中屬於 test 場景的列），同一 TTA。

    這些小圖不在 GT-aware 重組範圍內；判成油的像素全部是誤報，計入含背景的 Oil IoU。
    """
    import csv
    bg_path = ds_cfg.get('bg_coord_txt')
    if not bg_path or not Path(bg_path).exists():
        raise ValueError('[recon_gt_aware] bg_tiles 需要 dataset.bg_coord_txt')
    tiles = {}
    with open(bg_path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            if r['scene'] in by_scene:
                tiles.setdefault(r['scene'], []).append((int(r['x']), int(r['y'])))
    rows, fp_px, px = [], 0, 0
    for scene, coords in tqdm(sorted(tiles.items()), desc='gt_aware_recon bg tiles'):
        vrt_path = _find_vrt(vrt_dir, scene)
        if vrt_path is None:
            raise FileNotFoundError(f'[recon_gt_aware] bg tile VRT 找不到: {scene}')
        with rasterio.open(str(mask_dir / f'{scene}.tif')) as msrc:
            for x, y, tensor in _patch_iter(vrt_path, coords, patch_size, mean, std, device,
                                            in_channels, band_indices=band_indices,
                                            spectral_index=spectral_index):
                pred, _ = _tta_predict_prob(model, tensor)
                gt = msrc.read(1, window=Window(x, y, patch_size, patch_size),
                               boundless=True, fill_value=MASK_IGNORE)
                n_fp, n_px = bg_tile_counts(pred == oil_class, gt)
                rows.append({'scene': scene, 'x': x, 'y': y, 'fp_px': n_fp, 'px': n_px})
                fp_px += n_fp
                px += n_px
    with open(out_dir / 'bg_tiles.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scene', 'x', 'y', 'fp_px', 'px'])
        writer.writeheader()
        writer.writerows(rows)
    return {'recon_gtaware_bg_tiles': len(rows), 'recon_gtaware_bg_tiles_fp_px': fp_px,
            'recon_gtaware_bg_tiles_px': px,
            'recon_gtaware_bg_tiles_with_fp': sum(r['fp_px'] > 0 for r in rows),
            'recon_gtaware_bg_fp_per_M': f'{(1e6 * fp_px / px if px else 0.0):.1f}'}
