# 資料交付契約（DATA_MANIFEST）

這份文件回答「`DATA_DIR` 底下到底要放什麼」——`README.md` 第三節列出了資料種類，
這裡補上目錄結構、數量、大小，跟一個容易踩到的地雷（VRT 不能直接複製）。

---

## 一、`DATA_DIR` 預期目錄結構

```
<DATA_DIR>/
├── MS6_sen2like_vrt/                    # 訓練時實際讀取的來源
│   └── <事件資料夾>/
│       └── <場景檔名>.vrt
├── mask/                                # GT Mask TIF，攤平放，不分事件資料夾
│   └── <場景檔名>.tif
└── MS6_sen2like/                        # 原始 8-band per-band TIF（只有要重建 VRT 時才需要）
    └── <事件資料夾>/
        └── <場景檔名>_B01.tif ... _B12.tif（8 個 band 各一檔）
```

`<場景檔名>` 格式例：`2017年希臘薩龍灣Agia_Zoni_II油輪沉沒事件_20170918_S2_34SGG`
（事件描述 + 日期 + 衛星 + tile ID）。`mask/` 底下的檔名要跟對應 VRT 的檔名（不含
副檔名）完全一致，`recon_gt_aware_module.py` 是靠檔名比對兩邊的。

## 二、數量與大小（來自正式 repo 現況，2026-09）

| 項目 | 數量/大小 | 說明 |
|---|---|---|
| VRT 檔（`MS6_sen2like_vrt/`） | 789 個場景，共 ~6.3 MB | 純 XML 指標檔，體積小，好搬 |
| Mask TIF（`mask/`） | 789 個場景，共 ~873 MB | |
| 原始 per-band TIF（`MS6_sen2like/`） | 對應同一批場景，共 ~521 GB | 真正佔空間的是這裡 |
| 本 onboarding 目錄 `data_split/` 實際會用到的場景 | **355 個**（`data_split/patch_level_gt_expand/*.txt` 去除 patch 座標後去重的唯一場景數；其中 351 個場景另外有背景 patch） | 這是本包 yaml 範本（`experiments.example.yaml`）真正會讀到的子集，不是全部 789 個都要準備齊 |

新人不需要一次準備滿 789 個場景——只要 `data_split/` 底下的 fold 檔案（train/val/test
+ bg_coords.tsv）引用到的 355 個場景，在 `DATA_DIR/MS6_sen2like_vrt/` 跟
`DATA_DIR/mask/` 下都找得到同名檔案，就能跑本包預設的 3-fold 實驗。

## 三、⚠️ VRT 不能直接複製過去就用

`MS6_sen2like_vrt/` 底下的 `.vrt` 檔案很小、看起來很好搬，但它們是
**純 XML 指標檔**（見 `preprocess/build_vrt_ms6.py`），裡面的 `<SourceFilename>`
是建立當下那台機器上、指向原始 per-band TIF 的**絕對路徑**。也就是說：

- 如果你只複製 `.vrt`（6.3 MB）跟 `mask/`（873 MB），不複製 `MS6_sen2like/`
  原始 TIF（521 GB），或者複製了原始 TIF 但放的目錄結構跟原機器不同，
  訓練時 rasterio 打開 VRT 會找不到底層像素資料，直接報錯或讀出全零/全空。
- 正確做法二選一：
  1. 把 `MS6_sen2like/` 原始 TIF 用**完全相同的相對目錄結構**複製到新機器
     的對應位置，讓 VRT 內嵌的絕對路徑（或你調整後的路徑）仍然指得到；
  2. 或是把原始 TIF 複製到你自己想要的位置後，在新機器上重新跑一次
     `preprocess/build_vrt_ms6.py` 重新產生 VRT——這樣新產生的 VRT 才會
     指向新機器上正確的路徑。**不要偷懶只搬 VRT 檔案本身。**

`preprocess/preflight_check.py` 會檢查 VRT 是否真的讀得到資料，建議資料就位後
先跑一次這個再開始訓練，可以提早抓到「VRT 指標對不上」這類問題。

## 四、Prithvi 權重

不算在 `DATA_DIR` 裡，是獨立的一步：

```bash
python main/fetch_prithvi_checkpoint.py
```

會下載到 `<repo_root>/pretrained_ckpt/`（約 1.33 GB），`experiments.example.yaml`
裡的 `backbone_ckpt` 已經預設指向這個位置，跑完下載腳本不用再改路徑。
