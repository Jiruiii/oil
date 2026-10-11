# [TODO/15-dlv3] Prithvi-EO-2.0 encoder + torchvision DeepLabV3(ASPP) head
# （2026-07-15 Claude×Codex 協商定稿，兩輪收斂 ACCEPT/BLOCKING=[]；decision record 見 vault）
"""
定位：decoder 消融 arm——與 PrithviUperNetSegmenter 共用完全相同的 encoder 建構/載入
（deeplab_adapter._build_prithvi_encoder），唯一差異 = 分割頭換成 torchvision DeepLabHead。

組成：
    encoder(PrithviViT) → tap [23]（第 24 block 輸出、經 encoder terminal norm）
    → prepare_features_for_image_model（去 CLS reshape，T=1 時 (B,1024,16,16)）
    → torchvision DeepLabHead(1024, num_classes, atrous_rates=(6,12,18))
      （= ASPP + projection + dropout 0.5 + 3×3 conv + classifier，整包原結構不自組）
    → bilinear 上採樣回輸入解析度（align_corners=False，同 torchvision 慣例）
輸出契約與其他 family 相同：forward(x) -> {'out': logits}（輸入解析度）。

預註冊約束（協商 blocking 條款，違反即 raise，不得看結果後改）：
    - ASPP rates 預註冊允許集 = {(6,12,18) 現行, (2,4,6) P3 grid-適配候選}
      （P3 sensitivity arm，analysis/p3_aspp_rate；(2,4,6) 三條空洞跨度 5/9/13 全進 16×16 格）；
      其餘一律 raise、不得看結果後改。head 以「傳入的 rates」建構，非寫死常數。
    - tap 必須是最後一層（feature_indices=[depth-1]），保證 terminal norm 語意
    - 無 aux classifier / aux loss
    - 結論措辭限定「Prithvi 條件下的 decoder-family contrast」，無 ResNet50+UPerNet
      的完整 2×2，不得做 encoder/decoder 貢獻百分比分解
"""
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from prithvi_backbone import PrithviViT

REGISTERED_ASPP_RATES = (6, 12, 18)          # 現行預設值
# [P3] grid-適配候選加入預註冊允許集（analysis/p3_aspp_rate，Claude×codex 2026-07-29）
REGISTERED_ASPP_RATE_SETS = {(6, 12, 18), (2, 4, 6)}


class PrithviDeepLabV3Segmenter(nn.Module):
    """8-band 輸入 → 6-band 選擇 → Prithvi encoder → tap[-1] → DeepLabHead → {'out': logits}。

    - band_indices：輸入通道 index（8-band VRT 順序中的 [1,2,3,5,6,7]）。
      正規化在 dataset 端（yaml mean/std 槽）已完成，本層只做選擇。
    - feature_index：encoder block tap，必須 = depth-1（terminal norm 後的最終特徵）。
    """

    def __init__(self, encoder: PrithviViT, band_indices, feature_index: int,
                 num_classes: int, aspp_atrous_rates=(6, 12, 18)):
        super().__init__()
        import torch
        depth = len(encoder.blocks)
        if int(feature_index) != depth - 1:
            raise RuntimeError(f"[15-dlv3] feature_index 必須為最後一層 {depth - 1}"
                               f"（terminal norm 後），收到 {feature_index}")
        if tuple(aspp_atrous_rates) not in REGISTERED_ASPP_RATE_SETS:
            raise RuntimeError(f"[15-dlv3] ASPP rates 預註冊允許集={REGISTERED_ASPP_RATE_SETS}，"
                               f"收到 {tuple(aspp_atrous_rates)}；不得看結果後改")
        self.encoder = encoder
        self.register_buffer('band_indices',
                             torch.as_tensor(list(band_indices), dtype=torch.long),
                             persistent=False)
        self.feature_index = int(feature_index)
        # effective_time_dim = num_frames // tubelet(=1)；T=1 時特徵通道數 = embed_dim
        eff_t = encoder.patch_embed.input_size[0] // encoder.patch_embed.patch_size[0]
        in_channels = encoder.embed_dim * eff_t
        # 整包 torchvision DeepLabHead（ASPP+projection+dropout0.5+3×3conv+classifier），
        # 只參數化 in_channels 與 rates；無 aux 分支
        self.head = DeepLabHead(in_channels, num_classes,
                                atrous_rates=tuple(aspp_atrous_rates))

    def select_bands(self, x):
        return x.index_select(1, self.band_indices)

    def forward(self, x):
        in_hw = x.shape[-2:]
        x = self.select_bands(x)      # (B, 6, H, W)
        x = x.unsqueeze(2)            # (B, 6, 1, H, W)：顯式 T=1
        feats = self.encoder.forward_features(x)
        feat = feats[self.feature_index]                              # 最後層（含 terminal norm）
        feat = self.encoder.prepare_features_for_image_model([feat])[0]  # (B, embed*T, h, w)
        out = self.head(feat)                                         # (B, num_classes, h, w)
        if out.shape[-2:] != in_hw:
            out = F.interpolate(out, size=in_hw, mode='bilinear', align_corners=False)
        return {'out': out}
