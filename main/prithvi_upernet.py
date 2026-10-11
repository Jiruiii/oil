# [TODO/15] Prithvi-EO-2.0 官方式 UperNet 分割組件（2026-07-13 Claude×Codex 共識第 4 案）
"""
組成（對應官方 NASA-IMPACT/Prithvi-EO-2.0 configs/sen1floods11.yaml 的 300M 規格）：
    encoder(PrithviViT) → SelectIndices [5,11,17,23] → prepare_features_for_image_model
    → UperNetDecoder(channels=256, scale_modules=True) → SegmentationHead(dropout=0.1)
    → bilinear resize 回輸入解析度（官方 rescale: True）
輸出契約與 _SMPDeepLabV3PlusWrapper 相同：forward(x) -> {'out': logits}（輸入解析度）。

Vendor 溯源（上游 SHA256 見 prithvi_vendor_manifest.json；上游=terratorch v1.2.8，Apache-2.0）：
    - ConvModule                ← terratorch/models/decoders/utils.py
    - PPM / UperNetDecoder      ← terratorch/models/decoders/upernet_decoder.py
    - SegmentationHead          ← terratorch/models/heads/segmentation_head.py
與上游的差異（全列，防漂移；除此以外逐字未動，含 align_corners=True 預設與 scale_modules 的 fpn1-4 結構）：
    1. 移除 @TERRATORCH_DECODER_REGISTRY.register 裝飾器與 terratorch.registry / .utils import
    2. utils.py 只取 ConvModule（resize() 未使用故未 vendor）
    3. UperNetDecoder 的 scale_modules DeprecationWarning 保留原文（官方 300M config 仍用 scale_modules=true）
token→image reshape 不 vendor terratorch 的 ReshapeTokensToImage，改用官方 encoder 自帶的
prepare_features_for_image_model()（數學相同：去 CLS → rearrange "(t h w) e -> (t e) h w"；
T=1 時每層即 (B, 1024, 16, 16)），更貼官方 HF 實作、少一份要對 parity 的程式。
"""
import warnings

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from prithvi_backbone import PrithviViT


# ─────────────────────────────────────────────────────────────────────────────
# vendored: terratorch v1.2.8 terratorch/models/decoders/utils.py（ConvModule）
# ─────────────────────────────────────────────────────────────────────────────
class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        padding=0,
        dilation=1,
        stride=1,
        inplace=False,
        transpose=False,
        scale_factor=None,
    ) -> None:

        super().__init__()

        if transpose:
            kind = "Transpose"
        else:
            kind = ""

        conv_name = f"Conv{kind}2d"

        if transpose:
            stride = scale_factor
            padding = (kernel_size - scale_factor) // 2

        conv_template = getattr(nn, conv_name)
        self.conv = conv_template(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, stride=stride, bias=False
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=inplace)

    def forward(self, x):

        return self.act(self.norm(self.conv(x)))


# ─────────────────────────────────────────────────────────────────────────────
# vendored: terratorch v1.2.8 terratorch/models/decoders/upernet_decoder.py
# ─────────────────────────────────────────────────────────────────────────────
# Adapted from MMSegmentation
class UperNetDecoder(nn.Module):
    """UperNetDecoder. Adapted from MMSegmentation."""

    def __init__(
        self,
        embed_dim: list[int],
        pool_scales: tuple[int] = (1, 2, 3, 6),
        channels: int = 256,
        align_corners: bool = True,  # noqa: FBT001, FBT002
        scale_modules: bool = False,
    ):
        """Constructor

        Args:
            embed_dim (list[int]): Input embedding dimension for each input.
            pool_scales (tuple[int], optional): Pooling scales used in Pooling Pyramid
                Module applied on the last feature. Default: (1, 2, 3, 6).
            channels (int, optional): Channels used in the decoder. Defaults to 256.
            align_corners (bool, optional): Wheter to align corners in rescaling. Defaults to True.
            scale_modules (bool, optional): Whether to apply scale modules to the inputs. Needed for plain ViT.
                Defaults to False.
        """
        super().__init__()
        if scale_modules:
            # TODO: remove scale_modules before v1?
            warnings.warn(
                "DeprecationWarning: scale_modules is deprecated and will be removed in future versions. "
                "Use LearnedInterpolateToPyramidal neck instead.",
                stacklevel=1,
            )

        self.scale_modules = scale_modules
        if scale_modules:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim[0], embed_dim[0] // 2, 2, 2),
                nn.BatchNorm2d(embed_dim[0] // 2),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim[0] // 2, embed_dim[0] // 4, 2, 2),
            )
            self.fpn2 = nn.Sequential(nn.ConvTranspose2d(embed_dim[1], embed_dim[1] // 2, 2, 2))
            self.fpn3 = nn.Sequential(nn.Identity())
            self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
            self.embed_dim = [embed_dim[0] // 4, embed_dim[1] // 2, embed_dim[2], embed_dim[3]]
        else:
            self.embed_dim = embed_dim

        self.out_channels = channels
        self.channels = channels
        self.align_corners = align_corners
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.embed_dim[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        self.bottleneck = ConvModule(
            self.embed_dim[-1] + len(pool_scales) * self.channels, self.channels, 3, padding=1, inplace=True
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for embed_dim in self.embed_dim[:-1]:  # skip the top layer
            l_conv = ConvModule(
                embed_dim,
                self.channels,
                1,
                inplace=False,
            )
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                inplace=False,
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(len(self.embed_dim) * self.channels, self.channels, 3, padding=1, inplace=True)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """

        if self.scale_modules:
            scaled_inputs = []
            scaled_inputs.append(self.fpn1(inputs[0]))
            scaled_inputs.append(self.fpn2(inputs[1]))
            scaled_inputs.append(self.fpn3(inputs[2]))
            scaled_inputs.append(self.fpn4(inputs[3]))
            inputs = scaled_inputs
        # build laterals
        laterals = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + torch.nn.functional.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=self.align_corners
            )

        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = torch.nn.functional.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )
        fpn_outs = torch.cat(fpn_outs, dim=1)
        feats = self.fpn_bottleneck(fpn_outs)
        return feats


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet."""

    def __init__(self, pool_scales, in_channels, channels, align_corners):
        """Constructor

        Args:
            pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
                Module.
            in_channels (int): Input channels.
            channels (int): Channels after modules, before conv_seg.
            align_corners (bool): align_corners argument of F.interpolate.
        """
        super().__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels

        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(self.in_channels, self.channels, 1, inplace=True),
                )
            )

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = torch.nn.functional.interpolate(
                ppm_out, size=x.size()[2:], mode="bilinear", align_corners=self.align_corners
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


# ─────────────────────────────────────────────────────────────────────────────
# vendored: terratorch v1.2.8 terratorch/models/heads/segmentation_head.py
# ─────────────────────────────────────────────────────────────────────────────
class SegmentationHead(nn.Module):
    """Segmentation head"""

    def __init__(
        self, in_channels: int, num_classes: int, channel_list: list[int] | None = None, dropout: float = 0
    ) -> None:
        """Constructor

        Args:
            in_channels (int): Number of input channels
            num_classes (int): Number of output classes
            channel_list (list[int] | None, optional):  List with number of channels for each Conv
                layer to be created. Defaults to None.
            dropout (float, optional): Dropout value to apply. Defaults to 0.
        """
        super().__init__()
        self.num_classes = num_classes
        if channel_list is None:
            pre_head = nn.Identity()
        else:

            def block(in_channels, out_channels):
                return nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1), nn.ReLU()
                )

            channel_list = [in_channels, *channel_list]
            pre_head = nn.Sequential(
                *[block(channel_list[i], channel_list[i + 1]) for i in range(len(channel_list) - 1)]
            )
            in_channels = channel_list[-1]
        dropout = nn.Identity() if dropout == 0 else nn.Dropout(dropout)
        self.head = nn.Sequential(
            pre_head,
            dropout,
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_classes,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 本專案組裝層（非 vendor）
# ─────────────────────────────────────────────────────────────────────────────
class PrithviViTEncoder(PrithviViT):
    """薄 subclass：只加 gradient-checkpointing 開關，其餘完全沿用上游 PrithviViT。
    開關生效時 forward_features 與上游逐行同構，僅 block 呼叫包 torch.utils.checkpoint。"""

    def __init__(self, *args, grad_checkpointing: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_checkpointing = grad_checkpointing

    def forward_features(self, x, temporal_coords=None, location_coords=None):
        if not (self.grad_checkpointing and self.training):
            return super().forward_features(x, temporal_coords, location_coords)

        if len(x.shape) == 4 and self.patch_embed.input_size[0] == 1:
            # add time dim
            x = x.unsqueeze(2)
        sample_shape = x.shape[-3:]

        x = self.patch_embed(x)
        pos_embed = self.interpolate_pos_encoding(sample_shape)
        x = x + pos_embed[:, 1:, :]

        if self.temporal_encoding and temporal_coords is not None:
            num_tokens_per_frame = x.shape[1] // self.num_frames
            x = x + self.temporal_embed_enc(temporal_coords, num_tokens_per_frame)
        if self.location_encoding and location_coords is not None:
            x = x + self.location_embed_enc(location_coords)

        cls_token = self.cls_token + pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        out = []
        for block in self.blocks:
            x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            out.append(x.clone())

        x = self.norm(x)
        out[-1] = x
        return out


class PrithviUperNetSegmenter(nn.Module):
    """8-band 輸入 → 6-band 選擇 → Prithvi encoder → 官方式 UperNet → {'out': logits}。

    - band_indices：輸入通道 index（8-band VRT 順序中的 [1,2,3,5,6,7] = B02,B03,B04,B8A,B11,B12）。
      正規化在 dataset 端（yaml mean/std 槽）已完成，本層只做選擇。
    - feature_indices：encoder blocks 的 tap 位置（300M 官方=[5,11,17,23]，0-indexed）。
    - T=1 顯式建立時間維（[B,6,1,H,W]），不依賴上游 4D 隱式補維。
    """

    def __init__(self, encoder: PrithviViT, band_indices, feature_indices,
                 num_classes: int, decoder_channels: int = 256, head_dropout: float = 0.1):
        super().__init__()
        if len(feature_indices) != 4:
            raise RuntimeError(f"[prithvi] UperNet 需要 4 個 feature taps，收到 {feature_indices}")
        self.encoder = encoder
        self.register_buffer('band_indices',
                             torch.as_tensor(list(band_indices), dtype=torch.long),
                             persistent=False)
        self.feature_indices = list(feature_indices)
        # effective_time_dim = num_frames // tubelet(=1)；T=1 時每層 feature 通道數 = embed_dim
        eff_t = encoder.patch_embed.input_size[0] // encoder.patch_embed.patch_size[0]
        embed_dims = [encoder.embed_dim * eff_t] * 4
        self.decoder = UperNetDecoder(embed_dim=embed_dims, channels=decoder_channels,
                                      align_corners=True, scale_modules=True)
        self.head = SegmentationHead(decoder_channels, num_classes, dropout=head_dropout)

    def select_bands(self, x):
        return x.index_select(1, self.band_indices)

    def forward(self, x):
        in_hw = x.shape[-2:]
        x = self.select_bands(x)      # (B, 6, H, W)
        x = x.unsqueeze(2)            # (B, 6, 1, H, W)：顯式 T=1
        feats = self.encoder.forward_features(x)
        feats = [feats[i] for i in self.feature_indices]
        feats = self.encoder.prepare_features_for_image_model(feats)   # 4 × (B, embed*T, h, w)
        out = self.decoder(feats)
        out = self.head(out)
        if out.shape[-2:] != in_hw:   # 官方 rescale: True（decoder 出 H/4，補回輸入解析度）
            out = F.interpolate(out, size=in_hw, mode='bilinear', align_corners=False)
        return {'out': out}
