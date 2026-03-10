import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

from .dinov2_vit import DINOv2ViT
from .PQI import PSP
from .SAM import SAM

########################################################################################################################

class LocalBCP(nn.Module):
    """
    修改版 BCP: Spatial-Aware Bin Predictor
    不使用 Global Pooling，而是預測一個粗略的網格 (Grid) 的 Bins。
    """
    def __init__(
        self,
        max_depth: float,
        min_depth: float,
        in_features: int = 512,
        hidden_features: int = 512 * 4,
        out_features: int = 256,
        act_layer=nn.GELU,
        drop: float = 0.0,
        grid_size: tuple = (8, 8)  # 新增: 控制空間網格的大小
    ):
        super().__init__()
        self.grid_size = grid_size
        
        # 改用 Conv2d 來保留空間資訊，而不是 Linear
        self.conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout2d(drop) # 使用 2D Dropout
        
        self.register_buffer("min_depth", torch.tensor(float(min_depth)))
        self.register_buffer("max_depth", torch.tensor(float(max_depth)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x input: B x C x H x W
        
        # 1. Spatial Pooling: 變成 B x C x Gh x Gw (例如 8x8)
        # 這一步讓模型擁有 "區域性" 的視野
        x = F.adaptive_avg_pool2d(x, self.grid_size)
        
        # 2. MLP (實作為 1x1 Conv)
        x = self.drop(self.act(self.conv1(x)))
        x = self.drop(self.conv2(x))  # B x D x Gh x Gw

        # 3. 計算 Bins (邏輯與原本相同，但多了空間維度)
        bins = torch.softmax(x, dim=1)
        bins = bins / (bins.sum(dim=1, keepdim=True) + 1e-12)

        span = (self.max_depth - self.min_depth)
        bin_widths = span * bins  # B x D x Gh x Gw
        
        # Pad on "Channel" dimension (dim=1)
        # Pad 需要指定的順序是 (left, right, top, bottom, front, back...)
        # 我們只對 Channel pad，所以需要一點技巧或用 cat
        min_depth_map = self.min_depth.view(1, 1, 1, 1).expand_as(bin_widths[:, 0:1, :, :])
        
        # 累積求和得到 Edges
        # 先 concat min_depth
        bin_widths_padded = torch.cat([min_depth_map, bin_widths], dim=1) # B x (D+1) x Gh x Gw
        bin_edges = torch.cumsum(bin_widths_padded, dim=1)
        
        # 計算 Centers
        centers = 0.5 * (bin_edges[:, :-1, :, :] + bin_edges[:, 1:, :, :]) # B x D x Gh x Gw
        
        return centers

class SpatialDispHead(nn.Module):
    """
    修改版 DispHead: 支援 Spatial-Varying Centers
    """
    def __init__(self, input_dim: int = 100, temperature: float = 1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 256, kernel_size=3, padding=1)
        self.temperature = temperature

    def forward(self, x: torch.Tensor, centers: torch.Tensor, scale: int) -> torch.Tensor:
        # x: B x D x H x W (Feature Map)
        # centers: B x D x Gh x Gw (Grid Centers, e.g., 8x8)
        
        x = self.conv1(x)
        
        # 關鍵修改: 將粗糙的 centers 插值放大到跟 x 一樣的解析度 (H x W)
        # 使用 bilinear 插值，這樣 Bins 的變化會是平滑的
        if centers.shape[-2:] != x.shape[-2:]:
            centers = F.interpolate(centers, size=x.shape[-2:], mode='bilinear', align_corners=False)

        if self.temperature != 1.0:
            x = x / self.temperature
            
        x = F.softmax(x, dim=1)
        
        # 現在 x 和 centers 都是 B x D x H x W，可以做 Element-wise 的加權總和
        out = torch.sum(x * centers, dim=1, keepdim=True) # B x 1 x H x W
        
        if scale > 1:
            out = F.interpolate(out, scale_factor=scale, mode='bilinear', align_corners=False)
            
        return out


def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor"""
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)


class PixelFormer(nn.Module):
    def __init__(
        self,
        version=None,                         # 保留但用於選擇 DINOv2 版本
        inv_depth: bool = False,
        pretrained=None,                      # DINOv2 自動從 hub 載入，這個可以忽略
        frozen_stages: int = -1,
        min_depth: float = 0.1,
        max_depth: float = 100.0,
        use_amp: bool = True,
        amp_dtype: str = "bf16",
        use_compile: bool = False,
        enable_tf32: bool = True,
        # ===== 新增 DINOv2 相關參數 =====
        backbone_type: str = "dinov2",        # "dinov2" 或 "swin"
        dinov2_model: str = "dinov2_vitb14",  # 'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14'
        freeze_backbone: bool = False,        # 是否凍結 backbone
        **kwargs,
    ):
        super().__init__()

        self.inv_depth = inv_depth
        self.with_auxiliary_head = False
        self.with_neck = False
        self.backbone_type = backbone_type

        # 選擇 AMP dtype
        self.use_amp = use_amp
        if amp_dtype.lower() == "fp16":
            self._amp_dtype = torch.float16
        else:
            self._amp_dtype = torch.bfloat16

        # TF32 for matmul/conv（Ampere+）
        if enable_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        norm_cfg = dict(type='BN', requires_grad=True)

        # ===================================================================
        # Backbone 選擇：DINOv2 ViT 或 Swin Transformer
        # ===================================================================
        if backbone_type == "dinov2":
            # ------------- DINOv2 ViT Backbone -------------
            print(f"[PixelFormer] Using DINOv2 backbone: {dinov2_model}")
            
            # DINOv2 輸出維度配置
            dinov2_configs = {
                'dinov2_vits14': {
                    'out_indices': [3, 6, 9, 12],
                    'in_channels': [96, 192, 384, 384],      # 投影後的維度
                },
                'dinov2_vitb14': {
                    'out_indices': [3, 6, 9, 12],
                    'in_channels': [192, 384, 768, 768],
                },
                'dinov2_vitl14': {
                    'out_indices': [5, 12, 18, 24],
                    'in_channels': [256, 512, 1024, 1024],   # UniDepth 使用的配置
                },
                'dinov2_vitg14': {
                    'out_indices': [10, 20, 30, 40],
                    'in_channels': [384, 768, 1536, 1536],
                },
            }
            
            cfg = dinov2_configs.get(dinov2_model, dinov2_configs['dinov2_vitb14'])
            in_channels = cfg['in_channels']
            
            self.backbone = DINOv2ViT(
                model_name=dinov2_model,
                pretrained=True,
                out_indices=cfg['out_indices'],
                freeze_backbone=freeze_backbone,
            )
            
        else:
            # ------------- 原本的 Swin Transformer Backbone -------------
            from .swin_transformer import SwinTransformer
            
            window_size = int(version[-2:])

            if version[:-2] == 'base':
                embed_dim = 128
                depths = [2, 2, 18, 2]
                num_heads = [4, 8, 16, 32]
                in_channels = [128, 256, 512, 1024]
            elif version[:-2] == 'large':
                embed_dim = 192
                depths = [2, 2, 18, 2]
                num_heads = [6, 12, 24, 48]
                in_channels = [192, 384, 768, 1536]
            elif version[:-2] == 'tiny':
                embed_dim = 96
                depths = [2, 2, 6, 2]
                num_heads = [3, 6, 12, 24]
                in_channels = [96, 192, 384, 768]
            else:
                raise ValueError(f"Unknown version: {version}")

            backbone_cfg = dict(
                embed_dim=embed_dim,
                depths=depths,
                num_heads=num_heads,
                window_size=window_size,
                ape=False,
                drop_path_rate=0.3,
                patch_norm=True,
                use_checkpoint=False,
                frozen_stages=frozen_stages,
            )
            
            self.backbone = SwinTransformer(**backbone_cfg)

        # ===================================================================
        # Decoder 和 SAM (根據 backbone 的輸出維度調整)
        # ===================================================================
        dec_embed_dim = 512
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=dec_embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False,
        )

        self.decoder = PSP(**decoder_cfg)

        # --------------- SAM 維度調整 ---------------
        # SAM 的 input_dim 需要匹配 backbone 的輸出
        # embed_dim 和 v_dim 可以保持原樣或按比例調整
        
        if backbone_type == "dinov2":
            # DINOv2 的維度較大，SAM 維度也相應調整
            sam_dims = [128, 256, 512, 1024]
            v_dims = [64, 128, 256, dec_embed_dim]
            sam_heads = [4, 8, 16, 32]
        else:
            # Swin 原本的配置
            sam_dims = [128, 256, 512, 1024]
            v_dims = [64, 128, 256, dec_embed_dim]
            sam_heads = [4, 8, 16, 32]

        sam_win = 7

        self.sam4 = SAM(input_dim=in_channels[3], embed_dim=sam_dims[3], window_size=sam_win, v_dim=v_dims[3], num_heads=sam_heads[3])
        self.sam3 = SAM(input_dim=in_channels[2], embed_dim=sam_dims[2], window_size=sam_win, v_dim=v_dims[2], num_heads=sam_heads[2])
        self.sam2 = SAM(input_dim=in_channels[1], embed_dim=sam_dims[1], window_size=sam_win, v_dim=v_dims[1], num_heads=sam_heads[1])
        self.sam1 = SAM(input_dim=in_channels[0], embed_dim=sam_dims[0], window_size=sam_win, v_dim=v_dims[0], num_heads=sam_heads[0])

        # ----------------- Heads -----------------
        self.disp_head1 = SpatialDispHead(input_dim=sam_dims[0], temperature=1.0)
        self.bcp = LocalBCP(max_depth=max_depth, min_depth=min_depth, 
                    in_features=dec_embed_dim, out_features=256, grid_size=(8, 8))

        # Init weights (只有 Swin 需要手動載入)
        if backbone_type != "dinov2":
            self.init_weights(pretrained=pretrained)
        else:
            # DINOv2 已經自動載入預訓練權重，只需初始化 decoder
            self.decoder.init_weights()
            print(f"[PixelFormer] DINOv2 backbone loaded, decoder initialized")

        # ✅ 選配：torch.compile
        if use_compile and hasattr(torch, "compile"):
            try:
                self.forward = torch.compile(self.forward, mode="max-autotune")
            except Exception as e:
                print(f"[compile] disabled due to: {e}")

    def init_weights(self, pretrained=None):
        """只有使用 Swin backbone 時才需要"""
        if self.backbone_type == "dinov2":
            return
        print(f'== Load encoder backbone from: {pretrained}')
        self.backbone.init_weights(pretrained=pretrained)
        self.decoder.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def _autocast_context(self, device: torch.device):
        if not self.use_amp:
            return nullcontext()
        dev_type = "cuda" if device.type == "cuda" else "cpu"
        return torch.autocast(device_type=dev_type, dtype=self._amp_dtype, enabled=True)

    def forward(self, imgs: torch.Tensor):
        # AMP context（訓練/推論都可用）
        ctx = self._autocast_context(imgs.device)
        with ctx:
            enc_feats = self.backbone(imgs)
            if self.with_neck:
                enc_feats = self.neck(enc_feats)

            # PSP decoder 產生高階 query q4
            q4 = self.decoder(enc_feats)                # B x dec_embed_dim x H/32 x W/32

            # 自上而下的 SAM（使用 functional pixel_shuffle，避免反覆構建 module）
            q3 = self.sam4(enc_feats[3], q4)
            q3 = F.pixel_shuffle(q3, 2)

            q2 = self.sam3(enc_feats[2], q3)
            q2 = F.pixel_shuffle(q2, 2)

            q1 = self.sam2(enc_feats[1], q2)
            q1 = F.pixel_shuffle(q1, 2)

            q0 = self.sam1(enc_feats[0], q1)            # B x D x H/4 x W/4

            # Bin centers（由 q4 估計）
            bin_centers = self.bcp(q4)                  # B x D x 1 x 1

            # 期望值回歸為深度；scale=4 -> 輸出到 H x W
            f = self.disp_head1(q0, bin_centers, scale=4)

        return f


# ============================================================================
# 便捷函數：快速創建不同版本的模型
# ============================================================================

def pixelformer_dinov2_vitl14(min_depth=0.1, max_depth=100.0, freeze_backbone=False, **kwargs):
    """PixelFormer with DINOv2 ViT-L/14 (推薦，UniDepth 使用的版本)"""
    return PixelFormer(
        backbone_type="dinov2",
        dinov2_model="dinov2_vitl14",
        min_depth=min_depth,
        max_depth=max_depth,
        freeze_backbone=freeze_backbone,
        **kwargs
    )


def pixelformer_dinov2_vitb14(min_depth=0.1, max_depth=100.0, freeze_backbone=False, **kwargs):
    """PixelFormer with DINOv2 ViT-B/14 (較小，較快)"""
    return PixelFormer(
        backbone_type="dinov2",
        dinov2_model="dinov2_vitb14",
        min_depth=min_depth,
        max_depth=max_depth,
        freeze_backbone=freeze_backbone,
        **kwargs
    )


def pixelformer_swin_large(min_depth=0.1, max_depth=100.0, pretrained=None, **kwargs):
    """PixelFormer with Swin-Large (原本的版本)"""
    return PixelFormer(
        backbone_type="swin",
        version="large22",
        min_depth=min_depth,
        max_depth=max_depth,
        pretrained=pretrained,
        **kwargs
    )