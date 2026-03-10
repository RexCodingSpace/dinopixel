import torch
import torch.nn as nn
from contextlib import nullcontext

from .dinov2_vit import DINOv2ViT
from .pixel_decoder import PixelDecoderLayer
from .heads import SemanticAwareBinHead, LightweightPixelHead

class MambaPixelFormer(nn.Module):
    def __init__(
        self, 
        min_depth: float = 0.1, 
        max_depth: float = 100.0, 
        backbone_type: str = "dinov2",
        dinov2_model: str = "dinov2_vitl14", 
        freeze_backbone: bool = False,
        use_amp: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.use_amp = use_amp
        self._amp_dtype = torch.bfloat16 if use_amp else torch.float32
        
        print(f"[Model] Initializing MambaPixelFormer with {dinov2_model}...")
        
        # 1. Backbone (DINOv2)
        # 修正：這是經過投影後的維度，不是原始 ViT 維度
        dinov2_configs = {
            'dinov2_vits14': {
                'out_indices': [3, 6, 9, 12],
                'in_channels': [96, 192, 384, 384],
            },
            'dinov2_vitb14': {
                'out_indices': [3, 6, 9, 12],
                'in_channels': [192, 384, 768, 768],
            },
            'dinov2_vitl14': {
                'out_indices': [5, 12, 18, 24],
                'in_channels': [256, 512, 1024, 1024],
            }
        }
        
        cfg = dinov2_configs.get(dinov2_model, dinov2_configs['dinov2_vitl14'])
        in_channels = cfg['in_channels']
        out_indices = cfg['out_indices']
        
        self.backbone = DINOv2ViT(
            model_name=dinov2_model,
            pretrained=True,
            out_indices=out_indices,
            freeze_backbone=freeze_backbone
        )
        
        # 2. Decoder 架構設定
        embed_dim = 256  # Decoder 內部的統一維度
        
        # 初始投影層 (處理 E4)
        self.proj_e4 = nn.Sequential(
            nn.Conv2d(in_channels[3], embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU()
        )
        
        # Mamba Decoder Layers (U-Net 風格)
        # Layer 3: 融合 E3
        self.up3 = PixelDecoderLayer(dim=embed_dim, skip_dim=in_channels[2])
        # Layer 2: 融合 E2
        self.up2 = PixelDecoderLayer(dim=embed_dim, skip_dim=in_channels[1])
        # Layer 1: 融合 E1
        self.up1 = PixelDecoderLayer(dim=embed_dim, skip_dim=in_channels[0])
        
        # 3. Hybrid Regression Heads
        n_bins = 64
        self.bin_head = SemanticAwareBinHead(in_dim=embed_dim, n_bins=n_bins, min_depth=min_depth, max_depth=max_depth)
        self.pixel_head = LightweightPixelHead(in_dim=embed_dim, n_bins=n_bins)
        
    def _autocast_context(self, device: torch.device):
        if not self.use_amp:
            return nullcontext()
        dev_type = "cuda" if device.type == "cuda" else "cpu"
        return torch.autocast(device_type=dev_type, dtype=self._amp_dtype, enabled=True)

    def forward(self, imgs):
        ctx = self._autocast_context(imgs.device)
        with ctx:
            # 1. Backbone
            features = self.backbone(imgs)  # [E1, E2, E3, E4]
            
            # 2. Decoder Path
            x = self.proj_e4(features[3])
            
            x = self.up3(x, features[2])  # Fuse E3
            x = self.up2(x, features[1])  # Fuse E2
            x = self.up1(x, features[0])  # Fuse E1
            
            # 3. Hybrid Regression
            centers = self.bin_head(x)
            depth = self.pixel_head(x, centers, scale=4)
            
        return depth


# 便捷函數
def mamba_pixelformer_dinov2_vits14(**kwargs):
    return MambaPixelFormer(dinov2_model="dinov2_vits14", **kwargs)

def mamba_pixelformer_dinov2_vitb14(**kwargs):
    return MambaPixelFormer(dinov2_model="dinov2_vitb14", **kwargs)

def mamba_pixelformer_dinov2_vitl14(**kwargs):
    return MambaPixelFormer(dinov2_model="dinov2_vitl14", **kwargs)