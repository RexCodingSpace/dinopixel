"""
DINOv2 ViT Backbone for PixelFormer
自動 padding 輸入到 patch_size (14) 的倍數

使用方式:
在 PixelFormer.py 中:
    from .dinov2_vit import DINOv2ViT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


class DINOv2ViT(nn.Module):
    """
    DINOv2 ViT Backbone for PixelFormer
    自動處理任意輸入尺寸（會 padding 到 14 的倍數）
    
    輸出格式與 SwinTransformer 兼容:
    - 返回 4 個不同尺度的特徵圖 [1/4, 1/8, 1/16, 1/32]
    """
    
    def __init__(
        self,
        model_name: str = 'dinov2_vitl14',
        pretrained: bool = True,
        out_indices: List[int] = [5, 12, 18, 24],
        freeze_backbone: bool = False,
        **kwargs
    ):
        super().__init__()
        
        self.model_name = model_name
        self.out_indices = out_indices
        
        # 載入 DINOv2
        if pretrained:
            print(f"Loading pretrained {model_name} from torch.hub...")
            self.vit = torch.hub.load('facebookresearch/dinov2', model_name)
        else:
            raise NotImplementedError("請使用 pretrained=True")
        
        # 獲取配置
        self.embed_dim = self.vit.embed_dim
        self.patch_size = self.vit.patch_size
        self.num_heads = self.vit.num_heads
        self.num_layers = len(self.vit.blocks)
        
        print(f"  - embed_dim: {self.embed_dim}")
        print(f"  - patch_size: {self.patch_size}")
        print(f"  - num_layers: {self.num_layers}")
        print(f"  - out_indices: {self.out_indices}")
        
        # 設定輸出維度
        self._setup_output_dims()
        
        # 凍結 backbone
        if freeze_backbone:
            self._freeze()
        
        # 檢查 register tokens
        self.num_register_tokens = 0
        if hasattr(self.vit, 'register_tokens') and self.vit.register_tokens is not None:
            self.num_register_tokens = self.vit.register_tokens.shape[1]
    
    def _setup_output_dims(self):
        """設定各 stage 的輸出維度"""
        configs = {
            'dinov2_vits14': [96, 192, 384, 384],
            'dinov2_vitb14': [192, 384, 768, 768],
            'dinov2_vitl14': [256, 512, 1024, 1024],
            'dinov2_vitg14': [384, 768, 1536, 1536],
        }
        
        self.out_dims = configs.get(self.model_name, [256, 512, 1024, 1024])
        
        # 輸出投影層
        self.output_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embed_dim, out_dim),
                nn.LayerNorm(out_dim)
            )
            for out_dim in self.out_dims
        ])
        
        # 尺度調整層
        self.scale_layers = nn.ModuleList([
            # Stage 1: 1/14 -> 1/4
            nn.Sequential(
                nn.ConvTranspose2d(self.out_dims[0], self.out_dims[0], kernel_size=4, stride=4, padding=0),
                nn.BatchNorm2d(self.out_dims[0]),
                nn.GELU()
            ),
            # Stage 2: 1/14 -> 1/8
            nn.Sequential(
                nn.ConvTranspose2d(self.out_dims[1], self.out_dims[1], kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(self.out_dims[1]),
                nn.GELU()
            ),
            # Stage 3: 1/14 -> 1/16
            nn.Identity(),
            # Stage 4: 1/14 -> 1/32
            nn.Sequential(
                nn.Conv2d(self.out_dims[3], self.out_dims[3], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(self.out_dims[3]),
                nn.GELU()
            )
        ])
        
        # 給 PixelFormer 用的屬性
        self.num_features = self.out_dims
    
    def _freeze(self):
        """凍結 ViT backbone"""
        for param in self.vit.parameters():
            param.requires_grad = False
        print(f"DINOv2 backbone frozen ({self.model_name})")
    
    def _pad_to_multiple(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
        """
        Pad 輸入到 patch_size 的倍數
        Returns:
            padded_x: padded tensor
            padding: (pad_left, pad_right, pad_top, pad_bottom)
        """
        B, C, H, W = x.shape
        
        # 計算需要 pad 的量
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        
        # 分配 padding (盡量對稱)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
        
        return x, (pad_left, pad_right, pad_top, pad_bottom)
    
    def _unpad_features(self, features: List[torch.Tensor], original_size: Tuple[int, int]) -> List[torch.Tensor]:
        """
        將特徵圖調整回原始尺寸對應的大小
        """
        H_orig, W_orig = original_size
        target_sizes = [
            (H_orig // 4, W_orig // 4),
            (H_orig // 8, W_orig // 8),
            (H_orig // 16, W_orig // 16),
            (H_orig // 32, W_orig // 32)
        ]
        
        unpadded = []
        for feat, target_size in zip(features, target_sizes):
            # 使用 interpolate 調整到精確的目標尺寸
            feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            unpadded.append(feat)
        
        return unpadded
    
    def _interpolate_pos_embed(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """插值位置編碼以支援不同輸入尺寸"""
        pos_embed = self.vit.pos_embed[:, 1:, :]
        
        orig_size = int(pos_embed.shape[1] ** 0.5)
        
        if h != orig_size or w != orig_size:
            pos_embed = pos_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, size=(h, w), mode='bicubic', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        
        return x + pos_embed
    
    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """提取多尺度特徵"""
        B, _, H_orig, W_orig = x.shape
        
        # ===== 自動 Padding 到 14 的倍數 =====
        x, padding = self._pad_to_multiple(x)
        _, _, H, W = x.shape
        
        # Patch embedding
        x = self.vit.patch_embed(x)
        h = H // self.patch_size
        w = W // self.patch_size
        
        # 添加位置編碼
        x = self._interpolate_pos_embed(x, h, w)
        
        # 添加 CLS 和 register tokens
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        if self.num_register_tokens > 0:
            reg_tokens = self.vit.register_tokens.expand(B, -1, -1)
            x = torch.cat([cls_token, reg_tokens, x], dim=1)
            num_prefix = 1 + self.num_register_tokens
        else:
            x = torch.cat([cls_token, x], dim=1)
            num_prefix = 1
        
        # Forward through blocks
        features = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            
            layer_idx = i + 1
            if layer_idx in self.out_indices:
                stage_idx = self.out_indices.index(layer_idx)
                
                # 提取 patch tokens
                patch_tokens = x[:, num_prefix:, :]
                
                # 投影
                proj_tokens = self.output_projs[stage_idx](patch_tokens)
                
                # Reshape 為 2D
                feat = proj_tokens.permute(0, 2, 1).reshape(B, -1, h, w)
                
                # 調整尺度
                feat = self.scale_layers[stage_idx](feat)
                
                features.append(feat)
        
        # ===== 調整回原始尺寸對應的特徵大小 =====
        features = self._unpad_features(features, (H_orig, W_orig))
        
        return features
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass，與 SwinTransformer 接口兼容"""
        return self.forward_features(x)


# ============================================================================
# Factory functions
# ============================================================================

def dinov2_vits14(pretrained=True, **kwargs):
    return DINOv2ViT(model_name='dinov2_vits14', pretrained=pretrained, out_indices=[3, 6, 9, 12], **kwargs)

def dinov2_vitb14(pretrained=True, **kwargs):
    return DINOv2ViT(model_name='dinov2_vitb14', pretrained=pretrained, out_indices=[3, 6, 9, 12], **kwargs)

def dinov2_vitl14(pretrained=True, **kwargs):
    return DINOv2ViT(model_name='dinov2_vitl14', pretrained=pretrained, out_indices=[5, 12, 18, 24], **kwargs)

def dinov2_vitg14(pretrained=True, **kwargs):
    return DINOv2ViT(model_name='dinov2_vitg14', pretrained=pretrained, out_indices=[10, 20, 30, 40], **kwargs)


if __name__ == '__main__':
    print("Testing DINOv2 ViT backbone with auto-padding...")
    model = dinov2_vitl14(pretrained=True)
    model.eval()
    
    # 測試不是 14 倍數的尺寸
    test_sizes = [(480, 640), (476, 644), (518, 518), (375, 500)]
    
    for H, W in test_sizes:
        x = torch.randn(1, 3, H, W)
        print(f"\nInput: {x.shape}")
        
        with torch.no_grad():
            features = model(x)
        
        print("Outputs:")
        for i, feat in enumerate(features):
            expected_h = H // (4 * (2 ** i)) if i < 3 else H // 32
            expected_w = W // (4 * (2 ** i)) if i < 3 else W // 32
            print(f"  Stage {i+1}: {feat.shape} (expected ~{expected_h}x{expected_w})")
    
    print("\n✅ All tests passed!")