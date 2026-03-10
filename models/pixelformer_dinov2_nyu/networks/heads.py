import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticAwareBinHead(nn.Module):
    """
    亮點：語義感知動態分箱頭 (SADB)
    透過輕量化注意力機制，讓模型聚焦於關鍵幾何區域來優化深度區間分配。
    """
    def __init__(self, in_dim, n_bins, min_depth, max_depth, dropout=0.1):
        super().__init__()
        # 1. 空間權重分支 (Spatial Attention Branch)
        # 用於識別影像中對深度分佈影響較大的區域
        self.attention_map = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_dim // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 2. 特徵轉換 MLP (配合 ViT 使用 GELU 激活函數)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_bins)
        )
        
        self.register_buffer("min_depth", torch.tensor(min_depth))
        self.register_buffer("max_depth", torch.tensor(max_depth))

    def forward(self, x):
        # x: [B, C, H, W]
        
        # [亮點實作] 空間加權池化：不再只是平均，而是獲取具備語義權重的全域特徵
        weights = self.attention_map(x) # [B, 1, H, W]
        weighted_feat = torch.sum(x * weights, dim=(2, 3)) / (torch.sum(weights, dim=(2, 3)) + 1e-7)
        
        # 預測 Bin 比例
        out = self.mlp(weighted_feat)
        out = torch.softmax(out, dim=1)
        
        # 轉換為實際深度中心點 (AdaBins 邏輯優化)
        bin_widths = (self.max_depth - self.min_depth) * out
        bin_widths = F.pad(bin_widths, (1, 0), value=self.min_depth.item())
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        
        # 直接升維，方便 PixelWiseHead 做廣播運算 (Inference Friendly)
        return centers.unsqueeze(-1).unsqueeze(-1)

class LightweightPixelHead(nn.Module):
    """
    亮點：輕量化像素映射頭
    專為邊緣設備 (Jetson Orin) 優化，將運算負擔集中在 1x1 卷積。
    """
    def __init__(self, in_dim, n_bins):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(), # 維持與 Backbone 一致的激活函數
            nn.Conv2d(256, n_bins, kernel_size=1)
        )

    def forward(self, x, centers, scale=4):
        # x: [B, C, H, W], centers: [B, N_Bins, 1, 1]
        logits = self.decoder(x)
        probs = torch.softmax(logits, dim=1)
        
        # 混合迴歸計算：Depth = Sum(Prob * Center)
        depth = torch.sum(probs * centers, dim=1, keepdim=True)
        
        if scale > 1:
            depth = F.interpolate(depth, scale_factor=scale, mode='bilinear', align_corners=False)
            
        return depth