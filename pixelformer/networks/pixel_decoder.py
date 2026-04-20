import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossSSMBlock_V2(nn.Module):
    """
    Mock Mamba V2: High Accuracy Version
    ----------------------------------------------------------
    核心升級：
    1. Large Kernel (7x7): 擴大感受野，模擬 Attention Window。
    2. FFN (MLP): 增加特徵變換能力，彌補 V1 的不足。
    3. Cross Gating: 用 Decoder (Query) 控制 Encoder (Value)。
    ----------------------------------------------------------
    """
    def __init__(self, dim, expansion_ratio=4):
        super().__init__()
        self.norm_skip = nn.LayerNorm(dim) # 處理 Encoder 特徵
        self.norm_x = nn.LayerNorm(dim)    # 處理 Decoder 特徵
        
        # --- 1. Content Branch (Encoder / Value) ---
        # 關鍵升級：改用 7x7 Depthwise Conv，padding=3
        # 這能捕捉更長距離的空間關係 (例如牆壁的延伸)
        self.proj_content = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        
        # --- 2. Gate Branch (Decoder / Query) ---
        # 用 Decoder 的資訊來產生 "閘門"
        self.proj_gate = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        
        # --- 3. Global Context (Channel Attention) ---
        # 類似 SE-Block，讓模型知道哪些 Channel 重要
        self.global_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )
        
        # --- 4. FFN (Feed-Forward Network) ---
        # 增加模型複雜度，模擬 Transformer 的 MLP 層
        hidden_features = int(dim * expansion_ratio)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Dropout(0.0), # 如果資料量少，可設 0.1
            nn.Linear(hidden_features, dim)
        )
        
        # 最終投影
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, skip):
        # x: Decoder 特徵 (Query / Gate) -> 用來決定要看哪裡
        # skip: Encoder 特徵 (Value / Content) -> 實際的影像細節
        
        B, C, H, W = skip.shape
        
        # --- Part A: Cross Gating (模擬 Attention) ---
        
        # 1. 處理 Encoder (Content)
        # HWC -> LayerNorm -> CHW -> 7x7 Conv
        s = skip.permute(0, 2, 3, 1)
        s = self.norm_skip(s).permute(0, 3, 1, 2)
        content = self.proj_content(s) 
        
        # 2. 處理 Decoder (Gate)
        # HWC -> LayerNorm -> Linear -> CHW
        q = x.permute(0, 2, 3, 1)
        q = self.norm_x(q)
        gate = self.proj_gate(q).permute(0, 3, 1, 2)
        
        # 3. 交互作用 (Gating) 
        # Encoder 的細節 * Decoder 的注意力
        out = content * self.act(gate)
        
        # 4. Global Refinement (加強全域觀念)
        out = out * self.global_fc(out)
        
        # 5. Output Projection
        out = out.permute(0, 2, 3, 1)
        out = self.out_proj(out)
        
        # --- Part B: FFN Mixing (增加腦容量) ---
        # Residual Connection 2: FFN
        out = out + self.ffn(out)
        
        out = out.permute(0, 3, 1, 2)
        
        # Residual Connection 1: 接回 Skip (ResNet 風格)
        return skip + out


class PixelDecoderLayer(nn.Module):
    """
    負責融合 Skip Connection 並用 CrossSSM 處理特徵
    """
    def __init__(self, dim, skip_dim, dropout=0.2):
        super().__init__()
        
        # 降維層：先把 Encoder 的 Skip Connection 降到跟 Decoder 一樣的維度
        # 例如: 384 -> 256
        self.proj_skip = nn.Conv2d(skip_dim, dim, 1, bias=False)
        
        # 使用升級版的 Cross-Gating Block
        self.cross_ssm = CrossSSMBlock_V2(dim)

        #self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, x, skip):
        # x: 解碼器特徵 (低解析度)
        # skip: 編碼器特徵 (高解析度)
        
        # 1. Upsample x 到 skip 的大小 (對齊解析度)
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        
        # 2. Project skip (對齊 Channel)
        skip = self.proj_skip(skip)
        
        # 3. Cross Interaction (不再是簡單的 Concat)
        # 讓 x 去指導 skip
        x = self.cross_ssm(x, skip)
        #x = self.dropout(x)
        return x