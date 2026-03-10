import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window-based multi-head cross-attention with configurable output dimension."""
    
    def __init__(self, dim, window_size, num_heads, v_dim,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.v_dim = v_dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # 確保 v_dim 可以被 num_heads 整除
        assert v_dim % num_heads == 0, f"v_dim ({v_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim_v = v_dim // num_heads

        # Q from query, K/V from encoder
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, v_dim, bias=qkv_bias)  # V 投影到 v_dim

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(v_dim, v_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self._fa2_ready = False
        try:
            from flash_attn import flash_attn_func
            self._fa2_ready = True
        except:
            pass

        self.register_buffer("_logged_once", torch.zeros((), dtype=torch.bool), persistent=False)

    def forward(self, x, v_in, mask=None):
        """
        x: (B_, N, dim) - query input
        v_in: (B_, N, dim) - key/value input (encoder feature)
        """
        B_, N, C = x.shape
        h = self.num_heads
        d = C // h
        d_v = self.head_dim_v

        q = self.q(x).view(B_, N, h, d)
        k = self.k(v_in).view(B_, N, h, d)
        v = self.v(v_in).view(B_, N, h, d_v)

        q = q * self.scale

        if (not bool(self._logged_once)) and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            which = "FA2" if (self._fa2_ready and q.is_cuda) else "SDPA"
            print(f"[SAM/WindowAttention] {which} | dim={C}, v_dim={self.v_dim} | heads={h}")
            self._logged_once.fill_(True)

        drop_p = self.attn_drop.p if self.training else 0.0

        if self._fa2_ready and q.is_cuda:
            from flash_attn import flash_attn_func
            dtype = torch.bfloat16 if q.dtype == torch.bfloat16 else torch.float16
            q_f = q.to(dtype) if q.dtype not in (torch.float16, torch.bfloat16) else q
            k_f = k.to(q_f.dtype)
            v_f = v.to(q_f.dtype)
            out = flash_attn_func(q_f, k_f, v_f, dropout_p=drop_p, softmax_scale=1.0, causal=False)
            out = out.to(x.dtype)
        else:
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                attn_mask=None, dropout_p=drop_p, is_causal=False
            ).transpose(1, 2)

        x = out.reshape(B_, N, self.v_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SAMBLOCK(nn.Module):
    def __init__(self, dim, num_heads, v_dim, window_size=7, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., 
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.window_size = window_size
        self.dim = dim
        self.v_dim = v_dim
        self.num_heads = num_heads

        self.norm1 = norm_layer(dim)
        self.normv = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(window_size), num_heads=num_heads, v_dim=v_dim,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(v_dim)
        self.mlp = Mlp(in_features=v_dim, hidden_features=int(v_dim * mlp_ratio), 
                       out_features=v_dim, act_layer=nn.GELU, drop=drop)
        
        # Shortcut projection if dimensions differ
        self.shortcut_proj = nn.Linear(dim, v_dim) if dim != v_dim else nn.Identity()

    def forward(self, x, v, H, W):
        B, L, C = x.shape
        assert L == H * W

        shortcut = self.shortcut_proj(x)  # (B, L, v_dim)

        x = self.norm1(x).view(B, H, W, C)
        v = self.normv(v).view(B, H, W, C)

        # Padding
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        v = F.pad(v, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        # Window partition
        x_win = window_partition(x, self.window_size).view(-1, self.window_size**2, C)
        v_win = window_partition(v, self.window_size).view(-1, self.window_size**2, C)

        # Attention
        attn_win = self.attn(x_win, v_win)  # (nW*B, ws*ws, v_dim)

        # Window reverse
        attn_win = attn_win.view(-1, self.window_size, self.window_size, self.v_dim)
        x = window_reverse(attn_win, self.window_size, Hp, Wp)
        
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        
        x = x.view(B, L, self.v_dim)

        # Residual + FFN
        x = self.drop_path(x) + shortcut
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, H, W


class SAM(nn.Module):
    """
    Skip Attention Module (Fixed version)
    
    真正輸出 v_dim 維度，而不是 embed_dim。
    
    Args:
        input_dim: Encoder 特徵維度 (in_channels[i])
        embed_dim: 內部處理維度，也是 Query 輸入的期望維度
        v_dim: 輸出維度，用於 PixelShuffle
        window_size: Attention window size
        num_heads: Number of attention heads
    
    資料流:
        e: (B, input_dim, H, W) - Encoder 特徵
        q: (B, embed_dim, H, W) - Query (來自上一層 shuffle 後的輸出)
        output: (B, v_dim, H, W) - 給下一層 PixelShuffle
    """
    def __init__(self,
                 input_dim=96,
                 embed_dim=96,
                 v_dim=256,
                 window_size=7,
                 num_heads=4,
                 norm_layer=nn.LayerNorm):
        super().__init__()

        self.embed_dim = embed_dim
        self.v_dim = v_dim
        
        # 投影到統一的 embed_dim
        self.proj_e = nn.Conv2d(input_dim, embed_dim, 3, padding=1) if input_dim != embed_dim else nn.Identity()
        self.proj_q = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)  # 保持維度，但做特徵變換
        
        # SAM Block: 輸入 embed_dim，輸出 v_dim
        self.sam_block = SAMBLOCK(
            dim=embed_dim,
            num_heads=num_heads,
            v_dim=v_dim,
            window_size=window_size,
            mlp_ratio=4.,
            qkv_bias=True,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=norm_layer
        )

        self.norm_sam = norm_layer(v_dim)
        
        # 殘差投影到 v_dim
        self.res_e = nn.Conv2d(input_dim, v_dim, 1)
        self.res_q = nn.Conv2d(embed_dim, v_dim, 1) if embed_dim != v_dim else nn.Identity()

    def forward(self, e, q):
        """
        e: (B, input_dim, H, W) - Encoder feature
        q: (B, embed_dim, H, W) - Query from previous stage
        
        Returns: (B, v_dim, H, W)
        """
        B, _, H, W = e.shape
        
        # 殘差（投影到 v_dim）
        e_res = self.res_e(e)
        q_res = self.res_q(q)
        
        # 投影到 embed_dim
        e_proj = self.proj_e(e) if isinstance(self.proj_e, nn.Conv2d) else self.proj_e(e)
        q_proj = self.proj_q(q)

        # Flatten
        q_flat = q_proj.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)
        e_flat = e_proj.flatten(2).transpose(1, 2)  # (B, H*W, embed_dim)

        # SAM Block
        out, _, _ = self.sam_block(q_flat, e_flat, H, W)  # (B, H*W, v_dim)
        
        # Reshape
        out = self.norm_sam(out)
        out = out.view(B, H, W, self.v_dim).permute(0, 3, 1, 2).contiguous()

        # 殘差連接
        return out + e_res + q_res