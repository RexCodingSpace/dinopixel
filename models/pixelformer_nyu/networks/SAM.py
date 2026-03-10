import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
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
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# --- FlashAttention v2-powered WindowAttention (NO attn_bias) ---
class WindowAttention(nn.Module):
    """ Window-based multi-head attention (no bias/mask).
        Primary: FlashAttention v2
        Fallback: PyTorch 2.x SDPA (memory-efficient)
    """
    def __init__(self, dim, window_size, num_heads, v_dim,
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # q / kv projections
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q  = nn.Linear(dim, dim,     bias=qkv_bias)

        # drops & proj
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(v_dim, v_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # SDPA kernel preference (fallback only)
        try:
            torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
        except Exception:
            pass

        # lazy check for FA2 (no bias needed)
        self._fa2_ready = False
        try:
            from flash_attn import flash_attn_func  # noqa: F401
            self._fa2_ready = True
        except Exception:
            self._fa2_ready = False

        # log once
        self.register_buffer("_logged_once", torch.zeros((), dtype=torch.bool), persistent=False)

    def forward(self, x, v, mask=None):
        """
        x: (B_, N, C), v: (B_, N, C)
        mask: ignored (no bias/mask path)
        """
        B_, N, C = x.shape
        h = self.num_heads
        d = C // h

        # q, k, v -> (B_, N, h, d)
        q = self.q(x).view(B_, N, h, d)
        kv = self.kv(v).view(B_, N, 2, h, d).permute(2, 0, 1, 3, 4)
        k, v = kv[0], kv[1]

        # scale
        q = q * self.scale

        # one-time log
        if (not bool(self._logged_once)) and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            which = "FA2(no bias)" if (self._fa2_ready and q.is_cuda) else "SDPA(no bias)"
            print(f"[SAM/WindowAttention] using {which} | dtype={q.dtype} | head_dim={d} | B_={B_}, N={N}")
            self._logged_once.fill_(True)

        drop_p = self.attn_drop.p if self.training and self.attn_drop.p > 0 else 0.0

        if self._fa2_ready and q.is_cuda:
            # FlashAttention v2 (no attn_bias)
            from flash_attn import flash_attn_func
            tgt_dtype = torch.bfloat16 if q.dtype == torch.bfloat16 else torch.float16
            if q.dtype not in (torch.float16, torch.bfloat16):
                q_f = q.to(tgt_dtype); k_f = k.to(tgt_dtype); v_f = v.to(tgt_dtype)
            else:
                q_f, k_f, v_f = q, k.to(q.dtype), v.to(q.dtype)

            out = flash_attn_func(
                q_f, k_f, v_f,
                dropout_p=drop_p,
                softmax_scale=1.0,
                causal=False
            )  # (B_, N, h, d)
            out = out.to(dtype=x.dtype)
        else:
            # PyTorch SDPA (no attn_mask)
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),  # (B_, h, N, d)
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=None,
                dropout_p=drop_p,
                is_causal=False
            ).transpose(1, 2)      # -> (B_, N, h, d)

        # merge heads -> proj
        x = out.reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SAMBLOCK(nn.Module):
    """ 
    Args:
        dim (int): Number of feature channels
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 num_heads,
                 v_dim,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 ):
        super().__init__()
        self.window_size = window_size
        self.dim = dim
        self.num_heads = num_heads
        self.v_dim = v_dim
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        act_layer=nn.GELU
        norm_layer=nn.LayerNorm

        self.norm1 = norm_layer(dim)
        self.normv = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, v_dim=v_dim,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(v_dim)
        mlp_hidden_dim = int(v_dim * mlp_ratio)
        self.mlp = Mlp(in_features=v_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, v, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        shortcut_v = v
        v = self.normv(v)
        v = v.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        v = F.pad(v, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # partition windows
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        v_windows = window_partition(v, self.window_size)  # nW*B, window_size, window_size, C
        v_windows = v_windows.view(-1, self.window_size * self.window_size, v_windows.shape[-1])  # nW*B, window_size*window_size, C
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, v_windows, mask=None)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.v_dim)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, self.v_dim)

        # FFN
        x = self.drop_path(x) + shortcut
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, H, W


class SAM(nn.Module):
    def __init__(self,
                 input_dim=96,
                 embed_dim=96,
                 v_dim=64,
                 window_size=7,
                 num_heads=4,
                 patch_size=4,
                 in_chans=3,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True):
        super().__init__()

        self.embed_dim = embed_dim
        
        if input_dim != embed_dim:
            self.proj_e = nn.Conv2d(input_dim, embed_dim, 3, padding=1)
        else:
            self.proj_e = None

        if v_dim != embed_dim:
            self.proj_q = nn.Conv2d(v_dim, embed_dim, 3, padding=1)
        elif embed_dim % v_dim == 0:
            self.proj_q = None
        self.proj = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)

        v_dim = embed_dim
        self.sam_block = SAMBLOCK(
                dim=embed_dim,
                num_heads=num_heads,
                v_dim=v_dim,
                window_size=window_size,
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop=0.,
                attn_drop=0.,
                drop_path=0.,
                norm_layer=norm_layer)

        layer = norm_layer(embed_dim)
        layer_name = 'norm_sam'
        self.add_module(layer_name, layer)


    def forward(self, e, q):
        if self.proj_q is not None:
            q = self.proj_q(q)
        if self.proj_e is not None:
            e = self.proj_e(e)
        e_proj = e
        q_proj = q

        Wh, Ww = q.size(2), q.size(3)
        q = q.flatten(2).transpose(1, 2)
        e = e.flatten(2).transpose(1, 2)

        q_out, H, W = self.sam_block(q, e, Wh, Ww)
        norm_layer = getattr(self, f'norm_sam')
        q_out = norm_layer(q_out)
        q_out = q_out.view(-1, H, W, self.embed_dim).permute(0, 3, 1, 2).contiguous()

        return q_out+e_proj+q_proj