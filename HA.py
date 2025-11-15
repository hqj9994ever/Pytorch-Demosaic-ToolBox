import torch
import torch.nn.functional as F
import os
import os.path

import numpy as np

import torch

from typing import Optional, Tuple


# ------------------------------------
# HA
# ------------------------------------


def _conv_h(x, k):
    """水平 1D 卷积（same padding）"""
    device, dtype = x.device, x.dtype
    k = torch.tensor(k, dtype=dtype, device=device).view(1,1,1,-1)
    pad = (k.size(-1)//2, 0)  # (left/right, top/bottom) but torch uses (W,H) in padding arg ordering as (w_left,w_right,h_top,h_bottom)? No: F.pad uses (W_left,W_right,H_top,H_bottom).
    # 使用 F.conv2d 的padding更安全：
    return F.conv2d(x, k, padding=(0, k.size(-1)//2))

def _conv_v(x, k):
    """垂直 1D 卷积（same padding）"""
    device, dtype = x.device, x.dtype
    k = torch.tensor(k, dtype=dtype, device=device).view(1,1,-1,1)
    return F.conv2d(x, k, padding=(k.size(-2)//2, 0))

def _conv2d(x, k2d):
    """2D 卷积（same padding）"""
    device, dtype = x.device, x.dtype
    k = torch.tensor(k2d, dtype=dtype, device=device).view(1,1,*torch.tensor(k2d).shape)
    kh, kw = k.size(-2), k.size(-1)
    return F.conv2d(x, k, padding=(kh//2, kw//2))


def HA(bayer: torch.Tensor) -> torch.Tensor:
    """
    HA demosaic (RGGB):
      Input:  bayer [B,1,H,W] in [0,1], RGGB
      Output: rgb   [B,3,H,W] in [0,1] (linear)
    """
    assert bayer.dim() == 4 and bayer.size(1) == 1, "Expect [B,1,H,W]"
    B, _, H, W = bayer.shape
    S = bayer  # 单通道 Bayer 观测

    # --- RGGB masks ---
    mask_R  = torch.zeros_like(S); mask_R[..., 0::2, 0::2] = 1.0
    mask_G1 = torch.zeros_like(S); mask_G1[..., 0::2, 1::2] = 1.0  # (0,1)
    mask_G2 = torch.zeros_like(S); mask_G2[..., 1::2, 0::2] = 1.0  # (1,0)
    mask_G  = mask_G1 + mask_G2
    mask_B  = torch.zeros_like(S); mask_B[..., 1::2, 1::2] = 1.0

    # -------------------------
    # Step 1: Interpolate G in R/B (方向加权)
    # -------------------------
    S0  = _conv_h(S, [-1, 2, 2, 2, -1]) / 4.0
    S90 = _conv_v(S, [-1, 2, 2, 2, -1]) / 4.0

    g0  = torch.abs(_conv_h(S, [-1, 0, 1])) + torch.abs(_conv_h(S, [-1, 0, 2, 0, -1]))
    g90 = torch.abs(_conv_v(S, [-1, 0, 1])) + torch.abs(_conv_v(S, [-1, 0, 2, 0, -1]))

    G_pred = (S0 + S90) * 0.5
    G_pred = torch.where((g0 < g90), S0, G_pred)
    G_pred = torch.where((g0 > g90), S90, G_pred)
    # 已知 G 位置用观测值覆盖
    G = torch.where(mask_G.bool(), S, G_pred)

    # -------------------------
    # Step 2: Interpolate R in B & B in R（对角方向）
    # S45/S135 + 梯度选择
    # -------------------------
    # I1/I2（45°）
    I1_45 = [[0,0,1],
             [0,0,0],
             [1,0,0]]
    I2_45 = [[ 0, 0,-1],
             [ 0, 2, 0],
             [-1, 0, 0]]
    S45  = _conv2d(S, I1_45) / 2.0 + _conv2d(G, I2_45) / 4.0

    # I1/I2（135°）
    I1_135 = [[1,0,0],
              [0,0,0],
              [0,0,1]]
    I2_135 = [[-1, 0, 0],
              [ 0, 2, 0],
              [ 0, 0,-1]]
    S135 = _conv2d(S, I1_135) / 2.0 + _conv2d(G, I2_135) / 4.0

    g45  = torch.abs(_conv2d(S, I1_45))  + torch.abs(_conv2d(G, I2_45))
    g135 = torch.abs(_conv2d(S, I1_135)) + torch.abs(_conv2d(G, I2_135))

    # 先用方向选择得到 R、B 的初值
    RB_blend = (S45 + S135) * 0.5
    R = torch.where((g45 < g135), S45, RB_blend)
    R = torch.where((g45 > g135), S135, R)
    Bc = R.clone()  # B 初值与 R 相同的规则

    # 已知处覆盖
    R = torch.where(mask_R.bool(), S, R)   # R at (0,0)
    Bc = torch.where(mask_B.bool(), S, Bc) # B at (1,1)

    # -------------------------
    # Step 3: Interpolate R/B at G 位置（用水平/垂直 1D 插值）
    # -------------------------
    # R 在 G 位置：
    #   (0,1) 用 S0；(1,0) 用 S90
    R = torch.where(mask_G1.bool(), S0, R)
    R = torch.where(mask_G2.bool(), S90, R)

    # B 在 G 位置：
    #   (1,0) 用 S0；(0,1) 用 S90
    Bc = torch.where(mask_G2.bool(), S0, Bc)
    Bc = torch.where(mask_G1.bool(), S90, Bc)

    # 拼接输出
    rgb = torch.cat([R, G, Bc], dim=1).clamp_(0.0, 1.0)
    return rgb
