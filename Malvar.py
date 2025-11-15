import torch
import torch.nn.functional as F
import os
import os.path

import numpy as np

import torch

from typing import Optional, Tuple

# ------------------------------------
# Malvar
# ------------------------------------

@torch.no_grad()
def Malvar(bayer_btchw: torch.Tensor) -> torch.Tensor:
    """
    HQL demosaic (RGGB). 
    Args:
        bayer_btchw: torch.Tensor, [B,1,H,W], in [0,1], RGGB Bayer
    Returns:
        rgb: torch.Tensor, [B,3,H,W], in [0,1] (linear)
    """
    assert bayer_btchw.dim() == 4 and bayer_btchw.size(1) == 1, "Expect [B,1,H,W]"
    B, _, H, W = bayer_btchw.shape
    device = bayer_btchw.device
    dtype  = bayer_btchw.dtype

    S = bayer_btchw  # 与原实现中 S = R+G+B 等价（Bayer 上每点只有一个通道为真）

    # ---------- 实用：2D 卷积（5x5，零填充，保持大小） ----------
    def conv5x5(x, kernel_5x5, div):
        k = torch.tensor(kernel_5x5, dtype=dtype, device=device).view(1,1,5,5)
        y = F.conv2d(x, k, padding=2)
        if div != 1:
            y = y / div
        return y

    # ---------- 位置掩码（RGGB） ----------
    mask_R  = torch.zeros_like(S); mask_R[..., 0::2, 0::2] = 1.0
    mask_G1 = torch.zeros_like(S); mask_G1[..., 0::2, 1::2] = 1.0  # G at (0,1)
    mask_G2 = torch.zeros_like(S); mask_G2[..., 1::2, 0::2] = 1.0  # G at (1,0)
    mask_G  = mask_G1 + mask_G2
    mask_B  = torch.zeros_like(S); mask_B[..., 1::2, 1::2] = 1.0

    # =============== 1) 预测 G =================
    # I_G:
    I_G = [
        [0, 0, -1, 0, 0],
        [0, 0,  2, 0, 0],
        [-1,2,  4, 2,-1],
        [0, 0,  2, 0, 0],
        [0, 0, -1, 0, 0],
    ]
    G_pred = conv5x5(S, I_G, div=8.0)
    # 已知 G 处用观测值替换
    G = torch.where(mask_G.bool(), S, G_pred)

    # =============== 2) 预测 R =================
    R = torch.zeros_like(S)
    # 已知 R 位置 (0,0)
    R = R + S * mask_R

    # R at (0,1)
    I_R_01 = [
        [0, 0, 1, 0, 0],
        [0,-2, 0,-2, 0],
        [-2,8,10, 8,-2],
        [0,-2, 0,-2, 0],
        [0, 0, 1, 0, 0],
    ]
    R_01 = conv5x5(S, I_R_01, div=16.0)
    R = R + R_01 * mask_G1  # (0,1)

    # R at (1,0)
    I_R_10 = [
        [0, 0,-2, 0, 0],
        [0,-2, 8,-2, 0],
        [1, 0,10, 0, 1],
        [0,-2, 8,-2, 0],
        [0, 0,-2, 0, 0],
    ]
    R_10 = conv5x5(S, I_R_10, div=16.0)
    R = R + R_10 * mask_G2  # (1,0)

    # R at (1,1)
    I_R_11 = [
        [0, 0,-3, 0, 0],
        [0, 4, 0, 4, 0],
        [-3,0,12, 0,-3],
        [0, 4, 0, 4, 0],
        [0, 0,-3, 0, 0],
    ]
    R_11 = conv5x5(S, I_R_11, div=16.0)
    R = R + R_11 * mask_B  # (1,1)

    # =============== 3) 预测 B =================
    Bc = torch.zeros_like(S)
    # 已知 B 位置 (1,1)
    Bc = Bc + S * mask_B

    # B at (1,0)  —— 与 R 的 (0,1) 同核
    B_10 = conv5x5(S, I_R_01, div=16.0)
    Bc = Bc + B_10 * mask_G2  # (1,0)

    # B at (0,1)  —— 与 R 的 (1,0) 同核
    B_01 = conv5x5(S, I_R_10, div=16.0)
    Bc = Bc + B_01 * mask_G1  # (0,1)

    # B at (0,0)  —— 与 R 的 (1,1) 同核
    B_00 = conv5x5(S, I_R_11, div=16.0)
    Bc = Bc + B_00 * mask_R   # (0,0)

    # 组装输出 [B,3,H,W]
    rgb = torch.cat([R, G, Bc], dim=1).clamp_(0.0, 1.0)
    return rgb

