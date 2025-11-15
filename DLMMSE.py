import torch
import torch.nn.functional as F
import os
import os.path

import numpy as np

import torch

from typing import Optional, Tuple


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

# --------------------------
#  主函数：DLMMSE
# --------------------------

@torch.no_grad()
def DLMMSE(bayer: torch.Tensor) -> torch.Tensor:
    """
    DLMMSE demosaic for RGGB Bayer.
    Args:
        bayer: [B,1,H,W] in [0,1], RGGB (R at (0,0))
    Returns:
        rgb:   [B,3,H,W] in [0,1] (linear)
    """
    assert bayer.dim() == 4 and bayer.size(1) == 1, "Expect [B,1,H,W]"
    S = bayer  # 单通道 Bayer 观测
    B, _, H, W = S.shape
    eps = 1e-10

    # RGGB masks
    mask_R  = torch.zeros_like(S); mask_R[..., 0::2, 0::2] = 1.0
    mask_G1 = torch.zeros_like(S); mask_G1[..., 0::2, 1::2] = 1.0  # (0,1)
    mask_G2 = torch.zeros_like(S); mask_G2[..., 1::2, 0::2] = 1.0  # (1,0)
    mask_G  = mask_G1 + mask_G2
    mask_B  = torch.zeros_like(S); mask_B[..., 1::2, 1::2] = 1.0

    # 拆 R/G/B（观测处）
    R  = S * mask_R
    G0 = S * mask_G
    Bc = S * mask_B

    # ============================
    # Step 1: Interpolate G
    # ============================
    # 1.1 简单方向插值 + 构造 delta_H / delta_V
    Hf = _conv_h(S, [-1, 2, 2, 2, -1]) / 4.0
    Vf = _conv_v(S, [-1, 2, 2, 2, -1]) / 4.0

    delta_H = Hf - S
    delta_V = Vf - S
    # 在 G 位置翻转符号（与原 numpy 索引一致）
    delta_H = torch.where(mask_G.bool(), -delta_H, delta_H)
    delta_V = torch.where(mask_G.bool(), -delta_V, delta_V)

    # 1.2 低通滤波（高斯 9-tap）
    gaussian = [4, 9, 15, 23, 26, 23, 15, 9, 4]
    gsum = float(sum(gaussian))
    gaussian_H = _conv_h(delta_H, gaussian) / gsum
    gaussian_V = _conv_v(delta_V, gaussian) / gsum

    # 1.3 均值/方差（窗口为 9，沿 H/V 方向分别做）
    mean_f = [1]*9
    msum = float(sum(mean_f))

    mean_H = _conv_h(gaussian_H, mean_f) / msum
    mean_V = _conv_v(gaussian_V, mean_f) / msum

    var_value_H = _conv_h((gaussian_H - mean_H).pow(2), mean_f) / msum + eps
    var_value_V = _conv_v((gaussian_V - mean_V).pow(2), mean_f) / msum + eps

    var_noise_H = _conv_h((gaussian_H - delta_H).pow(2), mean_f) / msum + eps
    var_noise_V = _conv_v((gaussian_V - delta_V).pow(2), mean_f) / msum + eps

    # 1.4 LMMSE 细化
    new_H = mean_H + var_value_H / (var_noise_H + var_value_H) * (delta_H - mean_H)
    new_V = mean_V + var_value_V / (var_noise_V + var_value_V) * (delta_V - mean_V)

    # 1.5 两方向融合的权重
    var_x_H = (var_value_H - var_value_H / (var_value_H + var_noise_H)).abs() + eps
    var_x_V = (var_value_V - var_value_V / (var_value_V + var_noise_V)).abs() + eps
    w_H = var_x_V / (var_x_H + var_x_V)
    w_V = var_x_H / (var_x_H + var_x_V)

    final_delta = w_H * new_H + w_V * new_V

    # 1.6 在 R/B 处恢复 G
    G = G0.clone()
    G[..., 0::2, 0::2] = (R  + final_delta)[..., 0::2, 0::2]
    G[..., 1::2, 1::2] = (Bc + final_delta)[..., 1::2, 1::2]

    # ============================
    # Step 2: Interpolate R & B
    # ============================
    # 2.1 在 B 处恢复 R、在 R 处恢复 B（对角核）
    diag4 = [[1,0,1],
             [0,0,0],
             [1,0,1]]
    diag4 = torch.tensor(diag4, dtype=S.dtype, device=S.device)
    k_diag4 = diag4.view(1,1,3,3)

    delta_GR = F.conv2d(G - R,  k_diag4, padding=1) / 4.0
    delta_GB = F.conv2d(G - Bc, k_diag4, padding=1) / 4.0

    R1  = R.clone()
    B1  = Bc.clone()
    R1[..., 1::2, 1::2] = (G - delta_GR)[..., 1::2, 1::2]  # R at B sites
    B1[..., 0::2, 0::2] = (G - delta_GB)[..., 0::2, 0::2]  # B at R sites

    # 2.2 在 G 处恢复 R/B（十字核）
    cross4 = [[0,1,0],
              [1,0,1],
              [0,1,0]]
    cross4 = torch.tensor(cross4, dtype=S.dtype, device=S.device)
    k_cross4 = cross4.view(1,1,3,3)

    delta_GR_g = F.conv2d(G - R1, k_cross4, padding=1) / 4.0
    delta_GB_g = F.conv2d(G - B1, k_cross4, padding=1) / 4.0

    R2 = R1.clone()
    B2 = B1.clone()
    # R at G sites
    R2[..., 0::2, 1::2] = (G - delta_GR_g)[..., 0::2, 1::2]
    R2[..., 1::2, 0::2] = (G - delta_GR_g)[..., 1::2, 0::2]
    # B at G sites
    B2[..., 0::2, 1::2] = (G - delta_GB_g)[..., 0::2, 1::2]
    B2[..., 1::2, 0::2] = (G - delta_GB_g)[..., 1::2, 0::2]

    # 组合输出
    rgb = torch.cat([R2, G, B2], dim=1).clamp_(0.0, 1.0)
    return rgb
