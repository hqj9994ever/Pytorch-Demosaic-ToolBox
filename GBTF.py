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
#  主函数：GBTF（RGGB）
# --------------------------
@torch.no_grad()
def GBTF(bayer: torch.Tensor) -> torch.Tensor:
    """
    Gradient-Based Threshold Free demosaic for RGGB Bayer.

    Args:
        bayer: [B,1,H,W], values in [0,1], RGGB (R at (0,0))
    Returns:
        rgb:   [B,3,H,W], values in [0,1] (linear)
    """
    assert bayer.dim() == 4 and bayer.size(1) == 1, "Expect input shape [B,1,H,W]"
    B, _, H, W = bayer.shape
    S = bayer  # 单通道 Bayer 观测（对应原代码里的 R/G/B 打包为 S=R+G+B）

    # RGGB 掩码
    mask_R  = torch.zeros_like(S); mask_R[..., 0::2, 0::2] = 1.0
    mask_G1 = torch.zeros_like(S); mask_G1[..., 0::2, 1::2] = 1.0
    mask_G2 = torch.zeros_like(S); mask_G2[..., 1::2, 0::2] = 1.0
    mask_G  = mask_G1 + mask_G2
    mask_B  = torch.zeros_like(S); mask_B[..., 1::2, 1::2] = 1.0

    # 用分路张量装 R/G/B（初始为观测，其他位置为 0）
    R = S * mask_R
    G = S * mask_G
    Bc = S * mask_B

    # -----------------------
    # Step 1: Interpolate H/V
    # -----------------------
    # H 水平滤波、V 垂直滤波（对应 [-1,2,2,2,-1]/4）
    Hf = _conv_h(S, [-1, 2, 2, 2, -1]) / 4.0
    Vf = _conv_v(S, [-1, 2, 2, 2, -1]) / 4.0

    # G_H/G_V、R_H/R_V、B_H/B_V（用 Hf/Vf 替换相应位置）
    G_H = G.clone(); R_H = R.clone(); B_H = Bc.clone()
    G_H[..., 0::2, 0::2] = Hf[..., 0::2, 0::2]
    G_H[..., 1::2, 1::2] = Hf[..., 1::2, 1::2]
    R_H[..., 0::2, 1::2] = Hf[..., 0::2, 1::2]
    B_H[..., 1::2, 0::2] = Hf[..., 1::2, 0::2]

    G_V = G.clone(); R_V = R.clone(); B_V = Bc.clone()
    G_V[..., 0::2, 0::2] = Vf[..., 0::2, 0::2]
    G_V[..., 1::2, 1::2] = Vf[..., 1::2, 1::2]
    R_V[..., 1::2, 0::2] = Vf[..., 1::2, 0::2]
    B_V[..., 0::2, 1::2] = Vf[..., 0::2, 1::2]

    # -----------------------
    # Step 2: delta 与梯度
    # -----------------------
    delta_H = G_H - R_H - B_H
    delta_V = G_V - R_V - B_V

    D_H = torch.abs(_conv_h(delta_H, [-1, 0, 1]))
    D_V = torch.abs(_conv_v(delta_V, [-1, 0, 1]))

    # -----------------------
    # Step 3: 方向权重（9x9 半区窗口）
    # -----------------------
    # W_W: rows 2..6, cols 0..4 置 1
    K_W = torch.zeros((9, 9), dtype=S.dtype, device=S.device); K_W[2:7, 0:5] = 1
    # W_E: rows 2..6, cols 4..8 置 1
    K_E = torch.zeros((9, 9), dtype=S.dtype, device=S.device); K_E[2:7, 4:9] = 1
    # W_N: rows 0..4, cols 2..6 置 1
    K_N = torch.zeros((9, 9), dtype=S.dtype, device=S.device); K_N[0:5, 2:7] = 1
    # W_S: rows 4..8, cols 2..6 置 1
    K_S = torch.zeros((9, 9), dtype=S.dtype, device=S.device); K_S[4:9, 2:7] = 1

    W_W = _conv2d(D_H, K_W)
    W_E = _conv2d(D_H, K_E)
    W_N = _conv2d(D_V, K_N)
    W_S = _conv2d(D_V, K_S)

    eps = 1e-10
    W_W = 1.0 / torch.clamp(W_W, min=eps).pow(2)
    W_E = 1.0 / torch.clamp(W_E, min=eps).pow(2)
    W_N = 1.0 / torch.clamp(W_N, min=eps).pow(2)
    W_S = 1.0 / torch.clamp(W_S, min=eps).pow(2)
    W_T = W_W + W_E + W_N + W_S + eps

    # -----------------------
    # Step 4: 方向聚合 delta（Eq.5）
    # -----------------------
    f = torch.tensor([1, 1, 1, 1, 1], dtype=S.dtype, device=S.device)
    f = f / f.sum()

    # 需要把 delta_H/ delta_V 子采样到两个 parity（列/行偶奇）
    # 构建 two-pass（c=0/1）并用 1D 滤波器滑动到远端（V1,V2,V3,V4）
    each_delta = []

    # 1D 长度 9 的核（前 5/后 5）
    k_head = torch.zeros(9, dtype=S.dtype, device=S.device); k_head[0:5] = f
    k_tail = torch.zeros(9, dtype=S.dtype, device=S.device); k_tail[4:9] = torch.flip(f, dims=[0])

    for c in (0, 1):
        # 取 delta_H: 行从 c 开始隔 2；delta_V: 列从 c 开始隔 2
        now_delta_H = torch.zeros_like(delta_H)
        now_delta_V = torch.zeros_like(delta_V)

        now_delta_H[..., c::2, :] = delta_H[..., c::2, :]
        now_delta_V[..., :, c::2] = delta_V[..., :, c::2]

        # V1/V2: 对 now_delta_V 做水平 1D 卷积（相当于对列上的子序列向左/向右扩展）
        V1 = _conv_h(now_delta_V, k_head.tolist())      # 左侧（0..4）
        V2 = _conv_h(now_delta_V, k_tail.tolist())      # 右侧（4..8，翻转）

        # V3/V4: 对 now_delta_H 做水平 1D 卷积
        V3 = _conv_h(now_delta_H, k_head.tolist())
        V4 = _conv_h(now_delta_H, k_tail.tolist())

        delta = (V1 * W_N + V2 * W_S + V3 * W_E + V4 * W_W) / W_T
        each_delta.append(delta)

    delta_GR, delta_GB = each_delta  # 对应 Eq.(6) 聚合结果

    # -----------------------
    # Step 5: 恢复 G（Eq.8）
    # -----------------------
    new_G = G.clone()
    new_G[..., 0::2, 0::2] = R[..., 0::2, 0::2] + delta_GR[..., 0::2, 0::2]
    new_G[..., 1::2, 1::2] = Bc[..., 1::2, 1::2] + delta_GB[..., 1::2, 1::2]

    # -----------------------
    # Step 6: 恢复 R in B、B in R（Eq.9）
    # -----------------------
    # prb 核（7x7，四角-1，十字四点 10）
    prb = torch.zeros((7, 7), dtype=S.dtype, device=S.device)
    prb[0, 2] = prb[0, 4] = prb[2, 0] = prb[2, 6] = -1
    prb[6, 2] = prb[6, 4] = prb[4, 0] = prb[4, 6] = -1
    prb[2, 2] = prb[2, 4] = prb[4, 2] = prb[4, 4] = 10

    # 分别用 prb 对 delta_GR / delta_GB 做 2D 卷积
    # 原始代码除以 sum(prb)，保持一致
    prb_sum = prb.sum()
    new_R = R.clone()
    new_B = Bc.clone()

    now_delta = _conv2d(delta_GR, prb) / (prb_sum + 1e-12)
    new_R[..., 1::2, 1::2] = (new_G - now_delta)[..., 1::2, 1::2]

    now_delta = _conv2d(delta_GB, prb) / (prb_sum + 1e-12)
    new_B[..., 0::2, 0::2] = (new_G - now_delta)[..., 0::2, 0::2]

    # -----------------------
    # Step 7: 恢复 R/B at G（Eq.10）
    # -----------------------
    cross = torch.tensor([[0,1,0],
                          [1,0,1],
                          [0,1,0]], dtype=S.dtype, device=S.device)
    now_R = new_G - _conv2d(new_G - new_R, cross) / 4.0
    now_B = new_G - _conv2d(new_G - new_B, cross) / 4.0

    new_R[..., 0::2, 1::2] = now_R[..., 0::2, 1::2]
    new_R[..., 1::2, 0::2] = now_R[..., 1::2, 0::2]
    new_B[..., 0::2, 1::2] = now_B[..., 0::2, 1::2]
    new_B[..., 1::2, 0::2] = now_B[..., 1::2, 0::2]

    # 组合输出（线性 RGB, [0,1]）
    rgb = torch.cat([new_R, new_G, new_B], dim=1).clamp_(0.0, 1.0)
    return rgb
