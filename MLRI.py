import torch
import torch.nn.functional as F
import os
import os.path

import numpy as np

import torch

from typing import Optional, Tuple

from RI import RI



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
# Guided filter (RI/MLRI 共用)
# --------------------------
def _guided_filter_mlri(M, I, p, h=7, w=7, eps=1e-2):
    """
    MLRI 中使用的引导滤波（带 mask 与拉普拉斯约束）
    M: [B,1,H,W]  二值采样掩码
    I: [B,1,H,W]  引导（一般为 G 或插值过的通道）
    p: [B,1,H,W]  稀疏观测（R/B/子采样的 G ）
    """
    device, dtype = I.device, I.dtype
    Ksum = torch.ones((1,1,h,w), device=device, dtype=dtype)

    ok_I = I * M
    ok_p = p * M

    N1 = F.conv2d(M, Ksum, padding=(h//2, w//2)).clamp_min(1e-12)

    # 水平一维“拉普拉斯样”核 [-1, 0, 2, 0, -1]
    lap_k = [-1, 0, 2, 0, -1]
    lap_I = _conv_h(ok_I, lap_k) * M
    lap_p = _conv_h(ok_p, lap_k) * M

    # 求 a, b（窗口内最小二乘）
    sum_lap_II = F.conv2d(lap_I*lap_I, Ksum, padding=(h//2, w//2))
    sum_lap_Ip = F.conv2d(lap_I*lap_p, Ksum, padding=(h//2, w//2))
    a = sum_lap_Ip / (sum_lap_II + 1e-32)

    sum_I = F.conv2d(ok_I, Ksum, padding=(h//2, w//2))
    sum_p = F.conv2d(ok_p, Ksum, padding=(h//2, w//2))
    b = (sum_p - a * sum_I) / N1

    # 计算权重 W（式 10）
    W = ok_p - a * ok_I - b
    W = (F.conv2d(W, Ksum, padding=(h//2, w//2)).pow(2)) / N1
    W = 1.0 / (W + 1e-32)

    sum_W  = F.conv2d(W,    Ksum, padding=(h//2, w//2)).clamp_min(1e-12)
    mean_a = F.conv2d(W * a, Ksum, padding=(h//2, w//2)) / sum_W
    mean_b = F.conv2d(W * b, Ksum, padding=(h//2, w//2)) / sum_W

    return mean_a * I + mean_b


# --------------------------
# 计算 delta（水平或垂直方向）
# --------------------------
def _compute_delta_dir(R, G, B, horizontal=True):
    """
    输入 R,G,B 为 [B,1,H,W]。当 horizontal=False 时，调用前请先转置到 [B,1,W,H] 再传入；
    本函数内部只做“按行”的 1D 卷积。
    返回：delta_GR, delta_GB （与原实现一致的屏蔽行）
    """
    B_,C,H,W = G.shape

    # 掩码（此处仅按行/列奇偶划分，不引入 RGGB 列偶奇）
    R_Mask = torch.zeros_like(G); R_Mask[..., 0::2, :] = 1.0
    B_Mask = torch.zeros_like(G); B_Mask[..., 1::2, :] = 1.0

    GR = torch.zeros_like(G); GR[..., 0::2, :] = G[..., 0::2, :]
    GB = torch.zeros_like(G); GB[..., 1::2, :] = G[..., 1::2, :]

    GR_Mask = torch.zeros_like(G); GR_Mask[..., 0::2, 1::2] = 1.0
    GB_Mask = torch.zeros_like(G); GB_Mask[..., 1::2, 0::2] = 1.0

    # Step 1: 线性插值（[1,2,1]/2）
    lin_k = [1,2,1]
    intp_R = _conv_h(R, lin_k) / 2.0
    intp_G = _conv_h(G, lin_k) / 2.0
    intp_B = _conv_h(B, lin_k) / 2.0

    # Step 2: 引导滤波并补偿边界残差（同原文）
    new_R  = _guided_filter_mlri(R_Mask,  intp_G, R, 7, 7)
    deltaR = _conv_h(new_R*R_Mask - R, lin_k)/2.0
    new_R  = new_R - deltaR

    new_GR = _guided_filter_mlri(GR_Mask, intp_R, GR, 7, 7)
    deltaGR= _conv_h(new_GR*GR_Mask - GR, lin_k)/2.0
    new_GR = new_GR - deltaGR

    new_B  = _guided_filter_mlri(B_Mask,  intp_G, B, 7, 7)
    deltaB = _conv_h(new_B*B_Mask - B, lin_k)/2.0
    new_B  = new_B - deltaB

    new_GB = _guided_filter_mlri(GB_Mask, intp_B, GB, 7, 7)
    deltaGB= _conv_h(new_GB*GB_Mask - GB, lin_k)/2.0
    new_GB = new_GB - deltaGB

    # Step 3: R-G 与 B-G 的行选择
    delta_GR = (new_GR - new_R)
    delta_GB = (new_GB - new_B)

    # 与 numpy 版一致：delta_GR 保留偶数行，delta_GB 保留奇数行
    keep_GR = torch.zeros_like(G); keep_GR[..., 0::2, :] = 1.0
    keep_GB = torch.zeros_like(G); keep_GB[..., 1::2, :] = 1.0
    delta_GR = delta_GR * keep_GR
    delta_GB = delta_GB * keep_GB
    return delta_GR, delta_GB



# --------------------------
# 主函数：MLRI (RGGB)
# --------------------------


@torch.no_grad()
def MLRI(bayer: torch.Tensor) -> torch.Tensor:
    """
    MLRI demosaic (RGGB).
    Input : bayer [B,1,H,W] in [0,1]
    Output: rgb   [B,3,H,W] in [0,1] (linear)
    """
    assert bayer.dim()==4 and bayer.size(1)==1, "Expect [B,1,H,W]"
    S = bayer
    Bn,_,H,W = S.shape
    eps = 1e-10

    # RGGB masks
    mask_R  = torch.zeros_like(S); mask_R[..., 0::2, 0::2] = 1.0
    mask_G1 = torch.zeros_like(S); mask_G1[..., 0::2, 1::2] = 1.0
    mask_G2 = torch.zeros_like(S); mask_G2[..., 1::2, 0::2] = 1.0
    mask_G  = mask_G1 + mask_G2
    mask_B  = torch.zeros_like(S); mask_B[..., 1::2, 1::2] = 1.0

    # 观测 R/G/B
    R  = S * mask_R
    G0 = S * mask_G
    Bc = S * mask_B

    # ===== Step 1-3: 计算 (G-R)、(G-B) 的 H / V 两个方向的 delta =====
    # 水平方向（按行）
    dGR_H, dGB_H = _compute_delta_dir(R, G0, Bc, horizontal=True)

    # 垂直方向：转置到 [B,1,W,H] 做“按行”处理，再转回
    Rt, Gt, Bt = R.transpose(-2,-1), G0.transpose(-2,-1), Bc.transpose(-2,-1)
    dGR_V_t, dGB_V_t = _compute_delta_dir(Rt, Gt, Bt, horizontal=True)
    dGR_V, dGB_V = dGR_V_t.transpose(-2,-1), dGB_V_t.transpose(-2,-1)

    # ===== Step 4: GBTF 风格的方向权重（W/E/N/S）=====
    D_H = torch.abs(_conv_h(dGR_H + dGB_H, [-1,0,1]))
    D_V = torch.abs(_conv_v(dGR_V + dGB_V, [-1,0,1]))

    K_W = torch.zeros((9,9), dtype=S.dtype, device=S.device); K_W[2:7, 0:5] = 1
    K_E = torch.zeros((9,9), dtype=S.dtype, device=S.device); K_E[2:7, 4:9] = 1
    K_N = torch.zeros((9,9), dtype=S.dtype, device=S.device); K_N[0:5, 2:7] = 1
    K_S = torch.zeros((9,9), dtype=S.dtype, device=S.device); K_S[4:9, 2:7] = 1

    W_W = _conv2d(D_H, K_W) + eps
    W_E = _conv2d(D_H, K_E) + eps
    W_N = _conv2d(D_V, K_N) + eps
    W_S = _conv2d(D_V, K_S) + eps

    W_W = 1.0 / (W_W.pow(2))
    W_E = 1.0 / (W_E.pow(2))
    W_N = 1.0 / (W_N.pow(2))
    W_S = 1.0 / (W_S.pow(2))
    W_T = W_W + W_E + W_N + W_S + eps

    # ===== Step 5: 聚合 delta（式 15）=====
    # 论文中的 f；实现与你代码一致，这里使用平均权 f = [1/4]*4
    f = torch.ones(4, dtype=S.dtype, device=S.device) / 4.0

    # 7-tap 的两种方向核（注意 numpy 的卷积方向，这里等价实现）
    k_head7 = torch.zeros(7, dtype=S.dtype, device=S.device); k_head7[0:4] = f       # (... f0 f1 f2 f3 0 0 0)
    k_tail7 = torch.zeros(7, dtype=S.dtype, device=S.device); k_tail7[3:7] = torch.flip(f, dims=[0])  # (0 0 0 f3 f2 f1 f0)

    # 对 V/H 的 delta 进行“向前/向后”聚合
    # —— GR
    V1 = _conv_h(dGR_V, k_tail7.tolist())   # 北
    V2 = _conv_h(dGR_V, k_head7.tolist())   # 南
    V3 = _conv_h(dGR_H, k_tail7.tolist())   # 西
    V4 = _conv_h(dGR_H, k_head7.tolist())   # 东
    dGR = (V1*W_N + V2*W_S + V3*W_W + V4*W_E) / W_T

    # —— GB
    V1 = _conv_h(dGB_V, k_tail7.tolist())   # 北
    V2 = _conv_h(dGB_V, k_head7.tolist())   # 南
    V3 = _conv_h(dGB_H, k_tail7.tolist())   # 西
    V4 = _conv_h(dGB_H, k_head7.tolist())   # 东
    dGB = (V1*W_N + V2*W_S + V3*W_W + V4*W_E) / W_T

    # ===== Step 6: 恢复 G （在 R/B 位置）=====
    G = G0.clone()
    G[..., 0::2, 0::2] = (R  + dGR)[..., 0::2, 0::2]
    G[..., 1::2, 1::2] = (Bc + dGB)[..., 1::2, 1::2]
    G = G.clamp(0,1)

    # ===== Step 7: 用 RI 恢复 R / B =====
    RGB = RI(S, G)
    R_full, B_full = RGB[:, 0:1], RGB[:, 2:3]

    rgb = torch.cat([R_full, G, B_full], dim=1).clamp(0,1)
    return rgb
