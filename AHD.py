import torch
import torch.nn.functional as F
import os
import os.path

import numpy as np

import torch

from typing import Optional, Tuple


# -------------------------
# AHD
# -------------------------
def _pad_reflect(x, kH, kW):
    ph = (kH - 1) // 2
    pw = (kW - 1) // 2
    return F.pad(x, (pw, pw, ph, ph), mode="reflect")

def conv2d_same(x, kernel):
    """
    x: [B, C=1 or 3, H, W]
    kernel: [outC, inC, kH, kW]
    return: [B, outC, H, W]
    """
    kH, kW = kernel.shape[-2:]
    xpad = _pad_reflect(x, kH, kW)
    kernel = kernel.to(xpad.device, dtype=xpad.dtype)
    return F.conv2d(xpad, kernel)

def make_kernel(weights_2d):
    """
    weights_2d: list[list[float]] 或 1D list
    返回: [1,1,kH,kW]
    """
    w = torch.tensor(weights_2d, dtype=torch.float32)
    if w.dim() == 1:
        w = w.unsqueeze(0)  # [1,k]
    w = w.unsqueeze(0).unsqueeze(0)  # [1,1,kH,kW]
    return w

def make_bank(kernels_2d):
    """
    kernels_2d: list of 2D tensors/list
    return: [K,1,kH,kW]
    """
    ws = []
    for k in kernels_2d:
        w = torch.tensor(k, dtype=torch.float32)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        ws.append(w.unsqueeze(0).unsqueeze(0))
    return torch.cat(ws, dim=0)  # [K,1,kH,kW]

# -------------------------
# Bayer 掩码（广播到 batch）
# -------------------------
def bayer_masks(bayer, pattern: str):
    """
    bayer: [B,1,H,W] in [0,1]
    return Rm, Gm, Bm: [B,1,H,W] (0/1 掩码)
    """
    B, _, H, W = bayer.shape
    Rm = torch.zeros_like(bayer)
    Gm = torch.zeros_like(bayer)
    Bm = torch.zeros_like(bayer)

    # 索引切片
    if pattern.upper() == "RGGB":
        Rm[..., 0::2, 0::2] = 1
        Gm[..., 0::2, 1::2] = 1
        Gm[..., 1::2, 0::2] = 1
        Bm[..., 1::2, 1::2] = 1
    elif pattern.upper() == "GRBG":
        Gm[..., 0::2, 0::2] = 1
        Rm[..., 0::2, 1::2] = 1
        Bm[..., 1::2, 0::2] = 1
        Gm[..., 1::2, 1::2] = 1
    elif pattern.upper() == "GBRG":
        Gm[..., 0::2, 0::2] = 1
        Bm[..., 0::2, 1::2] = 1
        Rm[..., 1::2, 0::2] = 1
        Gm[..., 1::2, 1::2] = 1
    elif pattern.upper() == "BGGR":
        Bm[..., 0::2, 0::2] = 1
        Gm[..., 0::2, 1::2] = 1
        Gm[..., 1::2, 0::2] = 1
        Rm[..., 1::2, 1::2] = 1
    else:
        raise ValueError("pattern must be one of: RGGB, GRBG, GBRG, BGGR")

    return Rm, Gm, Bm

# -------------------------
# AH 插值（水平核 Hg + 3x3 Hr）
# -------------------------
def AH_interpolate(bayer, pattern: str, gamma: float):
    """
    bayer: [B,1,H,W] in [0,1]
    return R,G,B: [B,1,H,W]
    """
    X = bayer
    device = X.device
    Rm, Gm, Bm = bayer_masks(X, pattern)

    # green: Hg = [0, 1/2, 0, 1/2, 0] + gamma * [-1/4,0,1/2,0,-1/4]
    Hg1 = torch.tensor([0.0, 0.5, 0.0, 0.5, 0.0], dtype=torch.float32, device=device)
    Hg2 = torch.tensor([-0.25, 0.0, 0.5, 0.0, -0.25], dtype=torch.float32, device=device)
    Hg = (Hg1 + gamma * Hg2).view(1, 1, 1, 5)  # [1,1,1,5]

    # red/blue kernel Hr
    Hr = torch.tensor([[0.25, 0.5, 0.25],
                       [0.5,  1.0, 0.5 ],
                       [0.25, 0.5, 0.25]], dtype=torch.float32, device=device).view(1,1,3,3)

    # G = Gm*X + (Rm+Bm)*conv(X,Hg)
    G_from_rb = conv2d_same(X, Hg)
    G = Gm * X + (Rm + Bm) * G_from_rb

    # R = G + conv(Rm*(X-G), Hr), B 同理
    R = G + conv2d_same(Rm * (X - G), Hr)
    B = G + conv2d_same(Bm * (X - G), Hr)

    # 裁剪到 [0,1]
    R = R.clamp_(0.0, 1.0)
    G = G.clamp_(0.0, 1.0)
    B = B.clamp_(0.0, 1.0)
    return R, G, B

def AH_interpolateX(bayer, pattern: str, gamma: float):
    R, G, B = AH_interpolate(bayer, pattern, gamma)
    return torch.cat([R, G, B], dim=1)  # [B,3,H,W]

def AH_interpolateY(bayer, pattern: str, gamma: float):
    # 模式在转置后需要交换（与原代码一致）
    if pattern.upper() == "RGGB":
        new_pattern = "RGGB"
    elif pattern.upper() == "GRBG":
        new_pattern = "GBRG"
    elif pattern.upper() == "GBRG":
        new_pattern = "GRBG"
    elif pattern.upper() == "BGGR":
        new_pattern = "BGGR"
    else:
        raise ValueError("pattern must be one of: RGGB, GRBG, GBRG, BGGR")

    # 先对 H,W 置换，再做 AH，再转回
    imgT = bayer.transpose(-1, -2)  # [B,1,W,H]
    YT = AH_interpolate(imgT, new_pattern, gamma)  # R,G,B: [B,1,W,H]
    R = YT[0].transpose(-1, -2)
    G = YT[1].transpose(-1, -2)
    B = YT[2].transpose(-1, -2)
    return torch.cat([R, G, B], dim=1)  # [B,3,H,W]

# -------------------------
# LAB 近似（保持与你的矩阵/标度一致）
# -------------------------
def labf(t):
    d = torch.pow(t, 1/3)
    idx = (t <= 0.008856)
    d = torch.where(idx, 7.787 * t + 16.0/116.0, d)
    return d

def RGB2LAB_like(X_rgb):
    """
    X_rgb: [B,3,H,W] in [0,1]
    复刻你原始的矩阵/标度逻辑（先×255，再按/255使用）
    """
    B, C, H, W = X_rgb.shape
    # 矩阵（与你粘贴的一致）
    a = torch.tensor([[3.40479,  -1.537150, -0.498535],
                      [-0.969256, 1.875992, 0.041556],
                      [0.055648, -0.204043, 1.057311]],
                     dtype=torch.float32, device=X_rgb.device)
    ai = torch.inverse(a)

    X = X_rgb * 255.0  # 对齐原始实现的标度
    # [B,3,H,W] -> [B,3,HW]
    Xp = X.view(B, C, -1)
    # [3,3] @ [B,3,HW] => [B,3,HW]
    labp = torch.matmul(ai.unsqueeze(0), Xp)
    L1 = labp[:, 0, :].view(B, 1, H, W)
    L2 = labp[:, 1, :].view(B, 1, H, W)
    L3 = labp[:, 2, :].view(B, 1, H, W)

    # 和原代码一致的公式
    result_L = 116.0 * labf(L2 / 255.0) - 16.0
    result_a = 500.0 * (labf(L1 / 255.0) - labf(L2 / 255.0))
    result_b = 200.0 * (labf(L2 / 255.0) - labf(L3 / 255.0))
    return torch.cat([result_L, result_a, result_b], dim=1)  # [B,3,H,W]

# -------------------------
# MN 参数、同质性（向量化的卷积核组）
# -------------------------
def MNparamA(YxLAB, YyLAB):
    # 水平/垂直差分核
    H1 = make_kernel([1, -1, 0])    # [1,1,1,3]
    H2 = make_kernel([0, -1, 1])

    V1 = H1.transpose(-1, -2)       # [1,1,3,1]
    V2 = H2.transpose(-1, -2)

    # eL
    eLM1 = torch.maximum(torch.abs(conv2d_same(YxLAB[:, 0:1], H1)),
                         torch.abs(conv2d_same(YxLAB[:, 0:1], H2)))
    eLM2 = torch.maximum(torch.abs(conv2d_same(YyLAB[:, 0:1], V1)),
                         torch.abs(conv2d_same(YyLAB[:, 0:1], V2)))
    eL = torch.minimum(eLM1, eLM2)  # [B,1,H,W]

    # eC
    Cx1 = conv2d_same(YxLAB[:, 1:2], H1)
    Cx2 = conv2d_same(YxLAB[:, 2:3], H1)
    Cx3 = conv2d_same(YxLAB[:, 1:2], H2)
    Cx4 = conv2d_same(YxLAB[:, 2:3], H2)
    eCx = torch.maximum(Cx1.pow(2) + Cx2.pow(2), Cx3.pow(2) + Cx4.pow(2))

    Cy1 = conv2d_same(YyLAB[:, 1:2], V1)
    Cy2 = conv2d_same(YyLAB[:, 2:3], V1)
    Cy3 = conv2d_same(YyLAB[:, 1:2], V2)
    Cy4 = conv2d_same(YyLAB[:, 2:3], V2)
    eCy = torch.maximum(Cy1.pow(2) + Cy2.pow(2), Cy3.pow(2) + Cy4.pow(2))

    eC = torch.minimum(eCx, eCy).sqrt_()
    return eL, eC  # [B,1,H,W] each

def MNballset(delta: int, device):
    """
    返回一个盘形邻域的卷积核组 H: [K,1,kH,kW]
    """
    size = delta * 2 + 1
    kernels = []
    for i in range(-delta, delta + 1):
        for j in range(-delta, delta + 1):
            if (i**2 + j**2) ** 0.5 <= delta + 1e-6:
                k = torch.zeros(size, size, dtype=torch.float32)
                k[i + delta, j + delta] = 1.0
                kernels.append(k)
    bank = torch.stack(kernels, dim=0).unsqueeze(1)  # [K,1,k,k]
    return bank.to(device)

def MNhomogeneity(LAB_image, delta, epsilonL, epsilonC):
    """
    LAB_image: [B,3,H,W]
    epsilonL/epsilonC: [B,1,H,W]
    return K: [B,1,H,W]
    """
    B, _, H, W = LAB_image.shape
    device = LAB_image.device
    bank = MNballset(delta, device)  # [K,1,k,k]
    K = bank.shape[0]

    # conv bank
    L = LAB_image[:, 0:1]
    a = LAB_image[:, 1:2]
    b = LAB_image[:, 2:3]

    conv_L = conv2d_same(L, bank)  # [B,K,H,W]
    conv_a = conv2d_same(a, bank)
    conv_b = conv2d_same(b, bank)

    Ldiff = (conv_L - L) .abs() <= epsilonL  # [B,K,H,W]
    Cdiff = (conv_a - a).pow(2) + (conv_b - b).pow(2) <= (epsilonC ** 2)

    U = Ldiff & Cdiff
    Kmap = U.float().sum(dim=1, keepdim=True)  # [B,1,H,W]
    return Kmap

# -------------------------
# 去伪影（方向中值）
# -------------------------
def MNartifact(R, G, B, iterations: int):
    """
    R,G,B: [B,1,H,W]  (范围 [0,1])
    """
    device = R.device
    # 8邻域（不含中心）
    dirs8 = [
        [[1,0,0],[0,0,0],[0,0,0]],
        [[0,1,0],[0,0,0],[0,0,0]],
        [[0,0,1],[0,0,0],[0,0,0]],
        [[0,0,0],[1,0,0],[0,0,0]],
        [[0,0,0],[0,0,1],[0,0,0]],
        [[0,0,0],[0,0,0],[1,0,0]],
        [[0,0,0],[0,0,0],[0,1,0]],
        [[0,0,0],[0,0,0],[0,0,1]],
    ]
    bank8 = make_bank(dirs8).to(device)  # [8,1,3,3]

    # 4 个正轴方向（上、左、右、下）——对应你原代码的 kernel_2/4/6/8
    dirs4 = [
        [[0,1,0],[0,0,0],[0,0,0]],  # 上
        [[0,0,0],[1,0,0],[0,0,0]],  # 左
        [[0,0,0],[0,0,1],[0,0,0]],  # 右
        [[0,0,0],[0,0,0],[0,1,0]],  # 下
    ]
    bank4 = make_bank(dirs4).to(device)

    for _ in range(iterations):
        # R,B：对 (R-G)、(B-G) 的 8 方向邻域取中值
        Rt = conv2d_same(R - G, bank8)  # [B,8,H,W]
        Bt = conv2d_same(B - G, bank8)
        Rm = Rt.median(dim=1, keepdim=True).values  # [B,1,H,W]
        Bm = Bt.median(dim=1, keepdim=True).values
        R  = (G + Rm).clamp_(0.0, 1.0)
        B  = (G + Bm).clamp_(0.0, 1.0)

        # G：分两路（相对 R / 相对 B），四方向中值再平均
        Grt = conv2d_same(G - R, bank4)  # [B,4,H,W]
        Gbt = conv2d_same(G - B, bank4)
        Grm = Grt.median(dim=1, keepdim=True).values
        Gbm = Gbt.median(dim=1, keepdim=True).values
        G   = ((R + Grm) * 0.5 + (B + Gbm) * 0.5).clamp_(0.0, 1.0)

    return R, G, B


@torch.no_grad()
def AHD(bayer: torch.Tensor, pattern: str = "RGGB",
                 delta: int = 2, gamma: float = 1.0,
                 iterations: int = 2, pad: int = 10):
    """
    bayer: [B,1,H,W] in [0,1]
    return: RGB [B,3,H,W] in [0,1]
    """
    assert bayer.dim() == 4 and bayer.size(1) == 1, "要求输入 [B,1,H,W]"
    B, _, H, W = bayer.shape
    device = bayer.device

    # 反射填充，和原实现一致（便于 X/Y 两向插值）
    f = F.pad(bayer, (pad, pad, pad, pad), mode="reflect")  # [B,1,H+2p,W+2p]

    # X/Y 两向 AH
    Yx = AH_interpolateX(f, pattern, gamma)  # [B,3,H+2p,W+2p]
    Yy = AH_interpolateY(f, pattern, gamma)

    # LAB 近似
    YxLAB = RGB2LAB_like(Yx)
    YyLAB = RGB2LAB_like(Yy)

    # 同质性参数 & 地图
    epsilonL, epsilonC = MNparamA(YxLAB, YyLAB)  # [B,1,H+2p,W+2p]
    Hx = MNhomogeneity(YxLAB, delta, epsilonL, epsilonC)
    Hy = MNhomogeneity(YyLAB, delta, epsilonL, epsilonC)

    # 3x3 平滑（全 1 卷积）
    ones3 = make_kernel([[1,1,1],[1,1,1],[1,1,1]]).to(device)
    Hx = conv2d_same(Hx, ones3)
    Hy = conv2d_same(Hy, ones3)

    # 逐像素挑选 X/Y
    sel = (Hy >= Hx).float()  # [B,1,H+2p,W+2p]
    RGB = Yx * (1 - sel) + Yy * sel  # [B,3,H+2p,W+2p]

    # 去伪影
    R, G, Bc = MNartifact(RGB[:,0:1], RGB[:,1:2], RGB[:,2:2+1], iterations)
    Y = torch.cat([R, G, Bc], dim=1).clamp_(0.0, 1.0)

    # 裁回原尺寸
    Y = Y[..., pad:pad+H, pad:pad+W]  # [B,3,H,W]
    return Y
