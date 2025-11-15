import torch
import torch.nn.functional as F
import os
import os.path

import numpy as np

import torch

from typing import Optional, Tuple

from GBTF import GBTF


# --------------------------
#  主函数：RI
# --------------------------

def _box_kernel(h, w, device, dtype):
    return torch.ones((1,1,h,w), device=device, dtype=dtype)

# ---------- guided filter with sparse mask M ----------
def _guided_filter(M, I, p, h=11, w=11, eps=1e-2):
    """
    M: [B,1,H,W] binary mask (1 where p is observed)
    I: [B,1,H,W] guidance (ok_G)
    p: [B,1,H,W] sparse signal (R or B observed at masked sites)
    """
    device, dtype = I.device, I.dtype
    K = _box_kernel(h, w, device, dtype)

    N1 = F.conv2d(M, K, padding=(h//2, w//2))                    # support size on masked area
    N2 = F.conv2d(torch.ones_like(I), K, padding=(h//2, w//2))   # support size on full window

    nI = I * M
    mean_I  = F.conv2d(nI,     K, padding=(h//2, w//2)) / (N1.clamp_min(1e-12))
    mean_p  = F.conv2d(p,      K, padding=(h//2, w//2)) / (N1.clamp_min(1e-12))
    corr_I  = F.conv2d(nI*nI,  K, padding=(h//2, w//2)) / (N1.clamp_min(1e-12))
    corr_Ip = F.conv2d(nI*p,   K, padding=(h//2, w//2)) / (N1.clamp_min(1e-12))

    var_I  = corr_I  - mean_I*mean_I
    cov_Ip = corr_Ip - mean_I*mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = F.conv2d(a, K, padding=(h//2, w//2)) / (N2.clamp_min(1e-12))
    mean_b = F.conv2d(b, K, padding=(h//2, w//2)) / (N2.clamp_min(1e-12))

    return mean_a * I + mean_b



@torch.no_grad()
def RI(bayer: torch.Tensor,
    ok_G: Optional[torch.Tensor] = None,
    gf_hw: Tuple[int, int] = (11, 11),
    eps: float = 1e-2) -> torch.Tensor:
    """
    Residual Interpolation demosaic (RGGB).
    Args:
        bayer: [B,1,H,W] in [0,1], RGGB
        ok_G:  optional [B,1,H,W] pre-estimated green (linear, [0,1])
        gf_hw: guided filter window (h,w)
    Returns:
        rgb:   [B,3,H,W] in [0,1]
    """
    assert bayer.dim() == 4 and bayer.size(1) == 1, "Expect [B,1,H,W]"
    B, _, H, W = bayer.shape
    S = bayer

    # RGGB masks
    mask_R  = torch.zeros_like(S); mask_R[..., 0::2, 0::2] = 1.0
    mask_G1 = torch.zeros_like(S); mask_G1[..., 0::2, 1::2] = 1.0
    mask_G2 = torch.zeros_like(S); mask_G2[..., 1::2, 0::2] = 1.0
    mask_G  = mask_G1 + mask_G2
    mask_B  = torch.zeros_like(S); mask_B[..., 1::2, 1::2] = 1.0

    # Sparse R/B (observed)
    R_obs = S * mask_R
    B_obs = S * mask_B

    # Step 1: OK_G (use GBTF or provided)
    if ok_G is None:
        G_est = GBTF(bayer)[:,1:2]   # 建议替换为你的 demosaic_gbtf_rggb(bayer)[:,1:2]
    else:
        assert ok_G.shape == S.shape, "ok_G must be [B,1,H,W]"
        G_est = ok_G.clamp(0,1)

    # Step 2: guided filter to predict dense R & B from G_est
    h, w = gf_hw
    R_mask = mask_R
    B_mask = mask_B
    R_pred = _guided_filter(R_mask, G_est, R_obs, h, w, eps=eps)
    B_pred = _guided_filter(B_mask, G_est, B_obs, h, w, eps=eps)

    # Step 3: residuals (only at observed sites)
    delta_R = (R_obs - R_pred) * R_mask
    delta_B = (B_obs - B_pred) * B_mask

    # Step 4: interpolate residuals with separable 3x3 kernel
    # intp = [[1/4,1/2,1/4],[1/2,1,1/2],[1/4,1/2,1/4]]
    intp = torch.tensor([[0.25,0.5,0.25],
                         [0.5 ,1.0,0.5 ],
                         [0.25,0.5,0.25]], dtype=S.dtype, device=S.device)
    K = intp.view(1,1,3,3)

    delta_R_s = F.conv2d(delta_R, K, padding=1)
    denom_R   = F.conv2d(R_mask,  K, padding=1).clamp_min(1e-12)
    delta_R_i = delta_R_s / denom_R

    delta_B_s = F.conv2d(delta_B, K, padding=1)
    denom_B   = F.conv2d(B_mask,  K, padding=1).clamp_min(1e-12)
    delta_B_i = delta_B_s / denom_B

    # Step 5: add residuals back
    R_full = (R_pred + delta_R_i).clamp(0,1)
    B_full = (B_pred + delta_B_i).clamp(0,1)

    rgb = torch.cat([R_full, G_est, B_full], dim=1).clamp(0,1)
    return rgb
