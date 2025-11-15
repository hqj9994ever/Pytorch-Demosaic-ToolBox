import torch
import torch.nn.functional as F
import os
import os.path

import numpy as np

import torch

from typing import Optional, Tuple


import kornia

def AHD_DualDn(image, color_desc=None, color_mask=None, rgb_xyz_matrix=None):

    mask_r = torch.zeros(image.shape).to(image.device)
    mask_g = torch.zeros(image.shape).to(image.device)
    mask_b = torch.zeros(image.shape).to(image.device)
    color_mask = color_mask.to(image.device)
    rgb_xyz_matrix = rgb_xyz_matrix.to(image.device)

    for i, col in enumerate(color_desc):
        if col == 'R':
            mask_r = mask_r.masked_fill(torch.eq(color_mask, i), 1)
            image_r = torch.mul(image, mask_r)
        if col == 'G':
            mask_g = mask_g.masked_fill(torch.eq(color_mask, i), 1)
            image_g = torch.mul(image, mask_g)
        if col == 'B':
            mask_b = mask_b.masked_fill(torch.eq(color_mask, i), 1)
            image_b = torch.mul(image, mask_b)

    h0_row = torch.tensor([-1, 2, 2, 2, -1], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) / 4

    h0_col = torch.tensor([[-1],
                    [2],
                    [2],
                    [2],
                    [-1]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 4

    d0_row = torch.nn.functional.conv2d(image, h0_row, stride=1, padding="same")
    d0_col = torch.nn.functional.conv2d(image, h0_col, stride=1, padding="same")

    g_row = image_g + torch.mul(d0_row, mask_r) + torch.mul(d0_row, mask_b)
    g_col = image_g + torch.mul(d0_col, mask_r) + torch.mul(d0_col, mask_b)

    r_b_kernel1 = torch.tensor([[1, 0, 1],
                                [0, 0, 0],
                                [1, 0, 1]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 4.0
    r_b_kernel2 = torch.tensor([[0, 1, 0],
                                [1, 0, 1],
                                [0, 1, 0]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 2.0
    g_kernel = torch.tensor([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 4.0

    homo_row1 = torch.tensor([1, 2, -3, 0, 0], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) 
    homo_col1 = torch.tensor([[1],
                    [2],
                    [-3],
                    [0],
                    [0]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) 

    homo_row2 = torch.tensor([0, 0, -3, 2, 1], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) 
    homo_col2 = torch.tensor([[0],
                    [0],
                    [-3],
                    [2],
                    [1]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) 

    r_sample = image_r + torch.nn.functional.conv2d(image_r, r_b_kernel1, stride=(1, 1), padding="same") + torch.nn.functional.conv2d(image_r, r_b_kernel2, stride=(1, 1), padding="same")
    g_sample = image_g + torch.nn.functional.conv2d(image_g, g_kernel, stride=(1, 1), padding="same")
    b_sample = image_b + torch.nn.functional.conv2d(image_b, r_b_kernel1, stride=(1, 1), padding="same") + torch.nn.functional.conv2d(image_b, r_b_kernel2, stride=(1, 1), padding="same")

    r_row = kornia.filters.gaussian_blur2d((r_sample - g_sample), (3,3), (1.5,1.5)) + g_row
    r_col = kornia.filters.gaussian_blur2d((r_sample - g_sample), (3,3), (1.5,1.5)) + g_col

    b_row = kornia.filters.gaussian_blur2d((b_sample - g_sample), (3,3), (1.5,1.5)) + g_row
    b_col = kornia.filters.gaussian_blur2d((b_sample - g_sample), (3,3), (1.5,1.5)) + g_col

    image_sample = torch.concat([r_sample, g_sample, b_sample], dim=1)

    image_xyz = torch.matmul(rgb_xyz_matrix.unsqueeze(-3), image_sample.permute(0,2,3,1).unsqueeze(-1)).squeeze(-1).permute(0,3,1,2)
    xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32)[..., :, None, None].to(image.device).unsqueeze(0)
    xyz_normalized = torch.div(image_xyz, xyz_ref_white)

    threshold = 0.008856
    power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
    scale = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int = torch.where(xyz_normalized > threshold, power, scale)

    L = ((116.0 * xyz_int[..., 1, :, :]) - 16.0).unsqueeze(-3)
    a = (500.0 * (xyz_int[..., 0, :, :] - xyz_int[..., 1, :, :])).unsqueeze(-3)
    b = (200.0 * (xyz_int[..., 1, :, :] - xyz_int[..., 2, :, :])).unsqueeze(-3)

    row1 = torch.abs(torch.nn.functional.conv2d(L, homo_row1, stride=(1, 1), padding="same") ) + torch.sqrt(torch.square(torch.nn.functional.conv2d(a, homo_row1, stride=(1, 1), padding="same")) + torch.square(torch.nn.functional.conv2d(b, homo_row1, stride=(1, 1), padding="same")))
    col1 = torch.abs(torch.nn.functional.conv2d(L, homo_col1, stride=(1, 1), padding="same") ) + torch.sqrt(torch.square(torch.nn.functional.conv2d(a, homo_col1, stride=(1, 1), padding="same")) + torch.square(torch.nn.functional.conv2d(b, homo_col1, stride=(1, 1), padding="same")))
    row2 = torch.abs(torch.nn.functional.conv2d(L, homo_row2, stride=(1, 1), padding="same") ) + torch.sqrt(torch.square(torch.nn.functional.conv2d(a, homo_row2, stride=(1, 1), padding="same")) + torch.square(torch.nn.functional.conv2d(b, homo_row2, stride=(1, 1), padding="same")))
    col2 = torch.abs(torch.nn.functional.conv2d(L, homo_col2, stride=(1, 1), padding="same") ) + torch.sqrt(torch.square(torch.nn.functional.conv2d(a, homo_col2, stride=(1, 1), padding="same")) + torch.square(torch.nn.functional.conv2d(b, homo_col2, stride=(1, 1), padding="same")))

    row = row1 + row2
    col = col1 + col2

    image_r = torch.where(row >= col, r_col, r_row)
    image_g = torch.where(row >= col, g_col, g_row)
    image_b = torch.where(row >= col, b_col, b_row)

    demosaic_image = torch.concat([image_r, image_g, image_b], dim=1)

    return demosaic_image