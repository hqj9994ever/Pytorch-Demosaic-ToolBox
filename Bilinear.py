import torch
import torch.nn.functional as F
import os
import os.path

import numpy as np

import torch

from typing import Optional, Tuple


import kornia

def Bilinear(image, color_desc=None, color_mask=None, rgb_xyz_matrix=None):

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

    
    r_b_kernel1 = torch.tensor([[1, 0, 1],
                                [0, 0, 0],
                                [1, 0, 1]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 4.0
    r_b_kernel2 = torch.tensor([[0, 1, 0],
                                [1, 0, 1],
                                [0, 1, 0]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 2.0
    g_kernel = torch.tensor([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]], dtype=torch.float32).to(image.device).unsqueeze(0).unsqueeze(0) / 4.0
    image_r = image_r + torch.nn.functional.conv2d(image_r, r_b_kernel1, stride=(1, 1), padding="same") + torch.nn.functional.conv2d(image_r, r_b_kernel2, stride=(1, 1), padding="same")
    image_g = image_g + torch.nn.functional.conv2d(image_g, g_kernel, stride=(1, 1), padding="same")
    image_b = image_b + torch.nn.functional.conv2d(image_b, r_b_kernel1, stride=(1, 1), padding="same") + torch.nn.functional.conv2d(image_b, r_b_kernel2, stride=(1, 1), padding="same")


    demosaic_image = torch.concat([image_r, image_g, image_b], dim=1)

    return demosaic_image