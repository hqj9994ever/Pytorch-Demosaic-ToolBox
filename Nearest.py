import torch
import torch.nn.functional as F
import os
import os.path

import numpy as np

import torch

from typing import Optional, Tuple


import kornia

def Nearest(image, color_desc=None, color_mask=None, rgb_xyz_matrix=None):

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

    image_r = image_r + torch.roll(image_r, shifts=1, dims=-2) + torch.roll(image_r, shifts=1, dims=-1) \
    + torch.roll(image_r, shifts=(1, 1), dims=(-1, -2))
    image_g = image_g + torch.roll(image_g, shifts=1, dims=-1)
    image_b = image_b + torch.roll(image_b, shifts=1, dims=-2) + torch.roll(image_b, shifts=1, dims=-1) \
    + torch.roll(image_b, shifts=(1, 1), dims=(-1, -2))


    demosaic_image = torch.concat([image_r, image_g, image_b], dim=1)

    return demosaic_image