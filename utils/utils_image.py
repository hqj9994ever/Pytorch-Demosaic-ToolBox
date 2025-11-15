import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
# from torchvision.utils import make_grid
from datetime import datetime
# import torchvision.transforms as transforms
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


'''
# --------------------------------------------
# Kai Zhang (github: https://github.com/cszn)
# 03/Mar/2019
# --------------------------------------------
# https://github.com/twhui/SRGAN-pyTorch
# https://github.com/xinntao/BasicSR
# --------------------------------------------
'''


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', '.dng', '.DNG', '.raw', '.mat', '.CR2']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def imshow(x, title=None, cbar=False, figsize=None):
    plt.figure(figsize=figsize)
    plt.imshow(np.squeeze(x), interpolation='nearest', cmap='gray')
    if title:
        plt.title(title)
    if cbar:
        plt.colorbar()
    plt.show()


def surf(Z, cmap='rainbow', figsize=None):
    plt.figure(figsize=figsize)
    ax3 = plt.axes(projection='3d')

    w, h = Z.shape[:2]
    xx = np.arange(0,w,1)
    yy = np.arange(0,h,1)
    X, Y = np.meshgrid(xx, yy)
    ax3.plot_surface(X,Y,Z,cmap=cmap)
    #ax3.contour(X,Y,Z, zdim='z',offset=-2，cmap=cmap)
    plt.show()


'''
# --------------------------------------------
# get image pathes
# --------------------------------------------
'''


def get_image_paths(dataroot):
    paths = None  # return None if dataroot is None
    if isinstance(dataroot, str):
        paths = sorted(_get_paths_from_images(dataroot))
    elif isinstance(dataroot, list):
        paths = []
        for i in dataroot:
            paths += sorted(_get_paths_from_images(i))
    return paths


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


'''
# --------------------------------------------
# split large images into small images 
# --------------------------------------------
'''


def patches_from_image(img, p_size=512, p_overlap=64, p_max=800):
    w, h = img.shape[:2]
    patches = []
    if w > p_max and h > p_max:
        w1 = list(np.arange(0, w-p_size, p_size-p_overlap, dtype=np.int))
        h1 = list(np.arange(0, h-p_size, p_size-p_overlap, dtype=np.int))
        w1.append(w-p_size)
        h1.append(h-p_size)
        for i in w1:
            for j in h1:
                patches.append(img[i:i+p_size, j:j+p_size,:])
    else:
        patches.append(img)

    return patches


def imssave(imgs, img_path):
    """
    imgs: list, N images of size WxHxC
    """
    img_name, ext = os.path.splitext(os.path.basename(img_path))
    for i, img in enumerate(imgs):
        if img.ndim == 3:
            img = img[:, :, [2, 1, 0]]
        new_path = os.path.join(os.path.dirname(img_path), img_name+str('_{:04d}'.format(i))+'.png')
        cv2.imwrite(new_path, img)


def split_imageset(original_dataroot, taget_dataroot, n_channels=3, p_size=512, p_overlap=96, p_max=800):
    """
    split the large images from original_dataroot into small overlapped images with size (p_size)x(p_size), 
    and save them into taget_dataroot; only the images with larger size than (p_max)x(p_max)
    will be splitted.

    Args:
        original_dataroot:
        taget_dataroot:
        p_size: size of small images
        p_overlap: patch size in training is a good choice
        p_max: images with smaller size than (p_max)x(p_max) keep unchanged.
    """
    paths = get_image_paths(original_dataroot)
    for img_path in paths:
        # img_name, ext = os.path.splitext(os.path.basename(img_path))
        img = imread_uint(img_path, n_channels=n_channels)
        patches = patches_from_image(img, p_size, p_overlap, p_max)
        imssave(patches, os.path.join(taget_dataroot, os.path.basename(img_path)))
        #if original_dataroot == taget_dataroot:
        #del img_path

'''
# --------------------------------------------
# crop image from top left to bottom right 
# --------------------------------------------
'''

def crop_by_corners(img: np.ndarray,
                    top_left: tuple,
                    bottom_right: tuple,
                    *,
                    inclusive: bool = False,
                    allow_swap: bool = True,
                    clip: bool = True,
                    copy: bool = True) -> np.ndarray:
    """
    Crop an image using top-left and bottom-right corner coordinates.

    Parameters
    ----------
    img : np.ndarray
        Input array with shape [H, W] or [H, W, C].
    top_left : (x1, y1)
        Top-left corner (x = column index, y = row index).
    bottom_right : (x2, y2)
        Bottom-right corner (x = column index, y = row index).
    inclusive : bool
        If True, treat bottom_right as inclusive (i.e., add +1 to slice stop).
        If False, use half-open interval [y1:y2, x1:x2] (NumPy default).
    allow_swap : bool
        If True, automatically swap coordinates when they are reversed
        (i.e., use min/max to fix order).
    clip : bool
        If True, clip coordinates to image bounds; otherwise raise an error
        on out-of-bound coordinates.
    copy : bool
        If True, return a copy; if False, return a view (faster, less memory).

    Returns
    -------
    np.ndarray
        The cropped region.

    Raises
    ------
    ValueError
        If input dims are invalid, coordinates are out of bounds (when clip=False),
        or the resulting crop is empty.
    """
    if img.ndim not in (2, 3):
        raise ValueError(f"img must be 2D or 3D, got shape {img.shape}")

    H, W = img.shape[:2]

    x1, y1 = map(int, top_left)
    x2, y2 = map(int, bottom_right)

    if allow_swap:
        x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
        y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)

    # 右下角是否“包含”
    if inclusive:
        x2 += 1
        y2 += 1

    if clip:
        x1 = max(0, min(x1, W))
        x2 = max(0, min(x2, W))
        y1 = max(0, min(y1, H))
        y2 = max(0, min(y2, H))
    else:
        if not (0 <= x1 <= W and 0 <= x2 <= W and 0 <= y1 <= H and 0 <= y2 <= H):
            raise ValueError(f"Coordinates out of bounds for image size (W={W}, H={H}).")

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Empty crop: got x in [{x1}, {x2}) and y in [{y1}, {y2}).")

    cropped = img[y1:y2, x1:x2]
    return cropped.copy() if copy else cropped


@torch.no_grad()
def process_image_in_patches_v1(image, model, patch_size=512, overlap=32):
    """
    Process large image by splitting into patches and merging results
    Args:
        image: input tensor image (1, 1, H, W)
        model: denoising model
        patch_size: size of patches
        overlap: overlap between patches
    Returns:
        result: processed image
    """
    _, _, h, w = image.shape
    stride = patch_size - overlap

    # Calculate padding needed to make image divisible by stride
    pad_h = stride - (h - patch_size) % stride if (h - patch_size) % stride != 0 else 0
    pad_w = stride - (w - patch_size) % stride if (w - patch_size) % stride != 0 else 0

    # Pad image
    image_padded = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
    _, _, h_pad, w_pad = image_padded.shape

    # Initialize output tensor
    output = torch.zeros_like(image_padded).to(image.device)
    count = torch.zeros_like(image_padded).to(image.device)

    # Process each patch
    for i in range(0, h_pad - patch_size + 1, stride):
        for j in range(0, w_pad - patch_size + 1, stride):
            patch = image_padded[:, :, i:i+patch_size, j:j+patch_size]

            # Process patch
            processed_patch = model(patch)

            # Add patch to output with weighted overlap
            weight = torch.ones_like(processed_patch)
            if i > 0:
                weight[:, :, :overlap//2, :] *= torch.linspace(0, 1, overlap//2).view(1, 1, -1, 1).to(image.device)
            if i < h_pad - patch_size:
                weight[:, :, -overlap//2:, :] *= torch.linspace(1, 0, overlap//2).view(1, 1, -1, 1).to(image.device)
            if j > 0:
                weight[:, :, :, :overlap//2] *= torch.linspace(0, 1, overlap//2).view(1, 1, 1, -1).to(image.device)
            if j < w_pad - patch_size:
                weight[:, :, :, -overlap//2:] *= torch.linspace(1, 0, overlap//2).view(1, 1, 1, -1).to(image.device)


            output[:, :, i:i+patch_size, j:j+patch_size] += processed_patch * weight
            count[:, :, i:i+patch_size, j:j+patch_size] += weight

    # Normalize by count
    output = output / count

    # Remove padding
    output = output[:, :, :h, :w]

    # print(output.shape)

    return output  


@torch.no_grad()
def process_image_in_patches_v2(
    image: torch.Tensor,
    model,
    patch_size: int = 512,
    overlap: int = 32,
    *,
    pad_mode: str = "reflect",
):
    """
    Process a large image by tiling it into overlapping patches and
    stitching model outputs (which may be upsampled) with smooth blending.

    Args:
        image: input tensor with shape (1, C_in, H, W). For your case, C_in=8.
        model: a callable mapping (1, C_in, h, w) -> (1, C_out, scale*h, scale*w).
               For your case, C_out=3 and scale=2.
        patch_size: tile size on the INPUT resolution.
        overlap: overlap (in INPUT pixels) between adjacent tiles.
        pad_mode: padding mode passed to F.pad (e.g., "reflect", "replicate", "constant").

    Returns:
        Tensor of shape (1, C_out, scale*H, scale*W).
    """
    assert image.dim() == 4 and image.size(0) == 1, "image must be (1, C_in, H, W)"
    assert 0 <= overlap < patch_size, "overlap must be in [0, patch_size)"

    device = image.device
    _, _, H, W = image.shape
    stride = patch_size - overlap

    # --- Compute padded full size so that tiles cover the whole image ---
    # number of steps per dimension (at least 1)
    steps_h = max(1, math.ceil((H - patch_size) / stride) + 1)
    steps_w = max(1, math.ceil((W - patch_size) / stride) + 1)
    H_pad = (steps_h - 1) * stride + patch_size
    W_pad = (steps_w - 1) * stride + patch_size

    pad_bottom = H_pad - H
    pad_right  = W_pad - W
    # pad format: (left, right, top, bottom)
    image_padded = F.pad(image, (0, pad_right, 0, pad_bottom), mode=pad_mode)

    # Will be determined on first patch run
    output = None
    count  = None
    out_scale = None
    C_out = None

    # Precompute linear ramps for blending in OUTPUT space (scaled by out_scale later)
    # We will build them lazily after we know out_scale.
    lin_cache = {}

    for i in range(0, H_pad - patch_size + 1, stride):
        for j in range(0, W_pad - patch_size + 1, stride):
            patch = image_padded[:, :, i:i + patch_size, j:j + patch_size]  # (1, C_in, ps, ps)

            # Forward the model
            patch_out = model(patch)  # expected (1, C_out, ps*out_scale, ps*out_scale)

            if output is None:
                # Infer scale and channels from first output
                _, C_out, oh, ow = patch_out.shape
                # robust integer scale (assumes square pixels)
                out_scale_h = oh // patch_size
                out_scale_w = ow // patch_size
                assert out_scale_h == out_scale_w and oh % patch_size == 0 and ow % patch_size == 0, \
                    f"Cannot infer integer scale from patch_out shape {patch_out.shape} and patch_size={patch_size}"
                out_scale = out_scale_h

                # Allocate output canvas and counter at OUTPUT resolution
                full_oh = H_pad * out_scale
                full_ow = W_pad * out_scale
                output = torch.zeros((1, C_out, full_oh, full_ow), device=device)
                count  = torch.zeros_like(output)

                # Prepare 1D ramps (in OUTPUT pixels)
                ovr_out = overlap * out_scale  # width of overlap band in output space
                # We'll use half on each touching edge (top/bottom, left/right)
                # Ensure even split
                half = max(0, ovr_out // 2)
                if half > 0:
                    lin_cache["up"]    = torch.linspace(0, 1, steps=half, device=device).view(1, 1, -1, 1)
                    lin_cache["down"]  = torch.linspace(1, 0, steps=half, device=device).view(1, 1, -1, 1)
                    lin_cache["left"]  = torch.linspace(0, 1, steps=half, device=device).view(1, 1, 1, -1)
                    lin_cache["right"] = torch.linspace(1, 0, steps=half, device=device).view(1, 1, 1, -1)
                lin_cache["half"] = half
                lin_cache["ovr_out"] = ovr_out

            # Build weight mask = 1s, then taper on overlapping bands (OUTPUT space)
            weight = torch.ones_like(patch_out, device=device)
            half = lin_cache["half"]

            oi = i * out_scale
            oj = j * out_scale
            ps_out = patch_size * out_scale

            # Vertical blends
            if half > 0:
                # touch top neighbor?
                if i > 0:
                    weight[:, :, :half, :] *= lin_cache["up"]
                # touch bottom neighbor?
                if i < H_pad - patch_size:
                    weight[:, :, -half:, :] *= lin_cache["down"]
                # Horizontal blends
                if j > 0:
                    weight[:, :, :, :half] *= lin_cache["left"]
                if j < W_pad - patch_size:
                    weight[:, :, :, -half:] *= lin_cache["right"]

            # Accumulate
            output[:, :, oi:oi + ps_out, oj:oj + ps_out] += patch_out * weight
            count[:,  :, oi:oi + ps_out, oj:oj + ps_out] += weight

    # Normalize (avoid divide-by-zero if any)
    output = output / count.clamp_min(1e-8)

    # Crop back to the original OUTPUT size (scale * H/W before padding)
    out_h = H * out_scale
    out_w = W * out_scale
    output = output[:, :, :out_h, :out_w]
    return output

'''
# --------------------------------------------
# makedir
# --------------------------------------------
'''


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


'''
# --------------------------------------------
# read image from path
# opencv is fast, but read BGR numpy image
# --------------------------------------------
'''


# --------------------------------------------
# get uint8 image of size HxWxn_channles (RGB)
# --------------------------------------------
def imread_uint(path, n_channels=3):
    #  input: path
    # output: HxWx3(RGB or GGG), or HxWx1 (G)
    if n_channels == 1:
        img = cv2.imread(path, 0)  # cv2.IMREAD_GRAYSCALE
        img = np.expand_dims(img, axis=2)  # HxWx1
    elif n_channels == 3:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGR or G
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # GGG
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    return img


# --------------------------------------------
# matlab's imwrite
# --------------------------------------------
def imsave(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)


def imwrite(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]
    cv2.imwrite(img_path, img)



# --------------------------------------------
# get single image of size HxWxn_channles (BGR)
# --------------------------------------------
def read_img(path):
    # read image by cv2
    # return: Numpy float32, HWC, BGR, [0,1]
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # cv2.IMREAD_GRAYSCALE
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


'''
# --------------------------------------------
# image format conversion
# --------------------------------------------
# numpy(single) <--->  numpy(uint)
# numpy(single) <--->  tensor
# numpy(uint)   <--->  tensor
# --------------------------------------------
'''


# --------------------------------------------
# numpy(single) [0, 1] <--->  numpy(uint)
# --------------------------------------------


def uint2single(img):

    return np.float32(img/255.)


def single2uint(img):

    return np.uint8((img.clip(0, 1)*255.).round())


def uint162single(img):

    return np.float32(img/65535.)


def single2uint16(img):

    return np.uint16((img.clip(0, 1)*65535.).round())


# --------------------------------------------
# numpy(uint) (HxWxC or HxW) <--->  tensor
# --------------------------------------------


# convert uint to 4-dimensional torch tensor
def uint2tensor4(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)


# convert uint to 3-dimensional torch tensor
def uint2tensor3(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.)


# convert 2/3/4-dimensional torch tensor to uint
def tensor2uint(img, keep_dim=False):
    if keep_dim:
        img = img.data.float().clamp_(0, 1).cpu().numpy()
    else:
        img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 4:
        img = np.transpose(img, (0, 2, 3, 1))
    return np.uint8((img*255.0).round())


# --------------------------------------------
# numpy(single) (HxWxC) <--->  tensor
# --------------------------------------------


# convert single (HxWxC) to 3-dimensional torch tensor
def single2tensor3(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


# convert single (HxWxC) or (BxHxWxC) to 4-dimensional torch tensor
def single2tensor4(img):
    if img.ndim == 3:
        return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)
    elif img.ndim == 4:
        return torch.from_numpy(np.ascontiguousarray(img)).permute(0, 3, 1, 2).float()


# convert torch tensor to single
def tensor2single(img, keep_dim=False):
    if keep_dim:
        img = img.data.float().cpu().numpy()
    else:
        img = img.data.squeeze().float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 4:
        img = np.transpose(img, (0, 2, 3, 1))

    return img

# convert torch tensor to single
def tensor2single3(img, keep_dim=False):
    if keep_dim:
        img = img.data.float().cpu().numpy()
    else:
        img = img.data.squeeze().float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    elif img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img


def single2tensor5(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1, 3).float().unsqueeze(0)


def single32tensor5(img):
    return torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0).unsqueeze(0)


def single42tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1, 3).float()


# from skimage.io import imread, imsave
# def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
#     '''
#     Converts a torch Tensor into an image Numpy array of BGR channel order
#     Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
#     Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
#     '''
#     tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # squeeze first, then clamp
#     tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
#     n_dim = tensor.dim()
#     if n_dim == 4:
#         n_img = len(tensor)
#         img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
#         img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
#     elif n_dim == 3:
#         img_np = tensor.numpy()
#         img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
#     elif n_dim == 2:
#         img_np = tensor.numpy()
#     else:
#         raise TypeError(
#             'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
#     if out_type == np.uint8:
#         img_np = (img_np * 255.0).round()
#         # Important. Unlike matlab, numpy.uint8() WILL NOT round by default.
#     return img_np.astype(out_type)


'''
# --------------------------------------------
# Augmentation, flipe and/or rotate
# --------------------------------------------
# The following two are enough.
# (1) augmet_img: numpy image of WxHxC or WxH
# (2) augment_img_tensor4: tensor image 1xCxWxH
# --------------------------------------------
'''


def augment_img(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))


def augment_img_tensor4(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return img
    elif mode == 1:
        return img.rot90(1, [2, 3]).flip([2])
    elif mode == 2:
        return img.flip([2])
    elif mode == 3:
        return img.rot90(3, [2, 3])
    elif mode == 4:
        return img.rot90(2, [2, 3]).flip([2])
    elif mode == 5:
        return img.rot90(1, [2, 3])
    elif mode == 6:
        return img.rot90(2, [2, 3])
    elif mode == 7:
        return img.rot90(3, [2, 3]).flip([2])


def augment_img_tensor(img, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    img_size = img.size()
    img_np = img.data.cpu().numpy()
    if len(img_size) == 3:
        img_np = np.transpose(img_np, (1, 2, 0))
    elif len(img_size) == 4:
        img_np = np.transpose(img_np, (2, 3, 1, 0))
    img_np = augment_img(img_np, mode=mode)
    img_tensor = torch.from_numpy(np.ascontiguousarray(img_np))
    if len(img_size) == 3:
        img_tensor = img_tensor.permute(2, 0, 1)
    elif len(img_size) == 4:
        img_tensor = img_tensor.permute(3, 2, 0, 1)

    return img_tensor.type_as(img)


def augment_img_np3(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return img.transpose(1, 0, 2)
    elif mode == 2:
        return img[::-1, :, :]
    elif mode == 3:
        img = img[::-1, :, :]
        img = img.transpose(1, 0, 2)
        return img
    elif mode == 4:
        return img[:, ::-1, :]
    elif mode == 5:
        img = img[:, ::-1, :]
        img = img.transpose(1, 0, 2)
        return img
    elif mode == 6:
        img = img[:, ::-1, :]
        img = img[::-1, :, :]
        return img
    elif mode == 7:
        img = img[:, ::-1, :]
        img = img[::-1, :, :]
        img = img.transpose(1, 0, 2)
        return img


def augment_imgs(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


'''
# --------------------------------------------
# modcrop and shave
# --------------------------------------------
'''


def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img


def crop_from_top_left(img_in, h, w):
    H, W = img_in.shape[0], img_in.shape[1]
    
    if h > H or w > W:
        h = H
        w = W
    
    if img_in.ndim == 2: 
        img = img_in[:h, :w]
    else:  
        img = img_in[:h, :w, :]
    
    return img


def shave(img_in, border=0):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    h, w = img.shape[:2]
    img = img[border:h-border, border:w-border]
    return img


'''
# --------------------------------------------
# image processing process on numpy image
# channel_convert(in_c, tar_type, img_list):
# rgb2ycbcr(img, only_y=True):
# bgr2ycbcr(img, only_y=True):
# ycbcr2rgb(img):
# --------------------------------------------
'''


def rgb2ycbcr(img, only_y=False):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    rlt = np.clip(rlt, 0, 255)
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=False):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


# def rgb2yuv(img, only_y=False):
#     '''convert rgb to yuv
#     only_y: only return Y channel
#     Input:
#         uint8, [0, 255]
#         float, [0, 1]
#     '''
#     in_img_type = img.dtype
#     img.astype(np.float32)
#     if in_img_type != np.uint8:
#         img *= 255.
#     # convert
#     if only_y:
#         rlt = np.dot(img, [0.299, 0.587, 0.114])
#     else:
#         rlt = np.matmul(img, [[0.299, -0.14713, 0.615], [0.587, -0.28886, -0.51499],
#                               [0.114, 0.436, -0.10001]]) + [0., 128., 128.]
#     if in_img_type == np.uint8:
#         rlt = rlt.round()
#     else:
#         rlt /= 255.
#     return rlt.astype(in_img_type)


def rgb2yuv(img, only_y=False):
    '''
    Convert RGB to YUV for PyTorch tensor
    Input shape: [N, 3, H, W]
    Input range: uint8 [0, 255] or float [0, 1]
    Output: same type as input, YUV color space
    Args:
        img: Input tensor of shape [N, 3, H, W]
        only_y: If True, return only Y channel
    Returns:
        YUV tensor or Y channel tensor with same type as input
    '''
    in_dtype = img.dtype
    is_uint8 = in_dtype == torch.uint8
    
    # Convert to float for computation
    img = img.float()
    if not is_uint8:
        img *= 255.0
    
    # Define conversion matrices
    if only_y:
        # Y channel weights
        weights = torch.tensor([0.299, 0.587, 0.114], device=img.device).view(1, 3, 1, 1)
        rlt = torch.sum(img * weights, dim=1, keepdim=True)
    else:
        # Full YUV conversion matrix
        weights = torch.tensor([
            [0.299, -0.14713, 0.615],
            [0.587, -0.28886, -0.51499],
            [0.114, 0.436, -0.10001]
        ], device=img.device).view(3, 3, 1, 1)
        # Add bias for U and V channels
        bias = torch.tensor([0., 128., 128.], device=img.device).view(1, 3, 1, 1)
        # print(img.shape)
        rlt = torch.matmul(img.permute(0, 2, 3, 1), weights.view(3, 3)).permute(0, 3, 1, 2) + bias
    
    # Convert back to original type
    if is_uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.0
    
    return rlt.to(in_dtype)


def yuv2rgb(img):
    '''
    Convert YUV to RGB for PyTorch tensor
    Input shape: [N, 3, H, W]
    Input range: uint8 [0, 255] or float [0, 1]
    Output: same type as input, YUV color space
    Args:
        img: Input tensor of shape [N, 3, H, W]
        only_y: If True, return only Y channel
    Returns:
        YUV tensor or Y channel tensor with same type as input
    '''
    in_dtype = img.dtype
    is_uint8 = in_dtype == torch.uint8
    
    # Convert to float for computation
    img = img.float()
    if not is_uint8:
        img *= 255.0
    
    # Define conversion matrices
    
    weights = torch.tensor([
        [1., 1., 1.],
        [0., -0.39465, 2.03211],
        [1.13983, -0.58060, 0]
    ], device=img.device).view(3, 3, 1, 1)
    # Add bias for U and V channels
    bias = torch.tensor([-145.89824, 124.832, -260.11008], device=img.device).view(1, 3, 1, 1)
    # print(img.shape)
    rlt = torch.matmul(img.permute(0, 2, 3, 1), weights.view(3, 3)).permute(0, 3, 1, 2) + bias
    
    # Convert back to original type
    if is_uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.0
    
    return rlt.to(in_dtype)


# def yuv2rgb(img):
#     '''convert yuv to rgb
#     Input:
#         uint8, [0, 255]
#         float, [0, 1]
#     '''
#     in_img_type = img.dtype
#     img.astype(np.float32)
#     if in_img_type != np.uint8:
#         img *= 255.
#     # convert
#     rlt = np.matmul(img, [[1., 1., 1.], [0., -0.39465, 2.03211],
#                           [1.13983, -0.58060, 0]]) + [-145.89824, 124.832, -260.11008]     
#     rlt = np.clip(rlt, 0, 255)
#     if in_img_type == np.uint8:
#         rlt = rlt.round()
#     else:
#         rlt /= 255.
#     return rlt.astype(in_img_type)


def channel_convert(in_c, tar_type, img_list):
    # conversion among BGR, gray and y
    if in_c == 3 and tar_type == 'gray':  # BGR to gray
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in gray_list]
    elif in_c == 3 and tar_type == 'y':  # BGR to y
        y_list = [bgr2ycbcr(img, only_y=True) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in y_list]
    elif in_c == 1 and tar_type == 'RGB':  # gray/y to BGR
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
    else:
        return img_list


'''
# --------------------------------------------
# metric, PSNR, SSIM and PSNRB
# --------------------------------------------
'''


# --------------------------------------------
# PSNR
# --------------------------------------------
def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def _blocking_effect_factor(im):
    block_size = 8

    block_horizontal_positions = torch.arange(7, im.shape[3] - 1, 8)
    block_vertical_positions = torch.arange(7, im.shape[2] - 1, 8)

    horizontal_block_difference = (
                (im[:, :, :, block_horizontal_positions] - im[:, :, :, block_horizontal_positions + 1]) ** 2).sum(
        3).sum(2).sum(1)
    vertical_block_difference = (
                (im[:, :, block_vertical_positions, :] - im[:, :, block_vertical_positions + 1, :]) ** 2).sum(3).sum(
        2).sum(1)

    nonblock_horizontal_positions = np.setdiff1d(torch.arange(0, im.shape[3] - 1), block_horizontal_positions)
    nonblock_vertical_positions = np.setdiff1d(torch.arange(0, im.shape[2] - 1), block_vertical_positions)

    horizontal_nonblock_difference = (
                (im[:, :, :, nonblock_horizontal_positions] - im[:, :, :, nonblock_horizontal_positions + 1]) ** 2).sum(
        3).sum(2).sum(1)
    vertical_nonblock_difference = (
                (im[:, :, nonblock_vertical_positions, :] - im[:, :, nonblock_vertical_positions + 1, :]) ** 2).sum(
        3).sum(2).sum(1)

    n_boundary_horiz = im.shape[2] * (im.shape[3] // block_size - 1)
    n_boundary_vert = im.shape[3] * (im.shape[2] // block_size - 1)
    boundary_difference = (horizontal_block_difference + vertical_block_difference) / (
                n_boundary_horiz + n_boundary_vert)

    n_nonboundary_horiz = im.shape[2] * (im.shape[3] - 1) - n_boundary_horiz
    n_nonboundary_vert = im.shape[3] * (im.shape[2] - 1) - n_boundary_vert
    nonboundary_difference = (horizontal_nonblock_difference + vertical_nonblock_difference) / (
                n_nonboundary_horiz + n_nonboundary_vert)

    scaler = np.log2(block_size) / np.log2(min([im.shape[2], im.shape[3]]))
    bef = scaler * (boundary_difference - nonboundary_difference)

    bef[boundary_difference <= nonboundary_difference] = 0
    return bef


def calculate_psnrb(img1, img2, border=0):
    """Calculate PSNR-B (Peak Signal-to-Noise Ratio).
    Ref: Quality assessment of deblocked images, for JPEG image deblocking evaluation
    # https://gitlab.com/Queuecumber/quantization-guided-ac/-/blob/master/metrics/psnrb.py
    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
    Returns:
        float: psnr result.
    """

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    if img1.ndim == 2:
        img1, img2 = np.expand_dims(img1, 2), np.expand_dims(img2, 2)

    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # follow https://gitlab.com/Queuecumber/quantization-guided-ac/-/blob/master/metrics/psnrb.py
    img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0) / 255.
    img2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0) / 255.

    total = 0
    for c in range(img1.shape[1]):
        mse = torch.nn.functional.mse_loss(img1[:, c:c + 1, :, :], img2[:, c:c + 1, :, :], reduction='none')
        bef = _blocking_effect_factor(img1[:, c:c + 1, :, :])

        mse = mse.view(mse.shape[0], -1).mean(1)
        total += 10 * torch.log10(1 / (mse + bef))

    return float(total) / img1.shape[1]

'''
# --------------------------------------------
# matlab's bicubic imresize (numpy and torch) [0, 1]
# --------------------------------------------
'''


# matlab 'imresize' function, now only support 'bicubic'
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5*absx3 - 2.5*absx2 + 1) * ((absx <= 1).type_as(absx)) + \
        (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * (((absx > 1)*(absx <= 2)).type_as(absx))


def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


# --------------------------------------------
# imresize for tensor image [0, 1]
# --------------------------------------------
def imresize(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: pytorch tensor, CHW or HW [0,1]
    # output: CHW or HW [0,1] w/o round
    need_squeeze = True if img.dim() == 2 else False
    if need_squeeze:
        img.unsqueeze_(0)
    in_C, in_H, in_W = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_C, out_H, in_W)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(out_C):
            out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_C, out_H, out_W)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(out_C):
            out_2[j, :, i] = out_1_aug[j, :, idx:idx + kernel_width].mv(weights_W[i])
    if need_squeeze:
        out_2.squeeze_()
    return out_2


# --------------------------------------------
# imresize for numpy image [0, 1]
# --------------------------------------------
def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC or HW [0,1]
    # output: HWC or HW [0,1] w/o round
    img = torch.from_numpy(img)
    need_squeeze = True if img.dim() == 2 else False
    if need_squeeze:
        img.unsqueeze_(2)

    in_H, in_W, in_C = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        for j in range(out_C):
            out_1[i, :, j] = img_aug[idx:idx + kernel_width, :, j].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        for j in range(out_C):
            out_2[:, i, j] = out_1_aug[:, idx:idx + kernel_width, j].mv(weights_W[i])
    if need_squeeze:
        out_2.squeeze_()

    return out_2.numpy()




if __name__ == '__main__':
    img = imread_uint('test.bmp', 3)
    img = uint2tensor4(img)
    img = rgb2yuv(img)
    img = tensor2single(img)
    img = yuv2rgb(img)
    imsave(single2uint(img), 'output.png')
#    img = uint2single(img)
#    img_bicubic = imresize_np(img, 1/4)
#    imshow(single2uint(img_bicubic))
#
#    img_tensor = single2tensor4(img)
#    for i in range(8):
#        imshow(np.concatenate((augment_img(img, i), tensor2single(augment_img_tensor4(img_tensor, i))), 1))
    
#    patches = patches_from_image(img, p_size=128, p_overlap=0, p_max=200)
#    imssave(patches,'a.png')


    
    
    
    
    
