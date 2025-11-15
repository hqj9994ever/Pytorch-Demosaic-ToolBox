# coding=utf-8
import argparse
from utils import utils_image as util
from utils import utils_isp
from utils import utils_mat as mat

import numpy as np
import os
import torch
import rawpy


DEMO_LIST = ['Nearest', 'Bilinear', 'Malvar', 'HA', 'RI', 'MLRI', 'AHD', 'AHD_DualDn', 'DLMMSE']


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render sRGB from RAW."
    )
    # I/O
    parser.add_argument('--img_path', type=str, default='Raw/huawei_mate60.dng',
                        help='Path to the input RAW/DNG file.')
    parser.add_argument('--save_dir', type=str, default='sRGB',
                        help='Directory to save rendered sRGB images.')

    # Camera / profile
    parser.add_argument('--camera_type', type=str, default='huawei_mate60',
                        help='Camera type used to select the profile .mat.')
    parser.add_argument('--profile_type', type=str, default=None,
                        help='Profile type (.mat name). Defaults to camera_type if not set.')
    parser.add_argument('--tonecurve_idx', type=int, default=-1,
                        help='Tone curve index (set -1 for auto by camera type).')

    # ISP options
    parser.add_argument('--fm_weight', type=float, default=1.0,
                        help='Weight to blend ForwardMatrix1/2 in [0,1].')
    parser.add_argument('--demosaic', type=str, default='all',
                        choices=['all'] + DEMO_LIST,
                        help="Choose one demosaicing method or 'all' to run every method.")
    parser.add_argument('--exposure', type=float, default=None,
                        help='Override BaselineExposure. None = use from profile.')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Select device. "auto" uses CUDA if available.')

    return parser.parse_args()


def auto_tonecurve_idx(profile_type: str) -> int:
    if profile_type == 'huawei_v8':
        return 74
    elif profile_type == 'huawei_mate60':
        return 0
    elif profile_type == 'iPhone13pro':
        return 23
    elif profile_type in ['huawei_p20', 'huawei_mate30']:
        return 127
    elif profile_type == 'huawei_p30':
        return 126
    elif profile_type == 'canon_s90':
        return 71
    elif profile_type == 'canon_t3i':
        return 184
    elif profile_type == 'xiaomi12':
        return 167
    elif 'nikon' in profile_type or 'canon' in profile_type or 'olympus' in profile_type:
        return 127
    else:
        return 127


if __name__ == '__main__':
    args = parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    img_path = args.img_path
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # ---------------------------
    # load camera profile & tone
    # ---------------------------
    CameraType = args.camera_type
    ProfileType = args.profile_type or CameraType

    CameraProfile = mat.loadmat(os.path.join('cameraprofile', ProfileType + '.mat'))

    # tone curve
    ToneCurve_idx = args.tonecurve_idx if args.tonecurve_idx >= 0 else auto_tonecurve_idx(ProfileType)
    ToneCurves = mat.loadmat(os.path.join('cameraprofile', 'tonecurves.mat'))['ToneCurves']
    ToneCurve = ToneCurves[ToneCurve_idx, :]
    ToneCurve = np.reshape(ToneCurve, (2, -1), 'F')
    ToneCurveX, ToneCurveY = ToneCurve[0, :], ToneCurve[1, :]

    # ---------------------------
    # load profile
    # ---------------------------
    BlackLevel = CameraProfile['BlackLevel'][0]
    WhiteLevel = CameraProfile['WhiteLevel']
    BaselineExposure = CameraProfile['BaselineExposure'] if CameraProfile['BaselineExposure'] is not None else 0
    if args.exposure is not None:
        BaselineExposure = args.exposure

    ForwardMatrix1 = torch.from_numpy(CameraProfile['ForwardMatrix1']).float().reshape(3, 3) \
        if CameraProfile['ForwardMatrix1'] is not None else None
    ForwardMatrix2 = torch.from_numpy(CameraProfile['ForwardMatrix2']).float().reshape(3, 3) \
        if CameraProfile['ForwardMatrix2'] is not None else None
    CameraCalibration1 = torch.from_numpy(CameraProfile['CameraCalibration1']).float().reshape(3, 3) \
        if CameraProfile['CameraCalibration1'] is not None \
        else torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    # ---------------------------
    # read RAW
    # ---------------------------
    img_name, ext = os.path.splitext(os.path.basename(img_path))
    raw = rawpy.imread(img_path)
    raw_image = raw.raw_image
    color_desc = 'RGGB'

    # white balance
    AsShotCameraNeutral = 1. / np.array(raw.camera_whitebalance[:3])
    AsShotCameraNeutral = torch.FloatTensor(AsShotCameraNeutral).reshape(3, 1)

    # forward matrices
    NeedDemosaic = True
    ForwardMatrixWeightFactor = args.fm_weight

    # RAW2XYZ(D50)
    D = (1 / AsShotCameraNeutral / CameraCalibration1.diag()).diag()
    if 'iPhone' in CameraType:
        color_matrix = raw.color_matrix[0:3, 0:3]  # raw2srgb matrix
        xyz_srgb_matrix = np.array([[3.2404542, -1.5371385, -0.4985314],
                                    [-0.9692660, 1.8760108, 0.0415560],
                                    [0.0556434, -0.2040259, 1.0572252]], dtype=np.float32)
        srgb_xyz_matrix = np.linalg.inv(xyz_srgb_matrix)
        raw_xyz_matrix = np.matmul(srgb_xyz_matrix, color_matrix)
        CameraToXYZ_D50 = torch.from_numpy(raw_xyz_matrix)
    else:
        CameraToXYZ_D50 = ForwardMatrixWeightFactor * ForwardMatrix1 + (1 - ForwardMatrixWeightFactor) * ForwardMatrix2

    # normalize to [0,1]
    raw_image = (raw_image.astype(np.float32) - BlackLevel) / (WhiteLevel - BlackLevel)

    # CFA arrangement (RGGB variants)
    if (raw.raw_pattern == np.array([[2, 3], [1, 0]], dtype='uint8')).all():  # BGGR
        raw_image = raw_image[1:-1, 1:-1, np.newaxis]
    elif (raw.raw_pattern == np.array([[0, 1], [3, 2]], dtype='uint8')).all():  # RGGB
        raw_image = raw_image[..., np.newaxis]
    elif (raw.raw_pattern == np.array([[3, 2], [0, 1]], dtype='uint8')).all():  # GBRG
        raw_image = raw_image[1:-1, ..., np.newaxis]
    elif (raw.raw_pattern == np.array([[1, 0], [2, 3]], dtype='uint8')).all():  # GRBG
        raw_image = raw_image[..., 1:-1, np.newaxis]

    raw_image = util.modcrop(raw_image, scale=32)
    H, W, _ = raw_image.shape
    tile = torch.tensor([[0, 1], [2, 3]])
    color_mask = tile.tile(H // 2, W // 2).unsqueeze(0).unsqueeze(0)
    raw_image = util.single2tensor4(raw_image).to(device)
    raw_image = torch.clamp(raw_image, min=0.0, max=1.0)

    # save rotation/flip mode
    if raw.sizes.flip == 6:
        mode = 3
    elif raw.sizes.flip == 3:
        mode = 6
    else:
        mode = 0

    # choose demosaic list
    demo_list = DEMO_LIST if args.demosaic == 'all' else [args.demosaic]

    for item in demo_list:
        isp = utils_isp.ISP(
            CameraToXYZ_D50, ToneCurveX, ToneCurveY,
            NeedDemosaic=NeedDemosaic, demosaic=item,
            Exposure=BaselineExposure,
            R_gain=D[0], G_gain=D[1], B_gain=D[2],
            Inflection=1.0
        ).to(device)

        img_L_sRGB = isp.forward(raw_image, color_desc, color_mask, CameraToXYZ_D50.unsqueeze(0))
        out = util.augment_img(util.tensor2uint(img_L_sRGB), mode)
        util.imsave(out, os.path.join(save_dir, f'{img_name}_{item}.png'))

    print(f'Done. Results saved to: {save_dir}')
