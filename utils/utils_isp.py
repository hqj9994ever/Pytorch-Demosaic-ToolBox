# coding=utf-8

import numpy as np
from scipy.interpolate import interp1d
import torch
import torch.nn as nn

from utils import utils_color

from Nearest import Nearest
from Bilinear import Bilinear
from DLMMSE import DLMMSE
from AHD import AHD
from AHD_DualDn import AHD_DualDn
from GBTF import GBTF
from HA import HA
from Malvar import Malvar
from MLRI import MLRI
from RI import RI



class Demosaic(nn.Module):
    """ matlab demosaicking
    Args:
        x: Nx1xWxH with RGGB bayer pattern

    Returns:
        output: Nx3xWxH
    """
    def __init__(self, NeedDemosaic=True, mode='AHD'): # Malvar, GBTF, AHD, RI, DLMMSE, HA, MLRI
        super(Demosaic, self).__init__()
        self.NeedDemosaic = NeedDemosaic
        self.mode = mode

    def forward(self, x, color_desc=None, color_mask=None, rgb_xyz_matrix=None):  
        if self.NeedDemosaic:
            if self.mode == 'Nearest':
                output = Nearest(image=x, color_desc=color_desc, color_mask=color_mask, rgb_xyz_matrix=rgb_xyz_matrix)

            elif self.mode == 'Bilinear':
                output = Bilinear(image=x, color_desc=color_desc, color_mask=color_mask, rgb_xyz_matrix=rgb_xyz_matrix)

            elif self.mode == 'Malvar':
                output = Malvar(x)

            elif self.mode == 'AHD':
                output = AHD(x)
            
            elif self.mode == 'AHD_DualDn':
                output = AHD_DualDn(image=x, color_desc=color_desc, color_mask=color_mask, rgb_xyz_matrix=rgb_xyz_matrix)
            
            elif self.mode == 'HA':
                output = HA(x)

            elif self.mode == 'GBTF':
                output = GBTF(x)
            
            elif self.mode == 'RI':
                output = RI(x)

            elif self.mode == 'DLMMSE':
                output = DLMMSE(x)

            elif self.mode == 'MLRI':
                output = MLRI(x)


        else:
            output = torch.cat((x[:, 0:1, 0::2, 0::2], (x[:, 0:1, 0::2, 1::2] + x[:, 0:1, 1::2, 0::2]) / 2, x[:, 0:1, 1::2, 1::2]), 1)
        return output.clamp_(0, 1)
    
    

class ExposureCompensationWhiteBalance(nn.Module):
    '''
    Exposure Compensation and white balance 
    Exposure: BaselineExposure: from camera profile
    ''' 
    def __init__(self, Exposure=0.0, R_gain=torch.FloatTensor([1.5]), G_gain=torch.FloatTensor([1.0]), B_gain=torch.FloatTensor([1.5]), Inflection=0.9):
        super(ExposureCompensationWhiteBalance, self).__init__()
        self.exposure = Exposure
        self.red_gain = R_gain
        self.green_gain = G_gain
        self.blue_gain = B_gain
        self.Inflection = Inflection
        self.register_buffer('gains', torch.tensor([1.0 / self.red_gain, 1.0 / self.green_gain, 1.0 / self.blue_gain]).view(1, 3, 1, 1).div(2**self.exposure))

    def forward(self, image):

        if self.Inflection < 1.0:
            assert image.dim() == 4 and image.shape[1] == 3, "The input must be Nx3xHxW"
            gains = self.gains.to(image.device)

            gray = image.mean(dim=1, keepdim=True)  
            inflection = self.Inflection
            mask = (torch.clamp(gray - inflection, min=0.0) / (1.0 - inflection)) ** 2.0 

            safe_gains = torch.max(mask + (1.0 - mask) * gains, gains) 
            out = image / safe_gains
        else:
            out = image / self.gains

        return out.clamp_(0, 1)

    

class Raw2XYZ(nn.Module):
    """
    camera raw (after demosaicing+exposure&white gain) --> XYZ(D50)
    """  
    def __init__(self, weight):
        super(Raw2XYZ, self).__init__()
        weight_inv = torch.inverse(weight).unsqueeze(-1).unsqueeze(-1)
        self.register_buffer('weight', weight.unsqueeze(-1).unsqueeze(-1))
        self.register_buffer('weight_inv', weight_inv)

    def forward(self, x):
        return torch.matmul(x.permute(0, 2, 3, 1), self.weight.squeeze().t()).permute(0, 3, 1, 2)



class XYZ2LinearRGB(nn.Module):
    """
    XYZ(D50) --> linear sRGB(D65)
    """
    def __init__(self):
        super(XYZ2LinearRGB, self).__init__()
        weight = utils_color.xyz2linearrgb_weight(0, is_forward=True)
        weight_inv = torch.inverse(weight).unsqueeze(-1).unsqueeze(-1)
        self.register_buffer('weight', weight.unsqueeze(-1).unsqueeze(-1))
        self.register_buffer('weight_inv', weight_inv)

    def forward(self, x):
        return nn.functional.conv2d(x, self.weight, bias=None)


class ToneMapping(nn.Module):
    '''
    Differentiable Tone Mapping
    Args:
        ToneCurveX: numpy array of x-coordinates for tone curve
        ToneCurveY: numpy array of y-coordinates for tone curve
        delta: sampling interval for interpolation points (used for initialization)
    '''
    def __init__(self, ToneCurveX, ToneCurveY, delta=1e-6):
        super(ToneMapping, self).__init__()
        self.delta = delta
        # Generate interpolation points
        xi = np.linspace(0, 1, num=int(1/delta+1), endpoint=True)
        yi = interp1d(ToneCurveX, ToneCurveY, kind='cubic')(xi)
        yi_inv = interp1d(yi, xi, kind='cubic')(xi)
        
        # Register interpolation points as buffers
        self.register_buffer('xi', torch.from_numpy(xi).float())
        self.register_buffer('yi', torch.from_numpy(yi).float())
        self.register_buffer('yi_inv', torch.from_numpy(yi_inv).float())

    def forward(self, x):
        '''
        Apply tone mapping using differentiable linear interpolation
        Input: tensor of shape [N, C, H, W] or [N, H, W]
        Output: tensor of same shape with tone mapping applied
        '''
        
        # Normalize input to [0, len(xi)-1] for interpolation
        idx = x / self.delta
        
        # Perform linear interpolation using torch.lerp
        idx_floor = torch.floor(idx).long()
        idx_ceil = idx_floor + 1
        # Clamp indices to valid range
        idx_floor = torch.clamp(idx_floor, 0, len(self.xi) - 2)
        idx_ceil = torch.clamp(idx_ceil, 0, len(self.xi) - 1)
        
        # Get corresponding y values
        y_floor = self.yi[idx_floor]
        y_ceil = self.yi[idx_ceil]
        
        # Compute interpolation weights
        weight = idx - idx_floor.float()
        
        # Interpolate
        out = y_floor + (y_ceil - y_floor) * weight
        
        return torch.clamp(out, 0, 1)


class GammaCorrect(nn.Module):
    """
    Gamma correction
    linear RGB --> sRGB
    """
    def __init__(self):
        super(GammaCorrect, self).__init__()

    def forward(self, x):
        x = torch.clamp(x, min=1e-8) ** (1.0 / 2.2)
        return x.clamp_(0, 1)



            
class ISP(nn.Module):
    def __init__(self, weight_raw2xyz, ToneCurveX, ToneCurveY, NeedDemosaic=True, demosaic='AHD', Exposure=0, R_gain=2.0, G_gain=1.0, B_gain=1.8, Inflection=0.9):
        super(ISP, self).__init__()
        self.demosaic = Demosaic(NeedDemosaic=NeedDemosaic, mode=demosaic)
        self.exposurecompensationwhitebalance = ExposureCompensationWhiteBalance(Exposure=Exposure, R_gain=R_gain, G_gain=G_gain, B_gain=B_gain, Inflection=Inflection)
        self.raw2xyz = Raw2XYZ(weight=weight_raw2xyz)
        self.xyz2linearrgb = XYZ2LinearRGB()
        self.tonemapping = ToneMapping(ToneCurveX=ToneCurveX, ToneCurveY=ToneCurveY)
        self.gammacorrect = GammaCorrect()

    def forward(self, x, color_desc=None, color_mask=None, rgb_xyz_matrix=None):
        x = self.demosaic.forward(x, color_desc, color_mask, rgb_xyz_matrix)
        x = self.exposurecompensationwhitebalance.forward(x)
        x = self.raw2xyz.forward(x)
        x = self.xyz2linearrgb.forward(x)
        x = self.tonemapping.forward(x)
        x = self.gammacorrect.forward(x)

        return x
    



