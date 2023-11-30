#####################################################
#@ Author: MinhNH
#@ Create date: Nov 29, 2023
#@ Last modified: Nov 30, 2023
#####################################################

from typing import Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t
import torch.nn.functional as F
from torch import Tensor

class GaussConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, mu = 0, sigma = 1, stride: _size_2_t = 1, padding: _size_2_t | str = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', normalize = True, device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.weight.data = torch.zeros_like(self.weight.data)
        weight_shape = self.weight.shape
        self.gauss_weight = nn.Parameter(self.get_gaussian_kernel(weight_shape[1], weight_shape[0], kernel_size, mu, sigma, normalize), requires_grad= False)

    def forward(self, x: Tensor):
        weight = self.weight + self.gauss_weight
        return F.conv2d(x, weight= weight, bias = self.bias, stride= self.stride, padding= self.padding, dilation= self.dilation, groups= self.groups)

    def get_gaussian_kernel(self, in_channels, out_channels, k=3, mu=0, sigma=1, normalize=True):
        # compute 1 dimension gaussian
        gaussian_1D = np.linspace(-1, 1, k)
        # compute a grid distance from center
        x, y = np.meshgrid(gaussian_1D, gaussian_1D)
        distance = (x ** 2 + y ** 2) ** 0.5
        # compute the 2 dimension gaussian
        gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
        gaussian_2D = gaussian_2D / (2 * np.pi * sigma ** 2)

        # normalize part (mathematically)
        if normalize:
            gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
        
        gaussian_2D = torch.from_numpy(gaussian_2D).float().unsqueeze(0).unsqueeze(0)
        gaussian_2D = gaussian_2D.expand(out_channels, in_channels, k, k)
        return gaussian_2D
    
class SobelConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: _size_2_t | str = 0, dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, transpose = False, padding_mode: str = 'zeros', device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.weight.data = torch.zeros_like(self.weight.data)
        self.transpose = transpose
        weight_shape = self.weight.shape
        self.sobel_weight = nn.Parameter(self.get_sobel_kernel(weight_shape[1], weight_shape[0], kernel_size, transpose), requires_grad= False)

    def forward(self, x: Tensor):
        weight = self.weight + self.sobel_weight
        return F.conv2d(x, weight= weight, bias = self.bias, stride= self.stride, padding= self.padding, dilation= self.dilation, groups= self.groups)   
    
    def get_sobel_kernel(self, in_channels, out_channels, k=3, transpose = False):
        # get range
        range = np.linspace(-(k // 2), k // 2, k)
        # compute a grid the numerator and the axis-distances
        x, y = np.meshgrid(range, range)
        sobel_2D_numerator = x
        sobel_2D_denominator = (x ** 2 + y ** 2)
        sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
        sobel_2D = sobel_2D_numerator / sobel_2D_denominator

        sobel_2D = torch.from_numpy(sobel_2D).float().unsqueeze(0).unsqueeze(0)
        if transpose:
            sobel_2D = sobel_2D.transpose(-1, -2)
        sobel_2D = sobel_2D.expand(out_channels, in_channels, k, k)
        return sobel_2D

class PermuteConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels= in_channels, kernel_size= 1, stride= 1, padding= 0, dilation= 1, groups= 1, bias= False, padding_mode= 'zeros', device=device, dtype=dtype)
        self.weight = nn.Parameter(torch.eye(in_channels).unsqueeze(-1).unsqueeze(-1), requires_grad= True)

    
class NMSConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, device=None, dtype=None) -> None:
        super().__init__(in_channels, out_channels= 8*in_channels, kernel_size= 3, stride= 1, padding= 1, dilation= 1, groups= 1, bias= False, padding_mode= 'zeros', device=device, dtype=dtype)
        canny_nms = torch.tensor([
                    [[[0.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 0.0]]],
                    [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]],
                    [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]],
                    [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]],
                    [[[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
                    [[[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
                    [[[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
                    [[[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]],
                ])
        nms_kernels = nn.Parameter(torch.cat([torch.cat([canny_nms]*self.in_channels, dim= 0)]*(self.in_channels // self.groups), dim= 1),
                requires_grad= False)
        self.weight = nms_kernels

class HystersisConv2d(nn.Conv2d):
    def __init__(self, device=None, dtype=None) -> None:
        super().__init__(in_channels= 1, out_channels= 1, kernel_size= 3, stride= 1, padding= 1, dilation= 1, groups= 1, bias= False, padding_mode= 'reflect', device=device, dtype=dtype)
        hyst_kernel = torch.ones((1,1, 3,3), requires_grad= False).add(0.25)
        self.weight = nn.Parameter(hyst_kernel, requires_grad= False)


class CannyDetector(nn.Module):
    '''
    Trainable Canny Edge Detector followed https://discuss.pytorch.org/t/trying-to-train-parameters-of-the-canny-edge-detection-algorithm/154517 implementation.\n
    Modified to support any input channels (not just only 1) and add a support for arbitrary number of hidden channels. This model will have much higher flexibility 
    compared to the OP.\n
    Non-maximal suppression is taken from Kornia implementation https://kornia.readthedocs.io/en/latest/_modules/kornia/filters/canny.html#canny .\n
    Notes:
    - Due to atan2 operator cannot export to TFLite without extending default operators set, I approximate this function by a tanh and a dereived sigmoid function,
      which is well-supported as popular activation functions. 
    - The average error I get is around 0.0041 rad which is acceptable in my case, you can fallback to torch.atan2() if you don't need to export the model to TFLite 
      format. 
    '''
    def __init__(self,
                 in_channels=3,
                 hidden_channels=8,
                 k_gaussian=3,
                 k_sobel = 3,
                 mu=0,
                 sigma=3):
        super(CannyDetector, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.eps = 1e-10

        self.conv_expand = nn.Conv2d(in_channels= in_channels,
                                   out_channels= self.hidden_channels,
                                   kernel_size=1,
                                   bias= False)
        
        self.conv_expand.weight.data = torch.softmax(torch.rand_like(self.conv_expand.weight.data, requires_grad= True),dim = 0).clone()

        # Gaussian filter
        self.conv_gauss = GaussConv2d(  in_channels=self.hidden_channels,
                                        out_channels=self.hidden_channels,
                                        kernel_size=k_gaussian,
                                        padding=k_gaussian // 2,
                                        mu = mu,
                                        sigma= sigma,
                                        padding_mode="reflect",
                                        bias=False,
                                        groups=self.hidden_channels,)
        self.conv_permute_gauss = PermuteConv2d(in_channels= self.hidden_channels)

        # Sobel filter x direction
        self.conv_sobel_x = SobelConv2d(in_channels=self.hidden_channels,
                                        out_channels=self.hidden_channels,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        padding_mode="reflect",
                                        bias=False,
                                        groups= self.hidden_channels,
                                        transpose= False)
        self.conv_permute_sobel_x = PermuteConv2d(in_channels= self.hidden_channels)

        # Sobel filter y directions
        self.conv_sobel_y = SobelConv2d(in_channels=self.hidden_channels,
                                        out_channels=self.hidden_channels,
                                        kernel_size=k_sobel,
                                        padding=k_sobel // 2,
                                        padding_mode="reflect",
                                        bias=False,
                                        groups= self.hidden_channels,
                                        transpose= True)
        self.conv_permute_sobel_y = PermuteConv2d(in_channels= self.hidden_channels)

        # Hysteresis custom kernel
        self.conv_hystersis = HystersisConv2d()

        self.conv_merge = nn.Conv2d(in_channels= self.hidden_channels,
                                    out_channels= 1,
                                    kernel_size= 1,
                                    bias= False)

        # Threshold parameters
        self.lowThreshold  = nn.Parameter(torch.ones(self.hidden_channels,1,1).mul(0.10), requires_grad=True)
        self.highThreshold = nn.Parameter(torch.ones(self.hidden_channels,1,1).mul(0.20), requires_grad=True)

        self.conv_nms = NMSConv2d(in_channels= self.hidden_channels)
        


    def get_gaussian_kernel(self, in_channels, out_channels, k=3, mu=0, sigma=1, normalize=True):
        # compute 1 dimension gaussian
        gaussian_1D = np.linspace(-1, 1, k)
        # compute a grid distance from center
        x, y = np.meshgrid(gaussian_1D, gaussian_1D)
        distance = (x ** 2 + y ** 2) ** 0.5
        # compute the 2 dimension gaussian
        gaussian_2D = np.exp(-(distance - mu) ** 2 / (2 * sigma ** 2))
        gaussian_2D = gaussian_2D / (2 * np.pi * sigma ** 2)

        # normalize part (mathematically)
        if normalize:
            gaussian_2D = gaussian_2D / np.sum(gaussian_2D)
        
        gaussian_2D = torch.from_numpy(gaussian_2D).float().unsqueeze(0).unsqueeze(0)
        gaussian_2D = gaussian_2D.expand(out_channels, in_channels, k, k)
        return gaussian_2D

    def get_sobel_kernel(self, in_channels, out_channels, k=3):
        # get range
        range = np.linspace(-(k // 2), k // 2, k)
        # compute a grid the numerator and the axis-distances
        x, y = np.meshgrid(range, range)
        sobel_2D_numerator = x
        sobel_2D_denominator = (x ** 2 + y ** 2)
        sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
        sobel_2D = sobel_2D_numerator / sobel_2D_denominator

        sobel_2D = torch.from_numpy(sobel_2D).float().unsqueeze(0).unsqueeze(0)
        sobel_2D = sobel_2D.expand(out_channels, in_channels, k, k)
        return sobel_2D

    def threshold(self, img):
        """ Thresholds for defining weak and strong edge pixels """

        alpha = 1000
        weak = 0.5
        strong = 1

        res_strong = strong * torch.sigmoid(alpha * (img - self.highThreshold))
        res_weak_1 = weak * torch.sigmoid(alpha * (self.highThreshold - img))
        res_weak_2 = weak * torch.sigmoid(alpha * (self.lowThreshold - img))
        res_weak = res_weak_1 - res_weak_2
        res = res_weak + res_strong

        return res

    def hysteresis(self,img):

        # Create image that has strong pixels remain at one, weak pixels become zero
        img_strong = (img == 1.).clone()*img
        # Create masked image that turns all weak pixel into ones, rest to zeros
        masked_img = (img == 0.5).clone()
        # Calculate weak edges that are changed to strong edges + Add changed edges to already good edges
        changed_edges = (self.conv_hystersis(img_strong) > 1)*masked_img*img + img_strong

        return changed_edges

    def non_maximal_suppression(self, sobel_x, sobel_y, magnitude = None):
        # Kornia elegant implementation of non_maxiaml_suppression with high speed
        # https://kornia.readthedocs.io/en/latest/_modules/kornia/filters/canny.html#canny

        # Non-maximal suppression
        nms_magnitude: Tensor = self.conv_nms(magnitude)

        if magnitude is None:
            magnitude = torch.sqrt(torch.pow(sobel_x + self.eps, 2) + torch.pow(sobel_y + self.eps, 2))

        # By MinhNH: Approximate arctan by tanh function and symmetrical error function to reduce computational cost and able to export to TFLite
        diff = (sobel_y + self.eps)/(sobel_x + self.eps)
        magnified_diff = 1.2276*diff
        sigmoid_m_diff = torch.sigmoid(1.2276*diff)
        angle = torch.pi/2 * torch.tanh(0.2714 * diff) + 1.8022*magnified_diff*sigmoid_m_diff*(1-sigmoid_m_diff)

        # angle  = torch.atan2(sobel_y + self.eps, sobel_x + self.eps)

        # Radians to Degrees
        angle = 180.0 * angle / torch.pi
        # print(angle.shape)

        # Round angle to the nearest 45 degree
        angle = torch.round(angle / 45) * 45

        # Get the indices for both directions
        positive_idx: Tensor = (angle / 45) % 8
        positive_idx = positive_idx.long()

        negative_idx: Tensor = ((angle / 45) + 4) % 8
        negative_idx = negative_idx.long()

        

        # Apply the non-maximum suppression to the different directions
        channel_select_filtered_positive: Tensor = torch.gather(nms_magnitude, 1, positive_idx)
        channel_select_filtered_negative: Tensor = torch.gather(nms_magnitude, 1, negative_idx)

        
        channel_select_filtered: Tensor = torch.stack(
            [channel_select_filtered_positive, channel_select_filtered_negative], 1
        )

        is_max: Tensor = channel_select_filtered.min(dim=1)[0] > 0.0
        
        magnitude = magnitude * is_max
        return magnitude

    def forward(self, x: torch.Tensor):
        x = self.conv_expand(x)
        # Gaussian filter
        x = self.conv_gauss(x)
        x = self.conv_permute_gauss(x)

        # Sobel filter
        sobel_x = self.conv_sobel_x(x)
        sobel_y = self.conv_sobel_y(x)
        sobel_x = self.conv_permute_sobel_x(sobel_x)
        sobel_y = self.conv_permute_sobel_y(sobel_y)
        # Magnitude and angles
        
        grad_magnitude = torch.sqrt(torch.pow(sobel_x + self.eps, 2) + torch.pow(sobel_y + self.eps, 2))



        # Non-max-suppression
        thin_edges = self.non_maximal_suppression(sobel_x, sobel_y, grad_magnitude)
        # thin_edges = grad_magnitude


        # Double threshold
        thin_edges = thin_edges / torch.max(thin_edges)
        thresh = self.threshold(thin_edges)

        thresh = self.conv_merge(thresh)
        # Hysteresis
        result = self.hysteresis(thresh)

        return result

