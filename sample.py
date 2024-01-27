import numpy as np
from skimage import transform
from PIL import Image
import torch.nn as nn
import torch

def upsample1(image, w, h, order):
    """
    Upsampling by using interpolation

    Args:
        address (str): the address of image
        w1, h1(int): target width and height
        order (int): the degree of interpolation polynomial
    """
    image = image[0]
    res = transform.resize(image, (w, h), order=order, anti_aliasing=True, mode='constant')
    return res

def upsample(image, times, order):
    """
    Upsampling by using interpolation

    Args:
        address (str): the address of image
        times (int): magnification factor
        order (int): the degree of interpolation polynomial
    """
    w, h =image.shape[1], image.shape[2]
    image = image[0]
    res = transform.resize(image, (times * w, times * h), order=order, anti_aliasing=True, mode='constant')
    return res

class downsample(nn.Module):
    def __init__(self, input_channels, factor, kernel_mode, device):
        super(downsample, self).__init__()
        self.kernel = self.get_kernel(factor, input_channels, kernel_mode).to(device)
        self.factor = factor
        self.kernel_mode = kernel_mode
        
    def get_kernel(self, factor, input_channels, kernel_mode):
        assert kernel_mode in ['box', 'gaussian', 'tent']
        
        
        if kernel_mode == 'box':
            kernel_size = factor
            res = torch.zeros(size=(input_channels, input_channels, kernel_size, kernel_size))
            kernel = torch.ones(size=(kernel_size, kernel_size)).float()
            kernel /= kernel_size*kernel_size
            
            
        elif kernel_mode == 'gaussian':
            kernel_size = factor * 2 - 1
            sigma = 0.5
            res = torch.zeros(size=(input_channels, input_channels, kernel_size, kernel_size))
            kernel = torch.zeros(size=(kernel_size, kernel_size))
            mid = float((kernel_size - 1) / 2)
            for i in range(0, kernel_size):
                for j in range(0, kernel_size):
                    di = i * 1. - mid
                    dj = j * 1. - mid
                    kernel[i, j] = torch.exp(torch.tensor(-(di**2 + dj**2 ) / (2 * sigma * sigma)))
            kernel /= torch.sum(kernel)
            
        elif kernel_mode == 'tent':
            kernel_size = factor
            res = torch.zeros(size=(input_channels, input_channels, kernel_size, kernel_size))
            kernel = torch.ones(size=(factor, factor))
            n = int((factor + 1) / 2)
            for i in range(1, n):
                m = 2 * n + 1
                kernel[i: factor-i, i: factor-i] *= m
            kernel /= torch.sum(kernel)
        for i in range(input_channels):
            res[i, i] = kernel
        return res
    
    def forward(self, x):
        # print(type(x), type(self.kernel))
        if self.kernel_mode == 'gaussian':
            down = nn.functional.conv2d(x, self.kernel, bias=None, stride=self.factor, padding=int(self.factor/2))
            # print(down.shape)
        else:
            down = nn.functional.conv2d(x, self.kernel, bias=None, stride=self.factor, padding=0)
        # print(self.kernel)
        return down
    
class smooth(nn.Module):
    def __init__(self,in_channels, device):
        super(smooth, self).__init__()
        self.channels = in_channels
        self.kernel = self.get_kernel().to(device)
        
    def get_kernel(self):
        kernel_size = 3
        res = torch.zeros(size=(self.channels, self.channels, kernel_size, kernel_size))
        kernel = torch.ones(size=(kernel_size, kernel_size))
        mid = int(kernel_size / 2)
        kernel[mid, mid] = torch.tensor(20)
        kernel /= kernel.sum()
        # print(kernel)
        for i in range(self.channels):
            res[i, i] = kernel
        return res
    
    def forward(self, x):
        smoothness = nn.functional.conv2d(x, self.kernel, stride=1, padding=int(self.kernel.shape[-1]/2))
        return smoothness
    
    
class sharp(nn.Module):
    def __init__(self, in_channels, device):
        super(sharp, self).__init__()
        self.channels = in_channels
        self.kernel = self.get_kernel().to(device)
                
    def get_kernel(self):
        kernel_size = 3
        res = torch.zeros(size=(self.channels, self.channels, kernel_size, kernel_size))
        kernel = torch.ones(size=(kernel_size, kernel_size))* -.1
        mid = int(kernel_size / 2)
        kernel[mid, mid] = torch.tensor(1.8)
        # kernel /= kernel.sum()
        # print(kernel)
        for i in range(self.channels):
            res[i, i] = kernel.float()
        return res
    
    def forward(self, x):
        sharpness = nn.functional.conv2d(x, self.kernel, stride=1, padding=1)
        return sharpness