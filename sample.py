import numpy as np
from skimage import transform
from PIL import Image
import torch.nn as nn
import torch

def upsample(address, times, order):
    """
    Upsampling by using interpolation

    Args:
        address (str): the address of image
        times (int): magnification factor
        order (int): the degree of interpolation polynomial
    """
    image = Image.open(address,'r')
    image = np.array(image)
    w, h =image.shape[0], image.shape[1]
    res = transform.resize(image, (times * w, times * h), order=order, anti_aliasing=True, mode='constant')
    return res

class downsample(nn.Module):
    def __init__(self, input_channels, factor, kernel_mode):
        super(downsample, self).__init__()
        self.kernel = self.get_kernel(factor, input_channels, kernel_mode)
        self.factor = factor

    def get_kernel(self, factor, input_channels, kernel_mode):
        assert kernel_mode in ['box', 'gaussian', 'tent']
        res = torch.zeros(size=(input_channels, input_channels, factor, factor))
        
        if kernel_mode == 'box':
            kernel = torch.zeros(size=(factor, factor))
            kernel /= factor*factor
            
            
        elif kernel_mode == 'gaussian':
            kernel = torch.zeros(size=(factor, factor))
            for i in range(0, factor):
                for j in range(0, factor):
                    kernel[i, j] = torch.exp(-((i - float(factor/2))**2 + (j - float(factor/2))**2 ) / 2)
            kernel /= sum(kernel)
            
        elif kernel_mode == 'tent':
            kernel = torch.ones(size=(factor, factor))
            n = int((factor + 1) / 2)
            for i in range(1, n):
                m = 2 * n + 1
                kernel[i: factor-i, i: factor-i] *= m
            kernel /= sum(kernel)
             
        res[:, :] = kernel
        return res
    
    def forward(self, x):
        down = nn.functional.conv2d(x, self.kernel, bias=False, stride=self.factor)
        return down