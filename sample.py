import numpy as np
from skimage import transform
from PIL import Image
import torch.nn as nn

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
    def __init__(self, factor):
        super(downsample, self).__init__()