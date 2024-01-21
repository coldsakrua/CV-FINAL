import numpy as np
from skimage import transform
from PIL import Image

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

def downsample(image, size, order):
    res = transform.resize(image, size, order=order, anti_aliasing=True, mode='constant')
    return res