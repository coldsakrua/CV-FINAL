import cv2
import numpy as np
from PIL import Image


def bgr2gray(image):
    return 0.114*image[:,:,0]+0.587*image[:,:,1]+0.299*image[:,:,2]

def cal_sharp(address):
    image = np.array(Image.open(address))
    gray_image = bgr2gray(image)
    std_dev = np.std(gray_image)
    contrast = std_dev / 127
    return contrast


def cal_blur(address):
    image = np.array(Image.open(address))
    gray_image = bgr2gray(image)

    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    return laplacian_var

if __name__=='__main__':
    print(cal_sharp('./data/denoising/F16_GT.png'), cal_blur('./data/denoising/F16_GT.png'))