from PIL import Image
import numpy as np
import math

def cal_psnr(img1, img2, data_range):
   '''
   calculate the psnr of two images
   Args:
      img1, img2:  ndarray, m x n x 3
      data_range:  maximum possible pixel value of the image
      return:      100 if the two images are nearly the same; else psnr (<=100)
   '''
   assert img1.shape == img2.shape
   mse = np.mean( (img1/data_range - img2/data_range) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1.
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

if __name__ == '__main__':
   img1 = Image.open("C:/Users/Rose Xue/Desktop/background.jpg")
   img2 = Image.open("C:/Users/Rose Xue/Desktop/background2.jpg")
   print(cal_psnr(np.array(img1)/255., np.array(img2)/255., 1))