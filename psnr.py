from PIL import Image
import numpy as np
import math

def cal_psnr(img1, img2):
   '''
   img1, img2:  m x n x 3, input images
   return:      100 if the two images are nearly the same; else psnr (<=100)
   '''
   arr1 = np.array(img1)
   arr2 = np.array(img2)
   assert arr1.shape == arr2.shape
   mse = np.mean( (arr1/255. - arr2/255.) ** 2 )
   if mse < 1.0e-10:
      return 100
   PIXEL_MAX = 1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

if __name__ == '__main__':
   img1 = Image.open("background.jpg")
   img2 = Image.open("background2.jpg")
   print(cal_psnr(img1, img2))