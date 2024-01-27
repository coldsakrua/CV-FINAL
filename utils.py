import cv2
import numpy as np
from PIL import Image
# from scipy.signal import convolve2d as conv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class TVLoss(nn.Module):
    def __init__(self, weight: float=1, beta: float = 1) -> None:
        """Total Variation Loss

        Args:
            weight (float): weight of TV loss
        """
        super().__init__()
        self.weight = weight
        self.beta = beta
    
    def forward(self, x):
        batch_size, c, h, w = x.size()
        tv_h = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).sum()
        tv_w = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).sum()
        res = self.weight * (tv_h + tv_w) / (batch_size * c * h * w)
        return res ** (self.beta)


def conv2(in1, in2, mode='same'):
    in1_torch = torch.tensor(in1).unsqueeze(0).unsqueeze(0).float()
    in2_torch = in2.unsqueeze(0).unsqueeze(0).float()

    if torch.cuda.is_available():
        in1_torch = in1_torch.cuda()
        in2_torch = in2_torch.cuda()

    # PyTorch uses 'valid' mode by default
    if mode == 'full':
        out = F.conv2d(in1_torch, in2_torch, padding=in2_torch.shape[-1] - 1)
    elif mode == 'same':
        pad = max(in2_torch.shape) // 2
        out = F.conv2d(in1_torch, in2_torch, padding=pad)
    else:
        out = F.conv2d(in1_torch, in2_torch)

    return out.squeeze().cpu().detach().numpy()



def bgr2gray(image):
    return 0.114*image[:,:,0]+0.587*image[:,:,1]+0.299*image[:,:,2]

def cal_blur(img, FiltSize=9):

    if len(img.shape)==3:
        img = bgr2gray(img)

    m, n = img.shape[0],img.shape[1]
    Hv = 1.0/FiltSize*torch.ones((1,FiltSize),requires_grad=True)
    Hh = Hv.T
    Bver = conv2(img, Hv, 'same')
    Bhor = conv2(img, Hh, 'same')
    s_ind = int(np.ceil(FiltSize/2))
    e_ind = int(np.floor(FiltSize/2))
    
    Bver = Bver[s_ind:m-e_ind, s_ind:n-e_ind]
    Bhor = Bhor[s_ind:m-e_ind, s_ind:n-e_ind]
    img = img[s_ind:m-e_ind, s_ind:n-e_ind]
    m, n = img.shape[0],img.shape[1]

    Hv = torch.tensor([[1., -1.]],requires_grad=True)
    Hh = Hv.T
    D_Fver = abs(conv2(img, Hv, 'same')).squeeze()
    D_Fhor = abs(conv2(img, Hh, 'same')).squeeze()
    D_Fver = D_Fver[1:m-1, 1:n-1]
    D_Fhor = D_Fhor[1:m-1, 1:n-1]

    D_Bver = abs(conv2(Bver, Hv, 'same')).squeeze()
    D_Bhor = abs(conv2(Bhor, Hh, 'same')).squeeze()
    D_Bver = D_Bver[1:m-1, 1:n-1]
    D_Bhor = D_Bhor[1:m-1, 1:n-1]


    D_Vver = D_Fver-D_Bver
    D_Vver[D_Vver<0] = 0
    D_Vhor = D_Fhor-D_Bhor
    D_Vhor[D_Vhor<0] = 0

    s_Fver = np.sum(D_Fver)
    s_Fhor = np.sum(D_Fhor)
    s_Vver = np.sum(D_Vver)
    s_Vhor = np.sum(D_Vhor)

    b_Fver = (s_Fver - s_Vver)/s_Fver
    b_Fhor = (s_Fhor - s_Vhor)/s_Fhor

    IDM = max(b_Fver, b_Fhor)

    return IDM


def MLVMap(im):

    if len(im.shape)==3:
        im = bgr2gray(im)

    xs, ys = im.shape
    x=im

    x1=np.zeros((xs,ys))
    x2=np.zeros((xs,ys))
    x3=np.zeros((xs,ys))
    x4=np.zeros((xs,ys))
    x5=np.zeros((xs,ys))
    x6=np.zeros((xs,ys))
    x7=np.zeros((xs,ys))
    x8=np.zeros((xs,ys))
    x9=np.zeros((xs,ys))

    x1[1:xs-2,1:ys-2] = x[2:xs-1,2:ys-1]
    x2[1:xs-2,2:ys-1] = x[2:xs-1,2:ys-1]
    x3[1:xs-2,3:ys]   = x[2:xs-1,2:ys-1]
    x4[2:xs-1,1:ys-2] = x[2:xs-1,2:ys-1]
    x5[2:xs-1,2:ys-1] = x[2:xs-1,2:ys-1]
    x6[2:xs-1,3:ys]   = x[2:xs-1,2:ys-1]
    x7[3:xs,1:ys-2]   = x[2:xs-1,2:ys-1]
    x8[3:xs,2:ys-1]   = x[2:xs-1,2:ys-1]
    x9[3:xs,3:ys]     = x[2:xs-1,2:ys-1]

    x1=x1[2:xs-1,2:ys-1]
    x2=x2[2:xs-1,2:ys-1]
    x3=x3[2:xs-1,2:ys-1]
    x4=x4[2:xs-1,2:ys-1]
    x5=x5[2:xs-1,2:ys-1]
    x6=x6[2:xs-1,2:ys-1]
    x7=x7[2:xs-1,2:ys-1]
    x8=x8[2:xs-1,2:ys-1]
    x9=x9[2:xs-1,2:ys-1]

    d1=x1-x5
    d2=x2-x5
    d3=x3-x5
    d4=x4-x5
    d5=x6-x5
    d6=x7-x5
    d7=x8-x5
    d8=x9-x5

    dd=np.maximum(d1,d2)
    dd=np.maximum(dd,d3)
    dd=np.maximum(dd,d4)
    dd=np.maximum(dd,d5)
    dd=np.maximum(dd,d6)
    dd=np.maximum(dd,d7)
    dd=np.maximum(dd,d8)

    return dd

def cal_sharp(img):
    
    # img=np.array(Image.open(address))
    T=1000
    alpha=-0.01

    im_map = MLVMap(img)
    xs, ys = im_map.shape

    xy_number=xs*ys
    l_number=int(xy_number)
    vec = np.reshape(im_map,(xy_number))
    vec=sorted(vec.tolist(),reverse = True)
    svec=np.array(vec[1:l_number])

    a=range(1,xy_number)
    q=np.exp(np.dot(alpha,a))
    svec=svec*q
    svec=svec[1:T]
    sigma = np.sqrt(np.mean(np.power(svec,2)))

    return sigma


def name2add(name):
    
    t = 1
    mask_address = None
    if name == 'f16':
        address = './data/denoising/F16_GT.png'
    elif name == 'snail':
        address = './data/denoising/snail.jpg'
    elif name == 'kate':
        address = './data/inpainting/kate.png'
        mask_address = './data/inpainting/kate_mask.png'
    elif name == 'library':
        address = './data/inpainting/library.png'
        mask_address = './data/inpainting/library_mask.png'
    elif name == 'vase':
        address = './data/inpainting/vase.png'
        mask_address = './data/inpainting/vase_mask.png'
    elif name == 'zebra':
        address = './data/sr/zebra_crop.png'
        t = 4
    elif name == 'wps':
        address = './data/denoising/wps.png'
    elif name == 'man':
        address = './data/denoising/man.png'
    elif name == 'house':
        address = './data/denoising/house.png'
    elif name == 'barbara':
        address = './data/denoising/barbara.png'
    elif name == 'inpainting_img1':
        address = './data/inpainting/own_imgs_1.png'
        mask_address = './data/inpainting/own_imgs_1_mask.png'
    elif name == 'inpainting_img2':
        address = './data/inpainting/own_imgs_2.png'
        mask_address = './data/inpainting/own_imgs_2_mask.png'
    elif name == 'inpainting_img3':
        address = './data/inpainting/own_imgs_3.png'
        mask_address = './data/inpainting/own_imgs_3_mask.png'
    elif name == 'inpainting_img4':
        address = './data/inpainting/own_imgs_4.png'
        mask_address = './data/inpainting/own_imgs_4_mask.png'
    elif name == 'denoise_img1':
        address = './data/denoising/own_imgs_1.png'
    elif name == 'denoise_img2':
        address = './data/denoising/own_imgs_2.png'
    elif name == 'boat':
        address = './data/denoising/boat.png'
    elif name == 'baby':
        address = './data/sr/LRbicx4/baby.png'
        mask_address = './data/sr/original/baby.png'
        t = 4
    elif name == 'boat_sr':
        address = './data/sr/boat_sr.png'
        t = 4
    elif name == 'woman':
        address = './data/sr/LRbicx4/woman.png'
        mask_address = './data/sr/original/woman.png'
        t = 4
    elif name == 'man_sr':
        address = './data/sr/man_sr.png'
        t = 4
    elif name == 'peppers_sr':
        address = './data/sr/peppers.png'
        t = 4
    elif name == 'indian_sr':
        address = './data/sr/indian.png'
        t = 4
    elif name == 'house_sr':
        address = './data/sr/house.png'
        t = 4
    elif name == 'montage_sr':
        address = './data/sr/montage.png'
        t = 4
    elif name == 'cv':
        address = './data/sr/cv_sr.jpg'
        t = 4
    elif name == 'butterfly':
        address = './data/sr/LRbicx4/butterfly.png'
        mask_address = './data/sr/original/butterfly.png'
        t = 4
    elif name == 'bird':
        address = './data/sr/LRbicx4/bird.png'
        mask_address = './data/sr/original/bird.png'
        t = 4
    elif name == 'head':
        address = './data/sr/LRbicx4/head.png'
        mask_address = './data/sr/original/head.png'
        t = 4
    return address, mask_address, t    


def pepper_and_salt(img,percentage):
    num=int(percentage*img.shape[0]*img.shape[1])
    random.randint(0, img.shape[0])
    img2=img.copy()
    for i in range(num):
        X=random.randint(0,img2.shape[0]-1)
        Y=random.randint(0,img2.shape[1]-1)
        if random.randint(0,1) ==0: 
            img2[X,Y] = np.array([255,255,255])
        else:
            img2[X,Y] = np.array([0,0,0])
        img2[X,Y] = np.array([0,0,0])
    return img2

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



if __name__=='__main__':
    img=np.array(Image.open('./data/denoising/F16_GT.png'))
    print(cal_sharp(img), cal_blur(img))