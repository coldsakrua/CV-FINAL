import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sample import downsample
import torch
from utils import *

def process(address,name):
    downsampler = downsample(3, 4, kernel_mode='gaussian', device = 'cuda')
    img = Image.open(address)
    w,h=img.size
    img=img.resize((w - w % 32, h - h % 32), resample=Image.LANCZOS)
    img = np.array(img)
    img = np.stack((img,)*3,axis=-1)
    img = torch.from_numpy(img/255.).unsqueeze(0).float().transpose(2,3).transpose(1,2).to('cuda')
    img1 = downsampler(img)
    img2 = img1[0].cpu().transpose(0,1).transpose(1,2).data.numpy()
    assert img2.shape[2]==3
    plt.imsave('./data/sr/{}_sr.png'.format(name),img2)
    
    
if __name__ =='__main__':
    ## Warning DO NOT RUN THIS FILE!!!!
    l = ['barbara','boat','couple', 'fingerprint','hill','house','indian','lena','man','montage','peppers']
    for name in l:
        address,_,_=name2add(name+'_sr')
        print(name)
        process(address,name)
    pass