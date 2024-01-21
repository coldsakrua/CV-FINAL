import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import argparse
from torch import optim
import matplotlib.pyplot as plt
import skimage.transform as transform
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from model import Model



if __name__ == "__main__":
    device='cuda'
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--task', type=str, default='denoise')
    parser.add_argument('--name', type=str, default='f16')
    parser.add_argument('--epoch',type=int,default='2400')
    args = parser.parse_args()
    if args.name == 'f16':
        address = './data/denoising/F16_GT.png'
    elif args.name == 'snail':
        address = './data/denoising/snail.jpg'
    elif args.name == 'kate':
        address = './data/inpainting/kate.png'
        mask_address = './data/inpainting/kate_mask.png'
    image = Image.open(address)
    w, h = image.size
    image = image.resize((w - w % 32, h - h % 32), resample=Image.LANCZOS)
    # image = image.resize((512, 512), resample=Image.LANCZOS)
    image = torch.from_numpy(np.array(image) / 255.0).unsqueeze(0).float()
    if args.task == 'denoise':
        # print(image.shape)
        channels = 32
        model = Model(image.shape[1], image.shape[2],input_channel=32).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.MSELoss()
        corrupted_img = (image + torch.randn_like(image) * .1).clip(0, 1)
        corrupted_img = corrupted_img.transpose(2, 3).transpose(1, 2)
        z = torch.rand([1, channels, corrupted_img.shape[-2], corrupted_img.shape[-1]]) * 0.1
        z = z.to(device)
        # model = Model(input_channel=32, w = corrupted_img.shape[-2],h=corrupted_img.shape[-1]).to(device)
        corrupted_img = corrupted_img.to(device)
    elif args.task == 'inpaint':
        loss_fn = torch.nn.MSELoss()
        mask = Image.open(mask_address)
        mask = image.resize((512, 512), resample=Image.LANCZOS)
        img_with_mask = image * mask
        
        
    elif args.task=='super':
        loss_fn = torch.nn.MSELoss()
    # print(corrupted_img.shape)
    for epoch in tqdm(range(args.epoch)):
        if args.task == 'denoise':
            noise = torch.randn([1, channels, corrupted_img.shape[-2], corrupted_img.shape[-1]],device=device, requires_grad=True)/30
            input = z + noise
            img_pred = model.forward(input)
            loss = loss_fn(img_pred, corrupted_img)    
        
        if args.task == 'super':
            pass
        
        elif args.task == 'inpaint':
            pass
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.figure(figsize=(18, 3.5))
    plt.subplot(1, 3, 1)
    corr_img=corrupted_img[0].cpu().transpose(0, 1).transpose(1, 2).data.numpy(), 
    plt.imshow(corr_img[0])
    # print(corr_img[0].shape)
    plt.imsave(f'./Imgs/{args.name}_input{str(args.epoch)}.png', corr_img[0])
    plt.title('Input', fontsize=15)
    plt.subplot(1, 3, 2)
    pred=img_pred[0].cpu().transpose(0, 1).transpose(1, 2).data.numpy(), 
    plt.imshow(pred[0])
    plt.imsave(f'./Imgs/{args.name}_{args.task}{str(args.epoch)}.png', pred[0])
    plt.title('Prediction', fontsize=15)
    plt.subplot(1, 3, 3)
    origin = image[0].data.numpy()
    plt.imshow(origin)
    plt.imsave(f'./Imgs/{args.name}_gt.png', origin)
    plt.title('Ground truth', fontsize=15)
    plt.savefig(f'Imgs/{args.name}.png')
    
    # python main.py --task=denoise --name=f16 --epoch=3000