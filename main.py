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
from model_without_skip import Model_without_skip
from model_for_inpainting import Model_for_inpainting
from sample import upsample,downsample


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
    elif args.name == 'library':
        address = './data/inpainting/library.png'
        mask_address = './data/inpainting/library_mask.png'
    elif args.name == 'vase':
        address = './data/inpainting/vase.png'
        mask_address = './data/inpainting/vase_mask.png'
    elif args.name == 'zebra':
        address = './data/sr/zebra_crop.png'
        t = 4
    image = Image.open(address, 'r')
    w, h = image.size
    image = image.resize((w - w % 32, h - h % 32), resample=Image.LANCZOS)
    image = torch.from_numpy(np.array(image) / 255.0).unsqueeze(0).float()
    if args.task == 'denoise':
        # print(image.shape)
        channels = 32
        model = Model(image.shape[1], image.shape[2],input_channel=channels).to(device)
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
        mask = Image.open(mask_address, 'r')
        mask = mask.resize((w - w % 32, h - h % 32), resample=Image.LANCZOS)
        mask = torch.from_numpy(np.array(mask) / 255.0).unsqueeze(0).float()
        img_with_mask = image * mask.transpose(0,1).transpose(1,2)
        mask = mask.to(device)
        # print(img_with_mask.shape)
        corrupted_img = np.transpose(img_with_mask[0], (2,0,1))[None,:,:,:]
        corrupted_img = corrupted_img.to(device)

        if args.name == 'kate':
            channels = 32
            model = Model(image.shape[1], image.shape[2],input_channel=channels).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            z = torch.rand([1, channels, image.shape[1], image.shape[2]]) * 0.1
            z = z.to(device)
        elif args.name == 'vase':
            channels = 2
            model = Model_without_skip(image.shape[1], image.shape[2], input_channel=channels).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            X, Y = np.meshgrid(np.arange(0, image.shape[2]) / float(image.shape[2] - 1),
                               np.arange(0, image.shape[1]) / float(image.shape[1] - 1))
            meshgrid = np.concatenate([X[None, :], Y[None, :]])
            z = meshgrid.reshape(-1, channels, meshgrid.shape[1], meshgrid.shape[2])
            z = torch.from_numpy(z)
            z = z.to(torch.float32)
            z = z.to(device)
            # channels = 32
            #model = Model(image.shape[1], image.shape[2],input_channel=channels).to(device)
            #optimizer = optim.Adam(model.parameters(), lr=0.01)
            #z = torch.rand([1, channels, image.shape[1], image.shape[2]]) * 0.1
            #z = z.to(device)
        elif args.name == 'library':
            channels = 32
            model = Model_for_inpainting(image.shape[1], image.shape[2],input_channel=channels, u_mode='nearest').to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.1)
            z = torch.rand([1, channels, image.shape[1], image.shape[2]]) * 0.1
            z = z.to(device)

        print(z.shape)
        
    elif args.task=='super':
        loss_fn = torch.nn.MSELoss()
        channels = 32
        model = Model(t * image.shape[1], t * image.shape[2], input_channel=channels).to(device)
        corrupted_img = (image + torch.randn_like(image) * .1).clip(0, 1)
        corrupted_img = corrupted_img.transpose(2, 3).transpose(1, 2).to(device)
        z = torch.rand([1, channels, t * image.shape[1], t * image.shape[2]]) * 0.1
        z = z.to(device)
        corrupted_img = corrupted_img.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        downsampler = downsample(3, t, 'gaussian', device).to(device)

    for epoch in tqdm(range(args.epoch)):
        if args.task == 'denoise':
            noise = torch.randn([1, channels, corrupted_img.shape[-2], corrupted_img.shape[-1]],device=device, requires_grad=True)/30
            input = z + noise
            img_pred = model(input)
            loss = loss_fn(img_pred, corrupted_img)    
        
        elif args.task == 'super':
            noise = torch.randn(z.shape,device=device, requires_grad=True)/30
            input = z + noise
            # print(input.shape)
            img_pred = model(input)
            # pred = downsample(img_pred, (1, 3, image.shape[1], image.shape[2]), order=3, factor=t)
            # pred = torch.tensor(pred, device=device, requires_grad=True)
            pred = downsampler(img_pred)
            # print(pred.shape, corrupted_img.shape)
            loss = loss_fn(pred, corrupted_img)
        
        elif args.task == 'inpaint':
            if args.name == 'library':
                input = z
            else:
                noise = torch.randn(z.shape,device=device, requires_grad=True)/30
                input = z + noise
            img_pred = model.forward(input)
            print(img_pred.shape, mask.shape)
            loss = loss_fn(img_pred*mask[None,:,:,:], corrupted_img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if args.task == 'denoise' or 'inpaint':
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
    
    elif args.task == 'super':
        plt.figure(figsize=(18, 3.5))
        plt.subplot((1, 3, 1))
        interpolation1 = upsample(address, t, 1)
        plt.imshow(interpolation1)
        plt.title('order=1')
        plt.subplot((1, 3, 2))
        interpolation1 = upsample(address, t, 4)
        plt.imshow(interpolation1)
        plt.title('order=4')
        
        plt.subplot(1, 3, 3)
        pred=img_pred[0].cpu().transpose(0, 1).transpose(1, 2).data.numpy(), 
        plt.imshow(pred[0])
        plt.imsave(f'./Imgs/{args.name}_{args.task}{str(args.epoch)}.png', pred[0])
        plt.title('Prediction', fontsize=15)
        
        
        plt.savefig(f'Imgs/{args.name}.png')
    
    # python main.py --task=denoise --name=f16 --epoch=3000
    # python main.py --task=inpaint --name=kate --epoch=10000
    # python main.py --task=super --name=zebra --epoch=2000
    # python main.py --task=inpaint --name=vase --epoch=20000
    # python main.py --task=inpaint --name=library --epoch=5000
