import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
from torch import optim
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from model import Model
from model_without_skip import Model_without_skip
from model_for_inpainting import Model_for_inpainting
from sample import upsample,downsample, smooth, sharp
from psnr import cal_psnr


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
    elif args.name == 'wps':
        address = './data/denoising/wps.png'
    
    image = Image.open(address, 'r')
    w, h = image.size
    image = image.resize((w - w % 32, h - h % 32), resample=Image.LANCZOS)
    image = torch.from_numpy(np.array(image) / 255.0).unsqueeze(0).float()
    if args.task == 'denoise':
        # print(image.shape)
        channels = 32
        if args.name == 'snail':
            model = Model(image.shape[1], image.shape[2],input_channel=channels, channels=[128, 128, 128, 256, 256], skip=[4, 4, 4, 8, 8]).to(device)
        else:
            model = Model(image.shape[1], image.shape[2],input_channel=channels).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.02)
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

        # print(z.shape)
        
    elif args.task=='super':
        loss_fn = torch.nn.MSELoss()
        channels = 32
        model = Model(t * image.shape[1], t * image.shape[2], input_channel=channels).to(device)
        # corrupted_img = (image + torch.randn_like(image) * .1).clip(0, 1)
        corrupted_img = image
        corrupted_img = corrupted_img.transpose(2, 3).transpose(1, 2).to(device)
        z = torch.rand([1, channels, t * image.shape[1], t * image.shape[2]]) * 0.1
        z = z.to(device)
        corrupted_img = corrupted_img.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        downsampler = downsample(3, t, 'gaussian', device).to(device)

    ## train
    loss_ = []
    res = torch.zeros_like(corrupted_img).to(device)
    thre = 50
    smooth_model = smooth(3, device)
    sharp_model = sharp(3, device)
    for epoch in tqdm(range(args.epoch)):
        if args.task == 'denoise':
            noise = torch.randn([1, channels, corrupted_img.shape[-2], corrupted_img.shape[-1]],device=device, requires_grad=True)/30
            input = z + noise
            # input = z
            img_pred = model(input)
            loss = loss_fn(img_pred, corrupted_img)  
            if (epoch + 1) % 3 == 0:
                # img_pred = sharp_model(img_pred)
                img_pred = smooth_model(img_pred)  
                
            if epoch >= args.epoch - thre :
                res += img_pred / thre
            
        
        elif args.task == 'super':
            noise = torch.randn(z.shape,device=device, requires_grad=True)/30
            input = z + noise
            # input = z
            img_pred = model(input)
            pred = downsampler(img_pred)
            # print(pred.shape)
            loss = loss_fn(pred, corrupted_img)
        
        elif args.task == 'inpaint':
            if args.name == 'library':
                input = z
            else:
                noise = torch.randn(z.shape,device=device, requires_grad=True)/30
                input = z + noise
            img_pred = model.forward(input)
            # print(img_pred.shape, mask.shape)
            loss = loss_fn(img_pred*mask[None,:,:,:], corrupted_img)
        
        optimizer.zero_grad()
        loss.backward()
        loss_.append(loss.cpu().item())
        optimizer.step()

    if args.task == 'denoise' or args.task == 'inpaint':
        plt.figure(figsize=(18, 3.5))
        plt.subplot(1, 3, 1)
        corr_img=corrupted_img[0].cpu().transpose(0, 1).transpose(1, 2).data.numpy(), 
        plt.imshow(corr_img[0])
        # print(corr_img[0].shape)
        # plt.imsave(f'./Imgs/{args.name}_input{str(args.epoch)}.png', corr_img[0])
        plt.title('Input', fontsize=15)
        plt.subplot(1, 3, 2)
        # pred=img_pred[0].cpu().transpose(0, 1).transpose(1, 2).data.numpy()
        pred=res[0].cpu().transpose(0, 1).transpose(1, 2).data.numpy().clip(0,1)
        # plt.imshow(pred[0])
        plt.imshow(pred)
        plt.imsave(f'./Imgs/{args.name}_{args.task}{str(args.epoch)}.png', pred)
        plt.title('Prediction', fontsize=15)
        plt.subplot(1, 3, 3)
        origin = image[0].data.numpy()
        plt.imshow(origin)
        plt.imsave(f'./Imgs/{args.name}_gt.png', origin)
        plt.title('Ground truth', fontsize=15)
        plt.savefig(f'Imgs/{args.name}.png')


        # calculate psnr
        prediction = Image.open(f'./Imgs/{args.name}_{args.task}{str(args.epoch)}.png', 'r').convert("RGB")
        ground_truth = Image.open(f'./Imgs/{args.name}_gt.png', 'r').convert("RGB")
        # print((np.array(prediction)).shape)
        psnr = cal_psnr(np.array(prediction), np.array(ground_truth), 255)
    
    elif args.task == 'super':
        plt.figure(figsize=(18, 4))
        plt.subplot(1, 3, 1)
        interpolation1 = upsample(address, t, 1)
        plt.imshow(interpolation1)
        plt.title('order=1')
        plt.subplot(1, 3, 2)
        interpolation1 = upsample(address, t, 4)
        plt.imshow(interpolation1)
        plt.title('order=4')
        
        plt.subplot(1, 3, 3)
        pred=img_pred[0].cpu().transpose(0, 1).transpose(1, 2).data.numpy()
        
        # pred=img_pred[0].cpu().transpose(0, 1).transpose(1, 2).data.numpy()
        # print(pred.max())
        plt.imshow(pred)
        plt.imsave(f'./Imgs/{args.name}_{args.task}{str(args.epoch)}.png', pred)
        plt.title('Prediction', fontsize=15)
        
        plt.savefig(f'Imgs/{args.name}.png')

        origin = image[0].data.numpy()
        plt.imsave(f'./Imgs/{args.name}_gt.png', origin)


        # calculate psnr
        downsampler = downsample(3, t, 'gaussian', device).to(device)
        prediction = Image.open(f'./Imgs/{args.name}_{args.task}{str(args.epoch)}.png', 'r').convert("RGB")
        # print(prediction.shape)
        prediction = torch.from_numpy(np.array(prediction) / 255.0).unsqueeze(0).float()
        # print(prediction.shape)
        prediction = prediction.transpose(2, 3).transpose(1, 2).to(device)
        # print(prediction.shape)
        prediction = downsampler(prediction)
        prediction = prediction[0].cpu().transpose(0, 1).transpose(1, 2).data.numpy()
        # print(np.max(prediction))

        ground_truth = np.array(Image.open(f'./Imgs/{args.name}_gt.png', 'r').convert('RGB')) / 255.
        psnr = cal_psnr(prediction, ground_truth, 1)
    
    plt.figure(figsize=(18, 4))
    # save loss curve and psnr
    x_ = [i for i in range(1, args.epoch + 1)]
    plt.plot(x_, loss_)
    plt.title('loss: {}_{}{}, psnr={}'.format(args.name, args.task, args.epoch, psnr))
    # plt.legend()
    if not os.path.exists('./loss_curve'):
        os.makedirs('./loss_curve')
    plt.savefig('./loss_curve/{}_{}{}_loss.png'.format(args.name, args.task, args.epoch))

    print(f"psnr = {psnr}")

    # python main.py --task=denoise --name=f16 --epoch=3000
    # python main.py --task=inpaint --name=kate --epoch=10000
    # python main.py --task=super --name=zebra --epoch=2000
    # python main.py --task=inpaint --name=vase --epoch=20000
    # python main.py --task=inpaint --name=library --epoch=5000
    # python main.py --task=denoise --name=snail --epoch=3000
