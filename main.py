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
from utils import *

if __name__ == "__main__":
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--task', type=str, default='denoise')
    parser.add_argument('--name', type=str, default='f16')
    parser.add_argument('--epoch',type=int,default='2400')
    parser.add_argument('--auto',type=bool,default=False)
    args = parser.parse_args()
    address, mask_address, t = name2add(args.name)
    
    image = Image.open(address, 'r')
    w, h = image.size
    if args.task != 'super':
        image = image.resize((w - w % 32, h - h % 32), resample=Image.LANCZOS)
    image = np.array(image)
    # print(image.shape)
    black = len(image.shape)==2
    if black:
        image = np.stack((image,)*3, axis=-1)
        in_ch = 3
        # image_with_noise = torch.from_numpy(pepper_and_salt(image, percentage=0.06)).float()/255.
        
    else:
        in_ch = 3
    image = torch.from_numpy(image / 255.0).unsqueeze(0).float()
    auto_flag = args.auto
    record = False
    psnr_list = []
    smooth_model = smooth(in_ch, device)
    sharp_model = sharp(in_ch, device)
    
    if args.task == 'denoise':
        channels = 32
        ground_truth = Image.open(address, 'r').convert("RGB")
        # if args.name == 'snail':
        #     model = Model(image.shape[1], image.shape[2],input_channel=channels, channels=[128, 128, 128, 256, 256], skip=[4, 4, 4, 8, 8]).to(device)
        # else:
        #     model = Model(image.shape[1], image.shape[2],input_channel=channels,u_mode='gaussian').to(device)
        # model = Model(image.shape[1], image.shape[2],input_channel=channels,out_channels=in_ch, u_mode='gaussian').to(device)
        model = Model(image.shape[1], image.shape[2],input_channel=channels,out_channels=in_ch,skip=[8,8,4,4,4]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.02)
        loss_fn1 = torch.nn.SmoothL1Loss()
        loss_fn2 = TVLoss(weight=0.02)
        noise = torch.randn_like(image) * .1
        # if black:
        #     corrupted_img = image_with_noise.unsqueeze(0).transpose(2, 3).transpose(1, 2)
        #     # print(corrupted_img.shape)
        # else:
        #     print('yes')
        corrupted_img = (image + noise)
        corrupted_img = corrupted_img.transpose(2, 3).transpose(1, 2).clip(0, 1)
        black = False
        z = torch.rand([1, channels, corrupted_img.shape[-2], corrupted_img.shape[-1]]) * 0.1
        z = z.to(device)
        corrupted_img = corrupted_img.to(device)
        res = torch.zeros_like(corrupted_img).to(device)
        thre = 50
        
        auto_iter = 100
        
        coef_epsilon = 0.005
        coef_list = np.zeros((args.epoch))
        
    elif args.task == 'inpaint':
        loss_fn = torch.nn.MSELoss()
        mask = Image.open(mask_address, 'r')
        mask = mask.resize((w - w % 32, h - h % 32), resample=Image.LANCZOS)
        mask = torch.from_numpy(np.array(mask) / 255.0).unsqueeze(0).float()
        img_with_mask = image * mask.transpose(0,1).transpose(1,2)
        mask = mask.to(device)
        corrupted_img = np.transpose(img_with_mask[0], (2,0,1))[None,:,:,:]
        corrupted_img = corrupted_img.to(device)

        if args.name == 'kate' or args.name == 'inpainting_img1' \
            or args.name == 'inpainting_img2' or args.name == 'inpainting_img3' or args.name == 'inpainting_img4':#text
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
        
        else:
            channels = 32
            model = Model(image.shape[1], image.shape[2],input_channel=channels).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            z = torch.rand([1, channels, image.shape[1], image.shape[2]]) * 0.1
            z = z.to(device)
            


    elif args.task=='super':
        loss_fn = torch.nn.MSELoss()
        channels = 32
        model = Model(t * image.shape[1], t * image.shape[2], input_channel=channels,channels=[128, 128, 128, 256, 256], skip=[4, 4, 4, 8, 8]).to(device)
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

    for epoch in tqdm(range(args.epoch)):
        if args.task == 'denoise':
            noise = torch.randn([1, channels, corrupted_img.shape[-2], corrupted_img.shape[-1]],device=device, requires_grad=True)
            input = z + noise / 30
            # input = z
            img_pred = model(input)
            # if (epoch + 1) % 3 == 0:
            #     # img_pred = sharp_model(img_pred)
            #     img_pred = smooth_model(img_pred)  
            if epoch >= args.epoch - thre :
                res += img_pred / thre
                # if epoch == args.epoch - 1:
                #     res = sharp_model(res)
            # loss = loss_fn1(img_pred, corrupted_img) + loss_fn2(img_pred)  
            loss = loss_fn1(img_pred, corrupted_img)
            if auto_flag:
                pred1 = img_pred[0].cpu().transpose(0, 1).transpose(1, 2).data.numpy()
                sharpness = cal_sharp(pred1)
                blurness = cal_blur(pred1)
                coef = (blurness / sharpness)
                coef_list[epoch] = coef
            if record:
                pred1=img_pred[0].cpu().transpose(0, 1).transpose(1, 2).data.numpy()
                psnr = cal_psnr(pred1, np.array(ground_truth)/255., 1)
                psnr_list.append(psnr)
            if auto_flag and epoch>2*auto_iter:
                coef1 = np.mean(coef_list[epoch-2*auto_iter:epoch-auto_iter])
                coef2 = np.mean(coef_list[epoch-auto_iter+1: epoch])
                # print(blurness , sharpness, coef)
                if np.abs(coef1-coef2)<coef_epsilon:
                    print('auto stop!!!')
                    break
        
        elif args.task == 'super':
            noise = torch.randn(z.shape,device=device, requires_grad=True)/30
            input = z + noise
            # input = z
            img_pred = model(input)
            if (epoch + 1) % 3 == 0:
                # img_pred = sharp_model(img_pred)
                img_pred = smooth_model(img_pred)  
            pred = downsampler(img_pred)
            loss = loss_fn(pred, corrupted_img)
        
        elif args.task == 'inpaint':
            if args.name == 'library':
                input = z
            else:
                noise = torch.randn(z.shape,device=device, requires_grad=True)/30
                input = z + noise
            img_pred = model.forward(input)
            loss = loss_fn(img_pred*mask[None,:,:,:], corrupted_img)
        
        optimizer.zero_grad()
        loss.backward()
        loss_.append(loss.cpu().item())
        optimizer.step()

    if args.task == 'denoise' or args.task == 'inpaint':
        plt.figure(figsize=(18, 3.5))
        plt.subplot(1, 3, 1)
        corr_img=corrupted_img[0].cpu().transpose(0, 1).transpose(1, 2).data.numpy()
        if black:
            plt.imshow(corr_img[:,:,0])
            plt.imsave(f'./Imgs/{args.name}_input{str(epoch+1)}.png', corr_img[:,:,0])
        else:
            plt.imshow(corr_img)
            plt.imsave(f'./Imgs/{args.name}_input{str(epoch+1)}.png', corr_img)
        # plt.imshow(corr_img)
        plt.title('Input', fontsize=15)
        
        plt.subplot(1, 3, 2)
        if auto_flag or args.task == 'inpaint':
            pred =img_pred[0].cpu().transpose(0, 1).transpose(1, 2).data.numpy()
        else:
            pred=res[0].cpu().transpose(0, 1).transpose(1, 2).data.numpy().clip(0,1)
        if black:
            plt.imshow(pred[:,:,0])
            plt.imsave(f'./Imgs/{args.name}_{args.task}{str(epoch+1)}.png', pred[:,:,0])
        else:
            plt.imshow(pred)
            plt.imsave(f'./Imgs/{args.name}_{args.task}{str(epoch+1)}.png', pred)
        plt.title('Prediction', fontsize=15)
        
        plt.subplot(1, 3, 3)
        origin = image[0].data.numpy()
        plt.imshow(origin)
        plt.imsave(f'./Imgs/{args.name}_gt.png', origin)
        plt.title('Ground truth', fontsize=15)
        plt.savefig(f'Imgs/{args.name}.png')


        # calculate psnr
        prediction = Image.open(f'./Imgs/{args.name}_{args.task}{str(epoch+1)}.png', 'r').convert("RGB")
        psnr = cal_psnr(np.array(prediction), np.array(ground_truth), 255)
        
    elif args.task == 'super':
        plt.figure(figsize=(18, 4))
        plt.subplot(1, 3, 1)
        interpolation1 = upsample(image, t, 1)
        plt.imshow(interpolation1)
        plt.title('order=1')
        
        plt.subplot(1, 3, 2)
        interpolation1 = upsample(image, t, 4)
        plt.imshow(interpolation1)
        plt.title('order=4')
        
        plt.subplot(1, 3, 3)
        pred=img_pred[0].cpu().transpose(0, 1).transpose(1, 2).data.numpy()
        plt.imshow(pred)
        plt.imsave(f'./Imgs/{args.name}_{args.task}{str(epoch+1)}.png', pred)
        plt.title('Prediction', fontsize=15)
        plt.savefig(f'Imgs/{args.name}.png')
        origin = image[0].data.numpy()
        plt.imsave(f'./Imgs/{args.name}_gt.png', origin)


        # calculate psnr
        downsampler = downsample(3, t, 'gaussian', device).to(device)
        prediction = Image.open(f'./Imgs/{args.name}_{args.task}{str(epoch+1)}.png', 'r').convert("RGB")
        prediction = torch.from_numpy(np.array(prediction) / 255.0).unsqueeze(0).float()
        prediction = prediction.transpose(2, 3).transpose(1, 2).to(device)
        prediction = downsampler(prediction)
        prediction = prediction[0].cpu().transpose(0, 1).transpose(1, 2).data.numpy()
        ground_truth = np.array(Image.open(f'./Imgs/{args.name}_gt.png', 'r').convert('RGB')) / 255.
        psnr = cal_psnr(prediction, ground_truth, 1)
    
    plt.figure(figsize=(18, 4))
    # save loss curve and psnr
    if auto_flag:
        x_ = [i for i in range(1, epoch + 2)]
    else:
        x_ = [i for i in range(1, args.epoch + 1)]
    plt.plot(x_, loss_)
    plt.title('loss: {}_{}{}, psnr={}'.format(args.name, args.task, args.epoch, psnr))
    # plt.legend()
    if not os.path.exists('./loss_curve'):
        os.makedirs('./loss_curve')
    plt.savefig('./loss_curve/{}_{}{}_loss.png'.format(args.name, args.task, epoch+1))
    if record:
            plt.clf()
            plt.plot(x_,psnr_list)
            plt.title('pnsr')
            plt.savefig('./loss_curve/{}_{}_psnr'.format(args.name,epoch+1))
    
    print(f"psnr = {psnr}")

    # python main.py --task=denoise --name=f16 --epoch=3000
    # python main.py --task=inpaint --name=kate --epoch=10000
    # python main.py --task=super --name=zebra --epoch=2000
    # python main.py --task=inpaint --name=vase --epoch=20000
    # python main.py --task=inpaint --name=library --epoch=5000
    # python main.py --task=denoise --name=snail --epoch=3000
    
    # region inpainting
    # python main.py --task=inpaint --name=inpainting_img1 --epoch=15000
    # python main.py --task=inpaint --name=inpainting_img2 --epoch=4000

    # python main.py --task=denoise --name=denoise_img1 --epoch=3000
    # python main.py --task=denoise --name=denoise_img2 --epoch=1800

    # text
    # python main.py --task=inpaint --name=inpainting_img3 --epoch=5000
    # python main.py --task=inpaint --name=inpainting_img4 --epoch=10000
