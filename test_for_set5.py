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
from sample import *
from utils import *

if __name__ == "__main__":
    task = 'sr'
    device='cuda' if torch.cuda.is_available() else 'cpu'
    smooth_model = smooth(3,device)
    sharp_model = sharp(3, device)
    torch.cuda.empty_cache()
    # parser = argparse.ArgumentParser(description='parameters')
    # parser.add_argument('--task', type=str, default='denoise')
    # parser.add_argument('--name', type=str, default='f16')
    # parser.add_argument('--epoch',type=int,default='2400')
    # parser.add_argument('--auto',type=bool,default=False)
    # args = parser.parse_args()
    name_list = ['baby', 'woman', 'head', 'bird', 'butterfly']
    for name in name_list:
        loss_=[]
        auto_flag = False
        epochs = 1500
        address, gt_address, t = name2add(name)
        image = Image.open(address, 'r')
        w, h = image.size
        image = image.resize((w - w % 32 + 32, h - h % 32 + 32), resample=Image.LANCZOS)
        image = np.array(image)
        image = torch.from_numpy(image / 255.0).unsqueeze(0).float()
        ground_truth = Image.open(gt_address, 'r').convert('RGB')
        w1, h1 = ground_truth.size
        ground_truth = ground_truth.resize((4*(w - w % 32 + 32), 4*(h - h % 32 + 32)), resample=Image.LANCZOS)
        ground_truth = np.array(ground_truth)
        w1, h1 = ground_truth.shape[0], ground_truth.shape[1]
        # loss_fn = torch.nn.MSELoss()
        loss_fn = torch.nn.SmoothL1Loss()
        loss_fn1 = TVLoss(weight=0.01)
        channels = 32
        model = Model(w1, h1, input_channel=channels,skip=[8,8,4,4,4]).to(device)##不是严格四倍关系
        # corrupted_img = (image + torch.randn_like(image) * .05).clip(0, 1)
        corrupted_img = image
        corrupted_img = corrupted_img.transpose(2, 3).transpose(1, 2).to(device)
        z = torch.rand([1, channels, w1, h1]) * 0.1
        z = z.to(device)
        corrupted_img = corrupted_img.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.01)
        downsampler = downsample(3, t, 'gaussian', device).to(device)
        res = torch.zeros((1,3,w1,h1)).to(device)
        thre = 25
        # print(z.shape, corrupted_img.shape, image.shape)
        for epoch in tqdm(range(epochs)):
            # noise = torch.randn(z.shape,device=device, requires_grad=True)* 0.01
            # input = z + noise
            input = z
            img_pred = model(input)
            pred = img_pred
            if (epoch + 1) % 3 == 0:
            #     img_pred = sharp_model(img_pred)
                img_pred = smooth_model(pred) 
            loss1 = loss_fn1(img_pred) 
            pred = downsampler(img_pred)
            if epoch >= epochs - thre :
                res += img_pred / (thre) 
            # if epoch == epochs - 1:
            #         res = sharp_model(res)
            loss2 = loss_fn(pred, corrupted_img)
            # print(loss1, loss2)
            loss = loss2+loss1
            optimizer.zero_grad()
            loss.backward()
            loss_.append(loss.cpu().item())
            optimizer.step()
        
        
        plt.figure(figsize=(18, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(ground_truth)
        plt.title('gt')
        
        plt.subplot(1, 3, 2)
        interpolation1 = upsample1(image, w1, h1, 4)
        plt.imshow(interpolation1)
        p2 = cal_psnr(np.floor(255*interpolation1), ground_truth, 255)
        plt.title('order=4, psnr = {:.4f}'.format(p2))
        
        plt.subplot(1, 3, 3)
        pred=res[0].cpu().transpose(0, 1).transpose(1, 2).data.numpy().clip(0,1)
        plt.imshow(pred)
        plt.imsave(f'./Imgs/{name}_sr{str(epoch+1)}.png',pred)

        psnr = cal_psnr(np.floor(pred*255), ground_truth, 255)
        plt.title('Prediction, pnsr = {:.4f}'.format(psnr), fontsize=15)
        
        
        plt.savefig(f'Imgs/{name}.png')
        origin = image[0].data.numpy()
        plt.imsave(f'./Imgs/{name}_gt.png', origin)


        # calculate psnr
        # prediction = Image.open(f'./Imgs/{name}_sr{str(epoch+1)}.png', 'r').convert("RGB")
        # prediction = torch.from_numpy(np.array(prediction) / 255.0).unsqueeze(0).float()
        # prediction = prediction.transpose(2, 3).transpose(1, 2).to(device)
        # prediction = prediction[0].cpu().transpose(0, 1).transpose(1, 2).data.numpy()
        # print(prediction.shape, ground_truth.shape, w1,h1)
        # psnr = cal_psnr(prediction, ground_truth, 1)
        
        plt.figure(figsize=(18, 4))
        # save loss curve and psnr
        if auto_flag:
            x_ = [i for i in range(1, epoch + 2)]
        else:
            x_ = [i for i in range(1, epoch + 2)]
        plt.plot(x_, loss_)
        plt.title('loss: {}_{}{}, psnr={}'.format(name, task, epoch+1, psnr))
        # plt.legend()
        if not os.path.exists('./loss_curve'):
            os.makedirs('./loss_curve')
        plt.savefig('./loss_curve/{}_{}{}_loss.png'.format(name, task, epoch+1))

        print(f"psnr = {psnr}")