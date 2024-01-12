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


class D(nn.Module):
    """
    Downsample
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(D, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, bias=True),  # Downsample
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.model(x)


class S(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(S, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.model(x)


class U(nn.Module):
    """
    Upsample
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(U, self).__init__()

        self.model = nn.Sequential(nn.BatchNorm2d(in_channels),
                                   nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding=1,
                                             bias=True),
                                   nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
                                   nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True),
                                   nn.Upsample(scale_factor=2, mode='bilinear'))

    def forward(self, x):
        return self.model(x)


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.d1 = D(3, 8, 3)
        self.d2 = D(8, 16, 3)
        self.d3 = D(16, 32, 3)
        self.d4 = D(32, 64, 3)
        self.d5 = D(64, 128, 3)

        self.u1 = U(16, 8, 3)
        self.u2 = U(32, 16, 3)
        self.u3 = U(64, 32, 3)
        self.u4 = U(128 + 4, 64, 3)
        self.u5 = U(128 + 4, 128, 3)

        self.s4 = S(32, 4, 1)
        self.s5 = S(64, 4, 1)

        self.conv_out = nn.Conv2d(8, 3, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        h = self.d1(x)
        h = self.d2(h)
        h = self.d3(h)
        skip3 = self.s4(h)
        h = self.d4(h)
        skip4 = self.s5(h)
        h = self.d5(h)
        h = self.u5(torch.cat((skip4[:, :, 8:-8, 8:-8], h), dim=1))
        h = self.u4(torch.cat((skip3[:, :, 16:-16, 16:-16], h), dim=1))
        h = self.u3(h)
        h = self.u2(h)
        h = self.u1(h)

        return torch.sigmoid(self.conv_out(h))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--task', type=str, default='denoise')
    parser.add_argument('--name', type=str, default='f16')
    parser.add_argument('--address', type=str, default='./data/denoising/F16_GT.png')
    args = parser.parse_args()
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    image = Image.open(args.address)
    w, h = image.size
    image = image.resize((512, 512), resample=Image.LANCZOS)
    image = torch.from_numpy(np.array(image) / 255.0).unsqueeze(0).float()
    corrupted_img = (image + torch.randn_like(image) * .1).clip(0, 1)
    corrupted_img = corrupted_img.transpose(2, 3).transpose(1, 2)
    z = torch.randn(corrupted_img.shape) * .1
    # print(corrupted_img.shape)
    for epoch in tqdm(range(2400)):
        img_pred = model.forward(z)
        loss = torch.nn.functional.mse_loss(img_pred, corrupted_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.figure(figsize=(18, 3.5))
    plt.subplot(1, 3, 1)
    corr_img=transform.resize(corrupted_img[0].transpose(0, 1).transpose(1, 2).data.numpy(), 
                              (h, w), 
                              order=3, 
                              mode='constant', 
                              preserve_range=True, 
                              anti_aliasing=True)
    plt.imshow(corr_img)
    plt.imsave(f'./Imgs/{args.name}_input.png', corr_img)
    plt.title('Input', fontsize=15)
    plt.subplot(1, 3, 2)
    pred=transform.resize(img_pred[0].transpose(0, 1).transpose(1, 2).data.numpy(), 
                            (h, w), 
                            order=3, 
                            mode='constant', 
                            preserve_range=True, 
                            anti_aliasing=True)
    plt.imshow(pred)
    plt.imsave(f'./Imgs/{args.name}_{args.task}.png', pred)
    plt.title('Prediction', fontsize=15)
    plt.subplot(1, 3, 3)
    origin=transform.resize(image[0].data.numpy(),
                            (h, w), 
                            order=3, 
                            mode='constant', 
                            preserve_range=True, 
                            anti_aliasing=True)
    plt.imshow(origin)
    plt.imsave(f'./Imgs/{args.name}_gt.png', origin)
    plt.title('Ground truth', fontsize=15)
    plt.savefig(f'Imgs/{args.name}.png')