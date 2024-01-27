import torch
import torch.nn as nn
import torch.nn.functional as F

def get_kernel(kernel_width=5, sigma=0.5):

    kernel = torch.zeros([kernel_width, kernel_width])
    center = (kernel_width + 1.)/2.
    sigma_sq =  sigma * sigma

    for i in range(1, kernel.shape[0] + 1):
        for j in range(1, kernel.shape[1] + 1):
            di = (i - center)/2.
            dj = (j - center)/2.
            kernel[i - 1][j - 1] = torch.exp(torch.tensor(-(di * di + dj * dj)/(2 * sigma_sq)))
    kernel /= kernel.sum()

    return kernel

class gaussian(nn.Module):
    def __init__(self, n_planes,  kernel_width=5, sigma=0.5):
        super(gaussian, self).__init__()
        self.n_planes = n_planes
        self.kernel = get_kernel(kernel_width=kernel_width,sigma=sigma)

        convolver = nn.ConvTranspose2d(n_planes, n_planes, kernel_size=kernel_width, stride=2, padding=int(kernel_width/2), output_padding=1, groups=n_planes)
        convolver.weight.data[:] = 0
        convolver.bias.data[:] = 0
        convolver.weight.requires_grad = False
        convolver.bias.requires_grad = False
        for i in range(n_planes):
            convolver.weight.data[i, 0] = self.kernel
        self.upsampler = convolver

    def forward(self, x):
        
        x = self.upsampler(x)

        return x

class D(nn.Module):
    """
    Downsample
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(D, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, bias=True, padding_mode='reflect'),  # Downsample
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1, bias=True, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.model(x)


class S(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super(S, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding=0, bias=True, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        return self.model(x)


class U(nn.Module):
    """
    Upsample
    """

    def __init__(self, in_channels, out_channels, kernel_size, mode='bilinear'):
        super(U, self).__init__()
        if mode =='gaussian':
            self.model = nn.Sequential(nn.BatchNorm2d(in_channels),
                                    nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding=1,
                                                bias=True),
                                    nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
                                    nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True),
                                    #    nn.Upsample(scale_factor=2, mode=mode)
                                        gaussian(out_channels)
                                    )
        else:
            self.model = nn.Sequential(nn.BatchNorm2d(in_channels),
                                    nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding=1,
                                                bias=True),
                                    nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True),
                                    nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
                                    nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True),
                                    nn.Upsample(scale_factor=2, mode=mode)
                                        # gaussian(out_channels, kernel_size)
                                    )

    def forward(self, x):
        return self.model(x)



class Model(nn.Module):

    def __init__(self, w, h, input_channel=3, out_channels = 3, u_mode='bilinear', channels=[128, 128, 128, 128, 128], skip=[4, 4, 4, 4, 4]):
        super(Model, self).__init__()
        self.skip = skip
        self.d1 = D(input_channel, channels[0], 3)
        self.d2 = D(channels[0], channels[1], 3)
        self.d3 = D(channels[1], channels[2], 3)
        self.d4 = D(channels[2], channels[3], 3)
        self.d5 = D(channels[3], channels[4], 3)

        self.u1 = U(channels[1] + skip[0], channels[0], 3, mode=u_mode)
        self.u2 = U(channels[2] + skip[1], channels[1], 3, mode=u_mode)
        self.u3 = U(channels[3] + skip[2], channels[2], 3, mode=u_mode)
        self.u4 = U(channels[4] + skip[3], channels[3], 3, mode=u_mode)
        self.u5 = U(channels[4] + skip[4], channels[4], 3, mode=u_mode)

        self.s1 = S(channels[0], skip[0], 1)
        self.s2 = S(channels[1], skip[1], 1)
        self.s3 = S(channels[2], skip[2], 1)
        self.s4 = S(channels[3], skip[3], 1)
        self.s5 = S(channels[4], skip[4], 1)

        self.conv_out = nn.Conv2d(channels[0], out_channels, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        h = self.d1(x)
        if self.skip[0]!=0:
            skip1 = self.s1(h)
        h = self.d2(h)
        if self.skip[1]!=0:
            skip2 = self.s2(h)
        h = self.d3(h)
        if self.skip[2]!=0:
            skip3 = self.s3(h)
        h = self.d4(h)
        if self.skip[3]!=0:
            skip4 = self.s4(h)
        h = self.d5(h)
        if self.skip[4]!=0:
            skip5 = self.s5(h)

        
        if self.skip[4]==0:
            h = self.u5(h)
        else:
            if skip5.shape[2] == h.shape[2]:
                h = self.u5(torch.cat((skip5, h), dim=1))
            else:
                self.crop_w, self.crop_h = h.shape[2] // 2, h.shape[3] // 2
                h = self.u5(torch.cat((skip5[:,:,self.crop_w:-self.crop_w,self.crop_h:-self.crop_h],h),dim=1))
        if self.skip[3]==0:
            h = self.u4(h)
        else:
            # print(skip4.shape, h.shape)
            if skip4.shape[2] == h.shape[2]:
                h = self.u4(torch.cat((skip4, h), dim=1))
            else:
                self.crop_w, self.crop_h = h.shape[2] // 2, h.shape[3] // 2
                h = self.u4(torch.cat((skip4[:,:,self.crop_w:-self.crop_w,self.crop_h:-self.crop_h],h),dim=1))

        if self.skip[2]==0:
            h = self.u3(h)
        else:
            if skip3.shape[2] == h.shape[2]:
                h = self.u3(torch.cat((skip3, h), dim=1))
            else:
                self.crop_w, self.crop_h = h.shape[2] // 2, h.shape[3] // 2
                h = self.u3(torch.cat((skip3[:,:,self.crop_w:-self.crop_w,self.crop_h:-self.crop_h],h),dim=1))

        if self.skip[1]==0:
            h = self.u2(h)
        else:
            if skip2.shape[2] == h.shape[2]:
                h = self.u2(torch.cat((skip2, h), dim=1))
            else:
                self.crop_w, self.crop_h = h.shape[2] // 2, h.shape[3] // 2
                h = self.u2(torch.cat((skip2[:,:,self.crop_w:-self.crop_w,self.crop_h:-self.crop_h],h),dim=1))

        if self.skip[0]==0:
            h = self.u1(h)
        else:
            if skip1.shape[2] == h.shape[2]:
                h = self.u1(torch.cat((skip1, h), dim=1))
            else:
                self.crop_w, self.crop_h = h.shape[2] // 2, h.shape[3] // 2
                h = self.u1(torch.cat((skip1[:,:,self.crop_w:-self.crop_w,self.crop_h:-self.crop_h],h),dim=1))
        return torch.sigmoid(self.conv_out(h))