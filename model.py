import torch
import torch.nn as nn




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

    def __init__(self, in_channels, out_channels, kernel_size, mode='bilinear'):
        super(U, self).__init__()

        self.model = nn.Sequential(nn.BatchNorm2d(in_channels),
                                   nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding=1,
                                             bias=True),
                                   nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True),
                                   nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
                                   nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True),
                                   nn.Upsample(scale_factor=2, mode=mode))

    def forward(self, x):
        return self.model(x)


class Model(nn.Module):

    def __init__(self, w, h, input_channel=3, u_mode='bilinear'):
        super(Model, self).__init__()

        self.d1 = D(input_channel, 128, 3)
        self.d2 = D(128, 128, 3)
        self.d3 = D(128, 128, 3)
        self.d4 = D(128, 128, 3)
        self.d5 = D(128, 128, 3)

        self.u1 = U(128 + 4, 128, 3, mode=u_mode)
        self.u2 = U(128 + 4, 128, 3, mode=u_mode)
        self.u3 = U(128 + 4, 128, 3, mode=u_mode)
        self.u4 = U(128 + 4, 128, 3, mode=u_mode)
        self.u5 = U(128 + 4, 128, 3, mode=u_mode)

        self.s1 = S(128, 4, 1)
        self.s2 = S(128, 4, 1)
        self.s3 = S(128, 4, 1)
        self.s4 = S(128, 4, 1)
        self.s5 = S(128, 4, 1)

        self.conv_out = nn.Conv2d(128, 3, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        h = self.d1(x)
        skip1 = self.s1(h)
        h = self.d2(h)
        skip2 = self.s2(h)
        h = self.d3(h)
        skip3 = self.s3(h)
        h = self.d4(h)
        skip4 = self.s4(h)
        h = self.d5(h)
        skip5 = self.s5(h)

        if skip5.shape[2] == h.shape[2]:
            h = self.u5(torch.cat((skip5, h), dim=1))
        else:
            self.crop_w, self.crop_h = h.shape[2] // 2, h.shape[3] // 2
            h = self.u5(torch.cat((skip5[:,:,self.crop_w:-self.crop_w,self.crop_h:-self.crop_h],h),dim=1))

        if skip4.shape[2] == h.shape[2]:
            h = self.u4(torch.cat((skip4, h), dim=1))
        else:
            self.crop_w, self.crop_h = h.shape[2] // 2, h.shape[3] // 2
            h = self.u4(torch.cat((skip4[:,:,self.crop_w:-self.crop_w,self.crop_h:-self.crop_h],h),dim=1))

        if skip3.shape[2] == h.shape[2]:
            h = self.u3(torch.cat((skip3, h), dim=1))
        else:
            self.crop_w, self.crop_h = h.shape[2] // 2, h.shape[3] // 2
            h = self.u3(torch.cat((skip3[:,:,self.crop_w:-self.crop_w,self.crop_h:-self.crop_h],h),dim=1))

        if skip2.shape[2] == h.shape[2]:
            h = self.u2(torch.cat((skip2, h), dim=1))
        else:
            self.crop_w, self.crop_h = h.shape[2] // 2, h.shape[3] // 2
            h = self.u2(torch.cat((skip2[:,:,self.crop_w:-self.crop_w,self.crop_h:-self.crop_h],h),dim=1))

        if skip1.shape[2] == h.shape[2]:
            h = self.u1(torch.cat((skip1, h), dim=1))
        else:
            self.crop_w, self.crop_h = h.shape[2] // 2, h.shape[3] // 2
            h = self.u1(torch.cat((skip1[:,:,self.crop_w:-self.crop_w,self.crop_h:-self.crop_h],h),dim=1))

        return torch.sigmoid(self.conv_out(h))