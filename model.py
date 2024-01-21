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
        self.w1,self.h1 = int(w/64), int(h/64)
        self.w2,self.h2 = int(w/32), int(h/32)
        self.w3,self.h3 = int(w/16), int(h/16)
        self.w4,self.h4 = int(w/8), int(h/8)
        # print(w,self.w)
        self.d1 = D(input_channel, 128, 3)
        self.d2 = D(128, 128, 3)
        self.d3 = D(128, 128, 3)
        self.d4 = D(128, 128, 3)
        self.d5 = D(128, 128, 3)
        # self.d6 = D(128, 256, 3)

        self.u1 = U(128, 128, 3, mode=u_mode)
        self.u2 = U(128 + 4, 128, 3, mode=u_mode)
        self.u3 = U(128 + 4, 128, 3, mode=u_mode)
        self.u4 = U(128 + 4, 128, 3, mode=u_mode)
        self.u5 = U(128 + 4, 128, 3, mode=u_mode)
        # self.u6 = U(128 + 4, 256, 3, mode=mode)

        self.s2 = S(128, 4, 1)
        self.s3 = S(128, 4, 1)
        self.s4 = S(128, 4, 1)
        self.s5 = S(128, 4, 1)
        # self.s6 = S(128, 4, 1)

        self.conv_out = nn.Conv2d(128, 3, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        h = self.d1(x)
        skip1 = self.s2(h)
        h = self.d2(h)
        skip2 = self.s3(h)
        h = self.d3(h)
        skip3 = self.s4(h)
        h = self.d4(h)
        skip4 = self.s5(h)
        h = self.d5(h)
        # print(skip4[:,:,self.w1:-self.w1,self.h1:-self.h1].shape,h.shape,self.w1)
        h = self.u5(torch.cat((skip4[:,:,self.w1:-self.w1,self.h1:-self.h1],h),dim=1))
        h = self.u4(torch.cat((skip3[:,:,self.w2:-self.w2,self.h2:-self.h2],h),dim=1))
        h = self.u3(torch.cat((skip2[:,:,self.w3:-self.w3,self.h3:-self.h3],h),dim=1))
        # print(skip2[:,:,self.w4:-self.w4,self.h4:-self.h4].shape,h.shape,self.h4)
        h = self.u2(torch.cat((skip1[:,:,self.w4:-self.w4,self.h4:-self.h4],h),dim=1))
        h = self.u1(h)

        return torch.sigmoid(self.conv_out(h))