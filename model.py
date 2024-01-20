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

    def __init__(self, w, h, input_channel=3, mode='bilinear'):
        super(Model, self).__init__()
        self.w,self.h = int(w/64), int(h/64)
        # print(w,self.w)
        self.d1 = D(input_channel, 8, 3)
        self.d2 = D(8, 16, 3)
        self.d3 = D(16, 32, 3)
        self.d4 = D(32, 64, 3)
        self.d5 = D(64, 128, 3)
        # self.d6 = D(128, 256, 3)

        self.u1 = U(16, 8, 3, mode=mode)
        self.u2 = U(32, 16, 3, mode=mode)
        self.u3 = U(64, 32, 3, mode=mode)
        self.u4 = U(128 + 4, 64, 3, mode=mode)
        self.u5 = U(128 + 4, 128, 3, mode=mode)
        # self.u6 = U(128 + 4, 256, 3, mode=mode)

        self.s4 = S(32, 4, 1)
        self.s5 = S(64, 4, 1)
        # self.s6 = S(128, 4, 1)

        self.conv_out = nn.Conv2d(8, 3, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        h = self.d1(x)
        h = self.d2(h)
        h = self.d3(h)
        skip3 = self.s4(h)
        h = self.d4(h)
        skip4 = self.s5(h)
        h = self.d5(h)
        # h = self.u5(torch.cat((skip4[:, :, 8:-8, 8:-8], h), dim=1))
        # h = self.u4(torch.cat((skip3[:, :, 16:-16, 16:-16], h), dim=1))
        # skip5 = self.s6(h)
        # h = self.d6(h)
        
        # h = self.u6(torch.cat((skip5[:,:,int(self.w/2):-int(self.w/2),int(self.h/2):-int(self.h/2)],h),dim=1))
        # h = self.u6(torch.cat((skip5[:,:,self.w:-self.w,self.h:-self.h],h),dim=1))
        h = self.u5(torch.cat((skip4[:,:,self.w:-self.w,self.h:-self.h],h),dim=1))
        h = self.u4(torch.cat((skip3[:,:,2*self.w:-2*self.w,2*self.h:-2*self.h],h),dim=1))
        h = self.u3(h)
        h = self.u2(h)
        h = self.u1(h)

        return torch.sigmoid(self.conv_out(h))