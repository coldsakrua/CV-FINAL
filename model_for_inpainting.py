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


class Model_for_inpainting(nn.Module):

    def __init__(self, w, h, input_channel=3, u_mode='bilinear'):
        super(Model_for_inpainting, self).__init__()
        print("A")
        self.d1 = D(input_channel, 16, 3)
        self.d2 = D(16, 32, 3)
        self.d3 = D(32, 64, 3)
        self.d4 = D(64, 128, 3)
        self.d5 = D(128, 128, 3)
        self.d6 = D(128, 128, 3)

        self.u1 = U(32, 16, 3, mode=u_mode)
        self.u2 = U(64, 32, 3, mode=u_mode)
        self.u3 = U(128, 64, 3, mode=u_mode)
        self.u4 = U(128, 128, 3, mode=u_mode)
        self.u5 = U(128, 128, 3, mode=u_mode)
        self.u6 = U(128, 128, 3, mode=u_mode)


        self.conv_out = nn.Conv2d(16, 3, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        h = self.d1(x)
        h = self.d2(h)
        h = self.d3(h)
        h = self.d4(h)
        h = self.d5(h)
        h = self.d6(h)

        h = self.u6(h)
        h = self.u5(h)
        h = self.u4(h)
        h = self.u3(h)
        h = self.u2(h)
        h = self.u1(h)

        return torch.sigmoid(self.conv_out(h))