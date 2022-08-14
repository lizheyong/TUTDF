import torch
import torch.nn as nn
import torch.nn.functional as F

#IOPE-Net

class Encode_Net(nn.Module):

    def __init__(self, n_channels=1, n_classes=2):
        super(Encode_Net, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = Conv(n_channels, 16)
        self.down1 = Down(16, 32)
        # self.down2 = Down(32, 64)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        # x3 = self.down2(x2)

        return x2

class Decode_Net1(nn.Module):

    def __init__(self, n_channels=32, n_classes=1):
        super(Decode_Net1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.up1 = Up(n_channels, 16)
        # self.up2 = Up(32, 16)
        self.out = OutConv(16, n_classes)

    def forward(self, x):

        x4 = self.up1(x)
        # x5 = self.up2(x4)
        x6 = self.out(x4)
        return x6

class Decode_Net2(nn.Module):

    def __init__(self, n_channels=32, n_classes=1):
        super(Decode_Net2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.up1 = Up(n_channels, 16)
        # self.up2 = Up(32, 16)
        self.out = OutConv(16, n_classes)

    def forward(self, x):

        x4 = self.up1(x)
        # x5 = self.up2(x4)
        x6 = self.out(x4)
        return x6

# IOPE-Net part

class Conv(nn.Module):
    """(Conv => BN => ReLU )"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Pool => CONV"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            Conv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """ConvTranspose => BN => Relu"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=0, output_padding=0, dilation=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        x1 = self.up_conv(x)

        return x1

class OutConv(nn.Module):
    """ConvTranspose"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.outconv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.outconv(x)


