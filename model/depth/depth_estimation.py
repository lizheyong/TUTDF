import torch
import torch.nn as nn
import torch.nn.functional as F



class Depth_Encode_Net(nn.Module):

    def __init__(self, n_channels=1, n_classes=2):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = Conv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(1920,512)
        self.linear2 = nn.Linear(512,128)
        self.linear3 = nn.Linear(128,32)
        self.linear4 = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.sigmoid(x)

        return x

"""
Part
"""
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