import torch
import torch.nn as nn
import sys
sys.path.append('..')
import dataset

class Encode_Net(nn.Module):
    # encoder part, input 1 curve sequence(1 channel), output channel is 64
    def __init__(self):
        super(Encode_Net, self).__init__()
        self.inc = Conv(1, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        return x3

class Decode_Net1(nn.Module):
    # input is 64 channels, output is 1 channel, estimated 'a'
    def __init__(self):
        super(Decode_Net1, self).__init__()
        self.up1 = Up(64, 32)
        self.up2 = Up(32, 16)
        self.out = OutConv(16, 1)

    def forward(self, x):
        x4 = self.up1(x)
        x5 = self.up2(x4)
        x6 = self.out(x5)
        return x6


class Decode_Net2(nn.Module):
    # input is 64 channels, output is 1 channel, estimated 'bb'
    def __init__(self):
        super(Decode_Net2, self).__init__()
        self.up1 = Up(64, 32)
        self.up2 = Up(32, 16)
        self.out = OutConv(16, 1)

    def forward(self, x):
        x4 = self.up1(x)
        x5 = self.up2(x4)
        x6 = self.out(x5)
        return x6

"""
part
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
    """Pool => Conv"""
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
            # I set these parameters just for it can run
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up_conv(x)

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


if __name__ == '__main__':

    # select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load nets
    net0 = Encode_Net()
    net1 = Decode_Net1()
    net2 = Decode_Net2()
    # load nets to deivce
    net0.to(device=device)
    net1.to(device=device)
    net2.to(device=device)
    # load train_dataset
    HSI_dataset = dataset.HSI_Loader('../data/water_curves.npy')
    train_loader = torch.utils.data.DataLoader(dataset=HSI_dataset,batch_size=1024,shuffle=True)
    # show a batch of train_data
    for curve, label in train_loader:
        len_of_this_batch = curve.shape[0]
        # curve and label should load to device
        curve = curve.reshape(len_of_this_batch, 1, -1).to(device=device, dtype=torch.float32)
        # label didn't add the dim
        label = label.to(device=device, dtype=torch.float32)
        # print a batch curve's shape [this_batch_size, 1, wavelength]
        print(curve.shape)
        encode_curve = net0(curve)
        # a [this_batch_size, 1, wavelength]
        a = net1(encode_curve)
        # bb [this_batch_size, 1, wavelength]
        bb = net2(encode_curve)
        print(f'{a.shape},a:{a[0,0,:]},\n bb:{bb[0,0,:]}')
        break