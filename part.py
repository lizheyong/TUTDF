import torch
import torch.nn as nn
import torch.nn.functional as F


def bn_relu(channels, sequential, dropout=0, order=''):
    """bn => relu [=>drop out]"""
    net = sequential
    net.add_module(str(order)+'bn', nn.BatchNorm1d(channels))
    net.add_module(str(order)+'relu', nn.ReLU(inplace=True))
    if dropout:
        net.add_module(str(order)+'dropout', nn.Dropout(p=0.2))

class resnet_block(nn.Module):
    # x5
    """res = _bn_relu => conv => _bn_relu + drop out => conv"
       out =  res + shorcut(input)
    """
    def __init__(self, in_channel, zero_pad, sequential, order=''):
        super().__init__()

        res = sequential
        zero_pad = zero_pad
        if zero_pad:
            out_channel = in_channel*2
        else: out_channel = in_channel

        bn_relu(in_channel, res, order=order)
        res.add_module(str(order)+'conv1', nn.Conv1d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False))
        bn_relu(in_channel, res, dropout=1, order=str(order)+'+')
        res.add_module(str(order)+'conv2', nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=2, padding=1))

        self.res = res
        self.shortcut = nn.MaxPool1d(2, padding=1, dilation=2)
        self.zero_pad = zero_pad

    def forward(self, x):
        res = self.res(x)
        shortcut = self.shortcut(x)
        zero_pad = self.zero_pad

        def zeropad(x):
            y = torch.zeros_like(x)
            return torch.cat([x, y], 1)  # keras写的2，这块再确认下

        if zero_pad:
            shortcut = zeropad(shortcut)

        out = res + shortcut

        return out

class head_block(nn.Module):
    """head_1: conv => _bn_relu
       head_2 = conv => _bn_relu + dropout => conv
       out = head_2( head_1(input) ) + shortcut( head_1(input) )
    """
    def __init__(self):
        super().__init__()

        head_1 = nn.Sequential()

        head_1.add_module('conv1',nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=False))
        bn_relu(32, head_1)

        head_2 = nn.Sequential()

        head_2.add_module('conv2', nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, bias=False))
        bn_relu(32, head_2, dropout=1)
        head_2.add_module('conv3', nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1))

        self.head_1 = head_1
        self.head_2 = head_2
        self.shortcut = nn.MaxPool1d(2, padding=1, dilation=2)

    def forward(self, x):
        x = self.head_1(x)
        x = self.head_2(x) + self.shortcut(x)

        return x

class backbone_block(nn.Module):
    """resnet_block x5
    """
    def __init__(self):
        super().__init__()
        backbone = nn.Sequential()

        start_channel = 32
        for i in range(5):
            zero_pad = i%2  # 奇数变通道数，需要pad，concat，奇数除2刚好余1，zero_pad为true
            resnet_block(start_channel, zero_pad, backbone, order=i)
            if zero_pad:
                start_channel = start_channel*2

        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)

        return x

class dense_block(nn.Module):
    """ _bn_relu => dense => softmax
    """
    def __init__(self):
        super().__init__()

        dense = nn.Sequential()

        bn_relu(128, dense)
        dense.add_module('flatten', nn.Flatten())
        dense.add_module('linear1', nn.Linear(384, 128))
        dense.add_module('linear2', nn.Linear(128, 64))
        dense.add_module('linear3', nn.Linear(64, 32))
        dense.add_module('linear4', nn.Linear(32, 16))
        dense.add_module('linear5', nn.Linear(16, 2))

        self.dense = dense

    def forward(self, x):
        x = self.dense(x)

        return x