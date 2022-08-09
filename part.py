import torch
import torch.nn as nn
import torch.nn.functional as F


class resnet_block(nn.Module):
    """res = _bn_relu => conv => _bn_relu + drop out => conv"
       out =  res + shorcut(input)
    """
    def __init__(self):
        super().__init__()

        res = nn.Sequential()

        def bn_relu(channels, dropout=0):
            """bn => relu [=>drop out]"""
            res.add_module(nn.BatchNorm1d(channels))
            res.add_module(nn.ReLU(inplace=True))
            if dropout:
                res.add_module(nn.Dropout(p=0.2))

        bn_relu(channels)
        res.add_module(nn.Conv1d(in_channle, out_channel, kernel_size=16, padding=1))
        bn_relu(channels, dropout=1)
        res.add_module(nn.Conv1d(in_channle, out_channel, kernel_size=16, padding=1))

        self.res = res
        self.shortcut = nn.Maxpool1d(2)

    def forward(self, x, zero_pad=0):
        res = self.res(x)
        shortcut = self.shortcut(x)

        def zeropad(x):
            y = torch.zeros_like(x)
            return torch.cat([x, y], 1)  # keras写的2，这块再确认下

        if zero_pad:
            shortcut = zeropad(shortcut)

        out = res + shortcut

        return out

class 