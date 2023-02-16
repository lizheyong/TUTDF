import torch
import torch.nn as nn
import torch.nn.functional as F


def bn_relu(channels, sequential, dropout=0, order=''):
    """bn => relu [=>drop out]"""
    net = sequential # add _bn_relu to which sequential, str(order) to distinguish
    net.add_module(str(order)+'bn', nn.BatchNorm1d(channels))
    net.add_module(str(order)+'relu', nn.ReLU(inplace=True))
    if dropout:
        net.add_module(str(order)+'dropout', nn.Dropout(p=0.1))

class resnet_block(nn.Module):
    # this block times 5
    """_bn_relu => conv => _bn_relu + drop out => conv"""
    def __init__(self, in_channel, zero_pad, sequential, order=''):
        super().__init__()
        res = sequential
        zero_pad = zero_pad
        if zero_pad:
            out_channel = in_channel*2
        else: out_channel = in_channel

        bn_relu(in_channel, res, order=order)
        res.add_module(str(order)+'conv1', nn.Conv1d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, bias=False))
        bn_relu(in_channel, res, dropout=0, order=str(order)+'+')
        res.add_module(str(order)+'conv2', nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=2, padding=1))

        self.res = res

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
        bn_relu(32, head_2, dropout=0)
        head_2.add_module('conv3', nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1))

        self.head_1 = head_1
        self.head_2 = head_2
        self.shortcut = nn.MaxPool1d(2, padding=1, dilation=2)

    def forward(self, x):
        x = self.head_1(x)
        x = self.head_2(x) + self.shortcut(x)

        return x

class backbone_block(nn.Module):
    """resnet_block x2 """
    def __init__(self):
        super().__init__()
        start_channel = 32
        # Generate several 'sequential'，In 'foward' let x pass separately
        for i in range(3):
            exec("res%s = nn.Sequential()"%i)
            # Change the number of channels when odd, need pad, concat,
            # 1Odd number divided by 2 is exactly 1, zero_pad为true
            zero_pad = i%2
            exec("resnet_block(start_channel, zero_pad, res%s, order=i)"%i)
            if zero_pad:
                start_channel = start_channel*2
            exec("self.res%s = res%s" %(i,i))

        self.shortcut = nn.MaxPool1d(2, padding=1, dilation=2)

    def forward(self, x):
        # Define zeropad to double the number of channels
        def zeropad(x):
            y = torch.zeros_like(x)
            return torch.cat([x, y], 1)

        for i in range(2):
            shortcut = self.shortcut(x)
            # This implementation may be silly, but that's how I did it at the time
            exec("a = self.res%s(x)"%i)  # Here the result cannot be given directly to x, 'exec''s bug
            x = locals()['a'] # Here should use the locals() method

            # Processing requires zero_pad's
            zero_pad = i % 2
            if zero_pad:
                shortcut = zeropad(shortcut)
            x += shortcut

        return x

class dense_block(nn.Module):
    """ _bn_relu => dense => softmax"""
    def __init__(self):
        super().__init__()
        dense = nn.Sequential()
        bn_relu(64, dense)
        dense.add_module('flatten', nn.Flatten())
        # if If an error is reported and the dimensions don't match,
        # just change the input 960 of the full connection layer to what should match,
        # I'm too lazy to change it here, you can Replace 960 with a method to get the dimension at this point
        dense.add_module('linear1', nn.Linear(960, 512))
        dense.add_module('linear2', nn.Linear(512, 256))
        dense.add_module('linear3', nn.Linear(256, 128))
        dense.add_module('linear4', nn.Linear(128, 64))
        dense.add_module('linear5', nn.Linear(64, 32))
        dense.add_module('linear6', nn.Linear(32, 2))

        self.dense = dense

    def forward(self, x):
        x = self.dense(x)
        return x