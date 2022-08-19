import torch
import torch.nn as nn
import torch.nn.functional as F


def bn_relu(channels, sequential, dropout=0, order=''):
    """bn => relu [=>drop out]"""
    net = sequential # 给哪个网络序列添加_bn_relu
    net.add_module(str(order)+'bn', nn.BatchNorm1d(channels))
    net.add_module(str(order)+'relu', nn.ReLU(inplace=True))
    if dropout:
        net.add_module(str(order)+'dropout', nn.Dropout(p=0.1))

class resnet_block(nn.Module):
    # x5
    """res = _bn_relu => conv => _bn_relu + drop out => conv"
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
    """resnet_block x2
    """
    def __init__(self):
        super().__init__()

        start_channel = 32
        # 生成几个sequential，在foward里让x分别通过
        for i in range(3):
            exec("res%s = nn.Sequential()"%i)
            zero_pad = i%2  # 奇数变通道数，需要pad，concat，奇数除2刚好余1，zero_pad为true
            exec("resnet_block(start_channel, zero_pad, res%s, order=i)"%i)
            if zero_pad:
                start_channel = start_channel*2
            exec("self.res%s = res%s" %(i,i))

        self.shortcut = nn.MaxPool1d(2, padding=1, dilation=2)

    def forward(self, x):
        #定义个补0让通道数翻倍
        def zeropad(x):
            y = torch.zeros_like(x)
            return torch.cat([x, y], 1)

        for i in range(2):
            # 计算直接的shortcut, res
            shortcut = self.shortcut(x)

            exec("a = self.res%s(x)"%i)  # 这里不能将结果直接给x，exec的bug
            x = locals()['a'] # 这里也必须用locals()方法

            # 处理需要zero_pad的
            zero_pad = i % 2
            if zero_pad:
                shortcut = zeropad(shortcut)
            x += shortcut

        return x

class dense_block_A(nn.Module):
    """ _bn_relu => dense => softmax
    """
    def __init__(self):
        super().__init__()

        dense_A = nn.Sequential()

        bn_relu(64, dense_A)
        dense_A.add_module('flatten', nn.Flatten())
        dense_A.add_module('linear1', nn.Linear(960, 512))
        dense_A.add_module('linear2', nn.Linear(512, 256))
        dense_A.add_module('linear3', nn.Linear(256, 128))
        dense_A.add_module('linear4', nn.Linear(128, 64))
        dense_A.add_module('linear5', nn.Linear(64, 32))
        dense_A.add_module('linear6', nn.Linear(32, 2))

        self.dense = dense_A

    def forward(self, x):
        x = self.dense(x)

        return x

class dense_block_B(nn.Module):
    """ _bn_relu => dense => softmax
    """
    def __init__(self):
        super().__init__()

        dense_B = nn.Sequential()

        bn_relu(64, dense_B)
        dense_B.add_module('flatten', nn.Flatten())
        dense_B.add_module('linear1', nn.Linear(960, 512))
        dense_B.add_module('linear2', nn.Linear(512, 256))
        dense_B.add_module('linear3', nn.Linear(256, 128))
        dense_B.add_module('linear4', nn.Linear(128, 64))
        dense_B.add_module('linear5', nn.Linear(64, 32))
        dense_B.add_module('linear6', nn.Linear(32, 2))

        self.dense = dense_B

    def forward(self, x):
        x = self.dense(x)

        return x

class domain_pred(nn.Module):

    def __init__(self):
        super().__init__()

        dense = nn.Sequential()
        dense.add_module('conv1', nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=False))
        dense.add_module('pool1', nn.MaxPool1d(2, padding=1, dilation=2))
        bn_relu(32, dense, order=1)
        dense.add_module('conv2', nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False))
        dense.add_module('pool2',nn.MaxPool1d(2, padding=1, dilation=2))
        bn_relu(64, dense, order=2)
        # B,64,30
        dense.add_module('flatten', nn.Flatten())
        dense.add_module('linear1', nn.Linear(1920, 512))
        dense.add_module('linear2', nn.Linear(512, 256))
        dense.add_module('linear3', nn.Linear(256, 128))
        dense.add_module('linear4', nn.Linear(128, 64))
        dense.add_module('linear5', nn.Linear(64, 32))
        dense.add_module('linear6', nn.Linear(32, 2))

        self.dense = dense

    def forward(self, x):
        x = self.dense(x)

        return x