import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import dataset
from .part import *


class Head(nn.Module):
    def __init__(self):
        super().__init__()

        self.head = head_block()

    def forward(self, x):
        x = self.head(x)

        return x

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = backbone_block()

    def forward(self, x):
        x = self.backbone(x)
        return x

class DenseA(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_A = dense_block_A()

    def forward(self, x):
        x = self.dense_A(x)
        return x

class DenseB(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_B = dense_block_B()

    def forward(self, x):
        x = self.dense_B(x)
        return x


class Head_P(nn.Module):
    def __init__(self):
        super().__init__()

        self.head = head_block_P()

    def forward(self, x):
        x = self.head(x)

        return x

class DomainPred(nn.Module):
    def __init__(self):
        super().__init__()
        self.domain_pred = domain_pred()

    def forward(self, x):
        x = self.domain_pred(x)
        return x


if __name__ == '__main__':


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net()
    net.to(device=device)

    from torchsummary import summary
    summary(net, input_size=(1, 120)) # (channel, H, W)
    print(net)

    # 指定训练集地址，开始训练
    HSI_dataset = dataset.test_Loader('../dataset/water_10000.npy')
    train_loader = torch.utils.data.DataLoader(dataset=HSI_dataset,
                                               batch_size=1024,
                                               shuffle=True)
    for curve in train_loader:
        # 将数据拷贝到device中
        curve = curve.unsqueeze(1).to(device=device, dtype=torch.float32)
        # label = label.to(device=device, dtype=torch.float32)
        break
    print(curve.shape)
    out = net(curve)
    print(out.shape)