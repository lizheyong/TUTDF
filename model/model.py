import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import dataset
from part import *


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.head = head_block()
        self.backbone = backbone_block()
        self.dense = dense_block()

    def forward(self, x):
        x = self.head(x)
        x = self.backbone(x)
        x = self.dense(x)

        return x

# class CNN(nn.Module):


if __name__ == '__main__':


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = ResNet()
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