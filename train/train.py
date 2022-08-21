from model.model import *
from dataset.dataset import train_Loader
from torch import optim
import torch.nn as nn
import torch
import sys
import math
import torch.nn.functional as F
import numpy as np

def train_net(net, train_dataset, device, batch_size, lr, epochs):

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-2, amsgrad=False)
    best_loss = float('inf')

    for epoch in range(epochs):
        net.train()
        # i = 0
        for curve, label in train_loader:
            optimizer.zero_grad()
            # 将数据拷贝到device中
            curve = curve.unsqueeze(1).to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)
            # 使用网络参数，输出预测结果
            out = net(curve)
            loss = criterion(out, label)
            # pred = torch.max(out, dim=1)[1] #[0]是值,[1]是索引
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), r"C:\Users\423\Desktop\铁测试\0.1m\训练\resnet_0.1m.pth")
            # 更新参数
            loss.backward()
            optimizer.step()
            # i += 1
            # if i%30 == 0:
            #     print(loss.item())
            #     i = 0
        print(f'epoch:{epoch}/{epochs}, loss:{loss.item()}')
    print(f'best_loss:{best_loss.item()}')


if __name__ == "__main__":


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ResNet = ResNet().to(device=device)
    dataset = train_Loader(r"C:\Users\423\Desktop\铁测试\0.1m\训练\0.1m_train.npy",
                         r"C:\Users\423\Desktop\铁测试\0.1m\训练\0.1m_train_label.npy")

    train_net(net=ResNet, train_dataset=dataset, device=device,
              batch_size=1024, lr=0.0001, epochs=1000)
