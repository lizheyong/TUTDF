from model.model import *
from dataset.dataset import train_Loader
from torch import optim
import torch.nn as nn
import torch
import sys
import math
import torch.nn.functional as F
import numpy as np

def train_net(net0, net1, train_dataset, device, batch_size, lr, epochs):

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(net1.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-2, amsgrad=False)
    best_loss = float('inf')

    for epoch in range(epochs):
        net0.eval()
        net1.train()
        net0.load_state_dict(torch.load(r'D:/LiZheyong/resnet/model/Head.pth'))

        # i = 0
        for curve, label in train_loader:
            optimizer1.zero_grad()
            # 将数据拷贝到device中
            curve = curve.unsqueeze(1).to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)
            # 使用网络参数，输出预测结果
            with torch.no_grad():
                out = net0(curve)
            out = net1(out)
            loss = criterion(out, label)
            # pred = torch.max(out, dim=1)[1] #[0]是值,[1]是索引
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net1.state_dict(), f'D:/LiZheyong/resnet/model/DomainPred.pth')
            # 更新参数
            loss.backward()
            optimizer1.step()
            # i += 1
            # if i%10 == 0:
            #     print(loss)
            #     i = 0
        print(f'epoch:{epoch}/{epochs}, loss:{loss.item()}')
    print(f'best_loss:{best_loss.item()}')


if __name__ == "__main__":


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Head = Head()
    DomainPred = DomainPred()
    Head.to(device=device)
    DomainPred.to(device=device)

    # 加载域B的训练集
    dataset = train_Loader(r"C:\Users\423\Desktop\MLT\domain1+2.npy",
                         r"C:\Users\423\Desktop\MLT\domain1+2_label.npy")

    train_net(net0=Head, net1=DomainPred,  train_dataset=dataset, device=device,
              batch_size=32, lr=0.0001, epochs=100)
