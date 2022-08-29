from depth_estimation import *
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
    criterion = torch.nn.nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-2, amsgrad=False)
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, eps=1e-08,  weight_decay=1e-2,  momentum=0.9)
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
            h = net(curve)
            loss = 1e7 * criterion(h, label)

            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), fr"D:\resnet\model\depth\depth_estimation.pth")
            # 更新参数
            loss.backward()
            optimizer.step()

        print(f'epoch:{epoch}/{epochs}, loss:{loss.item()}')
    print(f'best_loss:{best_loss.item()}')


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Net = Depth_Encode_Net().to(device=device)
    dataset = train_Loader(fr"",
                         fr"")

    train_net(net=Net, train_dataset=dataset, device=device,
              batch_size=32, lr=0.00001, epochs=800)
