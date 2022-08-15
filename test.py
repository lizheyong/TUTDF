from model.model import Net
from dataset.dataset import train_Loader, test_Loader
from torch import optim
import torch.nn as nn
import torch
import sys
import math
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def test_net(net, test_dataset, device, batch_size):

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    pred_net = net
    pred_net.to(device=device)
    pred_net.load_state_dict(torch.load(r'model/best_resnet_model.pth', map_location=device))
    pred_net.eval()

    class_map = np.ones((1)) # 一个占位，往后拼接
    for curve in test_loader:
        curve = curve.unsqueeze(1).to(device=device, dtype=torch.float32)
        # 使用网络参数，输出预测结果
        out = pred_net(curve)
        # out = F.softmax(out, dim=1)
        # pred = torch.max(out, dim=1)[0] #[0]是值,[1]是索引
        pred = out[:,1]
        pred = pred.detach().cpu().numpy()
        class_map = np.hstack((class_map, pred))
    class_map = np.delete(class_map, 0, 0)  # 最后删了这个占位的
    class_map = np.resize(class_map, (100, 100))
    np.save(r'D:\LiZheyong\data\iron_30_water_30_depth_0-3_0.01\结果\class_map', class_map)
    plt.imshow(class_map)
    plt.show()




if __name__ == "__main__":


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net()
    net.to(device=device)

    dataset = test_Loader(r'D:\LiZheyong\data\iron_in_water_15x15\Iron_shallow.npy')

    print("数据个数：", len(dataset))

    test_net(net=net, test_dataset=dataset, device=device, batch_size=1024)
