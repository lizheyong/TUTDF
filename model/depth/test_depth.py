from depth_estimation import *
from dataset.dataset import test_Loader
from torch import optim
import torch.nn as nn
import torch
import sys
import math
import torch.nn.functional as F
import numpy as np
torch.set_printoptions(threshold=np.inf)


def test_net(net, train_dataset, device, batch_size):

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)

    net.load_state_dict(torch.load(fr"D:\resnet\model\depth\stone_depth_estimation.pth"))
    net.eval()


    for curve in train_loader:

        # 将数据拷贝到device中
        curve = curve.unsqueeze(1).to(device=device, dtype=torch.float32)
        # 使用网络参数，输出预测结果
        h = net(curve).squeeze()
        print(f'h{h}')



if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Net = Depth_Encode_Net().to(device=device)
    dataset = test_Loader(fr"C:\Users\zheyong\Desktop\高光谱目标检测报告\石测试\0.4m\测试\0.4m_stone_spatial.npy")

    test_net(net=Net, train_dataset=dataset, device=device,
              batch_size=1024)
