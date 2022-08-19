from model.model import *
from dataset.dataset import  test_Loader
from torch import optim
import torch.nn as nn
import torch
import sys
import math
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def test_net(net0, net1, net2, net3, net4, test_dataset, device, batch_size):

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    Head = net0()
    Backbone = net1()
    DenseA = net2()
    DenseB = net3()
    Pred = net4()

    Head.load_state_dict(torch.load(r'model/Head.pth'))
    Backbone.load_state_dict(torch.load(r'model/Backbone.pth'))
    DenseA.load_state_dict(torch.load(r'model/DenseA.pth'))
    DenseB.load_state_dict(torch.load(r'model/DenseB.pth'))
    Pred.load_state_dict(torch.load(r'model/DomainPred.pth'))

    Head.eval()
    Backbone.eval()
    DenseA.eval()
    DenseB.eval()
    Pred.eval()

    class_map = np.ones((1)) # 一个占位，往后拼接
    for curve in test_loader:
        curve = curve.unsqueeze(1).to(device=device, dtype=torch.float32)
        # 使用网络参数，输出预测结果
        headout = Head(curve)
        dp = Pred(out1)
        out0 = DenseA(headout)[:,0]*torch.max(dp, 1)
        out1 = DenseB(headout)[:,0]*torch.min(dp, 1)
        out = out0 + out1

        out = out.detach().cpu().numpy()
        class_map = np.hstack((class_map, out))
    class_map = np.delete(class_map, 0, 0)  # 最后删了这个占位的
    class_map = np.resize(class_map, (100, 100))
    np.save(r'D:\LiZheyong\data\iron_30_water_30_depth_0-3_0.01\结果\class_map', class_map)
    plt.imshow(class_map)
    plt.show()




if __name__ == "__main__":


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Head = Head()
    Backbone = Backbone()
    DenseA = DenseA()
    DenseB = DenseB()
    Pred = DomainPred()
    Head.to(device=device)
    Backbone.to(device=device)
    DenseA.to(device=device)
    DenseB.to(device=device)
    Pred.to(device=device)

    # 加载测试集
    dataset = test_Loader(r"C:\Users\423\Desktop\MLT\train_domain_1.npy")

    train_net(net0=Head, net1=Backbone, net2=DenseA, net3=DenseB, net4=Pred, test_dataset=dataset, device=device,
              batch_size=32)
