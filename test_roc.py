from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.datasets import make_blobs
from sklearn. model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
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
import time
import matplotlib

def test_net(net,  test_dataset, device, batch_size):

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    net.load_state_dict(torch.load(fr"C:\Users\zheyong\Desktop\高光谱目标检测报告\石测试\{this}\训练\resnet_{this}.pth"))
    net.eval()

    class_map = np.ones((1)) # 一个占位，往后拼接
    for curve in test_loader:
        curve = curve.unsqueeze(1).to(device=device, dtype=torch.float32)
        time_start = time.time()
        out = net(curve)
        # out= F.softmax(out,1)[:, 1]
        out = out[:, 1]

        time_end = time.time()
        print('time cost', time_end - time_start, 's')
        out = out.detach().cpu().numpy()
        class_map = np.hstack((class_map, out))
        # for i,j in enumerate(class_map):
        #     if j > 1:
        #         class_map[i] = 1
        #     else:
        #         class_map[i] = 0
    class_map = np.delete(class_map, 0, 0)  # 最后删了这个占位的
    class_map = np.resize(class_map, (100, 100))
    label = np.load(fr"C:\Users\zheyong\Desktop\高光谱目标检测报告\石测试\1.6m\传统方法结果\label.npy").flatten()
    TUTDF = class_map.flatten()
    auc = roc_auc_score(label, TUTDF)
    print(auc)

if __name__ == "__main__":

    this = '1.6m'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ResNet = ResNet().to(device=device)
    # dataset = test_Loader(fr"C:\Users\zheyong\Desktop\铁测试\{this}\测试\{this}.npy")
    dataset = test_Loader(fr"C:\Users\zheyong\Desktop\高光谱目标检测报告\石测试\{this}\测试\{this}_stone_spatial.npy")

    test_net(net=ResNet, test_dataset=dataset, device=device,
              batch_size=4000)

