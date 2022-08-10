import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import numpy as np
import random


class HSI_Loader(Dataset):

    def __init__(self, npyfile):
        self.all_curve = np.load(npyfile)

    def __getitem__(self, index):
        # 根据index读取pixel的光谱曲线
        pixel_curve = torch.tensor(self.all_curve[index, :])
        label = torch.tensor(self.all_curve[index, :])

        return pixel_curve, label

    def __len__(self):
        # 返回训练集大小
        return len(self.all_curve)


if __name__ == "__main__":
    HSI_dataset = HSI_Loader('water_10000.npy')
    print("数据个数：", len(HSI_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=HSI_dataset,
                                               batch_size=1024,
                                               shuffle=True)

    for pixel_curve, label in train_loader:
        # print(pixel_curve.shape)
        print(pixel_curve.unsqueeze(1).shape)

        break