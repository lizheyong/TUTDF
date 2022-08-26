import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
this = '2.2m'
source = np.load(fr"C:\Users\zheyong\Desktop\铁测试\{this}\测试\{this}.npy")
s = np.resize(source, (200, 200, 189))
for i in range(1,199):
    for j in range(1,199):
        s[i,j,:] = 0.125*(s[i-1,j-1,:]+s[i-1,j,:]+s[i-1,j+1,:]+s[i,j-1,:]+s[i,j+1,:]+s[i+1,j-1,:]+s[i+1,j,:]+s[i+1,j+1,:])
s = np.resize(s, (40000, 189))
np.save(fr"C:\Users\zheyong\Desktop\铁测试\{this}\测试\{this}_iron_spatial.npy", s)

# source = torch.from_numpy(source).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
# source = source.transpose(1,4)
# source = source.squeeze(4) # 1,189,100,100
# s = torch.zeros(1,1,100,100)
#
# spatial_preocess = nn.Conv2d(1, 1, (3, 3), stride=1, padding=1, bias=False)
#
# """不同的滤波核"""
# mean = 0.125*torch.Tensor([[[[1, 1, 1],[1, 0, 1],[1, 1, 1]]]])
# spatial_preocess.weight.data = mean
#
# """处理，转格式，画图"""
# for i in range(189):
#     a = source[:, i, :, :].unsqueeze(1)
#     s = torch.cat((s, spatial_preocess(a)), dim=1)
# s = s.unsqueeze(4)
# s = s.transpose(1,4) #1,1,100,100,190
# s = s.squeeze(0).squeeze(0)
# sn = s.detach().numpy() # 100,100,190
# sn = sn[:,:,1:]
# sn= np.resize(source, (10000, 189))
#
# np.save(r"C:\Users\zheyong\Desktop\石测试\0.4m\测试\0.4m_stone_spatial.npy", sn)

