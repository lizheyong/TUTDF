import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2

"""加载原始图"""
class_map = np.load(r"C:\Users\zheyong\Desktop\class_map.npy")
class_map = torch.from_numpy(class_map).to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)

"""定义一个核大小"""
spatial_preocess = nn.Conv2d(1, 1, (3, 3), stride=1, padding=1, bias=False)

"""不同的滤波核"""
mean = 0.125*torch.Tensor([[[[1, 1, 1],[1, 0, 1],[1, 1, 1]]]])
gauss = torch.Tensor(np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]).reshape(1, 1, 3, 3))
bianyuanjiance = torch.Tensor(np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]).reshape(1, 1, 3, 3))
caomao = torch.Tensor(np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]).reshape(1, 1, 3, 3))

spatial_preocess.weight.data = mean

"""处理，转格式，画图"""

class_spatial_map = spatial_preocess(class_map)



class_spatial_map = class_spatial_map.squeeze().detach().numpy()
np.save(r"C:\Users\zheyong\Desktop\class_spatial_map.npy", class_spatial_map)
plt.imshow(class_spatial_map)
plt.show()
