import torch.nn as nn
import torch

class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        kernel = [[0.125, 0.125, 0.125],
                  [0.125, 0, 0.125],
                  [0.125, 0.125, 0.125]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = F.conv2d(x.unsqueeze(1), self.weight, padding=2)
        return x
