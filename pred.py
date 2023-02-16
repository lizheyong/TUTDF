from model.model import *
from model.dataset import  test_Loader
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib

def test_net(net, test_dataset, device, batch_size):
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    net.load_state_dict(torch.load(fr"xxx.pth"))
    net.eval()
    class_map = np.ones((1))
    for curve in test_loader:
        curve = curve.unsqueeze(1).to(device=device, dtype=torch.float32)
        out = net(curve)
        out = out[:, 1]
        out = out.detach().cpu().numpy()
        class_map = np.hstack((class_map, out))
    class_map = np.delete(class_map, 0, 0)
    # the shape should set by yourself
    class_map = np.resize(class_map, (200, 200))
    np.save(fr'xxx\result', class_map)
    matplotlib.image.imsave('result.png', class_map)
    plt.imshow(class_map)
    plt.show()


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ResNet = ResNet().to(device=device)
    dataset = test_Loader(fr"xxx.npy")
    test_net(net=ResNet, test_dataset=dataset, device=device, batch_size=4000)