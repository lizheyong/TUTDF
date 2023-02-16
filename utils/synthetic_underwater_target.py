import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from math import pi, e, cos
from scipy.ndimage import gaussian_filter1d
from model.iope_net.iope_net import Encode_Net, Decode_Net1, Decode_Net2

"""
Select the target land pixel, water pixel, and depth 
to synthesize the reflectance curve of the target underwater
"""

def add_target_pixel(r_b, r_inf, h):
        """
        input:
                r_b: reflectance of target[1d numpy]
                r_inf : reflectanc of optically deep water[1d numpy]
                h: depth
        return:
                synthetic_underwater_target_reflectance: [1d npy]
        param:
                'r'     :   underwater target's off-water reflectance
                'a'     :   absorption coefficient (estimate by the IOPE_Net)
                'bb'     :   scatter coefficient (estimate by the IOPE_Net)
                'k_d'   :   downwelling attenuation coefficient
                'k-uc'  :   upwelling attenuation coefficient of water column
                'k-ub'  :   upwelling attenuation coefficient of target
        """
        # load trained IOPE_Net model, get a,b from 'r_inf'
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net0 = Encode_Net().to(device=device)
        net1 = Decode_Net1().to(device=device)
        net2 = Decode_Net2().to(device=device)
        net0.load_state_dict(torch.load(fr"xxx\best_model_net0.pth", map_location=device))
        net1.load_state_dict(torch.load(fr"xxx\best_model_net1.pth", map_location=device))
        net2.load_state_dict(torch.load(fr"xxx\best_model_net2.pth", map_location=device))
        net0.eval()
        net1.eval()
        net2.eval()

        # copy them for plot, because their format will change to input the net.
        plot_r_inf = r_inf
        plot_r_b = r_b

        r_inf = torch.from_numpy(r_inf).to(device=device, dtype=torch.float32).reshape(1, 1, -1) # tensor(1,1,189)
        r_b = torch.from_numpy(r_b).to(device=device, dtype=torch.float32).reshape(1, 1, -1)
        encode_out = net0(r_inf)
        a = net1(encode_out) # tensor(1,1,189)
        bb = net2(encode_out)

        # Calculating the reflectance of underwater targets out of water
        u = bb / (a + bb)
        k = a + bb
        k_uc = 1.03 * (1 + 2.4 * u) ** (0.5) * k
        k_ub = 1.04 * (1 + 5.4 * u) ** (0.5) * k
        theta = 0
        k_d = k / cos(theta)
        r = r_inf * (1 - e ** (-(k_d + k_uc) * h)) + r_b  * e ** (-(k_d + k_ub) * h) # tensor(1,1,189)
        r = r.squeeze().detach().cpu().numpy() # np(189)

        """PLOT"""
        wavelength = np.load(r"xxx\wavelength.npy")
        plt.figure()
        plt.plot(wavelength, gaussian_filter1d(r,sigma=1),
                 label='systhetic', color='r', marker='o', markersize=3)
        plt.plot(wavelength, gaussian_filter1d(plot_r_inf,sigma=1),
                 label='water_inf', color='b', marker='o', markersize=3)
        plt.plot(wavelength, gaussian_filter1d(plot_r_b,sigma=1),
                 label='r_b', color='g', marker='o', markersize=3)
        plt.xlabel('wavelength(nm)')
        plt.ylabel('reflectance')
        plt.legend()
        plt.show()
        """PLOT"""

        return r

if __name__ == '__main__':

        R_B = np.load(fr"xxx\stone.npy")[0:5]
        R_INF = np.load(fr"xxx\water.npy")[0:10]
        H = np.linspace(0.9, 1.7, 100)

        data_len = len(R_B)*len(R_INF)*len(H)
        synthetic_data = np.zeros((data_len, R_B.shape[1]))
        print(f'R_B:{len(R_B)}  R_INF:{len(R_INF)}  H:{len(H)}  number of synthetic data:{data_len}')
        s = 0
        for i in range(len(R_B)):
                for j in range(len(R_INF)):
                        for h in range(len(H)):
                                synthetic_data[s] = add_target_pixel(R_B[i], R_INF[j], H[h])
                                s += 1
                                if s%len(H)==0:
                                        print(f'[{j}/{len(R_INF)}][{i}/{len(R_B)}]')
        np.save(fr'xxx\synthetic_data.npy', synthetic_data)