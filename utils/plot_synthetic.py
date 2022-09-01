import torch
from math import pi, e, cos
from scipy.ndimage import gaussian_filter1d
from model.iop.iop_model import Encode_Net, Decode_Net1, Decode_Net2
import numpy as np
import matplotlib.pyplot as plt


def add_target_pixel(r_b, r_inf, h):
        """
        input:
                r_b: reflectance spectrum of target[1d numpy]
                r_inf : reflectance spectrum of water[1d numpy]
                h: depth
        return:
                new_curve: [1d npy]
        param:
                'r'     :   sensor-observed spectrum
                'a'     :   absorption rate (estimate by the IOPE_Net)
                'b'     :   scatter rate (estimate by the IOPE_Net)
                'k_d'   :   downwelling attenuation coefficients
                'k-uc'  :   upwelling attenuation coefficients of water column
                'k-ub'  :   upwelling attenuation coefficients of target column
        """
        # 加载IOPE模型,用r_inf得到a,b
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net0 = Encode_Net(n_channels=1, n_classes=64)
        net1 = Decode_Net1(n_channels=64, n_classes=1)
        net2 = Decode_Net2(n_channels=64, n_classes=1)
        # 将网络拷贝到deivce中
        net0.to(device=device)
        net1.to(device=device)
        net2.to(device=device)
        net0.load_state_dict(torch.load(fr"C:\Users\zheyong\Desktop\高光谱目标检测报告\铁测试\{this}\水100x100\best_model_net0.pth", map_location=device))  # 加载模型参数
        net1.load_state_dict(torch.load(fr"C:\Users\zheyong\Desktop\高光谱目标检测报告\铁测试\{this}\水100x100\best_model_net1.pth", map_location=device))  # 加载模型参数
        net2.load_state_dict(torch.load(fr"C:\Users\zheyong\Desktop\高光谱目标检测报告\铁测试\{this}\水100x100\best_model_net2.pth", map_location=device))
        net0.eval()
        net1.eval()
        net2.eval()

        # plot_r_inf = r_inf # 画图使用
        # plot_r_b = r_b # 画图使用
        r_inf = torch.from_numpy(r_inf).to(device=device, dtype=torch.float32).reshape(1, 1, -1) # tensor(1,1,189)
        r_b = torch.from_numpy(r_b).to(device=device, dtype=torch.float32).reshape(1, 1, -1)
        encode_out = net0(r_inf)
        a = net1(encode_out) # tensor(1,1,189)
        b = net2(encode_out)

        # 计算水下目标反射率
        u = b / (a + b)
        k = a + b
        k_uc = 1.03 * (1 + 2.4 * u) ** (0.5) * k
        k_ub = 1.04 * (1 + 5.4 * u) ** (0.5) * k
        theta = 0
        k_d = k / cos(theta)
        r = r_inf * (1 - e ** (-(k_d + k_uc) * h)) + r_b  * e ** (-(k_d + k_ub) * h) # tensor(1,1,189)
        r = r.squeeze().detach().cpu().numpy() # np(189)

        """画图测试"""
        # wavelength = np.load(r"C:\Users\423\Desktop\铁测试\wavelength.npy")
        # plt.figure()
        # plt.plot(wavelength, gaussian_filter1d(r,sigma=5),
        #          label='systhetic', color='r', marker='o', markersize=3)
        # plt.plot(wavelength, gaussian_filter1d(plot_r_inf,sigma=5),
        #          label='water_inf', color='b', marker='o', markersize=3)
        # plt.plot(wavelength, gaussian_filter1d(plot_r_b,sigma=5),
        #          label='r_b', color='g', marker='o', markersize=3)
        # plt.xlabel('wavelength(nm)')
        # plt.ylabel('reflect value')
        # plt.legend()
        # plt.show()
        """画图测试"""

        return r

if __name__ == '__main__':
        # 读取要合成的“目标曲线”，“水曲线”，”波长范围“
        this = '2.2m'
        R_B = np.load(fr"C:\Users\zheyong\Desktop\高光谱目标检测报告\铁测试\铁10x10\0.1m_Iron.npy")[10:20,9:129]
        R_INF = np.load(fr"C:\Users\zheyong\Desktop\高光谱目标检测报告\铁测试\{this}\水100x100\{this}_water.npy")[10:20,9:129]
        wavelength = np.load(r"C:\Users\zheyong\Desktop\高光谱目标检测报告\铁测试\wavelength.npy")
        # 设置深度，深度变化
        H = np.linspace(0,0.4, 10)
        # 合成目标水下反射率曲线
        synthetic_data = np.zeros((10, 120))
        s = 0 # 替换空合成数据，索引s从0开始
        for h in range(len(H)):
                synthetic_data[s] = add_target_pixel(R_B[0], R_INF[0], H[h])
                s += 1
        plt.figure()
        color = ['r','deeppink','m','violet','b','slategray','aqua','g','orange','peru']
        for s in range(10):
                plt.plot(wavelength, gaussian_filter1d(synthetic_data[s] ,sigma=5),
                         label=f'Systhetic{s}', color=color[s], marker='o', markersize=3)
        plt.plot(wavelength, gaussian_filter1d(R_INF[0],sigma=5),
                 label='Water_Inf', color='black', marker='o', markersize=3)

        plt.xlabel('Wavelength(nm)')
        plt.ylabel('Reflectence')
        plt.legend()
        plt.show()


