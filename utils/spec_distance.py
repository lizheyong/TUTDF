from add_target import add_target_pixel
import numpy as np
import matplotlib.pyplot as plt

def spec_distance(r_inf, r):
        return np.linalg.norm(r_inf - r)


if __name__ == '__main__':
        # 读取要合成的“目标曲线”，“水曲线”，”波长范围“
        R_B = np.load(r'D:\LiZheyong\data\iron_30_water_30_depth_0-3_0.01\iron.npy')
        R_INF = np.load(r'D:\LiZheyong\data\iron_30_water_30_depth_0-3_0.01\water.npy')
        wavelength = np.load(r'D:\LiZheyong\resnet\dataset\wavelength.npy')
        # 设置深度，深度变化
        H = np.linspace(0.01, 5, 500)
        # 合成目标水下反射率曲线
        data_len = len(H)
        R_synthetic_data = np.zeros((data_len, 189))
        Rrs_synthetic_data = np.zeros((data_len, 189))
        R_distance = np.zeros(data_len)
        Rrs_distance = np.zeros(data_len)

        print(f'H:{len(H)}  data_len:{data_len}')
        s = 0 # 替换空合成数据，索引s从0开始
        for h in range(len(H)):
                R_synthetic_data[s] = add_target_pixel(R_B[1], R_INF[1], H[h])
                Rrs_synthetic_data[s] = add_target_pixel(R_B[1], R_INF[1], H[h], Rrs=True)

                R_distance[s] = spec_distance(R_B[1],R_synthetic_data[s])
                Rrs_distance[s] = spec_distance(R_B[1],Rrs_synthetic_data[s])
                s += 1

        plt.figure()
        plt.plot(H, R_distance,
                 label='R', color='r', marker='o', markersize=3)
        plt.plot(H, Rrs_distance,
                 label='Rrs', color='b', marker='o', markersize=3)
        plt.xlabel('depth')
        plt.ylabel('spec distance')
        plt.legend()
        plt.show()
