from add_target import add_target_pixel
import numpy as np
import matplotlib.pyplot as plt

def spec_distance(r_inf, r):
        return np.linalg.norm(r_inf - r)


if __name__ == '__main__':
        # 读取要合成的“目标曲线”，“水曲线”，”波长范围“
        R_1 = np.load(r"C:\Users\zheyong\Desktop\铁测试\铁10x10\0.1m_Iron.npy")[:,:188]
        R_2 = np.load(r"C:\Users\zheyong\Desktop\石测试\原始HSI\stone.npy")[:,:188]
        R_3 = np.load(r"C:\Users\zheyong\Desktop\橡胶测试\原始HSI\rubber.npy")[:,:188]
        R_INF1 = np.load(r"C:\Users\zheyong\Desktop\铁测试\2.2m\水100x100\2.2m_water.npy")[:,:188]
        R_INF2 = np.load(r"C:\Users\zheyong\Desktop\石测试\0.4m\水100x100\0.4m_water.npy")[:,:188]
        R_INF3 = np.load(r"C:\Users\zheyong\Desktop\橡胶测试\0.1m\水100x100\0.1m_water.npy")[:,:188]
        # wavelength = np.load(r"C:\Users\zheyong\Desktop\铁测试\wavelength.npy")
        # 设置深度，深度变化
        H = np.linspace(0, 3, 100)
        # 合成目标水下反射率曲线
        data_len = len(H)
        R_synthetic_data_1 = np.zeros((data_len, 188))
        R_synthetic_data_2 = np.zeros((data_len, 188))
        R_synthetic_data_3 = np.zeros((data_len, 188))

        R_distance1 = np.zeros(data_len)
        R_distance2 = np.zeros(data_len)
        R_distance3 = np.zeros(data_len)


        print(f'H:{len(H)}  data_len:{data_len}')
        s = 0 # 替换空合成数据，索引s从0开始
        for h in range(len(H)):
                R_synthetic_data_1[s] = add_target_pixel(R_1[1], R_INF1[1], H[h])
                R_distance1[s] = spec_distance(R_1[1],R_synthetic_data_1[s])
                s += 1
        print(f'H:{len(H)}  data_len:{data_len}')
        s = 0 # 替换空合成数据，索引s从0开始
        for h in range(len(H)):
                R_synthetic_data_2[s] = add_target_pixel(R_2[1], R_INF2[1], H[h])
                R_distance2[s] = spec_distance(R_2[1],R_synthetic_data_2[s])
                s += 1
        print(f'H:{len(H)}  data_len:{data_len}')
        s = 0 # 替换空合成数据，索引s从0开始
        for h in range(len(H)):
                R_synthetic_data_3[s] = add_target_pixel(R_3[1], R_INF3[1], H[h])
                R_distance3[s] = spec_distance(R_3[1],R_synthetic_data_3[s])
                s += 1



        plt.figure()
        plt.plot(H, R_distance1,label='Iron', color='r', marker='o', markersize=2)
        plt.plot(H, R_distance2,label='Stone', color='g', marker='+', markersize=2)
        plt.plot(H, R_distance3,label='Rubber', color='b', marker='d', markersize=2)
        plt.xlabel('Depth (m)')
        plt.ylabel('Spec Distance')
        plt.legend()
        plt.savefig('sd.jpg', dpi=1000)

        plt.grid(axis="y")
        plt.show()
