from utils.synthetic_underwater_target import add_target_pixel
import matplotlib.pyplot as plt
import numpy as np


def spec_distance(r_inf, r):
        return np.linalg.norm(r_inf - r)


if __name__ == '__main__':

        R_1 = np.load(r"xxx_iron.npy")
        R_2 = np.load(r"xxx_stone.npy")
        R_3 = np.load(r"xxx_rubber.npy")

        R_INF1 = np.load(r"xxx_water1.npy")
        R_INF2 = np.load(r"xxx_water2.npy")
        R_INF3 = np.load(r"xxx_water3.npy")

        H = np.linspace(0, 3, 100)
        data_len = len(H)

        # bands = 188
        R_synthetic_data_1 = np.zeros((data_len, 188))
        R_synthetic_data_2 = np.zeros((data_len, 188))
        R_synthetic_data_3 = np.zeros((data_len, 188))

        R_distance1 = np.zeros(data_len)
        R_distance2 = np.zeros(data_len)
        R_distance3 = np.zeros(data_len)

        print(f'H:{len(H)}  data_len:{data_len}')
        s = 0
        for h in range(len(H)):
                R_synthetic_data_1[s] = add_target_pixel(R_1[1], R_INF1[1], H[h])
                R_distance1[s] = spec_distance(R_1[1], R_synthetic_data_1[s])
                s += 1
        print(f'H:{len(H)}  data_len:{data_len}')

        s = 0
        for h in range(len(H)):
                R_synthetic_data_2[s] = add_target_pixel(R_2[1], R_INF2[1], H[h])
                R_distance2[s] = spec_distance(R_2[1],R_synthetic_data_2[s])
                s += 1
        print(f'H:{len(H)}  data_len:{data_len}')

        s = 0
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