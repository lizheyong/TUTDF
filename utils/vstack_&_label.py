import numpy as np

"""
拼接水的数据集
"""
# file_name = ['0.1m_150x150.npy', '0.2m_150x150.npy', '0.3m_150x150.npy', '0.4m_150x150.npy',
#              '0.5m_150x150.npy', '0.6m_150x150.npy', '0.7m_150x150.npy', '0.8m_150x150.npy',
#              '0.9m_150x150.npy', '1.0m_150x150.npy', '1.3m_150x150.npy', '1.6m_150x150.npy']
# for i in range(len(file_name)):
#     file_name[i] = f'D:\LiZheyong\data\iron_30_water_30_depth_0-3_0.01\裁剪的水\{file_name[i]}'
#
# all_curve = np.zeros((1,189))       # 一个占位，往后拼接
#
# for file in file_name:
#     tmp = np.load(file)
#     all_curve = np.vstack((all_curve, tmp))
#     print(file)
#
# all_curve = np.delete(all_curve, 0, 0)  # 最后删了这个占位的
# print(all_curve.shape)
# np.save(r'D:\LiZheyong\data\iron_30_water_30_depth_0-3_0.01\裁剪的水\all_water', all_curve)

"""
拼接水和水下目标数据集，并将两个标签合到一起
"""
this = '0.9m'

water = np.load(fr"C:\Users\zheyong\Desktop\高光谱目标检测报告\橡胶测试\{this}\水100x100\{this}_water.npy")[:,9:129]
target = np.load(fr"C:\Users\zheyong\Desktop\高光谱目标检测报告\橡胶测试\{this}\合成数据\{this}_no_synthetic_data.npy")

train_data = np.vstack((water,target))

water_label = np.zeros(10000)
target_label = np.ones(10000)

train_label = np.hstack((water_label, target_label))

print(f'train_data:{train_data.shape}, train_label{train_label.shape}')
np.save(fr"C:\Users\zheyong\Desktop\高光谱目标检测报告\橡胶测试\{this}\训练\{this}_train.npy", train_data)
np.save(fr"C:\Users\zheyong\Desktop\高光谱目标检测报告\橡胶测试\{this}\训练\{this}_train_label.npy", train_label)
# train_data:(540000, 189), train_label(540000,)