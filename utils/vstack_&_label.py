import numpy as np


"""
拼接水和水下目标数据集，并将两个标签合到一起
"""
water = np.load(fr"xxx\water.npy")
target = np.load(fr"xxx\synthetic_data.npy")

train_data = np.vstack((water,target))

water_label = np.zeros(len(water))
target_label = np.ones(len(target))
train_label = np.hstack((water_label, target_label))

print(f'train_data:{train_data.shape}, train_label{train_label.shape}')
np.save(fr"xxx\train.npy", train_data)
np.save(fr"xxx\train_label.npy", train_label)