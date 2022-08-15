import numpy as np
"""
str格式的np变成float的
['1', '2'] ==> [1, 2]
"""

n_str = np.load(r'D:\LiZheyong\data\iron_30_water_30_depth_0-3_0.01\iron.npy')
n_float = n_str.astype(float)
np.save(r'D:\LiZheyong\data\iron_30_water_30_depth_0-3_0.01\iron.npy',n_float)