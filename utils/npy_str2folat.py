import numpy as np

"""
npy str2float
['1', '2'] ==> [1, 2]
"""

n_str = np.load(r"xxx.npy")
n_float = n_str.astype(float)
np.save(r"xxx.npy",n_float)