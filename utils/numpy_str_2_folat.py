import numpy as np
"""
str格式的np变成float的
['1', '2'] ==> [1, 2]
"""

n_str = np.load(r"C:\Users\423\Desktop\铁测试\wavelength.npy")
n_float = n_str.astype(float)
np.save(r"C:\Users\423\Desktop\铁测试\wavelength.npy",n_float)