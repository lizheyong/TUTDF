import numpy as np


file = np.load(fr"xxx.npy")

# if it is flatten format, first resize
s = np.resize(file, (200, 200, 189))

for i in range(1,199):
    for j in range(1,199):
        s[i,j,:] = 0.125*(s[i-1,j-1,:]+s[i-1,j,:]+s[i-1,j+1,:]+s[i,j-1,:]+s[i,j+1,:]+s[i+1,j-1,:]+s[i+1,j,:]+s[i+1,j+1,:])
s = np.resize(s, (40000, 189))
np.save(fr"xxxx_spatial.npy", s)
