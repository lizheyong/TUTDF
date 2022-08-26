import os
import glob
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
this = '0.9m'
# origin_image = Image.open(rf"C:\Users\zheyong\Desktop\{this}.jpg")  # 读取图片
origin_image = Image.open(rf"C:\Users\zheyong\Desktop\橡胶测试\{this}\传统方法结果\{this}_label.png")  # 读取图片

bw = origin_image.convert('L')# 转为 图
# 反转
bw = ImageOps.invert(bw)
bw = bw.convert('1')

# reverse_bw.show()
bw.show()
img = np.array(bw)
img = img+0


import matplotlib
plt.imshow(img)
plt.show()

np.save(rf"C:\Users\zheyong\Desktop\橡胶测试\{this}\传统方法结果\label_reverse",img)
# matplotlib.image.imsave(rf"C:\Users\zheyong\Desktop\橡胶测试\{this}\传统方法结果\label.png", img)



