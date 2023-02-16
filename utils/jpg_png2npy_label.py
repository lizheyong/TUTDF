from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt

# origin_image = Image.open(rf"xxx.jpg")
origin_image = Image.open(rf"xxx.png")

bw = origin_image.convert('L')#
bw.show()

# bw = bw.convert('1')

# invert
# invert_bw = ImageOps.invert(bw)
# invert_bw.show()

img = np.array(bw)
img = img+0

plt.imshow(img)
plt.show()

np.save(rf"xxxx",img)
# matplotlib.image.imsave(rf"xxx.png", img)



