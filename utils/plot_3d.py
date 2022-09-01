from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator

fig = plt.figure()
ax = Axes3D(fig)
X = np.array([5,10,20,50,100])
# X = np.repeat(np.array([5,10,20,50,100]), 5)
Y = np.array([5,10,20,50,100])


X, Y = np.meshgrid(X, Y)
Z = np.array([0.8758215574449492,0.8312929648867864,0.7646790344319665,0.5983238605682635,0.8149,
    0.8160962479391176,0.8296167547112899,0.8486444810723188,0.8509368302626151,0.8473028301793784,
0.8744575470082876,0.902300940794171,0.7603675800614282,0.7183246854104015,0.7230264255789567,
     0.8340436297840562,0.5354045892079189,0.7795771254116026,0.7158609737607631,0.8189573257835192,
0.7320042166291336,0.7556758749704889,0.7234985573917407,0.6560967913114019,0.72007751258317]).reshape(5,5)



# 具体函数方法可用 help(function) 查看，如：help(ax.plot_surface)
ax.set_zlabel('AUC values')  # 坐标轴
ax.set_ylabel(r'$\beta$')
ax.set_xlabel('$\lambda$')

x_major_locator = MultipleLocator(20)
#把x轴的刻度间隔设置为10，并存在变量里
y_major_locator = MultipleLocator(20)
#把y轴的刻度间隔设置为10，并存在变量里
ax=plt.gca()
#ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
#把x轴的主刻度设置为10的倍数
ax.yaxis.set_major_locator(y_major_locator)
#把y轴的主刻度设置为10的倍数
plt.xlim(0,100)
#把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
plt.ylim(0,100)
#把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白

# ax.plot_surface( Y,X,Z, cmap='Oranges')
ax.plot_surface( Y,X,Z, cmap='copper')
ax.view_init(25,45)

plt.show()
