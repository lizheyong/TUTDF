## TUTDF: A Transfer- Based Framework for Underwater Target Detection from Hyperspectral Imagery

The detection of underwater targets through hyperspectral imagery is a relatively novel topic as the assumption of target background independence is no longer valid, making it difficult to directly detect underwater targets using land target information. Meanwhile, deep-learning based methods have faced challenges regarding the availability of training datasets, especially in underwater conditions. To solve these problems, a transfer-based framework is proposed,  which exploits synthetic data to train deep-learning models and transfers them to real-world applications. However, the transfer becomes challenging due to the imparity in the distribution between real and synthetic data. To address this dilemma, the proposed framework, named the transfer-based underwater target detection framework (TUTDF), first divides the domains using the depth information, then trains models for different domains and develops an adaptive module to determine which model to use. Meanwhile, a spatial–spectral process is applied prior to detection, which is devoted to eliminating the adverse influence of background noise. Since there is no publicly available hyperspectral underwater target dataset, most of the existing methods only run on simulated data; therefore, we conducted expensive experiments to obtain datasets with accurate depths and use them for validation. Extensive experiments verify the effectiveness and efficiency of TUTDF in comparison with traditional methods.



<div align=center><img src="https://github.com/lijinchao98/TUTDF/blob/master/model/resnet_readme/fig.jpg" width="600px" alt="TUTDF"></div>



* **Rreference:** [2023, Zheyong Li, A Transfer- Based Framework for Underwater Target Detection from Hyperspectral Imagery](https://doi.org/10.3390/rs15041023). 
* This is a partial implementation of TUTDF, for reference only.

## Requirements
```
Python = 3.6.x
Pytorch >= 1.6.0
CUDA >= 10.1
```
## Data Format
* Hyperspectral data (.hdr) is converted to (.npy) by 'hdr_2_npy' and then flattened to the shape (pixels, bands) before use it.
* All data used are reflectance data, i.e. after reflectance correction.


## TUTDF: 基于迁移的高光谱水下目标检测框架

通过高光谱图像探测水下目标是一个相对新颖的课题，因为目标背景独立的假设不再有效，使得利用陆地目标信息直接探测水下目标变得困难。同时，基于深度学习的方法在训练数据集的可用性方面面临挑战，特别是在水下条件下。为了解决这些问题，提出了一个基于转移的框架，它利用合成数据来训练深度学习模型，并将其转移到真实世界的应用中。然而，由于真实数据和合成数据之间分布的不确定性，转移变得具有挑战性。为了解决这一难题，所提出的框架被命名为基于转移的水下目标检测框架（TUTDF），首先利用深度信息划分领域，然后为不同的领域训练模型，并开发一个自适应模块来决定使用哪个模型。同时，在检测前采用空间-光谱过程，致力于消除背景噪声的不利影响。由于没有公开可用的高光谱水下目标数据集，大多数现有的方法只在模拟数据上运行；因此，我们进行了昂贵的实验，以获得具有准确深度的数据集，并将其用于验证。大量的实验验证了TUTDF与传统方法相比的有效性和效率。



<div align=center><img src="https://github.com/lijinchao98/TUTDF/blob/master/model/resnet_readme/fig.jpg" width="600px" alt="TUTDF"></div>



* **参考文献:** [2023, Zheyong Li, A Transfer- Based Framework for Underwater Target Detection from Hyperspectral Imagery](https://doi.org/10.3390/rs15041023). 
* 这是部分实现，仅供参考。

## 环境需求
```
Python = 3.6.x
Pytorch >= 1.6.0
CUDA >= 10.1
```
## 数据格式
* 高光谱数据 (.hdr) 转换成 (.npy) 通过 'hdr_2_npy' 然后在输入网络前形状变为 (像元数, 波段数)。
* 所使用的数据都是反射率数据。
