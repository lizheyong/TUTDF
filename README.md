## TUTDF: A Transfer- Based Framework for Underwater Target Detection from Hyperspectral Imagery

The detection of underwater targets through hyperspectral imagery is a relatively novel topic as the assumption of target background independence is no longer valid, making it difficult to directly detect underwater targets using land target information. Meanwhile, deep-learning based methods have faced challenges regarding the availability of training datasets, especially in underwater conditions. To solve these problems, a transfer-based framework is proposed,  which exploits synthetic data to train deep-learning models and transfers them to real-world applications. However, the transfer becomes challenging due to the imparity in the distribution between real and synthetic data. To address this dilemma, the proposed framework, named the transfer-based underwater target detection framework (TUTDF), first divides the domains using the depth information, then trains models for different domains and develops an adaptive module to determine which model to use. Meanwhile, a spatial–spectral process is applied prior to detection, which is devoted to eliminating the adverse influence of background noise. Since there is no publicly available hyperspectral underwater target dataset, most of the existing methods only run on simulated data; therefore, we conducted expensive experiments to obtain datasets with accurate depths and use them for validation. Extensive experiments verify the effectiveness and efficiency of TUTDF in comparison with traditional methods.



<div align=center><img src="https://github.com/lijinchao98/TUTDF\model\resnet_readme/fig.jpg" width="600px" alt="TUTDF"></div>



* **Rreference:** [2023, Zheyong Li, A Transfer- Based Framework for Underwater Target Detection from Hyperspectral Imagery](https://doi.org/10.3390/rs15041023). 
* This is a partial implementation of TUTDF, for reference only.

## Requirements
```
Python = 3.6.x
Pytorch >= 1.6.0
CUDA >= 10.1
```
## Data Format
* Hyperspectral data (.hdr) is converted to (.npy) by 'hdr_2_npy' and then flattened to the shape (pixels, wavelength) before use it.
* All data used are reflectance data, i.e. after reflectance correction.
