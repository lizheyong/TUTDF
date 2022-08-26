from scipy import io, misc
import os
import spectral
import numpy as np


def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return misc.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))

if __name__ == "__main__":


    # thislist = ['0.1m','0.3m','0.6m','0.9m']
    thislist = ['stone10x10']
    for this in thislist:
        file = fr"C:\Users\zheyong\Desktop\石测试\原始HSI\\{this}.hdr"
        name,_ = os.path.splitext(file)
        HSI_image = open_file(file)
        all_curve = HSI_image.read_pixel(0, 0) # 读一个占位，往后拼接，最后删了这个占位的
        for raw in range(10):
            for col in range(10):
                pixel_curve = HSI_image.read_pixel(raw, col)
                all_curve = np.vstack((all_curve, pixel_curve))
            print(raw)
        print(file)
        all_curve = np.delete(all_curve, 0, 0)
        np.save(name, all_curve)
        print(all_curve.shape)