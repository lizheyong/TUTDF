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

    file_name = ['0.1m_150x150.hdr', '0.2m_150x150.hdr', '0.3m_150x150.hdr', '0.4m_150x150.hdr',
                '0.5m_150x150.hdr','0.6m_150x150.hdr', '0.7m_150x150.hdr', '0.8m_150x150.hdr',
                '0.9m_150x150.hdr', '1.0m_150x150.hdr','1.3m_150x150.hdr', '1.6m_150x150.hdr']
    for i in range(len(file_name)):
        file_name[i] = f'D:\LiZheyong\data\iron_30_water_30_depth_0-3_0.01\裁剪的水\{file_name[i]}'

    for file in file_name:
        name,_ = os.path.splitext(file)
        HSI_image = open_file(file)
        all_curve = HSI_image.read_pixel(0, 0) # 读一个占位，往后拼接，最后删了这个占位的
        for raw in range(150):
            for col in range(150):
                pixel_curve = HSI_image.read_pixel(raw, col)
                all_curve = np.vstack((all_curve, pixel_curve))
            print(raw)
        print(file)
        all_curve = np.delete(all_curve, 0, 0)
        np.save(name, all_curve)
        print(all_curve.shape)