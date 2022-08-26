from scipy import io, misc
import os
import spectral
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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
    thislist = ['9_mf','9_cem','9_ace','9_osp','9_sam_rule','9_tcimf']
    for this in thislist:
        file_name = [rf"C:\Users\zheyong\Desktop\{this}.hdr"]
        for file in file_name:
            name,_ = os.path.splitext(file)
            HSI_image = open_file(file)
            all_curve = HSI_image.read_pixel(0, 0) # 读一个占位，往后拼接，最后删了这个占位的
            for raw in range(100):
                for col in range(100):
                    pixel_curve = HSI_image.read_pixel(raw, col)
                    all_curve = np.vstack((all_curve, pixel_curve))
                print(raw)
            print(file)
            all_curve = np.delete(all_curve, 0, 0)
            np.save(name, all_curve)
            print(all_curve.shape)

            all_curve = np.resize(all_curve, (100, 100))
            np.save(rf"C:\Users\zheyong\Desktop\{this}", all_curve)
            # plt.imshow(all_curve)
            matplotlib.image.imsave(rf"C:\Users\zheyong\Desktop\{this}.png", all_curve)
            # plt.show()