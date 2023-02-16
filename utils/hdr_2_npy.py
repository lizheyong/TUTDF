from scipy import io, misc
import os
import spectral
import numpy as np


def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    # Load Matlab array
    if ext == '.mat':
        return io.loadmat(dataset)
    # Load TIFF file
    elif ext == '.tif' or ext == '.tiff':
        return misc.imread(dataset)
    # Recommend for '.hdr'
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))


if __name__ == "__main__":

    file = fr"xxxx\xxxx.hdr"
    name, _ = os.path.splitext(file)
    # type(HSI_image) is 'ImageArray' (Width, Height, Bands)
    HSI = open_file(file)
    # To numpy
    HSI = np.array(HSI)
    # Show the shape of input HSI
    print(f'The shape of the input HSI is:{HSI.shape}')
    # Save HSI as numpy, append the shape to filename
    np.save(f'{name}_{HSI.shape}', HSI)
    # Save the flatten HSI as numpy, append the shape to filename
    HSI_flatten = HSI.reshape((-1, HSI.shape[-1]))
    np.save(f'{name}_{HSI_flatten.shape}', HSI_flatten)