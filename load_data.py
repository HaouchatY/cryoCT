from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import numpy as np
import torch

def get_proj(sigma=1, pix_size=1, size=400):
    """
    Load the projection data from the .mat file and return it as a torch tensor.
    :param sigma: sigma value for the gaussian filter, in Angstrom
    :param pix_size: pixel size in Angstrom
    """
    data = loadmat("potential.mat")
    phase = data["output"]
    orig_pix_size = 0.25

    #result = np.sum(phase, axis=2)
    result = phase
    # Crop empty edges
    #result = crop(result, size=size)

    # Blur
    result = gaussian_filter(result, sigma=sigma / orig_pix_size)

    # Downsample
    result = result[::int(pix_size / orig_pix_size), ::int(pix_size / orig_pix_size)]

    return torch.tensor(result)

def crop(input_img, size):
    current_size = input_img.shape[0]
    if current_size > size:
        middle = int(current_size / 2)
        start = int(middle - size / 2)
        stop = start + size
        input_img = input_img[start:stop, start:stop]
    return input_img

