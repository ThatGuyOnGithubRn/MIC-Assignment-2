import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
from scipy.special import i0, i1
from enum import Enum
import h5py
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize,radon,iradon
from scipy.ndimage import map_coordinates
import os
from scipy.signal import convolve2d
from skimage.filters import gaussian



file1 = h5py.File('../data/assignmentMathImagingRecon_chestCT.mat', 'r')
image1 = np.array(file1['imageAC'])

file2 = h5py.File('../data/assignmentMathImagingRecon_myPhantom.mat', 'r')
image2 = np.array(file2['imageMyPhantomAC'])
print(image1.shape, image2.shape)
