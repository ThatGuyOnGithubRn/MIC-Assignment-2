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

plt.imshow(image1, cmap='gray')
# plt.title('Image 1: Chest CT')
plt.axis('off')
plt.savefig('image1.png', bbox_inches='tight', pad_inches=0)
plt.close()
plt.imshow(image2, cmap='gray')
# plt.title('Image 2: My Phantom')
plt.axis('off')
plt.savefig('image2.png', bbox_inches='tight', pad_inches=0)
plt.close()

def rrmse(recon, ref):
    return np.sqrt(np.sum((recon - ref)**2) / np.sum(ref**2))

base_theta = np.arange(0, 180, 1)%180
radon_transform1 = radon(image1, theta=base_theta, circle=False)
radon_transform2 = radon(image2, theta=base_theta, circle=False)    
recon1 = iradon(radon_transform1, theta=base_theta, circle=False)
recon2 = iradon(radon_transform2, theta=base_theta, circle=False)
recon1 = (recon1 - recon1.min()) / (recon1.max() - recon1.min())
recon2 = (recon2 - recon2.min()) / (recon2.max() - recon2.min())
print(f"RRMSE for Image 1 (unfiltered): {rrmse(recon1, image1):.6f}")
print(f"RRMSE for Image 2 (unfiltered): {rrmse(recon2, image2):.6f}")


plt.imshow(recon1, cmap='gray')
# plt.title('Unfiltered Reconstruction: Image 1')
plt.axis('off')
plt.savefig('full_degree_recon1.png', bbox_inches='tight', pad_inches=0)
plt.close()
plt.imshow(recon2, cmap='gray')
# plt.title('Unfiltered Reconstruction: Image 2')
plt.axis('off')
plt.savefig('full_degree_recon2.png', bbox_inches='tight', pad_inches=0)
plt.close()

for i, image in enumerate([image1, image2]):
    min_rrmse = float('inf')
    argmin_rrmse = None
    rrmses = []
    for theta_0 in range(181):
        theta = np.arange(theta_0, theta_0 + 151) % 180
        radon_transform = radon(image, theta=theta, circle=False)
        filtered = iradon(radon_transform, theta=theta, circle=False)
        filtered = (filtered - filtered.min()) / (filtered.max() - filtered.min())
        rrmses.append(rrmse(filtered, image))
        if rrmses[-1] < min_rrmse:
            min_rrmse = rrmses[-1]
            argmin_rrmse = theta_0
    plt.plot(range(181), rrmses)
    plt.xlabel('Starting Angle (degrees)')    
    plt.ylabel('RRMSE')
    # plt.title(f'RRMSE vs Starting Angle for Image {i+1}')
    plt.grid()
    plt.savefig(f'rrmse_image_{i+1}.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Image {i+1}: min_rrmse = {min_rrmse}, argmin_rrmse = {argmin_rrmse}")
    theta_opt = np.arange(argmin_rrmse, argmin_rrmse + 151) % 180
    radon_transform_opt = radon(image, theta=theta_opt, circle=False)
    filtered_opt = iradon(radon_transform_opt, theta=theta_opt, circle=False)
    filtered_opt = (filtered_opt - filtered_opt.min()) / (filtered_opt.max() - filtered_opt.min())
    plt.imshow(filtered_opt, cmap='gray')
    # plt.title(f'Optimal Reconstruction for Image {i+1}')
    plt.axis('off')
    plt.savefig(f'optimal_reconstruction_image_{i+1}.png', bbox_inches='tight', pad_inches=0)
    plt.close()