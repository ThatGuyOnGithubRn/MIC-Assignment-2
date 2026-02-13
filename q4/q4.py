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

image1/=image1.max()

def myXrayIntegration(f,t,theta_deg,delta_s,interpolation_scheme=1):
    theta_rad = np.deg2rad(theta_deg)
    costheta = np.cos(theta_rad)
    sintheta = np.sin(theta_rad)
    s = np.arange(-int((f.shape[0]+2)//2), int((f.shape[0]+2)//2), delta_s) 
    x_coords = ((f.shape[0]-1)/2 + t*costheta-s*sintheta)  
    y_coords = ((f.shape[1]-1)/2 + t*sintheta+s*costheta)  
    x_coords = np.clip(x_coords, 0, f.shape[0]-1)
    y_coords = np.clip(y_coords, 0, f.shape[1]-1)
    coords = np.vstack((y_coords, x_coords))      

    # print(coords)
    return coords

def rrmse(image1, image2):
    # image1 = (image1 - image1.min()) / (image1.max() - image1.min())
    # image2 = (image2 - image2.min()) / (image2.max() - image2.min())
    return np.sqrt(np.sum((image1 - image2) ** 2) / np.sum(image1 ** 2))

def integrate(f, coords):
    return map_coordinates(f, coords, order=1).sum()

rrmses = []
def myART(ordered_projections, lr, num_iterations=100):
    global image1, rrmses
    reconstructed_image = np.zeros(image1.shape)
    num_projections = ordered_projections.shape[0]
    for iteration in range(num_iterations):
        latest_rrmses = []
        for i in trange(num_projections):
            t, theta = ordered_projections[i]
            coords = myXrayIntegration(reconstructed_image, t, theta, delta_s=1)
            current_projection = integrate(reconstructed_image, coords)
            ordered_projection = integrate(image1, coords)
            error = ordered_projection - current_projection
            correction = lr * error / len(coords[0])  
            floored = np.floor(coords).astype(int)
            frac = coords - floored
            ceiled = np.ceil(coords).astype(int)
            wy = frac[0, :]
            wx = frac[1, :]            
            reconstructed_image[ceiled[0], ceiled[1]] += correction * (1 - wy) * (1 - wx)
            reconstructed_image[floored[0], ceiled[1]] += correction * wy * (1 - wx)
            reconstructed_image[ceiled[0], floored[1]] += correction * (1 - wy) * wx
            reconstructed_image[floored[0], floored[1]] += correction * wy * wx
            np.clip(reconstructed_image, 0, 1, out=reconstructed_image)
            # latest_rrmses.append(rrmses[-1])
            # print(i, rrmses[-1])
            # # if len(latest_rrmses) > 100 and max(latest_rrmses[-101:-1]) - min(latest_rrmses[-101:-1]) < 1e-5:
            # #     print("Convergence reached at iteration", iteration, "projection", i)
            # #     break
            # if len(latest_rrmses) > 100 and max(latest_rrmses[-101:-1]) < latest_rrmses[-1]:
            #     print("Convergence reached at iteration", iteration, "projection", i)
            #     break
        rrmses.append(rrmse(image1, reconstructed_image))
        print(f"Iteration {iteration+1}/{num_iterations}, RRMSE: {rrmses[-1]:.6f}")
        # lr /= 10
    # with open('reconstructed_image.txt', 'w') as f:
    #     for row in reconstructed_image:
    #         f.write(' '.join(map(str, row)) + '\n')
    plt.imshow(reconstructed_image, cmap='gray')
    plt.axis('off')
    plt.savefig(f'reconstructed_image_lambda_{lr}.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    return reconstructed_image


t = np.arange(-255, 255, 1)
thetha = np.arange(0, 180, 1)
coords = np.array(np.meshgrid(t, thetha)).reshape(2, -1).T
import random
random.shuffle(coords)

radon_transform = radon(image1, theta=thetha, circle=False)
iradon_transform = iradon(radon_transform, theta=thetha, circle=False, filter_name=None)
iradon_transform = (iradon_transform - iradon_transform.min()) / (iradon_transform.max() - iradon_transform.min())
print(rrmse(iradon_transform, image1))
plt.imshow(iradon_transform, cmap='gray')
plt.axis('off')
plt.savefig('iradon_transform.png', bbox_inches='tight', pad_inches=0)
plt.close()

# load the reconstructed image from reconstructed_image_lambda_0.1.txt
# with open('reconstructed_image.txt', 'r') as f:
#     reconstructed_image = np.array([[float(x) for x in line.split()] for line in f])
# print(rrmse(reconstructed_image, image1))

for lr in range(1, 11):
    myART(coords, lr=lr/10, num_iterations=10)
    plt.plot(rrmses)
    plt.xlabel('Iteration')
    plt.ylabel('RRMSE')
    # plt.title('RRMSE vs Iteration')
    plt.grid()
    plt.savefig(f'rrmse_plot_lambda_{lr}.png', bbox_inches='tight', pad_inches=0)
    rrmses = []
