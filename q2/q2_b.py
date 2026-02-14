import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize,radon,iradon
from scipy.ndimage import map_coordinates
import os
from scipy.signal import convolve2d
from skimage.filters import gaussian

def rrmse(recon, ref):
    return np.sqrt(np.sum((recon - ref)**2) / np.sum(ref**2))

def myFilter(f, filter_type='ram_lak', L=None):
    N = f.shape[0]
    w = np.fft.fftfreq(N).reshape(-1,1)
    # w*=(2*np.pi)
    if L is None:
        L = np.max(np.abs(w))
    ramp = np.abs(w)
    ramp[np.abs(w) > L] = 0 
    if filter_type == 'ram_lak':
        filter_response = ramp
    elif filter_type == 'shepp_logan':
        filter_response = ramp * np.sinc(w / (2*L*np.pi))
    elif filter_type == 'cosine':
        filter_response = ramp * np.cos(np.pi * w / (2*L))
    elif filter_type == 'none':
        filter_response = np.ones_like(w)
    # filter_response = np.fft.fftshift(filter_response, axes=0)
    f_fft = np.fft.fft(f, axis=0)
    filtered_fft = f_fft * filter_response
    filtered_f = np.real(np.fft.ifft(filtered_fft, axis=0))
    return filtered_f

def gaussian_kernel(size, sigma):
    ax = np.arange(-(size//2), size//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2)/(2*sigma**2))
    return kernel / np.sum(kernel)

mask1 = gaussian_kernel(11, 1)
mask5 = gaussian_kernel(51, 5)

def main():
    N = 128
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (N, N), mode='reflect', anti_aliasing=True)

    radon_transform = radon(phantom, theta=np.arange(0, 180, 3), circle=False)

    freqs = np.fft.fftfreq(radon_transform.shape[0])
    wmax = np.max(np.abs(freqs))
    filter_types = ['ram_lak', 'none']
    L_values = [wmax]
    if not os.path.exists('./output_q2_b'):
        os.makedirs('./output_q2_b')
    # filename = f'./output_q2/orig_phantom.png'
    # plt.imshow(phantom, cmap='gray')
    # # plt.title('Original Phantom')
    # plt.axis('off')
    # plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    # print(f"Saved {filename}")
    for mask, sigma in zip([None, mask1, mask5], [0, 1, 5]):
        if mask is not None:
            orig_image = convolve2d(phantom, mask, mode='same', boundary='symm')
        else:
            orig_image = phantom
        orig_image = (orig_image - orig_image.min()) / (orig_image.max() - orig_image.min())
        radon_transform = radon(orig_image, theta=np.arange(0, 180, 3), circle=False)
        plt.imshow(orig_image, cmap='gray')
        plt.axis('off')
        filename = f'./output_q2_b/radon_smoothed_sigma_{sigma}.png'
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

        for filter_type in filter_types:
            for L in L_values:
                filtered = myFilter(radon_transform, filter_type=filter_type, L=L)
                recon_img = iradon(filtered, theta=np.arange(0, 180, 3), circle=False, filter_name=None)
                # recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())
                

                # with open(f'recon_img_{filter_type}_L_{L:.3f}.txt', 'w') as f:
                #     for row in recon_img:
                #         f.write(' '.join(f'{val:.6f}' for val in row) + '\n')
                # recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())
                print(f"RRMSE for {filter_type} filter with L={L/(2*np.pi):.3f}: {rrmse(recon_img, orig_image):.6f}")
                filename = f'./output_q2_b/recon_{filter_type}_L_{L:.3f}_sigma_{sigma}.png'
                plt.imshow(recon_img, cmap='gray')
                plt.axis('off')
                plt.savefig(filename, bbox_inches='tight', pad_inches=0)
                plt.close()


if __name__ == "__main__":
    main()