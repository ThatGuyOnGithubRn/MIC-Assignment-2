import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize,radon,iradon
from scipy.ndimage import map_coordinates
import os

def rrmse(recon, ref):
    return np.sqrt(np.sum((recon - ref)**2) / np.sum(ref**2))

def myFilter(f, filter_type='ram_lak', L=None):
    N = f.shape[0]
    w = np.fft.fftfreq(N).reshape(-1,1)
    if L is None:
        L = np.max(np.abs(w))
    ramp = np.abs(w)
    ramp[np.abs(w) > L] = 0 
    if filter_type == 'ram_lak':
        filter_response = ramp
    elif filter_type == 'shepp_logan':
        filter_response = ramp * np.sinc(w / (2*L))
    elif filter_type == 'cosine':
        filter_response = ramp * np.cos(np.pi * w / (2*L))
    elif filter_type == 'none':
        filter_response = np.ones_like(w)
    # filter_response = np.fft.fftshift(filter_response, axes=0)
    f_fft = np.fft.fft(f, axis=0)
    filtered_fft = f_fft * filter_response
    filtered_f = np.real(np.fft.ifft(filtered_fft, axis=0))
    return filtered_f

def main():
    N = 128
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (N, N), mode='reflect', anti_aliasing=True)
    # help(radon)
    radon_transform = radon(phantom, theta=np.arange(0, 180, 3), circle=False)

    freqs = np.fft.fftfreq(radon_transform.shape[0])
    wmax = np.max(np.abs(freqs))
    filter_types = ['ram_lak', 'shepp_logan', 'cosine']
    L_values = [wmax, wmax/2]
    if not os.path.exists('./output_filtered'):
        os.makedirs('./output_filtered')
    filtered = myFilter(radon_transform, filter_type='none', L=wmax)
    recon_img = iradon(filtered, theta=np.arange(0, 180, 3), circle=False, filter_name=None)
    recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())
    # with open('recon_img_no_filter.txt', 'w') as f:
    #     for row in recon_img:
    #         f.write(' '.join(f'{val:.6f}' for val in row) + '\n')
    print(f"RRMSE for no filter: {rrmse(recon_img, phantom):.6f}")
    filename = f'./output_filtered/recon_no_filter.png'
    plt.imshow(recon_img, cmap='gray')
    # plt.title(f'Reconstruction: No Filter')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    print(f"Saved {filename}")  
    plt.close()
    for filter_type in filter_types:
        for L in L_values:
            filtered = myFilter(radon_transform, filter_type=filter_type, L=L)
            recon_img = iradon(filtered, theta=np.arange(0, 180, 3), circle=False, filter_name=None)
            recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())
            # with open(f'recon_img_{filter_type}_L_{L:.3f}.txt', 'w') as f:
            #     for row in recon_img:
            #         f.write(' '.join(f'{val:.6f}' for val in row) + '\n')
            # recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())
            print(f"RRMSE for {filter_type} filter with L={L:.3f}: {rrmse(recon_img, phantom):.6f}")
            filename = f'./output_filtered/recon_{filter_type}_L_{L:.3f}.png'
            plt.imshow(recon_img, cmap='gray')
            # plt.title(f'Reconstruction: {filter_type}, L={L:.3f}')
            plt.axis('off')
            # plt.colorbar(label='Intensity')
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Saved {filename}")
            



if __name__ == "__main__":
    main()