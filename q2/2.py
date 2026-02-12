import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize,radon,iradon
from scipy.ndimage import map_coordinates
import os




def prior(current_image, gamma=1e-3):
    abs_current_image = np.abs(current_image)
    return abs_current_image*gamma-gamma**2*np.log(1 + abs_current_image/gamma)

def apply_prior(orig_image):
    shifteds = np.stack([
        np.roll(orig_image, 1, axis=0),
        np.roll(orig_image, -1, axis=0),
        np.roll(orig_image, 1, axis=1),
        np.roll(orig_image, -1, axis=1)
    ], axis=0)
    diff = orig_image[np.newaxis, :, :] - shifteds
    smoothness_score = np.sum(prior(diff, gamma=1e-3)) / orig_image.size 
    #  divide by orig image size for better smoothness calc (o/w values are not comparable across diff img sizes)
    return smoothness_score

def myFilter(f, filter_type='ram_lak', L=None):
    N = f.shape[0]
    freqs = np.fft.fftfreq(N).reshape(-1,1)
    if L is None:
        L = np.max(np.abs(freqs))
    ramp = np.abs(freqs)
    ramp[freqs > L] = 0 
    if filter_type == 'ram_lak':
        filter_response = ramp
    elif filter_type == 'shepp_logan':
        filter_response = ramp * np.sinc(freqs / (2*L))
    elif filter_type == 'cosine':
        filter_response = ramp * np.cos(np.pi * freqs / (2*L))
    f_fft = np.fft.fft(f, axis=0)
    filtered_fft = f_fft * filter_response
    filtered_f = np.real(np.fft.ifft(filtered_fft, axis=0))
    return filtered_f

def main():
    N = 128
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (N, N), mode='reflect', anti_aliasing=True)
    # help(radon)
    radon_transform = radon(phantom, theta=np.arange(0, 180, 3), circle=True)
    print("Radon transform shape:", radon_transform.shape)

    freqs = np.fft.fftfreq(radon_transform.shape[0])
    wmax = np.max(np.abs(freqs))
    filter_types = ['ram_lak', 'shepp_logan', 'cosine']
    L_values = [wmax, wmax/2]
    if not os.path.exists('./output_filtered'):
        os.makedirs('./output_filtered')
    for filter_type in filter_types:
        for L in L_values:
            filtered = myFilter(radon_transform, filter_type=filter_type, L=L)
            recon_img = iradon(filtered, theta=np.arange(0, 180, 3), circle=True, filter_name=None)
            filename = f'./output_filtered/recon_{filter_type}_L_{L:.3f}.png'
            plt.imshow(recon_img, cmap='gray')
            plt.title(f'Reconstruction: {filter_type}, L={L:.3f}')
            plt.axis('off')
            plt.colorbar(label='Intensity')
            plt.savefig(filename)
            plt.clf()
            print(f"Saved {filename}")




if __name__ == "__main__":
    main()