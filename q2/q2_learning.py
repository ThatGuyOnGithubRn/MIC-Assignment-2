import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, resize

N=128
phantom = shepp_logan_phantom()
phantom = resize(phantom, (N, N), mode='reflect', anti_aliasing=True)
# plt.imshow(phantom, cmap='gray')
# plt.title("Shepp-Logan Phantom")
# plt.savefig("phantom.png")
# plt.close()
theta = np.arange(0, 180, 3)
R = radon(phantom, theta=theta, circle=True)
bp = iradon(R, theta=theta, circle=True)

wmax = 2*np.pi*np.max(np.abs(np.fft.fftfreq(R.shape[0])))
print("wmax:", wmax)
# plt.imshow(bp, cmap="gray")
# plt.title("Unfiltered BP")
# plt.savefig("bp.png")

def rrmse(recon, ref):
    return np.sqrt(np.sum((recon - ref)**2) / np.sum(ref**2))

def myFilter(f, filter_type='ram_lak', L=None):
    N = f.shape[0]
    w = 2*np.pi*np.fft.fftfreq(N).reshape(-1,1)
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
    filter_response = np.fft.fftshift(filter_response, axes=0)
    f_fft = np.fft.fft(f, axis=0)
    filtered_fft = f_fft * filter_response
    filtered_f = np.real(np.fft.ifft(filtered_fft, axis=0))
    return filtered_f

image = myFilter(R, filter_type='ram_lak', L=wmax)
recon = iradon(image, theta=theta, circle=True)
recon = (recon - recon.min()) / (recon.max() - recon.min())
with open('recon_img_mine.txt', 'w') as f:
    for row in recon:
        f.write(' '.join(f'{val:.6f}' for val in row) + '\n')
plt.imshow(recon, cmap="gray")
plt.title("Ram_Lak Filtered BP")
plt.axis('off')
plt.savefig("ram_lak_filtered_bp.png")
print("RRMSE for Ram_Lak Filtered BP:", rrmse(recon, phantom))

