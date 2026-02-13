import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize, radon, iradon
import os


# -----------------------------
# Relative Root Mean Squared Error
# -----------------------------
def rrmse(A, B):
    return np.sqrt(np.sum((A - B) ** 2)) / np.sqrt(np.sum(A ** 2))


# -----------------------------
# Custom FBP Filters
# -----------------------------
def myFilter(sinogram, filter_type='ram_lak', L=None):
    """
    sinogram : Radon transform (detector x angles)
    filter_type : 'ram_lak', 'shepp_logan', 'cosine'
    L : frequency cutoff
    """

    N = sinogram.shape[0]

    # Discrete frequency axis (cycles/pixel)
    w = np.fft.fftfreq(N).reshape(-1, 1)
    wmax = np.max(np.abs(w))

    if L is None:
        raise ValueError("L must be specified explicitly.")

    ramp = np.abs(w)
    H = np.zeros_like(w)

    mask = np.abs(w) <= L

    if filter_type == 'ram_lak':
        H[mask] = ramp[mask]

    elif filter_type == 'shepp_logan':
        H[mask] = ramp[mask] * np.sinc(w[mask] / L)

    elif filter_type == 'cosine':
        H[mask] = ramp[mask] * np.cos((np.pi / 2) * (w[mask] / L))

    else:
        raise ValueError("Unknown filter type")

    # FFT along detector direction
    S = np.fft.fft(sinogram, axis=0)
    S_filtered = S * H

    return np.real(np.fft.ifft(S_filtered, axis=0))


# -----------------------------
# Main
# -----------------------------
def main():

    N = 128

    # Generate phantom
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (N, N), mode='reflect', anti_aliasing=True)

    theta = np.arange(0, 180, 3)

    # Radon transform
    sinogram = radon(phantom, theta=theta, circle=False)

    # Highest discrete frequency
    freqs = np.fft.fftfreq(sinogram.shape[0])
    wmax = np.max(np.abs(freqs))

    L_values = [wmax, wmax / 2]
    filter_types = ['ram_lak', 'shepp_logan', 'cosine']

    if not os.path.exists('./output_filtered'):
        os.makedirs('./output_filtered')

    # -----------------------------
    # Unfiltered Backprojection
    # -----------------------------
    recon_unfiltered = iradon(
        sinogram,
        theta=theta,
        filter_name=None,
        circle=False,
        output_size=N
    )

    error_unfiltered = rrmse(phantom, recon_unfiltered)
    print(f"RRMSE (Unfiltered Backprojection): {error_unfiltered:.6f}")

    plt.imshow(recon_unfiltered, cmap='gray')
    plt.axis('off')
    plt.savefig('./output_filtered/recon_unfiltered.png',
                bbox_inches='tight', pad_inches=0)
    plt.close()

    # -----------------------------
    # Filtered Reconstructions
    # -----------------------------
    for filter_type in filter_types:
        for L in L_values:

            filtered_sinogram = myFilter(
                sinogram,
                filter_type=filter_type,
                L=L
            )

            recon = iradon(
                filtered_sinogram,
                theta=theta,
                filter_name=None,
                circle=False,
                output_size=N
            )

            error = rrmse(phantom, recon)

            print(f"RRMSE ({filter_type}, L={L:.5f}): {error:.6f}")

            plt.imshow(recon, cmap='gray')
            plt.axis('off')
            filename = f'./output_filtered/recon_{filter_type}_L_{L:.5f}.png'
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.close()


if __name__ == "__main__":
    main()
