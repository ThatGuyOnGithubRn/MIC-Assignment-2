import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
from scipy.ndimage import map_coordinates
import os

def myXrayIntegration(f,t,theta_deg,delta_s,interpolation_scheme=1):
    theta_rad = np.deg2rad(theta_deg)
    costheta = np.cos(theta_rad)
    sintheta = np.sin(theta_rad)
    s = np.arange(-f.shape[0], f.shape[0], delta_s) 
    x_coords = t*costheta-s*sintheta  #(256,)
    y_coords = t*sintheta+s*costheta  #(256,)
    coords = np.vstack((y_coords, x_coords))  #(2,256)

    # map coords wants [[y],[x]]
    line_integral = map_coordinates(f,coords,order=interpolation_scheme,mode='constant',cval=0.0)
    return line_integral.sum()*delta_s


def myXrayCTRadonTransform(f,delta_t,delta_theta,delta_s):
    t_values = np.arange(-90, 95, delta_t)
    theta_values = np.arange(0, 180, delta_theta)
    radon_transform = np.zeros((len(t_values),len(theta_values)))
    for i, t in enumerate(t_values):
        for j, theta in enumerate(theta_values):
            radon_transform[i, j] = myXrayIntegration(f,t,theta,delta_s=delta_s)
    return radon_transform

def main():
    
    N = 128
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (N, N), mode='reflect', anti_aliasing=True)
    # print(type(phantom))
    # print(phantom.shape)
    # np.set_printoptions(threshold=np.inf)
    # with open("array_output.txt", "w") as f:
    #     print(phantom, file=f)
    x = np.linspace(-(N-1)/2, (N-1)/2, N)
    y = np.linspace(-(N-1)/2, (N-1)/2, N)
    X, Y = np.meshgrid(x, y)
    # myXrayIntegration(phantom, t=0, theta_deg=30)

    delta_s_values = [0.1, 0.5, 1, 3, 10]
    if not os.path.exists('./output'):
        os.makedirs('./output')
    for delta_s in delta_s_values:
        radon_transform = myXrayCTRadonTransform(phantom,delta_t=5,delta_theta=5,delta_s=delta_s)
        plt.imshow(radon_transform, extent=(-90, 90, -90, 90), aspect='auto')
        plt.title(f'Radon Transform with Δs={delta_s}')
        plt.xlabel('θ (degrees)')
        plt.ylabel('t (pixel-width units)')
        plt.colorbar(label='Radon Transform Value')
        plt.savefig(f'./output/radon_transform_delta_s_{delta_s}.png')
        plt.clf()
        print("Log prior to calc smoothness...", delta_s)
        print(apply_prior(radon_transform))
        # plt.show()
        #    NOTE: 0.1 does worse than 0.5
        #    NOTE: 0.5 and 1 are similar, but 0.5 is slightly better
    '''
    Log prior to calc smoothness... 0.1
    0.007664408181411028
    Log prior to calc smoothness... 0.5
    0.007663613111978756
    Log prior to calc smoothness... 1
    0.007665728259402043
    Log prior to calc smoothness... 3
    0.007924217046504343
    Log prior to calc smoothness... 10
    0.013104456194064591
    '''


    delta_t_values = [0.1, 0.5, 2.5, 5, 10]
    delta_theta = 0.5
    delta_s = 0.1

    if not os.path.exists('./output_delta_t'):
        os.makedirs('./output_delta_t')
    for delta_t in delta_t_values:
        radon_transform = myXrayCTRadonTransform(phantom, delta_t=delta_t, delta_theta=delta_theta, delta_s=delta_s)
        plt.imshow(radon_transform, extent=(-90, 90, -90, 90), aspect='auto')
        plt.title(f'Radon Transform with Δt={delta_t}, Δθ={delta_theta}, Δs={delta_s}')
        plt.xlabel('θ (degrees)')
        plt.ylabel('t (pixel-width units)')
        plt.colorbar(label='Radon Transform Value')
        plt.savefig(f'./output_delta_t/radon_transform_delta_t_{delta_t}.png')
        plt.clf()
        print(f"Delta t = {delta_t}, Smoothness (log prior) = {apply_prior(radon_transform)}")

        '''
        Delta t = 0.1, Smoothness (log prior) = 0.000615221620951703
        Delta t = 0.5, Smoothness (log prior) = 0.0009827163160813109
        Delta t = 2.5, Smoothness (log prior) = 0.002559424417978858
        Delta t = 5, Smoothness (log prior) = 0.00416248252365935
        Delta t = 10, Smoothness (log prior) = 0.006850882674789374
        '''
        #  NOTE THESE VALUES ARE WORSE DUE TO MORE PIXELS --> fixed in later iterations
        #  NOTE: 0.1 is better than 0.5 now


    delta_theta_values = [0.1, 0.5, 2.5, 5, 10]
    delta_t = 0.5
    delta_s = 0.1 

    if not os.path.exists('./output_delta_theta'):
        os.makedirs('./output_delta_theta')
    for delta_theta in delta_theta_values:
        radon_transform = myXrayCTRadonTransform(phantom, delta_t=delta_t, delta_theta=delta_theta, delta_s=delta_s)
        plt.imshow(radon_transform, extent=(-90, 90, -90, 90), aspect='auto')
        plt.title(f'Radon Transform with Δt={delta_t}, Δθ={delta_theta}, Δs={delta_s}')
        plt.xlabel('θ (degrees)')
        plt.ylabel('t (pixel-width units)')
        plt.colorbar(label='Radon Transform Value')
        plt.savefig(f'./output_delta_theta/radon_transform_delta_theta_{delta_theta}.png')
        plt.clf()
        print(f"Delta theta = {delta_theta}, Smoothness (log prior) = {apply_prior(radon_transform)}")

        '''
        Delta theta = 0.1, Smoothness (log prior) = 0.0005663694007187037
        Delta theta = 0.5, Smoothness (log prior) = 0.0009827163160813109
        Delta theta = 2.5, Smoothness (log prior) = 0.002740957921060526
        Delta theta = 5, Smoothness (log prior) = 0.004570646500220572
        Delta theta = 10, Smoothness (log prior) = 0.007641838235531056
        '''
        #  NOTE THESE VALUES ARE WORSE DUE TO MORE PIXELS --> fixed in later iterations
        #  NOTE: 0.1 is better than 0.5 now

# ANSER FOR d

# del theta to be kept as small as possible (dont keep it too small due to computation costs and radiation exposure)
# del t to be kept small to reduce discretization errors (not too small excessive computation and noise sensitivity)
# since this is a phantom smaller delta t give better and better results
    


# ANSWER FOR e

# How would you choose the number of pixels, and the pixel size, in the scene grid ?
# More pixels--> better reconstruction BUT heavy computation + mpre error due to measurement noise
# each pixel collects lesss xray energy so I0/I is is more prone to changes as I0 is smaller adn I expected is smaller 

# ∆s << pixel width: error accumulation small improvement betyond a point
# ∆s >> pixel width: underestimates line integral -> bad blocky reconstruction



if __name__ == "__main__":
    main()

