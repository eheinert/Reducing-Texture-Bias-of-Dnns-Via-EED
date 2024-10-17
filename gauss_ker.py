import numpy as np
import matplotlib.pyplot as plt

import scipy as sp
from scipy import signal

import PIL as pil
from PIL import Image

def apply(pic, k_size, sigma=1, verbose=False, ret_Image = False):
    g_ker = gen_kernel(k_size,sigma, verbose)
    pic = np.array(pic)

    pic_smooth_1 =  np.pad(pic[:,:,0],int((k_size-1)/2)+1, mode='symmetric')         # pad
    pic_smooth_2 =  np.pad(pic[:,:,1],int((k_size-1)/2)+1, mode='symmetric')
    pic_smooth_3 =  np.pad(pic[:,:,2],int((k_size-1)/2)+1, mode='symmetric')

    pic_smooth_1 = sp.signal.convolve2d(g_ker, pic_smooth_1,mode='valid')
    pic_smooth_2 = sp.signal.convolve2d(g_ker, pic_smooth_2,mode='valid')
    pic_smooth_3 = sp.signal.convolve2d(g_ker, pic_smooth_3,mode='valid')

    if ret_Image:
        pic_png = Image.fromarray(np.uint8(np.stack((pic_smooth_1, pic_smooth_2, pic_smooth_3), axis=-1)))
        return pic_png
    else:
        pic = np.stack((pic_smooth_1, pic_smooth_2, pic_smooth_3), axis=-1)
        return pic


def gen_kernel(size, sigma=1, verbose=False):                  #create the diffusion kernel ...
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)

    kernel_2D *= 1.0 / kernel_2D.max()
    kernel_2D = kernel_2D/(kernel_2D.sum()).sum()

    if verbose:
        plt.imshow(kernel_2D, interpolation='none', cmap='gray')
        plt.title("Image")
        plt.show()

    return kernel_2D


def dnorm(x, mu, sd):                                               # ... using this norm
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)