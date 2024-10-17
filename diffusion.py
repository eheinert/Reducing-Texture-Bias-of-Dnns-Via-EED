#import scipy as sp
import PIL as pil
from PIL import Image
import time

import torch
from torchvision import transforms

#import diffusion_step_grey
import diffusion_step_rgb
import gauss_ker
import utils

class Matrix_Container:
    def __init__(self, dep, k_size, k_sig, device):
        #self.g_ker          = gauss_ker.gen_kernel(k_size,k_sig)
        
        self.f_g_ker_torch  = utils.get_gaussian_kernel(k_size, k_sig, dep, device)
        self.f_g_ker_1_dep  = utils.get_gaussian_kernel(k_size, k_sig, 1, device)
        
        self.pad_size       = int((k_size-1)/2)+1


def eed(pic_rgb, lam, k_size, k_sig, h, alpha, n_steps, tau = 'estimated', device = 'cpu', verbose = False): 
    
    if device != 'cpu':
        device = f'cuda:{device}'

    pic_torch = 255*transforms.ToTensor()(pic_rgb).permute(1,2,0).to(device)  # transform picture to array and transfer to device
    #pic_torch = pic_torch.repeat(64,1,1) #only for tests of size <-> runtime

    r, c, d = pic_torch.shape
    
    matrix_container = Matrix_Container(d, k_size, k_sig, device)
    
    for i in range(n_steps):
        start = time.time()
        # one diffusion step
        pic_torch, used_tau = diffusion_step_rgb.step_eed(pic_torch, k_size, k_sig, lam, h, tau, alpha, matrix_container, device, step_id = i, verbose = verbose)
        end = time.time()
        print(f'Step {i} took {1000*(end - start)} miliseconds \n')
        
    pic_torch = (1/255)*pic_torch
    pic_torch[pic_torch>1]  = 1
    pic_torch[pic_torch<0]  = 0

    pic_rgb = transforms.ToPILImage()(pic_torch.permute(2,0,1))

    return pic_rgb

def ced(pic_rgb, k_size, k_sig, h, alpha, n_steps, co_alpha, co_thr, tau = 'estimated', device = 'cpu', verbose = False): 
    
    if device != 'cpu':
        device = f'cuda:{device}'

    pic_torch = 255*transforms.ToTensor()(pic_rgb).permute(1,2,0).to(device)  # transform picture to array and transfer to device

    r, c, d = pic_torch.shape
    
    matrix_container = Matrix_Container(d, k_size, k_sig, device)
    
    for i in range(n_steps):
        pic_torch, used_tau = diffusion_step_rgb.step_ced(pic_torch, k_size, k_sig, h, tau, alpha, co_alpha, co_thr, matrix_container, step_id = i, verbose = verbose)  # one diffusion step
    
    pic_torch = (1/255)*pic_torch
    pic_torch[pic_torch>1]  = 1
    pic_torch[pic_torch<0]  = 0

    pic_rgb = transforms.ToPILImage()(pic_torch.permute(2,0,1))

    return pic_rgb

def lazy_mix(pic_rgb, lam, k_size, k_sig, h, alpha, n_steps, co_alpha, co_thr, mix = 0.5, tau = 'estimated', device = 'cpu', verbose = False):
    
    if device != 'cpu':
        device = f'cuda:{device}'

    pic_torch = 255*transforms.ToTensor()(pic_rgb).permute(1,2,0).to(device)  # transform picture to array and transfer to device

    r, c, d = pic_torch.shape
    
    matrix_container = Matrix_Container(d, k_size, k_sig, device)
    
    for i in range(n_steps):
        pic_torch_eed, used_tau = diffusion_step_rgb.step_new(pic_torch, k_size, k_sig, lam, h, tau, alpha, matrix_container, step_id = i, verbose = verbose)
        pic_torch_coh, used_tau = diffusion_step_rgb.step_new_coherence(pic_torch, k_size, k_sig, h, tau, alpha, co_alpha, co_thr, matrix_container, step_id = i, verbose = verbose)  # one diffusion step

        pic_torch = mix * pic_torch_eed + (1-mix) * pic_torch_coh
    
    pic_torch = (1/255)*pic_torch
    pic_torch[pic_torch>1]  = 1
    pic_torch[pic_torch<0]  = 0

    pic_rgb = transforms.ToPILImage()(pic_torch.permute(2,0,1))

    return pic_rgb


def eed_tensorblur(pic_rgb, lam, k_size, k_sig, h, alpha, n_steps, tau = 'estimated', device = 'cpu', verbose = False): 
    
    if device != 'cpu':
        device = f'cuda:{device}'

    pic_torch = 255*transforms.ToTensor()(pic_rgb).permute(1,2,0).to(device)  # transform picture to array and transfer to device

    r, c, d = pic_torch.shape
    
    matrix_container = Matrix_Container(d, k_size, k_sig, device)
    
    for i in range(n_steps):
        pic_torch = diffusion_step_rgb.step_eed_tensorblur(pic_torch, k_size, k_sig, lam, h, tau, alpha, matrix_container, step_id = i, verbose = verbose)  # one diffusion step
        
    pic_torch = (1/255)*pic_torch
    pic_torch[pic_torch>1]  = 1
    pic_torch[pic_torch<0]  = 0

    pic_rgb = transforms.ToPILImage()(pic_torch.permute(2,0,1))

    return pic_rgb