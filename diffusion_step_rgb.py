import torch
from torch import nn
import torch.nn.functional as fu

import matplotlib.pyplot as plt
#import scipy as sp
#from scipy import signal
import PIL as pil
from PIL import Image
import sys
import time

import gauss_ker
import stencil_gen
import utils


def step_eed(pic_torch,k_size, k_sig,lam,h,tau,alpha,matrix_container,device, step_id=None, verbose = False):
    start = time.time()

    #pic_torch = torch.from_numpy(pic)
    r, c, d = pic_torch.shape
    print(pic_torch.shape)
  #pad and smooth:
    pad_size  = matrix_container.pad_size 
    #pic_torch = utils.symm_pad_2D(pic_torch,[pad_size,pad_size,pad_size,pad_size])
    
    f_gauss_ker = matrix_container.f_g_ker_torch #gaussian filter function
    pic_tsm = utils.apply_gaussian_filter(f_gauss_ker, utils.symm_pad_2D(pic_torch,[pad_size,pad_size,pad_size,pad_size])) #pad and smooth
    
    end = time.time()
    print(f"padding and smoothing took {(end -start)*1000} miliseconds")
    start = time.time()
  #build derivatives:
    #smallest parts:
    pic_tsm_x1 = (1 / h) * (pic_tsm[0:r+1,0:c+1,:]-pic_tsm[0:r+1,1:c+2,:])  # mehrfach verwendetes vorher benennen
    pic_tsm_x2 = (1 / h) * (pic_tsm[1:r+2,0:c+1,:]-pic_tsm[1:r+2,1:c+2,:])  # ggf vorhandene Operation verwenden
                                                                            # ggf auch als Faltung formulieren
    pic_tsm_y1 = (1 / h) * (pic_tsm[0:r+1,0:c+1,:]-pic_tsm[1:r+2,0:c+1,:]) 
    pic_tsm_y2 = (1 / h) * (pic_tsm[0:r+1,1:c+2,:]-pic_tsm[1:r+2,1:c+2,:]) 
    

    #squares:
    pic_tsm_xx = (1-alpha)/2   * (torch.square(pic_tsm_x1) + torch.square(pic_tsm_x2)) \
                 + alpha       * (pic_tsm_x1 * pic_tsm_x2)
    pic_tsm_yy = (1-alpha)/2   * (torch.square(pic_tsm_y1) + torch.square(pic_tsm_y2)) \
                 + alpha       * (pic_tsm_y1 * pic_tsm_y2)
    pic_tsm_xy = 0.25 * (pic_tsm_x1 * pic_tsm_y1 + pic_tsm_x1 * pic_tsm_y2 + pic_tsm_x2 * pic_tsm_y1 + pic_tsm_x2 * pic_tsm_y2)

  #build kernel (formulas from https://hal.archives-ouvertes.fr/hal-01501221/document):
    #a,b,c before eigenvalue transformation:
    A = torch.sum(pic_tsm_xx, dim = 2)
    B = torch.sum(pic_tsm_xy, dim = 2)
    C = torch.sum(pic_tsm_yy, dim = 2)
    
    D = torch.sqrt(4*torch.square(B)+torch.square(A-C))
    ind_crit = (D < 1e-6)

    #eigenvalue transformation:
    EW_1, EW_2 = (A + C - D) / 2, (A + C + D) / 2
    g_EW_1 = torch.div(1,torch.sqrt(1+torch.square(torch.div(EW_1,lam))))
    g_EW_2 = torch.div(1,torch.sqrt(1+torch.square(torch.div(EW_2,lam))))
    
    #a,b,c after eigenvalue transformation:
    At = torch.div( (A - C + D) * g_EW_2 - (A - C - D) * g_EW_1, 2*D)
    At[ind_crit] = 1
    Ct = torch.div( (C - A + D) * g_EW_2 - (C - A - D) * g_EW_1, 2*D)
    Ct[ind_crit] = 1
    Bt = torch.div(B * (g_EW_2 - g_EW_1), D)
    Bt[ind_crit] = 0
    
    end = time.time()
    print(f"computing the gradients and forming the diffusion tensor took {(end-start)*1000} miliseconds")
    start = time.time()
    ###################################################################################################################
    
    if tau == 'estimated':
        tau = 0.9*stencil_gen.time_step_estimator(At, Bt, Ct, r, c, h, alpha)
    
    end = time.time()
    print(f"time step estimation took {(end-start)*1000} miliseconds")
    start = time.time()

    if device == 'cpu':
      pic_torch = torch.stack(( stencil_gen.vectorized(pic_torch[:, :, 0], At, Bt, Ct, r, c, h, tau,alpha),
                                stencil_gen.vectorized(pic_torch[:, :, 1], At, Bt, Ct, r, c, h, tau,alpha),
                                stencil_gen.vectorized(pic_torch[:, :, 2], At, Bt, Ct, r, c, h, tau,alpha))).permute(1,2,0)
    else:
      pic_torch = stencil_gen.vectorized_fullstack(pic_torch, At, Bt, Ct, r, c, h, tau,alpha)

    end = time.time()
    print(f"applying the stencil took {(end-start)*1000} miliseconds")

    if verbose:
        print(f"step {step_id}: used tau {tau}, min value: {pic_torch.min()}, max value: {pic_torch.max()}")


    return pic_torch, tau


def step_ced(pic_torch,k_size, k_sig,h,tau,alpha, co_alpha, co_thr, matrix_container,device , step_id=None, verbose = False):
    #pic_torch = torch.from_numpy(pic)
    r, c, d = pic_torch.shape
    
  #pad and smooth:
    pad_size  = matrix_container.pad_size 
    #pic_torch = utils.symm_pad_2D(pic_torch,[pad_size,pad_size,pad_size,pad_size])
    
    f_gauss_ker = matrix_container.f_g_ker_torch #gaussian filter function
    pic_tsm = utils.apply_gaussian_filter(f_gauss_ker, utils.symm_pad_2D(pic_torch,[pad_size,pad_size,pad_size,pad_size])) #pad and smooth
    
  #build derivatives:
    #smallest parts:
    pic_tsm_x1 = (1 / h) * (pic_tsm[0:r+1,0:c+1,:]-pic_tsm[0:r+1,1:c+2,:])
    pic_tsm_x2 = (1 / h) * (pic_tsm[1:r+2,0:c+1,:]-pic_tsm[1:r+2,1:c+2,:])

    pic_tsm_y1 = (1 / h) * (pic_tsm[0:r+1,0:c+1,:]-pic_tsm[1:r+2,0:c+1,:]) 
    pic_tsm_y2 = (1 / h) * (pic_tsm[0:r+1,1:c+2,:]-pic_tsm[1:r+2,1:c+2,:])  

    #squares:
    pic_tsm_xx = (1-alpha)/2   * (torch.square(pic_tsm_x1) + torch.square(pic_tsm_x2)) \
                 + alpha       * (pic_tsm_x1 * pic_tsm_x2)
    pic_tsm_yy = (1-alpha)/2   * (torch.square(pic_tsm_y1) + torch.square(pic_tsm_y2)) \
                 + alpha       * (pic_tsm_y1 * pic_tsm_y2)
    pic_tsm_xy = 0.25 * (pic_tsm_x1 * pic_tsm_y1 + pic_tsm_x1 * pic_tsm_y2 + pic_tsm_x2 * pic_tsm_y1 + pic_tsm_x2 * pic_tsm_y2)

  #build kernel (formulas from https://hal.archives-ouvertes.fr/hal-01501221/document):
    #a,b,c before eigenvalue transformation:
    A = torch.sum(pic_tsm_xx, dim = 2)
    B = torch.sum(pic_tsm_xy, dim = 2)
    C = torch.sum(pic_tsm_yy, dim = 2)

    f_gauss_ker_1D = matrix_container.f_g_ker_1_dep
    # first blurr A, B, C as described in Coherence-Enhancing Diffusion Filtering, JOACHIM WEICKERT, International Journal of Computer Vision 31(2/3), 111–127 (1999)
    A = torch.squeeze(f_gauss_ker_1D(torch.unsqueeze(torch.unsqueeze(utils.symm_pad_2D(A,[pad_size-1,pad_size-1,pad_size-1,pad_size-1]),0),0)))
    B = torch.squeeze(f_gauss_ker_1D(torch.unsqueeze(torch.unsqueeze(utils.symm_pad_2D(B,[pad_size-1,pad_size-1,pad_size-1,pad_size-1]),0),0)))
    C = torch.squeeze(f_gauss_ker_1D(torch.unsqueeze(torch.unsqueeze(utils.symm_pad_2D(C,[pad_size-1,pad_size-1,pad_size-1,pad_size-1]),0),0)))

    D = torch.sqrt(4*torch.square(B)+torch.square(A - C))
    ind_crit = (D < 1e-6)

    # eigenvalues
    EW_1, EW_2 = (A + C - D) / 2, (A + C + D) / 2

    # eigenvalue transformation, beginning with coherence metric
    K = torch.square(EW_1 - EW_2)

    g_EW_1, g_EW_2 = co_alpha*torch.ones_like(EW_1), co_alpha*torch.ones_like(EW_2)
    
    g_EW_1[(EW_1 < EW_2)] = co_alpha + (1 - co_alpha)*torch.exp(torch.div(-co_thr,K))[(EW_1 < EW_2)]
    g_EW_2[(EW_2 < EW_1)] = co_alpha + (1 - co_alpha)*torch.exp(torch.div(-co_thr,K))[(EW_2 < EW_1)]

    #a,b,c after eigenvalue transformation:
    At = torch.div( (A - C + D) * g_EW_2 - (A - C - D) * g_EW_1, 2*D)
    At[ind_crit] = 1
    Ct = torch.div( (C - A + D) * g_EW_2 - (C - A - D) * g_EW_1, 2*D)
    Ct[ind_crit] = 1
    Bt = torch.div(B * (g_EW_2 - g_EW_1), D)
    Bt[ind_crit] = 0
    
    if tau == 'estimated':
        tau = 0.9*stencil_gen.time_step_estimator(At, Bt, Ct, r, c, h, alpha)

    
    if device == 'cpu':
      pic_torch = torch.stack(( stencil_gen.vectorized(pic_torch[:, :, 0], At, Bt, Ct, r, c, h, tau,alpha),
                                stencil_gen.vectorized(pic_torch[:, :, 1], At, Bt, Ct, r, c, h, tau,alpha),
                                stencil_gen.vectorized(pic_torch[:, :, 2], At, Bt, Ct, r, c, h, tau,alpha))).permute(1,2,0)
    else:
      pic_torch = stencil_gen.vectorized_fullstack(pic_torch, At, Bt, Ct, r, c, h, tau,alpha)


    if verbose:
        print(f"step {step_id}: used tau {tau}, min value: {pic_torch.min()}, max value: {pic_torch.max()}")


    return pic_torch, tau


def step_isod(pic_torch, h,tau,alpha, At, Bt, Ct, device , step_id=None, verbose = False):
  #At, Bt, Ct have to be formerly allocated ones, zeros and ones on device (unity instead of anisotropic component)
  
    r, c, d = pic_torch.shape    
  

    if device == 'cpu':
      pic_torch = torch.stack(( stencil_gen.vectorized(pic_torch[:, :, 0], At, Bt, Ct, r, c, h, tau,alpha),
                                stencil_gen.vectorized(pic_torch[:, :, 1], At, Bt, Ct, r, c, h, tau,alpha),
                                stencil_gen.vectorized(pic_torch[:, :, 2], At, Bt, Ct, r, c, h, tau,alpha))).permute(1,2,0)
    else:
      pic_torch = stencil_gen.vectorized_fullstack(pic_torch, At, Bt, Ct, r, c, h, tau,alpha)


    if verbose:
        print(f"step {step_id}: used tau {tau}, min value: {pic_torch.min()}, max value: {pic_torch.max()}")


    return pic_torch



def step_eed_tensorblur(pic_torch,k_size, k_sig,lam,h,tau,alpha,matrix_container,device , step_id=None, verbose = False):
    #pic_torch = torch.from_numpy(pic)
    r, c, d = pic_torch.shape
    
  #pad and smooth:
    pad_size  = matrix_container.pad_size 
    #pic_torch = utils.symm_pad_2D(pic_torch,[pad_size,pad_size,pad_size,pad_size])
    
    f_gauss_ker = matrix_container.f_g_ker_torch #gaussian filter function
    pic_tsm = utils.apply_gaussian_filter(f_gauss_ker, utils.symm_pad_2D(pic_torch,[pad_size,pad_size,pad_size,pad_size])) #pad and smooth
    
  #build derivatives:
    #smallest parts:
    pic_tsm_x1 = (1 / h) * (pic_tsm[0:r+1,0:c+1,:]-pic_tsm[0:r+1,1:c+2,:])
    pic_tsm_x2 = (1 / h) * (pic_tsm[1:r+2,0:c+1,:]-pic_tsm[1:r+2,1:c+2,:])

    pic_tsm_y1 = (1 / h) * (pic_tsm[0:r+1,0:c+1,:]-pic_tsm[1:r+2,0:c+1,:]) 
    pic_tsm_y2 = (1 / h) * (pic_tsm[0:r+1,1:c+2,:]-pic_tsm[1:r+2,1:c+2,:]) 

    #squares:
    pic_tsm_xx = (1-alpha)/2   * (torch.square(pic_tsm_x1) + torch.square(pic_tsm_x2)) \
                 + alpha       * (pic_tsm_x1 * pic_tsm_x2)
    pic_tsm_yy = (1-alpha)/2   * (torch.square(pic_tsm_y1) + torch.square(pic_tsm_y2)) \
                 + alpha       * (pic_tsm_y1 * pic_tsm_y2)
    pic_tsm_xy = 0.25 * (pic_tsm_x1 * pic_tsm_y1 + pic_tsm_x1 * pic_tsm_y2 + pic_tsm_x2 * pic_tsm_y1 + pic_tsm_x2 * pic_tsm_y2)

  #build kernel (formulas from https://hal.archives-ouvertes.fr/hal-01501221/document):
    #a,b,c before eigenvalue transformation:
    A = torch.sum(pic_tsm_xx, dim = 2)
    B = torch.sum(pic_tsm_xy, dim = 2)
    C = torch.sum(pic_tsm_yy, dim = 2)
    
    f_gauss_ker_1D = matrix_container.f_g_ker_1_dep
    # first blurr A, B, C as described in Coherence-Enhancing Diffusion Filtering, JOACHIM WEICKERT, International Journal of Computer Vision 31(2/3), 111–127 (1999)
    A = torch.squeeze(f_gauss_ker_1D(torch.unsqueeze(torch.unsqueeze(utils.symm_pad_2D(A,[pad_size-1,pad_size-1,pad_size-1,pad_size-1]),0),0)))
    B = torch.squeeze(f_gauss_ker_1D(torch.unsqueeze(torch.unsqueeze(utils.symm_pad_2D(B,[pad_size-1,pad_size-1,pad_size-1,pad_size-1]),0),0)))
    C = torch.squeeze(f_gauss_ker_1D(torch.unsqueeze(torch.unsqueeze(utils.symm_pad_2D(C,[pad_size-1,pad_size-1,pad_size-1,pad_size-1]),0),0)))
    
    D = torch.sqrt(4*torch.square(B)+torch.square(A-C))
    ind_crit = (D < 1e-6)

    #eigenvalue transformation:
    EW_1, EW_2 = (A + C - D) / 2, (A + C + D) / 2
    g_EW_1 = torch.div(1,torch.sqrt(1+torch.square(torch.div(EW_1,lam))))
    g_EW_2 = torch.div(1,torch.sqrt(1+torch.square(torch.div(EW_2,lam))))

    #a,b,c after eigenvalue transformation:
    At = torch.div( (A - C + D) * g_EW_2 - (A - C - D) * g_EW_1, 2*D)
    At[ind_crit] = 1
    Ct = torch.div( (C - A + D) * g_EW_2 - (C - A - D) * g_EW_1, 2*D)
    Ct[ind_crit] = 1
    Bt = torch.div(B * (g_EW_2 - g_EW_1), D)
    Bt[ind_crit] = 0
    
    ###################################################################################################################
    
    if tau == 'estimated':
        tau = 0.9*stencil_gen.time_step_estimator(At, Bt, Ct, r, c, h, alpha)
        #tau = min(tau, 0.05+0.01*step_id)
    
    if device == 'cpu':
      pic_torch = torch.stack(( stencil_gen.vectorized(pic_torch[:, :, 0], At, Bt, Ct, r, c, h, tau,alpha),
                                stencil_gen.vectorized(pic_torch[:, :, 1], At, Bt, Ct, r, c, h, tau,alpha),
                                stencil_gen.vectorized(pic_torch[:, :, 2], At, Bt, Ct, r, c, h, tau,alpha))).permute(1,2,0)
    else:
      pic_torch = stencil_gen.vectorized_fullstack(pic_torch, At, Bt, Ct, r, c, h, tau,alpha)


    if verbose:
        print(f"step {step_id}: used tau {tau}, min value: {pic_torch.min()}, max value: {pic_torch.max()}")


    return pic_torch








######################################################################################################################
#for batch-wise network implementation
def step_eed_batchwise_nn(pic_torch,k_size, k_sig,lam,h,tau,alpha,matrix_container,device, step_id=None, verbose = False):
    start = time.time()

    #pic_torch = torch.from_numpy(pic)
    b, r, c, d = pic_torch.shape
    print(pic_torch.shape)
  #pad and smooth:
    pad_size  = matrix_container.pad_size 
    #pic_torch = utils.symm_pad_2D(pic_torch,[pad_size,pad_size,pad_size,pad_size])
    
    f_gauss_ker = matrix_container.f_g_ker_torch #gaussian filter function
    pic_tsm = utils.apply_gaussian_filter(f_gauss_ker, utils.symm_pad_2D(pic_torch,[pad_size,pad_size,pad_size,pad_size])) #pad and smooth
    
    end = time.time()
    print(f"padding and smoothing took {(end -start)*1000} miliseconds")
    start = time.time()
  #build derivatives:
    #smallest parts:
    pic_tsm_x1 = (1 / h) * (pic_tsm[0:r+1,0:c+1,:]-pic_tsm[0:r+1,1:c+2,:])
    pic_tsm_x2 = (1 / h) * (pic_tsm[1:r+2,0:c+1,:]-pic_tsm[1:r+2,1:c+2,:])

    pic_tsm_y1 = (1 / h) * (pic_tsm[0:r+1,0:c+1,:]-pic_tsm[1:r+2,0:c+1,:]) 
    pic_tsm_y2 = (1 / h) * (pic_tsm[0:r+1,1:c+2,:]-pic_tsm[1:r+2,1:c+2,:]) 

    #squares:
    pic_tsm_xx = (1-alpha)/2   * (torch.square(pic_tsm_x1) + torch.square(pic_tsm_x2)) \
                 + alpha       * (pic_tsm_x1 * pic_tsm_x2)
    pic_tsm_yy = (1-alpha)/2   * (torch.square(pic_tsm_y1) + torch.square(pic_tsm_y2)) \
                 + alpha       * (pic_tsm_y1 * pic_tsm_y2)
    pic_tsm_xy = 0.25 * (pic_tsm_x1 * pic_tsm_y1 + pic_tsm_x1 * pic_tsm_y2 + pic_tsm_x2 * pic_tsm_y1 + pic_tsm_x2 * pic_tsm_y2)

  #build kernel (formulas from https://hal.archives-ouvertes.fr/hal-01501221/document):
    #a,b,c before eigenvalue transformation:
    A = torch.sum(pic_tsm_xx, dim = 2)
    B = torch.sum(pic_tsm_xy, dim = 2)
    C = torch.sum(pic_tsm_yy, dim = 2)
    
    D = torch.sqrt(4*torch.square(B)+torch.square(A-C))
    ind_crit = (D < 1e-6)

    #eigenvalue transformation:
    EW_1, EW_2 = (A + C - D) / 2, (A + C + D) / 2
    g_EW_1 = torch.div(1,torch.sqrt(1+torch.square(torch.div(EW_1,lam))))
    g_EW_2 = torch.div(1,torch.sqrt(1+torch.square(torch.div(EW_2,lam))))

    #a,b,c after eigenvalue transformation:
    At = torch.div( (A - C + D) * g_EW_2 - (A - C - D) * g_EW_1, 2*D)
    At[ind_crit] = 1
    Ct = torch.div( (C - A + D) * g_EW_2 - (C - A - D) * g_EW_1, 2*D)
    Ct[ind_crit] = 1
    Bt = torch.div(B * (g_EW_2 - g_EW_1), D)
    Bt[ind_crit] = 0
    
    end = time.time()
    print(f"computing the gradients and forming the diffusion tensor took {(end-start)*1000} miliseconds")
    start = time.time()
    ###################################################################################################################
    
    if tau == 'estimated':
        tau = 0.9*stencil_gen.time_step_estimator(At, Bt, Ct, r, c, h, alpha)
    
    end = time.time()
    print(f"time step estimation took {(end-start)*1000} miliseconds")
    start = time.time()

    if device == 'cpu':
      pic_torch = torch.stack(( stencil_gen.vectorized(pic_torch[:, :, 0], At, Bt, Ct, r, c, h, tau,alpha),
                                stencil_gen.vectorized(pic_torch[:, :, 1], At, Bt, Ct, r, c, h, tau,alpha),
                                stencil_gen.vectorized(pic_torch[:, :, 2], At, Bt, Ct, r, c, h, tau,alpha))).permute(1,2,0)
    else:
      pic_torch = stencil_gen.vectorized_fullstack(pic_torch, At, Bt, Ct, r, c, h, tau,alpha)

    end = time.time()
    print(f"applying the stencil took {(end-start)*1000} miliseconds")

    if verbose:
        print(f"step {step_id}: used tau {tau}, min value: {pic_torch.min()}, max value: {pic_torch.max()}")


    return pic_torch, tau