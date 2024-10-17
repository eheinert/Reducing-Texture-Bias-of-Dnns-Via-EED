import torch
from torch import nn
import torch.nn.functional as fu

import matplotlib.pyplot as plt
import PIL as pil
from PIL import Image

import utils

def vectorized(pic_torch,A,B,C,pic_row,pic_col,h,tau,alpha): # faster for cpu

    ap=alpha
    am=1-ap
    

    pic_torch = utils.symm_pad_2D(pic_torch, [1,1,1,1])

    pic_torch = pic_torch[1:pic_row+1,1:pic_col+1] + tau/(h*h)*torch.stack([
                    (ap*A[0:pic_row,0:pic_col]      +ap*C[0:pic_row,0:pic_col]       +B[0:pic_row,0:pic_col])     *    pic_torch[0:pic_row,0:pic_col],          #ul
                    (am*C[0:pic_row,0:pic_col]      -ap*A[0:pic_row,0:pic_col]       
                    +am*C[0:pic_row,1:pic_col+1]    -ap*A[0:pic_row,1:pic_col+1])                                 *    pic_torch[0:pic_row,1:pic_col+1],        #um
                    (ap*A[0:pic_row,1:pic_col+1]    +ap*C[0:pic_row,1:pic_col+1]     -B[0:pic_row,1:pic_col+1])   *    pic_torch[0:pic_row,2:pic_col+2],        #ur
                    (am*A[1:pic_row+1,0:pic_col]    -ap*C[1:pic_row+1,0:pic_col]     
                    +am*A[0:pic_row,0:pic_col]      -ap*C[0:pic_row,0:pic_col])                                   *    pic_torch[1:pic_row+1,0:pic_col],        #ml
                    (-am*A[1:pic_row+1,1:pic_col+1]  -am*C[1:pic_row+1,1:pic_col+1]  -B[1:pic_row+1,1:pic_col+1]
                    -am*A[1:pic_row+1,0:pic_col]    -am*C[1:pic_row+1,0:pic_col]     +B[1:pic_row+1,0:pic_col]
                    -am*A[0:pic_row,1:pic_col+1]    -am*C[0:pic_row,1:pic_col+1]     +B[0:pic_row,1:pic_col+1]
                    -am*A[0:pic_row,0:pic_col]      -am*C[0:pic_row,0:pic_col]       -B[0:pic_row,0:pic_col])     *     pic_torch[1:pic_row+1,1:pic_col+1],     #mm
                    (am*A[1:pic_row+1,1:pic_col+1]  -ap*C[1:pic_row+1,1:pic_col+1]   
                    +am*A[0:pic_row,1:pic_col+1]    -ap*C[0:pic_row,1:pic_col+1])                                 *     pic_torch[1:pic_row+1,2:pic_col+2],     #mr
                    (ap*A[1:pic_row+1,0:pic_col]    +ap*C[1:pic_row+1,0:pic_col]     -B[1:pic_row+1,0:pic_col])   *     pic_torch[2:pic_row+2,0:pic_col],       #bl
                    (am*C[1:pic_row+1,1:pic_col+1]  -ap*A[1:pic_row+1,1:pic_col+1]   
                    +am*C[1:pic_row+1,0:pic_col]    -ap*A[1:pic_row+1,0:pic_col])                                 *     pic_torch[2:pic_row+2,1:pic_col+1],     #bm
                    (ap*A[1:pic_row+1,1:pic_col+1]  +ap*C[1:pic_row+1,1:pic_col+1]   +B[1:pic_row+1,1:pic_col+1]) *     pic_torch[2:pic_row+2,2:pic_col+2]      #br
                    ], dim = 0).sum(dim=0)
    
    return pic_torch



def vectorized_fullstack(pic_torch,A,B,C,pic_row,pic_col,h,tau,alpha): #faster for gpu

    ap=alpha
    am=1-ap
    

    pic_torch = utils.symm_pad_2D(pic_torch, [1,1,1,1])

    pic_torch = pic_torch[1:pic_row+1,1:pic_col+1] + tau/(h*h)*torch.stack([
                    (ap*A[0:pic_row,0:pic_col]      +ap*C[0:pic_row,0:pic_col]       +B[0:pic_row,0:pic_col]).repeat(3,1,1).permute(1,2,0) \
                                                                                                                *    pic_torch[0:pic_row,0:pic_col,:],          #ul
                    (am*C[0:pic_row,0:pic_col]      -ap*A[0:pic_row,0:pic_col]       
                    +am*C[0:pic_row,1:pic_col+1]    -ap*A[0:pic_row,1:pic_col+1]).repeat(3,1,1).permute(1,2,0) \
                                                                                                                *    pic_torch[0:pic_row,1:pic_col+1,:],        #um
                    (ap*A[0:pic_row,1:pic_col+1]    +ap*C[0:pic_row,1:pic_col+1]     -B[0:pic_row,1:pic_col+1]).repeat(3,1,1).permute(1,2,0) \
                                                                                                                *    pic_torch[0:pic_row,2:pic_col+2,:],        #ur
                    (am*A[1:pic_row+1,0:pic_col]    -ap*C[1:pic_row+1,0:pic_col]     
                    +am*A[0:pic_row,0:pic_col]      -ap*C[0:pic_row,0:pic_col]).repeat(3,1,1).permute(1,2,0) \
                                                                                                                *    pic_torch[1:pic_row+1,0:pic_col,:],        #ml
                    (-am*A[1:pic_row+1,1:pic_col+1]  -am*C[1:pic_row+1,1:pic_col+1]  -B[1:pic_row+1,1:pic_col+1]
                    -am*A[1:pic_row+1,0:pic_col]    -am*C[1:pic_row+1,0:pic_col]     +B[1:pic_row+1,0:pic_col]
                    -am*A[0:pic_row,1:pic_col+1]    -am*C[0:pic_row,1:pic_col+1]     +B[0:pic_row,1:pic_col+1]
                    -am*A[0:pic_row,0:pic_col]      -am*C[0:pic_row,0:pic_col]       -B[0:pic_row,0:pic_col]).repeat(3,1,1).permute(1,2,0) \
                                                                                                                *     pic_torch[1:pic_row+1,1:pic_col+1,:],     #mm
                    (am*A[1:pic_row+1,1:pic_col+1]  -ap*C[1:pic_row+1,1:pic_col+1]   
                    +am*A[0:pic_row,1:pic_col+1]    -ap*C[0:pic_row,1:pic_col+1]).repeat(3,1,1).permute(1,2,0) \
                                                                                                                *     pic_torch[1:pic_row+1,2:pic_col+2,:],     #mr
                    (ap*A[1:pic_row+1,0:pic_col]    +ap*C[1:pic_row+1,0:pic_col]     -B[1:pic_row+1,0:pic_col]).repeat(3,1,1).permute(1,2,0) \
                                                                                                                *     pic_torch[2:pic_row+2,0:pic_col,:],       #bl
                    (am*C[1:pic_row+1,1:pic_col+1]  -ap*A[1:pic_row+1,1:pic_col+1]   
                    +am*C[1:pic_row+1,0:pic_col]    -ap*A[1:pic_row+1,0:pic_col]).repeat(3,1,1).permute(1,2,0) \
                                                                                                                *     pic_torch[2:pic_row+2,1:pic_col+1,:],     #bm
                    (ap*A[1:pic_row+1,1:pic_col+1]  +ap*C[1:pic_row+1,1:pic_col+1]   +B[1:pic_row+1,1:pic_col+1]).repeat(3,1,1).permute(1,2,0) \
                                                                                                                *     pic_torch[2:pic_row+2,2:pic_col+2,:]      #br
                    ], dim = 0).sum(dim=0)
    
    return pic_torch



def time_step_estimator(A,B,C,pic_row,pic_col,h,alpha):

    ap=alpha
    am=1-ap
    

    mu_max_torch = torch.amax(torch.abs(torch.stack([
                    (ap*A[0:pic_row,0:pic_col]      +ap*C[0:pic_row,0:pic_col]       +B[0:pic_row,0:pic_col])    ,        #ul
                    (am*C[0:pic_row,0:pic_col]      -ap*A[0:pic_row,0:pic_col]       
                    +am*C[0:pic_row,1:pic_col+1]    -ap*A[0:pic_row,1:pic_col+1])                                ,        #um
                    (ap*A[0:pic_row,1:pic_col+1]    +ap*C[0:pic_row,1:pic_col+1]     -B[0:pic_row,1:pic_col+1])  ,        #ur
                    (am*A[1:pic_row+1,0:pic_col]    -ap*C[1:pic_row+1,0:pic_col]     
                    +am*A[0:pic_row,0:pic_col]      -ap*C[0:pic_row,0:pic_col])                                  ,        #ml
                    (-am*A[1:pic_row+1,1:pic_col+1]  -am*C[1:pic_row+1,1:pic_col+1]  -B[1:pic_row+1,1:pic_col+1]
                    -am*A[1:pic_row+1,0:pic_col]    -am*C[1:pic_row+1,0:pic_col]     +B[1:pic_row+1,0:pic_col]
                    -am*A[0:pic_row,1:pic_col+1]    -am*C[0:pic_row,1:pic_col+1]     +B[0:pic_row,1:pic_col+1]
                    -am*A[0:pic_row,0:pic_col]      -am*C[0:pic_row,0:pic_col]       -B[0:pic_row,0:pic_col])    ,     #mm
                    (am*A[1:pic_row+1,1:pic_col+1]  -ap*C[1:pic_row+1,1:pic_col+1]   
                    +am*A[0:pic_row,1:pic_col+1]    -ap*C[0:pic_row,1:pic_col+1])                                ,     #mr
                    (ap*A[1:pic_row+1,0:pic_col]    +ap*C[1:pic_row+1,0:pic_col]     -B[1:pic_row+1,0:pic_col])  ,       #bl
                    (am*C[1:pic_row+1,1:pic_col+1]  -ap*A[1:pic_row+1,1:pic_col+1]   
                    +am*C[1:pic_row+1,0:pic_col]    -ap*A[1:pic_row+1,0:pic_col])                                ,     #bm
                    (ap*A[1:pic_row+1,1:pic_col+1]  +ap*C[1:pic_row+1,1:pic_col+1]   +B[1:pic_row+1,1:pic_col+1])      #br
                    ], dim = 0)).sum(dim=0))
    
    return (2/mu_max_torch)