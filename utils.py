import torch 
import torch.nn as nn
from torchvision import transforms

import PIL as pil
from PIL import Image
import math

from typing import Tuple
import time

#@jit
def symm_pad_2D(IM: torch.Tensor, padding: Tuple[int, int, int, int]):
    #IM2 = IM.clone()

    if 0 in padding:
        raise NotImplementedError('The function is not suitable for 0 paddings yet')
    h, w = IM.shape[-2:]
    left, right, top, bottom = padding

    T = torch.flip(IM[0:top,:],[0])
    B = torch.flip(IM[-bottom:,:],[0])

    IM = torch.cat((T,IM,B),0)

    L = torch.flip(IM[:,0:top],[1])
    R = torch.flip(IM[:,-right:],[1])

    IM = torch.cat((L,IM,R),1)

    return IM

def channel_wise_symm_pad_2d(IM: torch.Tensor, padding: Tuple[int, int, int, int]):
    return symm_pad_2D(IM.permute(1,2,0),padding).permute(2,0,1)


def create_padding_indices(height,width,top,bottom,left,right,device):      #possibly faster on gpu when created for container
                          #application simply via pic_pad = torch.index_select(torch.index_select(pic_torch, 0, ind_y),1,ind_x)
    ind_x = torch.cat((torch.arange(left,0,-1), torch.arange(0,width), torch.arange(width-2, width-right-2, -1)))
    ind_y = torch.cat((torch.arange(top,0,-1), torch.arange(0,height), torch.arange(height-2, height-bottom-2, -1)))

    return ind_x.to(device), ind_y.to(device)


def get_gaussian_kernel(kernel_size, sigma, channels, device):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size, device = device)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    
    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter
def apply_gaussian_filter(f_gauss_ker, image):
    image = image.permute(2,0,1)
    image = torch.unsqueeze(image,0)
    image = f_gauss_ker(image)
    image = torch.squeeze(image)
    return image.permute(1,2,0)



if __name__ == "__main__":
    device = torch.device('cuda:0')
    A=torch.rand([3,2,2]).to(device)
    print(A[0,:,:])
    A=channel_wise_symm_pad_2d(A,[1,1,1,1])
    print(A[0,:,:])

    exit()

    B = torch.rand([2,2,3]).to(device)
    print(B.shape)
    print(B.permute(2,0,1))
    B = symm_pad_2D(B,[1,1,1,1])
    print(B.shape)
    print(B.permute(2,0,1))

    exit()

    A = torch.rand([1256,376,3]).to(device)

    start = time.time()
    for i in range(10000):
        B = symm_pad_2D(A,[1,1,1,1])
        B = symm_pad_2D(A,[3,3,3,3])
        B, C = None, None
    end = time.time()
    print(((end-start)*3*10000)/(60*60))



