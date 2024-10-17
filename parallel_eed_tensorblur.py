import os
import shutil
import time
from time import time

import torch
from torchvision import transforms

import multiprocessing
from joblib import Parallel, delayed
from multiprocessing import Array

import numpy as np
import scipy as sp
import random

import PIL as pil
from PIL import Image
import distutils
from time import time
from pathlib import Path

import diffusion_step_rgb
import diffusion
from diffusion import Matrix_Container


def iso_diffusion(lam,tau, k_size, k_sig, n_steps, save_points, alpha, source_dirs, tar_path, tar_dirs, Mp, Np, device, h=1):
    #set environment variables:
    Np=str(Np)
    os.environ["OMP_NUM_THREADS"] = Np  # export OMP_NUM_THREADS=2
    os.environ["OPENBLAS_NUM_THREADS"] = Np  # export OPENBLAS_NUM_THREADS=2
    os.environ["MKL_NUM_THREADS"] = Np  # export MKL_NUM_THREADS=2
    os.environ["VECLIB_MAXIMUM_THREADS"] = Np  # export VECLIB_MAXIMUM_THREADS=2
    os.environ["NUMEXPR_NUM_THREADS"] = Np  # export NUMEXPR_NUM_THREADS=2

    #function to be run in parallel
    def func_par(image_name, source_dir, tar_path, tar_dir, h, lam, tau, k_size, k_sig, alpha, n_steps, save_points, device):
        pic_torch = 255*transforms.ToTensor()(Image.open(source_dir+'/'+image_name).convert('RGB')).permute(1,2,0).to(device)
        matrix_container = Matrix_Container(3, k_size, k_sig, device)
        print('starting')
        if not os.path.isfile(tar_path+str(save_points[-1])+tar_dir+image_name):
            for i in range(n_steps):
                pic_torch = diffusion_step_rgb.step_eed_tensorblur(pic_torch, k_size, k_sig, lam, h, tau, alpha, matrix_container, device, step_id = i, verbose = False)
                if i in save_points:
                    image = (1/255)*pic_torch.clone()
                    image[image>1] = 1
                    image[image<0] = 0
                    image = (1/(image.max()-image.min()))*(image-image.min())
                    image = transforms.ToPILImage()(image.permute(2,0,1))

                    image.save(tar_path+str(i)+tar_dir+image_name)
        print('done')

    #create image_list
    image_lists=[]
    for source_path in source_dirs:
        image_lists.append(os.listdir(source_path))


    for i in range(len(image_lists)):
        Parallel(n_jobs=Mp)(delayed(func_par)(image_name, source_dirs[i],tar_path, tar_dirs[i], h, lam,
                                                                    tau, k_size, k_sig, alpha, n_steps, save_points, device) for image_name in image_lists[i])

    

if __name__ == "__main__":
    device = 'cpu'
    source_path = '/home/eheinert/NatalieAnnikaPASCALVOC/'#'/home/eheinert/Annika20000Carla/'
    #'/home/eheinert/PycharmProjects04/bdd100k_clear_daytime_partitioned/'
    #source_path = '/home/eheinert/PycharmProjects04/Folder_BDD100K_clear_Gauss_vs_EED/Gauss/127/'
    source_dirs =  ['/home/eheinert/NatalieAnnikaPASCALVOC/JPEGImages']#['/home/eheinert/Annika20000Carla/Town01_1']
                   #['/home/eheinert/PycharmProjects04/bdd100k_clear_daytime_partitioned/part1/images']#,
                   #'/home/eheinert/PycharmProjects04/bdd100k_clear_daytime_partitioned/part2/images',
                   #'/home/eheinert/PycharmProjects04/bdd100k_clear_daytime_partitioned/part3/images']
    tar_path = '/home/eheinert/NatalieAnnikaPASCALVOC/Diffused/'#/home/eheinert/Annika20000Carla/Town01_1_EED/'
    #'/home/eheinert/PycharmProjects04/Folder_BDD100K_clear_Gauss_vs_EED/EED/'
    tar_dirs = ['/']#['/part1/']#, '/part2/', '/part3/']

    lam     = 1/15 #0.1
    k_size  = 5    #9
    k_sig   = np.sqrt(k_size)
    h       = 1                               # artificial physical step length
    tau     = 0.2
    alpha   = 0.49
    step_0  = 0
    n_steps = 8195               # number of processing steps
    save_points = [255,1023,2047,4095,8191]

    Mp, Np = 60, 1

    #r, c = 720, 1280
    #A = torch.ones(r+1,c+1).to(device)
    #C = A
    #B = torch.zeros(r+1,c+1).to(device)
    
    if True:
        for save_point in save_points:
            for tar_dir in tar_dirs:
                os.makedirs(tar_path+str(save_point)+tar_dir, exist_ok=True)
            #    shutil.copytree(source_path+tar_dir[1:]+'labels', tar_path+str(save_point)+tar_dir+'labels')
            #    shutil.copyfile(source_path+tar_dir[1:]+tar_dir[1:-1]+'.txt', tar_path+str(save_point)+tar_dir+tar_dir[1:-1]+'.txt')
            #shutil.copyfile(source_path+'train_part1_part2.txt', tar_path+str(save_point)+'/train_part1_part2.txt')
            #shutil.copyfile(source_path+'train_part1_part3.txt', tar_path+str(save_point)+'/train_part1_part3.txt')
            #shutil.copyfile(source_path+'train_part2_part3.txt', tar_path+str(save_point)+'/train_part2_part3.txt')
    #
            #shutil.copyfile(source_path+'val_part1.txt', tar_path+str(save_point)+'/val_part1.txt')
            #shutil.copyfile(source_path+'val_part2.txt', tar_path+str(save_point)+'/val_part2.txt')
            #shutil.copyfile(source_path+'val_part3.txt', tar_path+str(save_point)+'/val_part3.txt')
    
    iso_diffusion(lam, tau, k_size, k_sig, n_steps, save_points, alpha, source_dirs, tar_path, tar_dirs, Mp, Np, device, h=1)