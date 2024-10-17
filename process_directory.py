import os
import argparse
import math

import torch
from torchvision import transforms
import PIL as pil
from PIL import Image

import diffusion_step_rgb
import diffusion
from diffusion import Matrix_Container

from joblib import Parallel, delayed

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    
    ## custom arguments for AL:
    parser.add_argument('recursive_source_dir', type=str,
                        help = 'Path to image directory. All images in this directory and its subdirectories will be processed.')
    parser.add_argument('--target_dir', type=str,
                        help = 'Processed clones of the recursive source directories will be created here.')
    parser.add_argument('--save_points', nargs='+', type=int, default=[512, 724, 1024, 1448, 2048, 2896, 4096, 5792],
                help="List of saving steps. For each of those there will be a complete clone of your data created. The highest number sets the runtime.")
    parser.add_argument('--contrast_parameter', type=float, default=1/15, help='Only needed for eed type diffusion.')
    parser.add_argument('--gauss_ker_size', type=int, default=5, help='Size of the full kernel - always odd.')
    parser.add_argument('--time_step', type=float, default=0.2, help='PDE time step size.')
                                  # artificial physical step length
    parser.add_argument('--diffusion_type', type=str, default='eed_tensorblur', help='Choose type of diffusion. We typically recommend eed_tensorblur.')
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--N_processes', type=int, default=40, help='Only used for cpu atm.')
    
    args = parser.parse_args()
    return args


def process_dir(source_dir, target_dir, device, N_proc, diffusion_type, save_points, n_steps, lam, k_size, k_sig, h, tau, alpha):
    
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    
    rest_paths = []
    roots = []
    # Walk through the directory recursively
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Check if the file has one of the valid image extensions
            if any(file.lower().endswith(ext) for ext in image_extensions):
                # Save the relative path of each found image
                rest_paths.append(os.path.join(root,file).replace(source_dir,'').lstrip('/'))
    
    def func_par(source_dir, target_dir, device, rest_path, h, lam, tau, k_size, k_sig, alpha, n_steps, save_points):
        pic_torch = 255*transforms.ToTensor()(Image.open(os.path.join(source_dir,rest_path)).convert('RGB')).permute(1,2,0).to(device)
        matrix_container = Matrix_Container(3, k_size, k_sig, device)
        print(f'starting image {os.path.join(source_dir,rest_path)}')
        if not os.path.isfile(os.path.join(target_dir, str(save_points[-1]+1), rest_path)):
            for i in range(n_steps):
                pic_torch = diffusion_step_rgb.step_eed_tensorblur(pic_torch, k_size, k_sig, lam, h, tau, alpha, matrix_container, device, step_id = i, verbose = False)
                if (i+1) in save_points:
                    os.makedirs(os.path.dirname(os.path.join(target_dir,str(i+1), rest_path)), exist_ok = True)
                    image = (1/255)*pic_torch.clone()
                    image[image>1] = 1
                    image[image<0] = 0
                    image = (1/(image.max()-image.min()))*(image-image.min())
                    image = transforms.ToPILImage()(image.permute(2,0,1))

                    image.save(os.path.join(target_dir,str(i+1), rest_path))
                    
                    
        print(f'done with image {os.path.join(source_dir,rest_path)}.')

    Parallel(n_jobs=N_proc)(delayed(func_par)(source_dir,target_dir, device, rest_path, h, lam,
                                                    tau, k_size, k_sig, alpha, n_steps, save_points) for rest_path in rest_paths)

    

if __name__ == "__main__":
    args = parse_args()
    
    source_dir = args.recursive_source_dir
    target_dir = args.target_dir
    if not target_dir:
        if source_dir.endswith(os.sep):
            target_dir = source_dir.rstrip(os.sep)+'_EED'
        else:
            target_dir = source_dir+'_EED'
    
    save_points = args.save_points
    n_steps = max(save_points)+1
    lam = args.contrast_parameter
    k_size = args.gauss_ker_size
    k_sig = math.sqrt(k_size)
    h = 1 #artificial spatial step size
    tau = args.time_step
    alpha = 0.49
    diffusion_type = args.diffusion_type
    device = args.device
    N_proc = args.N_processes
    
    process_dir(source_dir, target_dir, device, N_proc, diffusion_type, save_points, n_steps, lam, k_size, k_sig, h, tau, alpha)