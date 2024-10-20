# Reducing Texture Bias of Deep Neural Networks via Edge Enhancing Diffusion

In this repository you find the torch implementation of our variant of Edge Enhancing Diffusion (EED) as used in the paper "Reducing Texture Bias of Deep Neural Networks via Edge Enhancing Diffusion" by Edgar Heinert, Matthias Rottmann, Kira Maag and Karsten Kahl. It was published at the European Conference on Artifical Intelligence (ECAI2024) and you can find the full paper [here](https://ebooks.iospress.nl/doi/10.3233/FAIA240540).

## EED Example

| Method                            | Image                                                                                  |
|------------------------------------|----------------------------------------------------------------------------------------|
| Original Cityscapes Image          | <img src="Example/Original_frankfurt_000001_028335_leftImg8bit.jpg" width="500"/>       |
| Standard EED                       | <img src="Example/DiffusionNOOrientation5792_frankfurt_000001_028335_leftImg8bit.jpg" width="500"/> |
| Our Variation with Orientation Smoothing | <img src="Example/DiffusionWithOrientation5792_frankfurt_000001_028335_leftImg8bit.png" width="500"/>

## Usage

The installation should be pretty straightforward as we have not encountered a setting in which it caused any problems yet. A simple `pip install -r requirements.txt` in the terminal with some possible local variations should do the trick. 

Before you create EED duplicates of a directory via
```
nohup python process_directory.py [recursive_source_dir]
```
you should check the argparser

```
def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    
    ## custom arguments for AL:
    parser.add_argument('recursive_source_dir', type=str,
                        help = 'Path to image directory. All images in this directory and its subdirectories will be processed.')
    parser.add_argument('--target_dir', type=str,
                        help = 'Processed clones of the recursive source directories will be created here.')
    parser.add_argument('--save_points', nargs='+', type=int, default=[512, 724, 1024, 1448, 2048, 2896, 4096, 5792],
                help="List of saving steps. For each of those there will be a complete clone of your data created. The highest number sets the runtime.")
    parser.add_argument('--N_processes', type=int, default=40, help='Only used for cpu atm.')
    parser.add_argument('--contrast_parameter', type=float, default=1/15, help='Only needed for eed type diffusion.')
    parser.add_argument('--gauss_ker_size', type=int, default=5, help='Size of the full kernel - always odd.')
    parser.add_argument('--time_step', type=float, default=0.2, help='PDE time step size.')
                                  # artificial physical step length
    parser.add_argument('--diffusion_type', type=str, default='eed_tensorblur', help='Choose type of diffusion. We typically recommend eed_tensorblur.')
    parser.add_argument('--device', default="cpu")
    
    args = parser.parse_args()
    return args
```
If you don't specify a target directory the script will create a folder in the super-directory as sour source dir.The computation is still quite heavy in terms of computing ressources and the speed will very much depend on the number of CPU processors you have available.

## Citation

If you use this repository or find our research relevant to your work, please cite our paper:

```bibtex
@article{heinert2024reducing,
  title={Reducing Texture Bias of Deep Neural Networks via Edge Enhancing Diffusion},
  author={Heinert, Edgar and Rottmann, Matthias and Maag, Kira and Kahl, Karsten},
  journal={arXiv preprint arXiv:2402.09530},
  year={2024}
}
```
