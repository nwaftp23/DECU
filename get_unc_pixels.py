import pickle
import random
import argparse
import torch
import os
from collections import Counter
from omegaconf import OmegaConf

import numpy as np 
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from einops import rearrange
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import seaborn as sns

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm_test import DDPMSampler
from save_ensemble import load_ensemble
from sample_model import sample_model, get_model, seed_everything
from make_histogram import get_min_max, norm_uncs, split_uncs
from uncertainty_estimation.uncertainty_estimators import pairwise_exp
from create_helper_variables import create_synset_dicts, create_bin_dicts, check_synset

def get_count(subsets, class_num, idx2synset):
    synset = idx2synset[class_num]
    count = [k for k,v in subsets.items() if synset in v]
    return count[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Uncertainty Per Synset')
    parser.add_argument('--path', type=str, help='path to model', required=True)
    parser.add_argument('--sampler', type=str, help='which smapler to use', default='DDIM')
    parser.add_argument('--unc_branch', type=int, help='where to branch for uncertainty', 
        default=0)
    parser.add_argument('--ddim_eta', type=float, help='controls stdev for generative process', 
        default=0.00)
    # ddim_eta 0-1, 1=DDPM 
    parser.add_argument('--ddim_steps', type=int, help='number of steps to take in ddim', 
        default=200)
    parser.add_argument('--base_comp', type=int, help='comp to start from before branching', 
        default=-1)
    parser.add_argument('--scale', type=float, help='controls the amount of unconditional guidance',
        default=5.0)
    parser.add_argument('--dataset', type=str, help='which dataset was used to train the ensemble', default='binned_classes')
    args = parser.parse_args()
    seed_everything(42)

    numb_comps = 5
    model = load_ensemble(args.path, numb_comps)
    ensemble_size = model.model.diffusion_model.ensemble_size
    comp_idx = random.randint(0, (ensemble_size-1))
    subsets, human_synset, idx2synset, synset2idx = create_synset_dicts(args.dataset)
    df = pd.DataFrame(human_synset.items(), columns=['synset', 'text'])
    if args.dataset == 'binned_classes':
        syn1_human, syn10_human, syn100_human, syn1300_human = create_bin_dicts(human_synset, subsets)
        # missing synset 'n02012849'
        # double cranes n02012849, n03126707
        classes_1 = [synset2idx[k] for k in subsets[1]]
        classes_10 = [synset2idx[k] for k in subsets[10]]
        classes_100 = [synset2idx[k] for k in subsets[100]]
        classes_1300 = [synset2idx[k] for k in subsets[1300]]
        classes_1 = [392, 708, 729, 854]
        classes_10 = [120, 445, 726, 943]
        classes_100 = [24, 187, 830, 995]
        classes_1300 = [25, 447, 991, 992]
        classes2sample = []
        for idx_1, idx_10, idx_100, idx_1300 in zip(classes_1, classes_10, classes_100, classes_1300):
            classes2sample.append((idx_1, idx_10, idx_100, idx_1300))
    sampler = DDIMSampler(model)
    ddim_steps = args.ddim_steps
    ddim_eta = args.ddim_eta
    scale = args.scale
    n_samples_per_class = 1 

    filelists = [f'filelist_comp{i}.txt' for i in range(ensemble_size)]
    synsets2sample = {i:idx2synset[i] for i in list(sum(classes2sample, ()))}


    all_classes = classes2sample
    with torch.no_grad():
        with model.ema_scope():
            all_ucs = []
            for _ in range(ensemble_size):
                uc = model.get_learned_conditioning(
                    {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)},
                    comp_idx = _)
                all_ucs.append(uc)
            
            pic_num = 0
            for idx, label_batch in enumerate(classes2sample):
                if args.dataset == 'binned_classes':
                    batch_latent = []
                    batch_unc_latent = []
                    batch_pix = []
                    batch_unc_pix = []
                    batch_numb_samples = []
                    certain_comp = ([],[],[],[])
                for j, class_label in enumerate(label_batch):
                    print(f"rendering {n_samples_per_class} examples of class '{class_label}' in"\
                        f" {ddim_steps} steps and using s={scale:.2f}.")
                    print(f'{human_synset[idx2synset[class_label]]}')
                    xc = torch.tensor(n_samples_per_class*[class_label])
                    x_T = torch.randn((n_samples_per_class,3,64,64), device=model.device) 
                    all_cs = []
                    for _ in range(ensemble_size):
                        c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)},
                            comp_idx = _)
                        all_cs.append(c)
                    seed_everything(42)
                    comp_idx = random.randint(0, ensemble_size-1)
                    samples_ddim, epi_unc, inter, dist_mat  = sampler.sample(S=ddim_steps,
                                                     conditioning=all_cs[comp_idx],
                                                     batch_size=n_samples_per_class,
                                                     shape=[3, 64, 64],
                                                     verbose=False,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=all_ucs[comp_idx],
                                                     eta=ddim_eta,
                                                     ensemble_comp=comp_idx,
                                                     return_distribution=True,
                                                     return_unc=True,
                                                     branch_split=args.unc_branch,
                                                     all_ucs=all_ucs,
                                                     all_cs=all_cs,
                                                     unc_per_pixel=True, 
                                                     x_T=x_T, 
                                                     certain_comps=certain_comp[j])
                    if len(samples_ddim.shape)== 5:
                        samples_ddim = samples_ddim.reshape(-1, samples_ddim.shape[2],
                            samples_ddim.shape[3], samples_ddim.shape[4])

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0,
                                                 min=0.0, max=1.0)
                    batch_latent.append(samples_ddim)
                    batch_unc_latent.append(epi_unc)
                    batch_pix.append(x_samples_ddim)
                    unc_pix, _ = pairwise_exp(x_samples_ddim.unsqueeze(1), 0, 'Wass_0', numb_comps, unc_per_pixel=True)
                    batch_unc_pix.append(unc_pix)
                    batch_numb_samples.append(get_count(subsets, class_label, idx2synset))
                pixel_dir = os.path.join(args.path,'pixel_unc')
                if not os.path.exists(pixel_dir):
                    os.mkdir(pixel_dir)
                iter_num = len(classes2sample[0])
                file_name = os.path.join(pixel_dir, (f'pixel_unc_{idx}.png'))
                pics = [p[0,:,:,:] for p in batch_pix]
                uncs = [p[0,:,:,:] for p in batch_unc_pix]
                grid = torch.stack(pics, 0)
                #grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=4)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                grid_img = Image.fromarray(grid.astype(np.uint8))
                image_width = grid_img.width
                image_height = grid_img.height
                pixel_unc_image = Image.new("RGB", (image_width, image_height*2),
                    (255, 255, 255))
                pixel_unc_image.paste(grid_img, (0, 0))
                pixel_unc_image.save(file_name)
                single_image_height = image_height
                single_image_width = int(image_width/4+1)
                draw = ImageDraw.Draw(pixel_unc_image)
                max_unc = torch.stack(uncs).mean(0).max()
                min_unc = torch.stack(uncs).mean(0).min()
                for k, unc in enumerate(uncs):
                    unc = unc.mean(0)
                    max_unc = unc.mean(0).max()
                    min_unc = unc.mean(0).min()
                    #unc = (unc-min_unc)/(max_unc-min_unc)
                    px = 1/plt.rcParams['figure.dpi']
                    plt.subplots(figsize=(single_image_width*px, single_image_height*px))
                    plt.imshow(unc.cpu().numpy(), cmap='cividis', interpolation='nearest', vmin=min_unc, vmax=max_unc)
                    plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
                    plt.tight_layout(pad=0.0)
                    plt.savefig(os.path.join(args.path,'heat_map.png'))
                    im = Image.open(os.path.join(args.path,'heat_map.png'))
                    pixel_unc_image.paste(im, (single_image_width*k, single_image_height))

                pixel_unc_image.save(file_name)
