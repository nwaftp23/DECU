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
import matplotlib as mpl

mpl.rcParams['axes.linewidth'] = 1.5 



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
    parser.add_argument('--ensemble_name', type=str, help='which ensemble to load', default='bootstrapped')
    parser.add_argument('--full_ensemble', action='store_true', help='ensemble from scratch')
    args = parser.parse_args()
    seed_everything(42)

    train_path = '/home/nwaftp23/scratch/uncertainty_estimation/imagenet/ILSVRC2012_train'
    numb_comps = 10 
    if not args.full_ensemble:
        #model = get_model(args.path)
        model = load_ensemble(args.path, True, numb_comps)
        ensemble_comps = []
        ensemble_size = model.model.diffusion_model.ensemble_size
        comp_idx = random.randint(0, (ensemble_size-1))
    else:
        comp_idx = random.randint(0, (numb_comps-1))
        ensemble_comps = [get_model(os.path.join(args.path, i)) for i in
            os.listdir(args.path) if args.ensemble_name in i]
        model = ensemble_comps[comp_idx]
        ensemble_size = len(ensemble_comps)
    with open(os.path.join(train_path, 'filelist.txt')) as f:
        filelist = f.readlines()
    count_per_synset = [f.split('/')[0] for f in filelist]
    count_per_synset = dict(Counter(count_per_synset))
    
    with open(os.path.join(train_path, 'subsets.pkl'), 'rb') as fp:
        subsets = pickle.load(fp)
    # TODO:
    #could figure out which index is which by conditioning and then checking
    human_synset = {} 
    with open(os.path.join(train_path, 'synset_human.txt')) as f:
        for line in f:
            items = line.split()
            key, values = items[0], ' '.join(items[1:])
            human_synset[key] = values
    syn1_human = {k: human_synset[k] for k in subsets[1]}
    syn1300_human = {k: human_synset[k] for k in subsets[1300]}
    idx2synset = OmegaConf.load(os.path.join(train_path, 'index_synset.yaml'))
    synset2idx = {y: x for x, y in idx2synset.items()}
    df = pd.DataFrame(human_synset.items(), columns=['synset', 'text'])
    # missing synset 'n02012849'
    # double cranes n02012849, n03126707
    classes_1 = [synset2idx[k] for k in subsets[1]]
    classes_10 = [synset2idx[k] for k in subsets[10]]
    classes_100 = [synset2idx[k] for k in subsets[100]]
    classes_1300 = [synset2idx[k] for k in subsets[1300]]
    #classes_1 = random.sample(classes_1, 4)
    #print('redo class values as the subsets have changed')
    #import pdb; pdb.set_trace()
    #classes_1 = random.sample(classes_1, 10)
    #classes_10 = random.sample(classes_10, 10)
    #classes_100 = random.sample(classes_100, 10)
    #classes_1300 = random.sample(classes_1300, 100)
    
    classes_1 = [892, 503, 147]
    classes_10 = [936, 379, 616]
    classes_100 = [767, 118, 36]
    classes_1300 = [141, 466, 992]

    #classes2sample= classes_1+classes_10+classes_100+classes_1300
    classes2sample = interlaced = [item for sublist in zip(classes_1, classes_10, classes_100, classes_1300) for item in sublist]
    #classes2sample = classes_1+classes_10+classes_100+classes_1300
    if args.sampler == 'DDIM':
        sampler = DDIMSampler(model)
    else: 
        print('Not setup for DDPM')
        sampler = DDPMSampler(model)
    ddim_steps = args.ddim_steps
    ddim_eta = args.ddim_eta
    scale = args.scale
    n_samples_per_class = 1

    # --path /home/nwaftp23/scratch/logs/bootstrapped_imagenet_10 --unc_branch 200
    base_dir = '/home/nwaftp23/scratch/uncertainty_estimation/imagenet/ILSVRC2012_train'
    filelists = [f'filelist_comp{i}.txt' for i in range(ensemble_size)]
    synsets2sample = {i:idx2synset[i] for i in classes2sample}

    comp_files =[]
    for f in filelists:
        with open(os.path.join(base_dir, f)) as fl:
            comp_filelist = fl.readlines()
        comp_files.append(comp_filelist)
    numbfilesbycomp = {}
    for k,v in synsets2sample.items():
        numb_files = []
        for cf in comp_files:
            numb_files.append(len([i for i in cf if v in i]))
        numbfilesbycomp[k]=numb_files

    with torch.no_grad():
        with model.ema_scope():
            all_ucs = []
            for _ in range(ensemble_size):
                if not ensemble_comps:
                    uc = model.get_learned_conditioning(
                        {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)},
                        comp_idx = _)
                    all_ucs.append(uc)
                else:
                    uc = ensemble_comps[_].get_learned_conditioning(
                        {ensemble_comps[_].cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)})
                    all_ucs.append(uc)
            
            x_T = torch.randn((n_samples_per_class,3,64,64), device=model.device) 
            j = 0
            batch_latent = []
            batch_unc_latent = []
            batch_pix = []
            batch_unc_pix = []
            batch_numb_samples = []
            for class_label in classes2sample:
                print('make grid')
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in"\
                f" {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class*[class_label])
                all_cs = []
                for _ in range(ensemble_size):
                    if not ensemble_comps:
                        c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)},
                            comp_idx = _)
                        all_cs.append(c)
                    else:
                        c = ensemble_comps[_].get_learned_conditioning(
                            {ensemble_comps[_].cond_stage_key: xc.to(model.device)})
                        all_cs.append(c)
                seed_everything(42)

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
                                                 ensemble_comps=ensemble_comps,
                                                 unc_per_pixel=True, 
                                                 x_T=x_T)
                if len(samples_ddim.shape)== 5:
                    samples_ddim = samples_ddim.reshape(-1, samples_ddim.shape[2],
                        samples_ddim.shape[3], samples_ddim.shape[4])

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                #x_samples_ddim = model.decode_first_stage(samples_ddim[:,:,:,:])
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0,
                                             min=0.0, max=1.0)
                batch_latent.append(samples_ddim)
                batch_unc_latent.append(epi_unc)
                batch_pix.append(x_samples_ddim)
                n_comps = 5
                unc_pix, _ = pairwise_exp(x_samples_ddim[:n_comps,:,:,:].unsqueeze(1), 0, 'Wass_0', n_comps, unc_per_pixel=True)
                batch_unc_pix.append(unc_pix)
                batch_numb_samples.append(np.mean(numbfilesbycomp[class_label]))
            for i in range(len(classes_1)):
                file_name = os.path.join(os.path.join(args.path,'pixel_unc_paper'), 
                    (f'pixel_unc_{i}.png'))
                pics = [p[0,:,:,:] for p in batch_pix[i*4:(i+1)*4]]
                uncs = [p[0,:,:,:] for p in batch_unc_pix[i*4:(i+1)*4]]
                
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
                max_unc = torch.stack(uncs).max()
                min_unc = torch.stack(uncs).min()
                for k, unc in enumerate(uncs):
                    unc = unc.mean(0)
                    unc = (unc-min_unc)/(max_unc-min_unc)
                    px = 1/plt.rcParams['figure.dpi']
                    plt.subplots(figsize=(single_image_width*px, single_image_height*px))
                    plt.imshow(unc.cpu().numpy(), cmap='cividis', interpolation='nearest', vmin=min_unc, vmax=max_unc)
                    plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
                    plt.tight_layout(pad=0.0)
                    plt.savefig(os.path.join(args.path,'heat_map.png'))
                    im = Image.open(os.path.join(args.path,'heat_map.png'))
                    pixel_unc_image.paste(im, (single_image_width*k, single_image_height))
                
                pixel_unc_image.save(file_name)
