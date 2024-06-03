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

from save_ensemble import load_ensemble
from create_helper_variables import create_synset_dicts, create_bin_dicts
from create_subset_masked import check_synset
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm_test import DDPMSampler


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model(path):
    timestamp = path.split('/')[-1].split('_')[0]
    config_path = os.path.join(path, f'configs/{timestamp}-project.yaml')
    config = OmegaConf.load(config_path)
    model_path = os.path.join(path, f'checkpoints/last.ckpt')
    model = load_model_from_config(config, model_path)
    return model

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def sample_model(model, n_samples_per_class, classes2sample, ddim_steps, ddim_eta, scale, 
        ensemble_size, path, unc_branch, sampler, comp_idx, part, dataset,
        certain_comps, uncertain_comps, certain_classes, uncertain_classes):
    path = os.path.join(path, 'certain_vs_uncertain')
    if not os.path.exists(path):
        os.mkdir(path)
    with torch.no_grad():
        with model.ema_scope():
            all_ucs = []
            for _ in range(ensemble_size):
                uc = model.get_learned_conditioning(
                    {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)},
                    comp_idx = _)
                all_ucs.append(uc)
            all_samples_rows = []
            certain_all_samples_rows = []
            uncertain_all_samples_rows = []
            for idx, class_label in enumerate(classes2sample):
                seed_everything(42)
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in"\
                    f" {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class*[class_label])
                all_cs = []
                for _ in range(ensemble_size):
                    c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)}, 
                        comp_idx = _)
                    all_cs.append(c)
               
                comp_idx = random.randint(0, ensemble_size-1)
                samples_ddim, epi_unc, intermediates, dist_mat = sampler.sample(S=ddim_steps,
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
                                                 branch_split=unc_branch, 
                                                 all_ucs=all_ucs,
                                                 all_cs=all_cs) 
                samples_ddim = samples_ddim[:5,:,:,:,:]
                if len(samples_ddim.shape)== 5:
                    samples_ddim = samples_ddim.reshape(-1, samples_ddim.shape[2], 
                        samples_ddim.shape[3], samples_ddim.shape[4])

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                             min=0.0, max=1.0)
                if class_label in certain_classes:
                    certain_all_samples_rows.append(x_samples_ddim)
                elif class_label in uncertain_classes:
                    uncertain_all_samples_rows.append(x_samples_ddim)
                else:
                    all_samples_rows.append(x_samples_ddim)
            if certain_comps:
                certain_all_samples_rows = []
                uncertain_all_samples_rows = []
                for rw_num, rw in enumerate(all_samples_rows):
                    certain_all_samples_rows.append([rw[i] for i in certain_comps[rw_num]])
                    uncertain_all_samples_rows.append([rw[i] for i in uncertain_comps[rw_num]])
            
            if type(certain_all_samples_rows[0])==list:
                certain_grid = [torch.stack(rw) for rw in certain_all_samples_rows]
            else:
                certain_grid = certain_all_samples_rows
            certain_grid = torch.stack(certain_grid, 0)
            num_per_row = certain_grid.shape[1]
            certain_grid = rearrange(certain_grid, 'n b c h w -> (n b) c h w')
            certain_grid = make_grid(certain_grid, nrow=num_per_row)

            # to image
            certain_grid = 255. * rearrange(certain_grid, 'c h w -> h w c').cpu().numpy()
            certain_grid_class_img = Image.fromarray(certain_grid.astype(np.uint8))
            file_name = os.path.join(path,f'certain_images5_{dataset}_p{part}.png')
            certain_grid_class_img.save(file_name)
            print(file_name)
            certain_grid_class_img.close()
            
            if type(uncertain_all_samples_rows[0])==list:
                uncertain_grid = [torch.stack(rw) for rw in uncertain_all_samples_rows]
            else:
                uncertain_grid = uncertain_all_samples_rows
            uncertain_grid = torch.stack(uncertain_grid, 0)
            num_per_row = uncertain_grid.shape[1]
            uncertain_grid = rearrange(uncertain_grid, 'n b c h w -> (n b) c h w')
            uncertain_grid = make_grid(uncertain_grid, nrow=num_per_row)

            # to image
            uncertain_grid = 255. * rearrange(uncertain_grid, 'c h w -> h w c').cpu().numpy()
            uncertain_grid_class_img = Image.fromarray(uncertain_grid.astype(np.uint8))
            file_name = os.path.join(path,f'uncertain_images5_{dataset}_p{part}.png')
            uncertain_grid_class_img.save(file_name)
            uncertain_grid_class_img.close()
            print(file_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Uncertainty Per Synset')
    parser.add_argument('--path', type=str, help='path to model', required=True)
    parser.add_argument('--sampler', type=str, help='which smapler to use', default='DDIM')
    parser.add_argument('--ddim_eta', type=float, help='controls stdev for generative process', 
        default=0.00)
    # ddim_eta 0-1, 1=DDPM 
    parser.add_argument('--scale', type=float, help='controls the amount of unconditional guidance', 
        default=5.0)
    # higher scale less diversity
    parser.add_argument('--ddim_steps', type=int, help='number of steps to take in ddim', 
        default=200)
    parser.add_argument('--unc_branch', type=int, help='when to split for generative proccess', 
        default=0)
    parser.add_argument('--dataset', type=str, help='which dataset was used to train the ensemble', default='binned_classes')
    parser.add_argument('--part', type=int, help='one for main paper other for appendix', 
        default=1)
    args = parser.parse_args()
    seed_everything(42)

    numb_comps = 5
    model = load_ensemble(args.path, numb_comps)
    ensemble_size = model.model.diffusion_model.ensemble_size
    comp_idx = random.randint(0, (ensemble_size-1))
    subsets, human_synset, idx2synset, synset2idx = create_synset_dicts(args.dataset)
    df = pd.DataFrame(human_synset.items(), columns=['synset', 'text'])
    certain_comps = []
    uncertain_comps = []
    certain_classes = []
    uncertain_classes = [] 
    syn1_human, syn10_human, syn100_human, syn1300_human = create_bin_dicts(human_synset, subsets)
    # missing synset 'n02012849'
    # double cranes n02012849, n03126707
    classes_1 = [synset2idx[k] for k in subsets[1]]
    classes_10 = [synset2idx[k] for k in subsets[10]]
    classes_100 = [synset2idx[k] for k in subsets[100]]
    classes_1300 = [synset2idx[k] for k in subsets[1300]]
    classes_10 = []
    classes_100 = []
    if args.part == 1:
        classes_1 = [499, 190, 631, 789, 901]
        classes_1300 = [30, 959, 280, 510, 986]
    elif args.part == 2:
        classes_1 = [708, 798, 662, 811, 577]
        classes_1300 = [346, 595, 89, 25, 864] 
    classes2sample = classes_1+classes_10+classes_100+classes_1300 
    certain_classes = classes_1300
    uncertain_classes = classes_1
    synsets2sample = {i:idx2synset[i] for i in classes2sample}
    if args.sampler == 'DDIM':
        sampler = DDIMSampler(model)
    else: 
        print('Not setup for DDPM')
        sampler = DDPMSampler(model)
    ddim_steps = args.ddim_steps
    ddim_eta = args.ddim_eta
    scale = args.scale
    n_samples_per_class = 1 
    
    pic_path = args.path
    sample_model(model, n_samples_per_class, classes2sample, ddim_steps, ddim_eta, 
        scale, ensemble_size, pic_path, args.unc_branch, sampler, comp_idx, 
        args.part, args.dataset, certain_comps, uncertain_comps,
        certain_classes, uncertain_classes)
