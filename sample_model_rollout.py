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
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm_test import DDPMSampler
from create_helper_variables import create_synset_dicts, create_bin_dicts
from create_subset_masked import check_synset


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
    #model_path = os.path.join(path, f'checkpoints/epoch=000012.ckpt')
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
        ensemble_size, path, synsets, unc_branch, sampler, bins, comp_idx, uncertain):
    path = os.path.join(path, 'rollout_sample')
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
                 
            for bin_idx, class_label in enumerate(classes2sample):
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in"\
                    f" {ddim_steps} steps and using s={scale:.2f}.")
                print(f'human synset: {synsets[bin_idx]}')
                xc = torch.tensor(n_samples_per_class*[class_label])
                all_cs = []
                for _ in range(ensemble_size):
                    c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)}, 
                        comp_idx = _)
                    all_cs.append(c)
                all_samples_class =[]
                comp_idx_per_class = comp_idx[bin_idx]
                print(comp_idx_per_class)
                samples_ddim, epi_unc, intermediates, dist_mat = sampler.sample(S=ddim_steps,
                                                 conditioning=all_cs[comp_idx_per_class],
                                                 batch_size=n_samples_per_class,
                                                 shape=[3, 64, 64],
                                                 verbose=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=all_ucs[comp_idx_per_class], 
                                                 eta=ddim_eta,
                                                 ensemble_comp=comp_idx_per_class,
                                                 return_distribution=True,
                                                 return_unc=True,
                                                 branch_split=unc_branch, 
                                                 all_ucs=all_ucs,
                                                 all_cs=all_cs, 
                                                 log_every_t=10)
                inter_zs = [i for i in intermediates['x_inter'] if i.shape[0] != 1]
                if unc_branch == 0:
                    del inter_zs[1]
                num_imgs_row_branch = len(inter_zs)
                pre_zs = [i for i in intermediates['x_inter'] if i.shape[0] == 1]
                if unc_branch >0:
                    del pre_zs[1]
                num_imgs_row = len(pre_zs)
                pre_zs = torch.stack(pre_zs) 
                inter_zs = torch.stack(inter_zs)[:,:5,:,:,:,:]
                if len(pre_zs.shape)== 5:
                    pre_zs = pre_zs.reshape(-1, pre_zs.shape[2], 
                        pre_zs.shape[3], pre_zs.shape[4])
                x_samples_ddim = model.decode_first_stage(pre_zs)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                all_samples_class.append(x_samples_ddim)
                grid = torch.stack(all_samples_class, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=num_imgs_row)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                grid_class_img = Image.fromarray(grid.astype(np.uint8))
                if bins:
                    file_name = os.path.join(path, (f'prebranch_bin{bins[bin_idx]}_{synsets[bin_idx]}.png'))
                else:
                    if uncertain: 
                        file_name = os.path.join(path, (f'prebranch_uncertain_{synsets[bin_idx]}.png'))
                    else:
                        file_name = os.path.join(path, (f'prebranch_certain_{synsets[bin_idx]}.png'))
                grid_class_img.save(file_name)

                for i in range(5):
                    all_samples_class = []
                    inter_zs_comp = inter_zs[:,i,:,:,:,:]
                    if len(inter_zs_comp.shape)== 5:
                        inter_zs_comp = inter_zs_comp.reshape(-1, inter_zs_comp.shape[2], 
                            inter_zs_comp.shape[3], inter_zs_comp.shape[4])

                    x_samples_ddim = model.decode_first_stage(inter_zs_comp)
                    #x_samples_ddim = model.decode_first_stage(samples_ddim[:,:,:,:])
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                                 min=0.0, max=1.0)
                    all_samples_class.append(x_samples_ddim)
                    
                    grid = torch.stack(all_samples_class, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=num_imgs_row_branch)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    grid_class_img = Image.fromarray(grid.astype(np.uint8))
                    if bins:
                        file_name = os.path.join(path, (f'branch_bin{bins[bin_idx]}_{synsets[bin_idx]}_comp{i}.png'))
                    else:
                        if uncertain: 
                            file_name = os.path.join(path, (f'branch_uncertain_{synsets[bin_idx]}_comp{i}.png'))
                        else:
                            file_name = os.path.join(path, (f'branch_certain_{synsets[bin_idx]}_comp{i}.png'))
                    grid_class_img.save(file_name)


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
        default=199)
    parser.add_argument('--dataset', type=str, help='which dataset was used to train the ensemble', 
        default='binned_classes')
    parser.add_argument('--uncertain_comp', action='store_true', 
        help='before branch rollout from uncertain comp')
    args = parser.parse_args()
    seed_everything(42)

    numb_comps = 5 
    model = load_ensemble(args.path, numb_comps)
    ensemble_size = model.model.diffusion_model.ensemble_size
    comp_idx = random.randint(0, (ensemble_size-1))
    subsets, human_synset, idx2synset, synset2idx = create_synset_dicts(args.dataset)
    bins = []
    if args.dataset == 'binned_classes':
        syn1_human, syn10_human, syn100_human, syn1300_human = create_bin_dicts(human_synset, 
            subsets)
        df = pd.DataFrame(human_synset.items(), columns=['synset', 'text'])
        # missing synset 'n02012849'
        # double cranes n02012849, n03126707
        classes_1 = [synset2idx[k] for k in subsets[1]]
        classes_10 = [synset2idx[k] for k in subsets[10]]
        classes_100 = [synset2idx[k] for k in subsets[100]]
        classes_1300 = [synset2idx[k] for k in subsets[1300]]
        #import pdb; pdb.set_trace()
        #classes_1 = [663] 
        #classes_10 = [821]
        #classes_100 = [377]
        #classes_1300 = [991, 76]
        classes_1 = classes_1[10:20] 
        classes_10 = []
        classes_100 = []
        classes_1300 = classes_1300[10:50]
        classes2sample = classes_1300+classes_100+classes_10+classes_1
        bins = [1300]*len(classes_1300)+[100]*len(classes_100)
        bins += [10]*len(classes_10)+[1]*len(classes_1)
        comp_idx = [random.randint(0, (ensemble_size-1)) for i in range(len(bins))]
    elif args.dataset == 'masked_classes':
        synsets = [item for ls in subsets.values() for item in ls]
        synsets = list(set(synsets))
        intersection_dict = {s:check_synset(s,subsets) for s in synsets}
        three_comps = [k for k,v in intersection_dict.items() if len(v) == 3]
        three_comps.sort()
        classes2sample = random.sample(three_comps, 5)
        certain_comps = [[k for k,v in subsets.items() if c in v] for c in classes2sample]
        uncertain_comps = [[k for k,v in subsets.items() if c not in v] for c in classes2sample]
        for c in classes2sample:
            print(f'{human_synset[c]}')
        classes2sample = [synset2idx[c] for c in classes2sample]
        if args.uncertain_comp:
            comp_idx = [random.choice(uc) for uc in uncertain_comps]
        else:
            comp_idx = [random.choice(cc) for cc in certain_comps]
        print(certain_comps)
    synsets = [idx2synset[c] for c in classes2sample]
    synsets = [human_synset[syn] for syn in synsets]
    sampler = DDIMSampler(model)
    ddim_steps = args.ddim_steps
    ddim_eta = args.ddim_eta
    scale = args.scale
    n_samples_per_class = 1 
    
    pic_path = args.path
    pic_path = os.path.join(pic_path, f'samps_uncs_branch{args.unc_branch}')
    os.makedirs(pic_path, exist_ok=True)
    ## sample examples
    sample_model(model, n_samples_per_class, classes2sample, ddim_steps, ddim_eta, scale,
        ensemble_size, pic_path, synsets, args.unc_branch, sampler, 
        bins, comp_idx, args.uncertain_comp)
