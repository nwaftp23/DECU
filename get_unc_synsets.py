import pickle
import random
import argparse
import torch
import os
from collections import Counter
from omegaconf import OmegaConf

import numpy as np 
import pandas as pd
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm_test import DDPMSampler
from save_ensemble import load_ensemble
from sample_model import sample_model, get_model, seed_everything
from make_histogram import get_min_max, norm_uncs, split_uncs
from create_helper_variables import create_synset_dicts, create_bin_dicts, check_synset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Uncertainty Per Synset')
    parser.add_argument('--path', type=str, help='path to model', required=True)
    parser.add_argument('--sampler', type=str, help='which smapler to use', default='DDIM')
    parser.add_argument('--unc_branch', type=int, help='where to branch for uncertainty', 
        default=199)
    parser.add_argument('--ddim_eta', type=float, help='controls stdev for generative process', 
        default=0.00)
    # ddim_eta 0-1, 1=DDPM 
    parser.add_argument('--ddim_steps', type=int, help='number of steps to take in ddim', 
        default=200)
    parser.add_argument('--base_comp', type=int, help='comp to start from before branching', 
        default=-1)
    parser.add_argument('--quickrun', action='store_true', help='run quick test')
    parser.add_argument('--scale', type=float, help='controls the amount of unconditional guidance',
        default=5.0)
    parser.add_argument('--dataset', type=str, help='which dataset was used to train the ensemble', default='binned_classes')
    args = parser.parse_args()
    seed_everything(42)
    dir_path = os.path.join(args.path, f'samps_uncs_branch{args.unc_branch}')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    numb_comps = 5
    #numb_comps = 10
    model = load_ensemble(args.path, numb_comps)
    ensemble_size = model.model.diffusion_model.ensemble_size
    comp_idx = random.randint(0, (ensemble_size-1))
    subsets, human_synset, idx2synset, synset2idx = create_synset_dicts(args.dataset)
    df = pd.DataFrame(human_synset.items(), columns=['synset', 'text'])
    all_classes = [i for i in range(1000)]
    if args.dataset == 'binned_classes':
        syn1_human, syn10_human, syn100_human, syn1300_human = create_bin_dicts(human_synset, subsets)
        if args.quickrun:
            classes_10 = []
            classes_100 = []
            classes_1 = [392, 708, 729, 854]
            classes_1300 = [25, 187, 448, 992]
            
            classes2sample = classes_1+classes_10+classes_100+classes_1300
            all_classes = classes2sample
    sampler = DDIMSampler(model)
    ddim_steps = args.ddim_steps
    ddim_eta = args.ddim_eta
    scale = args.scale
    #n_samples_per_class = 7
    n_samples_per_class = 10 
        
    all_samples = list()
    epi_uncs = {}
    imgs_path = os.path.join(dir_path, 'images')
    img_space_path = os.path.join(dir_path, 'images_img_space')
    pair_dist_path = os.path.join(dir_path, 'pair_dists')
    if not os.path.exists(imgs_path):
        os.mkdir(imgs_path)
    if not os.path.exists(img_space_path):
        os.mkdir(img_space_path)
    if not os.path.exists(pair_dist_path):
        os.mkdir(pair_dist_path)
    if args.dataset == 'masked_classes':
        filename_certain_uncs = os.path.join(dir_path, f'certain_uncs_dict_eta{args.ddim_eta}'\
            f'_branch{args.unc_branch}_sampler{args.sampler}_ddimsteps{args.ddim_steps}_scale{scale}.pkl')
        filename_uncertain_uncs = os.path.join(dir_path, f'uncertain_uncs_dict_eta{args.ddim_eta}'\
            f'_branch{args.unc_branch}_sampler{args.sampler}_ddimsteps{args.ddim_steps}_scale{scale}.pkl')
        epi_certain_uncs = {}
        epi_uncertain_uncs = {}
    filename_uncs = os.path.join(dir_path, f'uncs_dict_eta{args.ddim_eta}'\
        f'_branch{args.unc_branch}_sampler{args.sampler}_ddimsteps{args.ddim_steps}_scale{scale}.pkl')
    with torch.no_grad():
        with model.ema_scope():
            all_ucs = []
            for _ in range(ensemble_size):
                uc = model.get_learned_conditioning(
                    {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)},
                    comp_idx = _)
                all_ucs.append(uc)

            for class_label in tqdm(all_classes):
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in"\
                    f" {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class*[class_label])
                x_T = torch.randn((n_samples_per_class,3,64,64), device=model.device) 
                all_cs = []
                for _ in range(ensemble_size):
                    c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)},
                        comp_idx = _)
                    all_cs.append(c)

                all_samples_class =[]
                certain_comps = []
                if args.dataset == 'masked_classes':
                    certain_comps = idx2certaincomps[class_label] 
                samples_ddim, epi_unc, dist, dist_mat = sampler.sample(S=ddim_steps,
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
                                                         x_T = x_T,
                                                         certain_comps = certain_comps)
                torch.save(samples_ddim.cpu(), 
                    os.path.join(imgs_path, f'imgs_class{class_label}.pt'))
                og_shape = samples_ddim.shape
                if len(samples_ddim.shape)== 5:
                    samples_ddim = samples_ddim.reshape(-1, samples_ddim.shape[2],
                        samples_ddim.shape[3], samples_ddim.shape[4])

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = 255*torch.clamp((x_samples_ddim+1.0)/2.0,
                                             min=0.0, max=1.0)
                x_samples_ddim = x_samples_ddim.reshape(og_shape[0], og_shape[1], x_samples_ddim.shape[1], 
                    x_samples_ddim.shape[2], x_samples_ddim.shape[3])
                torch.save(x_samples_ddim.cpu(), 
                    os.path.join(img_space_path, f'imgs_class{class_label}.pt'))
                np.save(os.path.join(pair_dist_path, f'pair_dist_class{class_label}.npy'), 
                    dist_mat.cpu().numpy())
                ### Uncertainty for graph
                #yo = [eu.cpu().numpy() for eu in epi_unc['Wass']]
                #epi_unc['Wass']=yo
                #with open('uncertainty_data_branch_point0.pkl', 'wb') as fp:
                #    pickle.dump(epi_unc, fp)
                ###
                if certain_comps:
                    uncs_2_write = {}
                    if epi_unc['Wass']['certain'][0] != None:
                        uncs_2_write['Wass'] = [epi_unc['Wass']['certain'][0].cpu()]
                        epi_certain_uncs[class_label] = uncs_2_write
                        if not args.quickrun:
                            with open(filename_certain_uncs, 'wb') as fp:
                                pickle.dump(epi_certain_uncs, fp)
                    uncs_2_write = {}
                    if epi_unc['Wass']['uncertain'][0] != None:
                        uncs_2_write['Wass'] = [epi_unc['Wass']['uncertain'][0].cpu()]
                        epi_uncertain_uncs[class_label] = uncs_2_write
                        if not args.quickrun:
                            with open(filename_uncertain_uncs, 'wb') as fp:
                                pickle.dump(epi_uncertain_uncs, fp)
                else:
                    epi_unc['Wass'] = [epi_unc['Wass'][0].cpu()]
                    epi_uncs[class_label] = epi_unc
                    if not args.quickrun:
                        with open(filename_uncs, 'wb') as fp:
                            pickle.dump(epi_uncs, fp)
    ## sample examples
    '''mean_uncs = {}
    std_uncs = {}
    kl_exist = 'KL' in epi_uncs[187].keys()
    uncs_bin_wass, uncs_bin_kl, uncs_bin_bhatt = split_uncs(epi_uncs, subsets, synset2idx, kl_exist)
    min_wass, max_wass = get_min_max(uncs_bin_wass)
    mean_std_wass = norm_uncs(uncs_bin_wass, min_wass, max_wass)
    print(f'epi unc Wass 1 \nmean:{mean_std_wass[1]["mean"]:.3f}       std:{mean_std_wass[1]["std"]:.3f}')
    print(f'epi unc Wass 1300 \nmean:{mean_std_wass[1300]["mean"]:.3f}       std:{mean_std_wass[1300]["std"]:.3f}')
    if kl_exist:
        min_kl, max_kl = get_min_max(uncs_bin_kl)
        min_bhatt, max_bhatt = get_min_max(uncs_bin_bhatt)
        mean_std_kl = norm_uncs(uncs_bin_kl, min_kl, max_kl)
        mean_std_bhatt = norm_uncs(uncs_bin_bhatt, min_bhatt, max_bhatt)
        print(f'epi unc KL 1 \nmean: {mean_std_kl[1]["mean"]:.3f}       std:{mean_std_kl[1]["std"]:.3f}')
        print(f'epi unc KL 1300 \nmean: {mean_std_kl[1300]["mean"]:.3f}       std:{mean_std_kl[1300]["std"]:.3f}')
        print(f'epi unc Bhatt 1 \nmean: {mean_std_bhatt[1]["mean"]:.3f}       std:{mean_std_bhatt[1]["std"]:.3f}')
        print(f'epi unc Bhatt 1300 \nmean: {mean_std_bhatt[1300]["mean"]:.3f}       std:{mean_std_bhatt[1300]["std"]:.3f}')'''
