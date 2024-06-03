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
from create_helper_variables import create_synset_dicts, create_bin_dicts
from create_subset_masked import check_synset

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
    parser.add_argument('--paper', action='store_true', help='images for the paper')
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
    else:
        synsets = [item for ls in subsets.values() for item in ls]
        synsets = list(set(synsets))
        intersection_dict = {s:check_synset(s,subsets) for s in synsets}
        three_comps = [k for k,v in intersection_dict.items() if len(v) == 3]
        three_comps.sort()
        two_comps = [k for k,v in intersection_dict.items() if len(v) == 2]
        two_comps.sort()
        num_certain_comps = 3
        if num_certain_comps == 3:
            classes2sample = random.sample(three_comps, 8)
        elif num_certain_comps == 2:
            classes2sample = random.sample(two_comps, 8)
        certain_comps = [[k for k,v in subsets.items() if c in v] for c in classes2sample]
        uncertain_comps = [[k for k,v in subsets.items() if c not in v] for c in classes2sample]
        classes2sample = [synset2idx[c] for c in classes2sample]
        classes2sample = [(classes2sample[i], classes2sample[i+1]) for i in range(0, len(classes2sample), 2)]
        certain_comps = [(certain_comps[i], certain_comps[i+1]) for i in range(0, len(certain_comps), 2)]
        uncertain_comps = [(uncertain_comps[i], uncertain_comps[i+1]) for i in range(0, len(uncertain_comps), 2)]
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
                if args.dataset == 'masked_classes':
                    certain_comp = certain_comps[idx]
                    uncertain_comp = uncertain_comps[idx]
                    batch_certain_latent = []
                    batch_certain_unc_latent = []
                    batch_certain_pix = []
                    batch_certain_unc_pix = []
                    batch_uncertain_latent = []
                    batch_uncertain_unc_latent = []
                    batch_uncertain_pix = []
                    batch_uncertain_unc_pix = []
                    batch_numb_samples = []

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
                    if args.dataset == 'masked_classes':
                        comp_idx = random.choice(certain_comp[j]) 
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
                    if args.dataset == 'masked_classes':
                        samples_certain_ddim =  samples_ddim[certain_comp[j]]
                        samples_uncertain_ddim =  samples_ddim[uncertain_comp[j]]
                        if len(samples_certain_ddim.shape)== 5:
                            samples_certain_ddim = samples_certain_ddim.reshape(-1, samples_certain_ddim.shape[2],
                                samples_certain_ddim.shape[3], samples_certain_ddim.shape[4])
                            samples_uncertain_ddim = samples_uncertain_ddim.reshape(-1, samples_uncertain_ddim.shape[2],
                                samples_uncertain_ddim.shape[3], samples_uncertain_ddim.shape[4])

                        x_certain_samples_ddim = model.decode_first_stage(samples_certain_ddim)
                        x_certain_samples_ddim = torch.clamp((x_certain_samples_ddim+1.0)/2.0,
                                                     min=0.0, max=1.0)
                        x_uncertain_samples_ddim = model.decode_first_stage(samples_uncertain_ddim)
                        x_uncertain_samples_ddim = torch.clamp((x_uncertain_samples_ddim+1.0)/2.0,
                                                     min=0.0, max=1.0)
                        batch_certain_latent.append(samples_certain_ddim)
                        batch_certain_unc_latent.append(epi_unc['Wass']['certain'])
                        batch_certain_pix.append(x_certain_samples_ddim)
                        unc_certain_pix, _ = pairwise_exp(x_certain_samples_ddim.unsqueeze(1), 0, 'Wass_0', numb_comps, unc_per_pixel=True)
                        batch_certain_unc_pix.append(unc_certain_pix)
                        batch_uncertain_latent.append(samples_uncertain_ddim)
                        batch_uncertain_unc_latent.append(epi_unc['Wass']['uncertain'])
                        batch_uncertain_pix.append(x_uncertain_samples_ddim)
                        unc_uncertain_pix, _ = pairwise_exp(x_uncertain_samples_ddim.unsqueeze(1), 0, 'Wass_0', numb_comps, unc_per_pixel=True)
                        batch_uncertain_unc_pix.append(unc_uncertain_pix)
                        batch_numb_samples.append(num_certain_comps)
                    else:
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
                if args.paper:
                    pixel_dir = os.path.join(args.path,'pixel_unc_paper')
                    if not os.path.exists(pixel_dir):
                        os.mkdir(pixel_dir)
                    if args.dataset=='masked_classes':
                        iter_num = 1
                    else:
                        iter_num = len(classes2sample[0])
                    file_name = os.path.join(pixel_dir, (f'pixel_unc_{idx}.png'))
                    for i in range(iter_num):
                        if args.dataset == 'masked_classes':
                            pics = [p[0,:,:,:] for p in batch_uncertain_pix[i*2:(i+1)*2]]
                            pics += [p[0,:,:,:] for p in batch_certain_pix[i*2:(i+1)*2]]
                            uncs = [p[0,:,:,:] for p in batch_uncertain_unc_pix[i*2:(i+1)*2]]
                            uncs += [p[0,:,:,:] for p in batch_certain_unc_pix[i*2:(i+1)*2]]
                        else: 
                            pics = [p[0,:,:,:] for p in batch_pix[i*4:(i+1)*4]]
                            uncs = [p[0,:,:,:] for p in batch_unc_pix[i*4:(i+1)*4]]
                        import pdb; pdb.set_trace() 
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
                else:
                    pics = [p[0,:,:,:] for p in batch_pix]
                    grid = torch.stack(pics, 0)
                    #grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                    grid = make_grid(grid, nrow=1)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    grid_class_img = Image.fromarray(grid.astype(np.uint8))
                    image_height = grid_class_img.height
                    image_width = 4*grid_class_img.width
                    combined_image = Image.new("RGB", (image_width, image_height),
                        (255, 255, 255))
                    font_color = (0, 0, 0)
                    font_size = 20
                    border_width = 1
                    font = ImageFont.truetype("Roboto-Thin.ttf", font_size)
                    draw = ImageDraw.Draw(combined_image)
                    combined_image.paste(grid_class_img, (grid_class_img.width, 0))
                    # Define the text for each image

                    texts = [idx2synset[l] for l in label_batch]
                    single_image_height = int(image_height/len(texts))+1
                    single_image_width = grid_class_img.width
                    for row in range(len(texts)):
                        # Calculate the position for the current image
                        image_x = 0 
                        image_y = row * single_image_height

                        # Create a new image with a white background for the current cell
                        image = Image.new("RGB", (single_image_width, single_image_height), (255, 255, 255))

                        # Create a draw object for the current image
                        image_draw = ImageDraw.Draw(image)
                        
                        # Calculate the position for the text in the middle of the current image
                        human_txt = human_synset[texts[row]].split(',')[0]
                        text_width, text_height = image_draw.textsize(f'{human_txt}',
                            font=font)
                        text_x = (single_image_width - text_width) // 2
                        text_y = (single_image_height - text_height) // 2

                        # Write the text in the middle of the current image
                        image_draw.text((text_x, text_y), f'{human_txt}',
                            font=font, fill=font_color)

                        border_box = [(0, 0), (single_image_width - 1, single_image_height - 1)]
                        image_draw.rectangle(border_box, outline=(0, 0, 0), width=border_width)
                        # Paste the current image onto the grid image at the calculated position
                        combined_image.paste(image, (image_x, image_y))
                    for row in range(len(label_batch)):
                        # Calculate the position for the current image
                        image_x = 3*single_image_width
                        image_y = row * single_image_height

                        # Create a new image with a white background for the current cell
                        image = Image.new("RGB", (single_image_width, single_image_height), (255, 255, 255))

                        # Create a draw object for the current image
                        image_draw = ImageDraw.Draw(image)

                        # Calculate the position for the text in the middle of the current image
                        text_width, text_height = image_draw.textsize(f'{batch_numb_samples[row]} images',
                            font=font)
                        text_x = (single_image_width - text_width) // 2
                        text_y = (single_image_height - text_height) // 2

                        # Write the text in the middle of the current image
                        image_draw.text((text_x, text_y), f'{batch_numb_samples[row]} images',
                            font=font, fill=font_color)

                        border_box = [(0, 0), (single_image_width - 1, single_image_height - 1)]
                        image_draw.rectangle(border_box, outline=(0, 0, 0), width=border_width)
                        # Paste the current image onto the grid image at the calculated position
                        combined_image.paste(image, (image_x, image_y))
                    max_unc = torch.stack(batch_unc_pix).max()
                    min_unc = torch.stack(batch_unc_pix).min()
                    for row in range(len(label_batch)):
                        uncs = batch_unc_pix[row][0,:,:,:]
                        uncs = uncs.mean(0)
                        #max_unc = uncs.max()
                        #min_unc = uncs.min()
                        uncs = (uncs-min_unc)/(max_unc-min_unc)
                        px = 1/plt.rcParams['figure.dpi']
                        plt.subplots(figsize=(single_image_width*px, single_image_height*px))
                        plt.imshow(uncs.cpu().numpy(), cmap='cividis', interpolation='nearest', vmin=min_unc, vmax=max_unc)
                        plt.tick_params(left = False, right = False, labelleft = False, labelbottom = False, bottom = False)
                        plt.tight_layout(pad=0.0)
                        plt.savefig(os.path.join(args.path,'heat_map.png'))
                        im = Image.open(os.path.join(args.path,'heat_map.png'))
                        combined_image.paste(im, (single_image_width*2, single_image_height*row))
                    file_name = os.path.join(os.path.join(args.path,'pixel_unc'), 
                        (f'pixel_uncertainty_{j}.png'))
                    combined_image.save(file_name)
                    pic_num += 1
