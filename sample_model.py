import pickle
import random
import argparse
import torch
import os
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
from create_helper_variables import create_synset_dicts
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
        ensemble_size, path, idx2synset, subsets, unc_branch, sampler, comp_idx, 
        ensemble_comps):
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
                 
            all_samples_row1 =[]
            all_samples_row2 =[]
            all_samples_row3 =[]
            all_samples_row4 =[]
            all_samples_row5 =[]
            for class_label in classes2sample:
                seed_everything(42)
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
                                                 all_cs=all_cs, 
                                                 ensemble_comps = ensemble_comps)
                if len(samples_ddim.shape)== 5:
                    samples_ddim = samples_ddim.reshape(-1, samples_ddim.shape[2], 
                        samples_ddim.shape[3], samples_ddim.shape[4])

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                #x_samples_ddim = model.decode_first_stage(samples_ddim[:,:,:,:])
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                             min=0.0, max=1.0)
                all_samples_row1.append(x_samples_ddim[0,:,:,:])
                all_samples_row2.append(x_samples_ddim[1,:,:,:])
                all_samples_row3.append(x_samples_ddim[2,:,:,:])
                all_samples_row4.append(x_samples_ddim[3,:,:,:])
                all_samples_row5.append(x_samples_ddim[4,:,:,:])
                
            #import pdb; pdb.set_trace()
            grid_row1 = torch.stack(all_samples_row1)
            grid_row2 = torch.stack(all_samples_row2)
            grid_row3 = torch.stack(all_samples_row3)
            grid_row4 = torch.stack(all_samples_row4)
            grid_row5 = torch.stack(all_samples_row5)
            grid = torch.stack([grid_row1, grid_row2, grid_row3, grid_row4, grid_row5], 0)
            #original_shape = grid.shape
            #tensor_2d = tensor.view(-1, original_shape[1], original_shape[3], original_shape[4])
            #all_samples = 
            #grid = rearrange(all_samples, 'n b c h w -> (n b) c h w').reshape(-1,5,3,256,256)
            #grid = itorch.stack((a,b), dim=2).view(2,4)
            #grid0 = torch.stack(all_samples, 0)
            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
            grid = make_grid(grid, nrow=len(classes2sample))

            # to image
            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
            grid_class_img = Image.fromarray(grid.astype(np.uint8))
            file_name = os.path.join(path,'images5_p2.png')
            #combined_image.paste(grid_class_img, (0, 0))
            #combined_image.save(file_name)
            grid_class_img.save(file_name)
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
        default=199)
    parser.add_argument('--dataset', default = 'binned_classes', type=str, help='binned or masked classes')
    args = parser.parse_args()
    seed_everything(54)

    numb_comps = int(args.path.split('_')[-1]) 
    #model = get_model(args.path)
    model = load_ensemble(args.path, numb_comps)
    ensemble_comps = []
    ensemble_size = model.model.diffusion_model.ensemble_size
    comp_idx = random.randint(0, (ensemble_size-1))
    
    subsets, human_synset, idx2synset, synset2idx = create_synset_dicts(args.dataset)
    # missing synset 'n02012849'
    # double cranes n02012849, n03126707
    if args.dataset == 'binned_classes':
        syn1_human = {k: human_synset[k] for k in subsets[1]}
        syn1300_human = {k: human_synset[k] for k in subsets[1300]}
        syn100_human = {k: human_synset[k] for k in subsets[100]}
        syn10_human = {k: human_synset[k] for k in subsets[10]}
        # NOTE all classes
        #classes_1 = [synset2idx[k] for k in subsets[1]]
        #classes_10 = [synset2idx[k] for k in subsets[10]]
        #classes_100 = [synset2idx[k] for k in subsets[100]]
        #classes_1300 = [synset2idx[k] for k in subsets[1300]]
        # Note subset
        # Uncertain Classes
        classes_1 = [392, 708, 729, 854]
        classes_10 = []
        classes_100 = []
        classes_1300 = []
        # Certain Classes
        #classes_1 = []
        #classes_1300 = [25, 447, 991, 992]
        classes2sample = classes_1+classes_10+classes_100+classes_1300
    elif args.dataset=='masked_classes':
        synsets = [item for ls in subsets.values() for item in ls]
        synsets = list(set(synsets))
        intersection_dict = {s:check_synset(s,subsets) for s in synsets}
        three_comps = [k for k,v in intersection_dict.items() if len(v) == 3]
        three_comps.sort()
        classes2sample = [three_comps[-6], three_comps[-7], three_comps[-8], three_comps[-21], three_comps[-23]]
        import pdb;
        certain_comps = []
        uncertain_comps = []
        #classes2sample = random.sample(three_comps, 20)
        for c in classes2sample:
            print(f'{human_synset[c]}')
        classes2sample = [synset2idx[c] for c in classes2sample]
    df = pd.DataFrame(human_synset.items(), columns=['synset', 'text'])

    synsets2sample = {i:idx2synset[i] for i in classes2sample}
    if args.sampler == 'DDIM':
        sampler = DDIMSampler(model)
    else: 
        print('Not setup for DDPM')
    ddim_steps = args.ddim_steps
    ddim_eta = args.ddim_eta
    scale = args.scale
    #n_samples_per_class = 5 
    n_samples_per_class = 1 
    
    pic_path = args.path
    pic_path = os.path.join(pic_path, f'certain_vs_uncertain')
    os.makedirs(pic_path, exist_ok=True)
    ## sample examples
    sample_model(model, n_samples_per_class, classes2sample, ddim_steps, ddim_eta, scale,
        ensemble_size, pic_path, idx2synset, subsets, args.unc_branch, sampler, comp_idx, 
        ensemble_comps)
