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

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    #config_path = f'/home/nwaftp23/scratch/ldm/models/ldm/cin256/'\
    #    'config.yaml'
    config_path = f'/home/nwaftp23/latent-diffusion/configs/latent-diffusion/'\
        'cin256-v2.yaml'
    config = OmegaConf.load(config_path)
    #model_path = '/home/nwaftp23/scratch/ldm/models/ldm/cin256/model.ckpt'
    model_path = '/home/nwaftp23/scratch/ldm/models/ldm/cin256-v2/model.ckpt'
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

if __name__ == '__main__':
    seed_everything(42)

    train_path = '/home/nwaftp23/scratch/uncertainty_estimation/imagenet/ILSVRC2012_train'
    model = get_model()
    with open(os.path.join(train_path, 'filelist.txt')) as f:
        filelist = f.readlines()
    
    with open(os.path.join(train_path, 'subsets.pkl'), 'rb') as fp:
        subsets = pickle.load(fp)
    human_synset = {} 
    with open(os.path.join(train_path, 'synset_human.txt')) as f:
        for line in f:
            items = line.split()
            key, values = items[0], ' '.join(items[1:])
            human_synset[key] = values
    syn100_human = {k: human_synset[k] for k in subsets[100]}
    syn1300_human = {k: human_synset[k] for k in subsets[1300]}
    idx2synset = OmegaConf.load(os.path.join(train_path, 'index_synset.yaml'))
    synset2idx = {y: x for x, y in idx2synset.items()}
    df = pd.DataFrame(human_synset.items(), columns=['synset', 'text'])
    # missing synset 'n02012849'
    # double cranes n02012849, n03126707
    classes_100 = [synset2idx[k] for k in subsets[100]]
    classes_1300 = [synset2idx[k] for k in subsets[1300]]
    classes_100 = random.sample(classes_100, 3)
    classes_1300 = [25, 187, 448, 992]

    classes = classes_100+classes_1300
    sampler = DDIMSampler(model)
    n_samples_per_class = 6
    #ensemble_size = model.model.diffusion_model.ensemble_size
    #ensemble_size = model.model.diffusion_model.ensemble_size

    ddim_steps = 200
    ddim_eta = 0.00
    scale = 5.0
    all_samples = list()

    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)})
            
            uncs = {}
            for class_label in classes:
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in"\
                    f" {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class*[class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})
                
                ensemble_samples = {}
                ensemble_means = {}
                ensemble_stds = {}
                all_samples_class =[]
                samples_ddim, dist = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=n_samples_per_class,
                                                 shape=[3, 64, 64],
                                                 verbose=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc, 
                                                 eta=ddim_eta,
                                                 ensemble_comp=-3,
                                                 return_distribution=True)
                #ensemble_samples[i] = samples_ddim
                #ensemble_means[i] = dist['mean'] 
                #ensemble_stds[i] = dist['std'] 
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                             min=0.0, max=1.0)
                all_samples_class.append(x_samples_ddim)
                
                #unc here!!!!!!
                grid = torch.stack(all_samples_class, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_samples_per_class)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                grid_class_img = Image.fromarray(grid.astype(np.uint8))
                if idx2synset[class_label] in subsets[100]:
                    file_name = os.path.join('/home/nwaftp23/scratch/ldm/models/ldm/cin256-v2', 
                        (f'{syn100_human[idx2synset[class_label]]}_images_'\
                        f'{idx2synset[class_label]}.png'))
                elif idx2synset[class_label] in subsets[1300]:
                    file_name = os.path.join('/home/nwaftp23/scratch/ldm/models/ldm/cin256-v2', 
                        (f'{syn1300_human[idx2synset[class_label]]}_images_'\
                        f'{idx2synset[class_label]}.png'))
                grid_class_img.save(file_name)
                print(file_name)
                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                             min=0.0, max=1.0)
                all_samples.append(x_samples_ddim)


    import pdb; pdb.set_trace()
    # display as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=n_samples_per_class)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    Image.fromarray(grid.astype(np.uint8))
    import pdb; pdb.set_trace()
    sns.displot(count_per_synset, color = 'skyblue', edgecolor='black', label='train_data')
    file_name = os.path.join(args.path, ('unc_per_synset.png'))
    plt.xticks(rotation=90)
    plt.savefig(file_name)
    plt.close()
