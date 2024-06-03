import gc
import os
import pickle
import argparse
import psutil

import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
from ignite.engine import Engine
from ignite.metrics import PSNR, SSIM, FID, InceptionScore

from save_ensemble import load_ensemble
from ldm.util import instantiate_from_config
mpl.rc('font',family='Times New Roman')

def divide_chunks(l, n):
      
    # looping till length l
    for i in range(0, len(l), n): 
        yield l[i:i + n]

def process_function(engine, batch):
    # ...
    return y_pred, y

def eval_step(engine, batch):
    return batch


class WrapperInceptionV3(nn.Module):
    
    def __init__(self, fid_incv3):
        super().__init__()
        self.fid_incv3 = fid_incv3
    
    @torch.no_grad()
    def forward(self, x):
        y = self.fid_incv3(x)
        y = y[0]
        y = y[:, :, 0, 0]
        return y



if __name__ == '__main__':
    print('plot img diff graph and latex for table')
    meta_path = '/home/nwaftp23/scratch/uncertainty_estimation/imagenet/ILSVRC2012_train' 
    with open(os.path.join(meta_path, 'filelist.txt')) as f:
        filelist = f.readlines()
    #count_per_synset = [f.split('/')[0] for f in filelist]
    #count_per_synset = dict(Counter(count_per_synset))
    with open(os.path.join(meta_path, 'subsets.pkl'), 'rb') as fp:
        subsets = pickle.load(fp)
    # TODO:
    #could figure out which index is which by conditioning and then checking
    human_synset = {}
    with open(os.path.join(meta_path, 'synset_human.txt')) as f:
        for line in f:
            items = line.split()
            key, values = items[0], ' '.join(items[1:])
            human_synset[key] = values
    syn1_human = {k: human_synset[k] for k in subsets[1]}
    syn1300_human = {k: human_synset[k] for k in subsets[1300]}
    syn100_human = {k: human_synset[k] for k in subsets[100]}
    syn10_human = {k: human_synset[k] for k in subsets[10]}
    idx2synset = OmegaConf.load(os.path.join(meta_path, 'index_synset.yaml'))
    synset2idx = {y: x for x, y in idx2synset.items()}
    df = pd.DataFrame(human_synset.items(), columns=['synset', 'text'])
    # missing synset 'n02012849'
    # double cranes n02012849, n03126707
    classes_1 = [synset2idx[k] for k in subsets[1]]
    classes_10 = [synset2idx[k] for k in subsets[10]]
    classes_100 = [synset2idx[k] for k in subsets[100]]
    classes_1300 = [synset2idx[k] for k in subsets[1300]]
    model_path = '/home/nwaftp23/scratch/logs/'
    model_path = os.path.join(model_path, 'bootstrapped_imagenet_10')
    unc_paths = os.listdir(model_path)
    unc_paths = [p for p in unc_paths if 'samps_uncs_branch' in p]
    ssim_mean = {}
    ssim_std = {}
    metric = SSIM(data_range=1.0)
    engine = Engine(process_function)
    default_evaluator = Engine(eval_step)
    metric.attach(default_evaluator, 'ssim')
    #metric.attach(engine, 'ssim')
    for up in tqdm(unc_paths):
        bp_dir = os.path.join(model_path, up)
        img_dir = os.path.join(bp_dir, 'images_img_space')
        img_files = os.listdir(img_dir)
        ssim_1 = []
        ssim_10 = []
        ssim_100 = []
        ssim_1300 = []
        img_file_chunks = divide_chunks(img_files, 200)
        memory_usage = psutil.virtual_memory().used / (1024 ** 2)
        tqdm.write(f'memory_usage in MB:{memory_usage}')
        for chunk in img_file_chunks:
            imgs = []
            idx_2_cl = {i:int(f.split('class')[1].split('.')[0]) for i, f in enumerate(chunk)} 
            for f in tqdm(chunk):
                img = torch.load(os.path.join(img_dir, f))
                imgs.append(img)#.cpu().detach())
            imgs = torch.stack(imgs)
            imgs = imgs[:,:5,:,:,:,:]
            #mu_ensemble = imgs.mean(1)
            ssim_dict = {}
            for c_idx in range(imgs.shape[0]):
                class_ssims = []
                for rand_noise in range(imgs.shape[2]):
                    pred = torch.zeros((10,3,256,256))
                    targ = torch.zeros((10,3,256,256))
                    count = 0
                    for ec in range(imgs.shape[1]):
                        for j in range(ec+1, imgs.shape[1]):
                            pred[count] = imgs[c_idx,ec,rand_noise,:,:,:]
                            targ[count] = imgs[c_idx,j,rand_noise,:,:,:]
                            count +=1
                    state = default_evaluator.run([[pred, targ]])
                    class_ssims.append(state.metrics['ssim'])
                ssim_dict[idx_2_cl[c_idx]] = class_ssims
            ssim_1 += [ssim_dict[c] for c in classes_1 if c in ssim_dict.keys()]
            ssim_10 += [ssim_dict[c] for c in classes_10 if c in ssim_dict.keys()]
            ssim_100 += [ssim_dict[c] for c in classes_100 if c in ssim_dict.keys()]
            ssim_1300 += [ssim_dict[c] for c in classes_1300 if c in ssim_dict.keys()]
            del imgs
            del pred 
            del targ
            del img
            torch.cuda.empty_cache()
            gc.collect()
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            memory_usage = psutil.virtual_memory().used / (1024 ** 2)
            tqdm.write(f'memory_usage in MB:{memory_usage}')
            tqdm.write(f'GPU_memory_usage in MB:{a}')
        print(f'{up}')
        bp = int(up.split('branch')[1])
        ssim_mean[bp] = {1:np.mean(ssim_1), 10:np.mean(ssim_10), 100:np.mean(ssim_100), 
                1300:np.mean(ssim_1300)}
        ssim_std[bp] = {1:np.std(ssim_1), 10:np.std(ssim_10), 100:np.std(ssim_100), 
                1300:np.std(ssim_1300)}

    with open(os.path.join(model_path, 'img_diversity_data_mean.pkl'), 'wb') as fp:
        pickle.dump(ssim_mean, fp)
    with open(os.path.join(model_path, 'img_diversity_data_std.pkl'), 'wb') as fp:
        pickle.dump(ssim_std, fp)
