import os
import pickle
import argparse

import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from save_ensemble import load_ensemble
from ldm.util import instantiate_from_config
mpl.rc('font',family='Times New Roman')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Uncertainty Per Synset')
    parser.add_argument('--path', type=str, help='path to model', required=True)
    parser.add_argument('--unc_branch', type=int, help='when to split for generative proccess',
        default=200)
    args = parser.parse_args()
    imagenet = True
    numb_comps = 10
    model = load_ensemble(args.path, imagenet, numb_comps)
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
    #unc_paths = [p for p in unc_paths if 'samps_uncs_branch' in p]
    unc_paths = ['samps_uncs_branch0']
    for up in tqdm(unc_paths):
        bp_dir = os.path.join(model_path, up)
        #img_dir = os.path.join(bp_dir, 'images')
        #img_space_dir = os.path.join(bp_dir, 'images_img_space')
        img_dir = os.path.join(bp_dir, 'larger_samp')
        img_space_dir = os.path.join(bp_dir, 'images_larger_samp')
        if not os.path.exists(img_space_dir):
            os.mkdir(img_space_dir)
        img_files = os.listdir(img_dir)
        imgs = []
        for f in tqdm(img_files):
            img = torch.load(os.path.join(img_dir, f))
            og_shape = img.shape
            if len(img.shape)== 5:
                img = img.reshape(-1, img.shape[2],
                    img.shape[3], img.shape[4])
            img = img.to('cuda')
            if img.shape[0] > 70:
                imgp1 = model.decode_first_stage(img[:70])
                imgp2 = model.decode_first_stage(img[70:])
                img = torch.vstack([imgp1, imgp2])
            else:
                img = model.decode_first_stage(img)
            #print('scale images between -1 and 1 for FID shit')
            #img = torch.clamp((img+1.0)/2.0, min=0.0, max=1.0)
            img = torch.clamp(img, min=-1.0, max=1.0)
            new_shape = list(og_shape)
            new_shape[4] = 256
            new_shape[3] = 256
            tqdm.write(f'{new_shape}')
            img = img.reshape(new_shape)
            class_label = int(f.split('class')[1].split('.')[0])
            torch.save(img, os.path.join(img_space_dir, f'imgs_class{class_label}.pt'))
            #imgs.append(img.cpu())
