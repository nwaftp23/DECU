import os
from collections import Counter
from omegaconf import OmegaConf
import pickle


def check_synset(s,comps):
    seen_by = []
    for k,v in comps.items():
        if s in v:
            seen_by.append(k)
    return seen_by

def create_synset_dicts(dataset):
    train_path = './imagenet_data/data_train'
    #train_path = '/home/nwaftp23/scratch/uncertainty_estimation/imagenet/ILSVRC2012_train'
    #train_path = '/home/nwaftp23/projects/def-dpmeger/nwaftp23/epi-diffusion/setup_files'
    with open(os.path.join(train_path, 'filelist.txt')) as f:
        filelist = f.readlines()
    count_per_synset = [f.split('/')[0] for f in filelist]
    count_per_synset = dict(Counter(count_per_synset))

    with open(os.path.join(train_path, f'subsets_{dataset}.pkl'), 'rb') as fp:
        subsets = pickle.load(fp)
    # TODO:
    #could figure out which index is which by conditioning and then checking
    human_synset = {}
    with open(os.path.join(train_path, 'synset_human.txt')) as f:
        for line in f:
            items = line.split()
            key, values = items[0], ' '.join(items[1:])
            human_synset[key] = values
    idx2synset = OmegaConf.load(os.path.join(train_path, 'index_synset.yaml'))
    synset2idx = {y: x for x, y in idx2synset.items()}
    return subsets, human_synset, idx2synset, synset2idx

def create_bin_dicts(human_synset, subsets):
    syn1_human = {k: human_synset[k] for k in subsets[1]}
    syn10_human = {k: human_synset[k] for k in subsets[10]}
    syn100_human = {k: human_synset[k] for k in subsets[100]}
    syn1300_human = {k: human_synset[k] for k in subsets[1300]}
    return syn1_human, syn10_human, syn100_human, syn1300_human

