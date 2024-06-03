import itertools
import pickle
import os
import argparse
from omegaconf import OmegaConf
import itertools



import seaborn as sns
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
from scipy.stats import gaussian_kde

from create_helper_variables import create_synset_dicts, create_bin_dicts, check_synset

plt.rcParams['text.usetex'] = True
mpl.rc('font',family='Times New Roman')
#color_hexes = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1',
#        '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
#color_hexes = sns.color_palette('colorblind')
color_hexes = ['#377eb8', '#ff7f00', '#4daf4a',
              '#f781bf', '#a65628', '#984ea3',
              '#999999', '#e41a1c', '#dede00'] 

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

#contents = pickle.load(f) becomes...
#contents = CPU_Unpickler(f).load()

def get_min_max(uncs_bin):
    all_dist = [v for k, v in uncs_bin.items()]
    all_dist = torch.stack(list(itertools.chain(*all_dist)))
    max_dist = all_dist.max()
    min_dist = all_dist.min()
    return min_dist, max_dist

def norm_uncs(uncs_bin, min_dist, max_dist):
    mean_std ={}
    for k, v in uncs_bin.items():
        bin_uncs = torch.stack(v)
        #normed_uncs = (bin_uncs-min_dist)/(max_dist-min_dist)
        normed_uncs = bin_uncs
        mean_std[k]={'mean':normed_uncs.mean().item(), 'std':normed_uncs.std().item()}
    return mean_std

def split_uncs(uncs, subsets, synset2idx, kl_exist):
    uncs_bin_wass = {}
    uncs_bin_kl = {}
    uncs_bin_bhatt = {}
    for k, v in subsets.items():
        unc_per_bin_wass = [uncs[synset2idx[synset]]['Wass'][-1] for synset in v if synset2idx[synset] in uncs]
        if unc_per_bin_wass:
            uncs_bin_wass[k] = unc_per_bin_wass
        if kl_exist:
            unc_per_bin_kl = [uncs[synset2idx[synset]]['KL'][-1] for synset in v if synset2idx[synset] in uncs]
            if unc_per_bin_kl:
                uncs_bin_kl[k] = unc_per_bin_kl
            unc_per_bin_bhatt = [uncs[synset2idx[synset]]['Bhatt'][-1] for synset in v if synset2idx[synset] in uncs]
            if unc_per_bin_bhatt:
                uncs_bin_bhatt[k] = unc_per_bin_bhatt
    return uncs_bin_wass, uncs_bin_kl, uncs_bin_bhatt

def split_uncs_eu(uncs, subsets, synset2idx):
    uncs_bin = {}
    for k, v in subsets.items():
        unc_per_bin_wass = [uncs[synset2idx[synset]].to('cpu') for synset in v if synset2idx[synset] in uncs]
        if unc_per_bin_wass:
            uncs_bin[k] = unc_per_bin_wass
    return uncs_bin


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Uncertainty Per Synset')
    parser.add_argument('--path', type=str, help='path to model', required=True)
    parser.add_argument('--unc_branch', type=int, help='where to branch for uncertainty',
        default=200)
    parser.add_argument('--ddim_eta', type=float, help='controls stdev for generative process',
        default=0.00)
    # ddim_eta 0-1, 1=DDPM
    parser.add_argument('--ddim_steps', type=int, help='number of steps to take in ddim',
        default=200)
    parser.add_argument('--sampler', type=str, help='which smapler to use', default='DDIM')
    parser.add_argument('--scale', type=float, help='controls the amount of unconditional guidance',
        default=5.0)
    parser.add_argument('--dataset', type=str, help='which dataset was used to train the ensemble', default='binned_classes')
    args = parser.parse_args()
    #train_path = '/home/nwaftp23/scratch/uncertainty_estimation/imagenet/ILSVRC2012_train'
    train_path = '/home/lucas/latent-diffusion/results'
    unc_path = os.path.join(args.path, 
        f'uncs_dict_eta{args.ddim_eta}_branch{args.unc_branch}_sampler{args.sampler}_ddimsteps{args.ddim_steps}_scale{args.scale}.pkl')
    pair_dist_mat = np.load(os.path.join(args.path, 'pair_dist_mat.npy'))
    weight = np.array([1/5])
    uncs = {}
    for i in range(pair_dist_mat.shape[0]):
        dist = pair_dist_mat[i,:5,:5,:]
        pairwise_dist = np.log(weight)+weight*np.log(np.exp(-dist).sum(1)).sum(0)
        uncs.update({i:{'Wass':[torch.tensor(-pairwise_dist)]}})
    with open(os.path.join(train_path, 'subsets.pkl'), 'rb') as fp:
        subsets = pickle.load(fp)
    #pair_dist_mat = np.load(os.path.join(args.path, 'pair_dist_mat.npy'))
    #uncs = {k:{list(v.keys())[0]:[list(v.values())[0][0].cpu()]} for k, v in uncs.items()}
    #unc_path_cpu = os.path.join(args.path,
    #    f'uncs_dict_eta{args.ddim_eta}_branch{args.unc_branch}_'\
    #    f'sampler{args.sampler}_ddimsteps{args.ddim_steps}_scale{args.scale}_cpu.pkl')
    #with open(unc_path_cpu, 'wb') as fp:
    #    pickle.dump(uncs, fp)
    eu_wass = {k:uncs[k]['Wass'][0].mean() for k in uncs.keys()}
    idx2synset = OmegaConf.load(os.path.join(train_path, 'index_synset.yaml'))
    synset2idx = {y: x for x, y in idx2synset.items()}
    uncs_bin_wass={}
    uncs_bin_kl={}
    uncs_bin_bhatt={}
    kl_exist = 'KL' in uncs[268].keys() 
    #uncs_bin_wass, uncs_bin_kl, uncs_bin_bhatt = split_uncs(uncs, subsets, synset2idx, kl_exist)
    uncs_bin_wass = split_uncs_eu(eu_wass, subsets, synset2idx)
    min_wass, max_wass = get_min_max(uncs_bin_wass)
    mean_std_wass = norm_uncs(uncs_bin_wass, min_wass, max_wass)  
    if kl_exist:
        min_kl, max_kl = get_min_max(uncs_bin_kl)
        min_bhatt, max_bhatt = get_min_max(uncs_bin_bhatt)
        mean_std_kl = norm_uncs(uncs_bin_kl, min_kl, max_kl)  
        mean_std_bhatt = norm_uncs(uncs_bin_bhatt, min_bhatt, max_bhatt)  
    fig=plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(111)
    print('pixel unc 0')
    print(f'1300 unc {uncs[959]["Wass"][0].mean():.3f} \pm {uncs[959]["Wass"][0].std():.3f}')
    print(f'100 unc {uncs[767]["Wass"][0].mean():.3f} \pm {uncs[767]["Wass"][0].std():.3f}')
    print(f'10 unc {uncs[936]["Wass"][0].mean():.3f} \pm {uncs[936]["Wass"][0].std():.3f}')
    print(f'1 unc {uncs[892]["Wass"][0].mean():.3f} \pm {uncs[892]["Wass"][0].std():.3f}')
    print('pixel unc 1')
    print(f'1300 unc {uncs[466]["Wass"][0].mean():.3f} \pm {uncs[466]["Wass"][0].std():.3f}')
    print(f'100 unc {uncs[118]["Wass"][0].mean():.3f} \pm {uncs[118]["Wass"][0].std():.3f}')
    print(f'10 unc {uncs[379]["Wass"][0].mean():.3f} \pm {uncs[379]["Wass"][0].std():.3f}')
    print(f'1 unc {uncs[503]["Wass"][0].mean():.3f} \pm {uncs[503]["Wass"][0].std():.3f}')
    print('pixel unc 2')
    print(f'1300 unc {uncs[992]["Wass"][0].mean():.3f} \pm {uncs[992]["Wass"][0].std():.3f}')
    print(f'100 unc {uncs[376]["Wass"][0].mean():.3f} \pm {uncs[376]["Wass"][0].std():.3f}')
    print(f'10 unc {uncs[341]["Wass"][0].mean():.3f} \pm {uncs[341]["Wass"][0].std():.3f}')
    print(f'1 unc {uncs[147]["Wass"][0].mean():.3f} \pm {uncs[147]["Wass"][0].std():.3f}')
    #ax2 = ax1.twinx()
    fig.savefig(os.path.join(args.path, 'unc_plot_2.png'))
    

    def plot_mean_line(data, **kwargs):
        #sns.kdeplot(data['EU'], fill=True, color=kwargs['color'], alpha=0.7, edgecolor='black')
        ax = g.axes_dict[kwargs['label']]
        kdeline = ax.lines[0]
        xs = kdeline.get_xdata()
        ys = kdeline.get_ydata()
        middle = data['EU'].mean() 
        sdev = data['EU'].std()
        left = middle - sdev
        right = middle + sdev
        ax.vlines(middle, 0, np.interp(middle, xs, ys), color='black', ls=':')
        #ax.fill_between(xs, 0, ys, facecolor=kwargs['color'], alpha=0.2)
        ax.fill_between(xs, 0, ys, where=(left <= xs) & (xs <= right), 
            interpolate=True, facecolor=kwargs['color'], alpha=0.5)
    
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), 'axes.linewidth':2}) 
    g = sns.FacetGrid(eu_df, row="Bin", hue="Bin", aspect=9, height=1.2, palette=color_hexes[:4])
    g.map_dataframe(sns.kdeplot, x="EU", fill=True, alpha=0.4)
    g.map_dataframe(sns.kdeplot, x="EU", color='black')
    g.map_dataframe(plot_mean_line)
    #g.map(sns.kdeplot, "EU", bw_adjust=.5, clip_on=False, fill=True, alpha=1, linewidth=1.5)
    
    #g.map(sns.kdeplot, "EU", clip_on=False, color="w", lw=2, bw_adjust=.5)
    #g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
    def label(x, color, label):
        ax = plt.gca()
        # Note could include color in color
        ax.text(0, .1, label, fontsize=13, fontweight="bold", color='black',
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "EU")

    # Set the subplots to overlap
    #g.figure.subplots_adjust(hspace=-.5)
    # Remove axes details that don't play well with overlap
    g.fig.subplots_adjust(hspace=-.5)
    g.set_titles("")
    #g.set(yticks=[], ylabel="", xlim=(-.1, 1.1), xlabel=r"$I_W(Y,\theta)$")
    g.set(yticks=[], ylabel="",  xlim=(0.0, 0.08), xlabel=r"$\hat{I}_W(z_0,\theta|z_5,x,b=5)$")
    g.despine(left=True)
    g.fig.suptitle('Uncertainty Distribution According to Bin')
    #plt.suptitle('Uncertainty by Number of Images', y=0.98)
    plt.savefig(os.path.join(args.path, 'bin_distributions_2.png'), bbox_inches='tight', dpi=800)
