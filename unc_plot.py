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

from create_helper_variables import create_synset_dicts, create_bin_dicts
from create_subset_masked import check_synset

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
    args = parser.parse_args()

    dataset = 'binned_classes'
    subsets, human_synset, idx2synset, synset2idx = create_synset_dicts(dataset)
    df = pd.DataFrame(human_synset.items(), columns=['synset', 'text'])
    path = '/home/nwaftp23/projects/def-dpmeger/nwaftp23/epi-diffusion/bootstrapped_imagenet_10'
    samp_dir = os.path.join(path, f'samps_uncs_branch{args.unc_branch}')
    pair_dist_path = os.path.join(samp_dir, 'pair_dists')
    pair_dist_files = os.listdir(pair_dist_path)
    unc_path = os.path.join(samp_dir,
            f'uncs_dict_eta0.0_branch50_samplerDDIM_ddimsteps200_scale5.0.pkl')
    with open(unc_path, 'rb') as fp:
            uncs = pickle.load(fp)
    import pdb; pdb.set_trace()

    pair_dists = {}
    
    for pdf in pair_dist_files:
        class_idx = pdf.split('class')[-1].split('.')[0]
        full_path = os.path.join(pair_dist_path, pdf)
        pair_dists[class_idx] = np.load(full_path)
    import pdb; pdb.set_trace()
    eu_wass = {k:uncs[k]['Wass'][0].mean() for k in uncs.keys()}
    uncs_bin_wass={}
    uncs_bin_kl={}
    uncs_bin_bhatt={}
    kl_exist = 'KL' in uncs[268].keys() 
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

    # changes here
    x = list(range(1,5))
    x2 = list(range(0,4))
    bin_count = [1, 10, 100, 1300]
    y_mean_unc_wass = np.array([mean_std_wass[k]['mean'] for k in bin_count])
    y_std_unc_wass = np.array([mean_std_wass[k]['std'] for k in bin_count])
    
    
    combined_data = []
    bins_4_df = []
    for xval in bin_count:
        #combined_data.append((torch.stack(uncs_bin_wass[xval])-min_wass)/(max_wass-min_wass))
        combined_data.append(torch.stack(uncs_bin_wass[xval]))
        bins_4_df.append([xval]*len(uncs_bin_wass[xval]))
    eu_df = pd.DataFrame({'EU':torch.cat(combined_data), 'Bin':list(itertools.chain(*bins_4_df))})
    sns.violinplot(data=eu_df, x="Bin", y="EU", palette=color_hexes[:4], ax=ax1)
    sns.stripplot(x="Bin", y="EU", data=eu_df,
              color="black", edgecolor="gray", jitter=0.2)

    
    # Add labels and title
    ax1.set_xlabel('Number of Images')
    ax1.set_ylabel('Epistemic Uncertainty')
    if kl_exist:
        y_mean_unc_kl = [mean_std_kl[k]['mean'] for k in bin_count]
        y_mean_unc_bhatt = [mean_std_bhatt[k]['mean'] for k in bin_count]
        ax2.plot(bin_count, y_mean_unc_kl, color=color_hexes[2])
        ax2.plot(bin_count, y_mean_unc_bhatt, color=color_hexes[3])
        
    fig.savefig(os.path.join(args.path, 'unc_plot.png'))
    

    def plot_mean_line(data, **kwargs):
        ax = g.axes_dict[kwargs['label']]
        kdeline = ax.lines[0]
        xs = kdeline.get_xdata()
        ys = kdeline.get_ydata()
        middle = data['EU'].mean() 
        sdev = data['EU'].std()
        left = middle - sdev
        right = middle + sdev
        ax.vlines(middle, 0, np.interp(middle, xs, ys), color='black', ls=':')
        ax.fill_between(xs, 0, ys, where=(left <= xs) & (xs <= right), 
            interpolate=True, facecolor=kwargs['color'], alpha=0.5)
    
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0), 'axes.linewidth':2}) 
    g = sns.FacetGrid(eu_df, row="Bin", hue="Bin", aspect=9, height=1.2, palette=color_hexes[:4])
    g.map_dataframe(sns.kdeplot, x="EU", fill=True, alpha=0.4)
    g.map_dataframe(sns.kdeplot, x="EU", color='black')
    g.map_dataframe(plot_mean_line)
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .1, label, fontsize=13, fontweight="bold", color='black',
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "EU")

    # Remove axes details that don't play well with overlap
    g.fig.subplots_adjust(hspace=-.5)
    g.set_titles("")
    g.set(yticks=[], ylabel="",  xlim=(0.0, 0.08), xlabel=r"$\hat{I}_W(z_0,\theta|z_5,x)$")
    g.despine(left=True)
    g.fig.suptitle('Uncertainty Distribution According to Bin')
    plt.savefig(os.path.join(samp_dir, 'bin_distributions.png'), bbox_inches='tight', dpi=800)
