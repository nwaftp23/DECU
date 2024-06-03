import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from omegaconf import OmegaConf

mpl.rc('font',family='Times New Roman')

if __name__ == '__main__':
    print('plot img diff graph and latex for table')
    model_path = '/home/lucas/latent-diffusion/results/bootstrapped_imagenet_10'
    with open(os.path.join(model_path, 'img_diversity_data_mean.pkl'), 'rb') as f:
        mus = pickle.load(f)
    with open(os.path.join(model_path, 'img_diversity_data_std.pkl'), 'rb') as f:
        sigs = pickle.load(f)
    x = [1,10,100,1300]
    mus_0 = [mus[0][1], mus[0][10], mus[0][100], mus[0][1300]]
    mus_50 = [mus[50][1], mus[50][10], mus[50][100], mus[50][1300]]
    mus_100 = [mus[100][1], mus[100][10], mus[100][100], mus[100][1300]]
    mus_150 = [mus[150][1], mus[150][10], mus[150][100], mus[150][1300]]
    mus_199 = [mus[199][1], mus[199][10], mus[199][100], mus[199][1300]]
    marker_size = 15
    linew = 5
    tick_size = 12
    title_size = 18
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(x, mus_0, '-o', label=r'b=1000', markersize=marker_size, linewidth=linew)
    ax.plot(x, mus_50, '-o', label=r'b=750', markersize=marker_size, linewidth=linew)
    ax.plot(x, mus_100, '-o', label=r'b=500', markersize=marker_size, linewidth=linew)
    ax.plot(x, mus_150, '-o', label=r'b=250', markersize=marker_size, linewidth=linew)
    #ax.plot(x, mus_199, '-o', label=r'b=1', markersize=marker_size, linewidth=linew)
    ax.set_xscale('log')
    ax.set_xlabel("Bin", fontsize=title_size)
    ax.set_ylabel("SSIM",  fontsize=title_size)
    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    lines = ax.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1]
    leg = fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(.5, -.25), ncol=4, 
            fontsize="14")
    leg.get_frame().set_linewidth(1.0)
    leg.get_frame().set_edgecolor('black')
    plt.savefig(os.path.join('/home/lucas/latent-diffusion/graphs', 
        'img_diversity_bp.png'), dpi=400, bbox_inches="tight")
    plt.close()
    df = pd.DataFrame(mus).T.sort_index()
    df = df.iloc[:5,:]
    print(df.to_latex())
    print(sigs)
