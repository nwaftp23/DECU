# DECU (Diffusion Ensembles for Capturing Uncertainty)
[arXiv](https://arxiv.org/abs/2406.18580) | [BibTeX](#bibtex)

This repository contains the code used for the research paper titled "Shedding Light on Large Generative Networks:
Estimating Epistemic Uncertainty in Diffusion Models". The code is heavily based on the [latent-diffusion](https://github.com/CompVis/latent-diffusion) repository by CompVis.

### Certain Rollout
<p align="center">
<img src=assets/certain_rollout.png />
</p>

### Uncertain Rollout
<p align="center">
<img src=assets/uncertain_rollout.png />
</p>


## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Training](#training)
- [Sample](#sample)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

Generative diffusion models, notable for their large parameter count (exceeding 100 million) and operation within high-dimensional image spaces, pose significant challenges for traditional uncertainty estimation methods due to computational demands. In this work, we introduce an innovative framework, Diffusion Ensembles for Capturing Uncertainty (DECU), designed for estimating epistemic uncertainty for diffusion models. The DECU framework introduces a novel method that efficiently trains ensembles of conditional diffusion models by incorporating a static set of pre-trained parameters, drastically reducing the computational burden and the number of parameters that require training. Additionally, DECU employs Pairwise-Distance Estimators (PaiDEs) to accurately measure epistemic uncertainty by evaluating the mutual information between model outputs and weights in high-dimensional spaces. The effectiveness of this framework is demonstrated through experiments on the ImageNet dataset, highlighting its capability to capture epistemic uncertainty, specifically in under-sampled image classes.

## Requirements

- python 3.8.10
- pip intall -r requirements.txt
- run prep_data.sh which will:
   - download imagenet dataset [imagenet](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php)
   - download pretrained networks: [vae](https://ommer-lab.com/files/latent-diffusion/vq-f4.zip) and [diffusion](https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt)
- change paths in configs/latent-diffusion/cin256-v2.yaml (ckpt_path:, data_root:)

## Training

### Model Pipeline
<p align="center">
<img src=assets/flow_chart.png />
</p>

Change ```seed``` and ```component``` to train an ensemble:

```
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/latent-diffusion/cin256-v2.yaml -t --gpus 0, --scale_lr False --start_from_pretrained --logdir /path/to/storelogs  --seed 0 --component 0 --ensemble_name bootstrapped --overwrite_data_root ./imagenet_data
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/latent-diffusion/cin256-v2.yaml -t --gpus 0, --scale_lr False --start_from_pretrained --logdir /path/to/storelogs  --seed 1 --component 1 --ensemble_name bootstrapped --overwrite_data_root ./imagenet_data
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/latent-diffusion/cin256-v2.yaml -t --gpus 0, --scale_lr False --start_from_pretrained --logdir /path/to/storelogs  --seed 2 --component 2 --ensemble_name bootstrapped --overwrite_data_root ./imagenet_data
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/latent-diffusion/cin256-v2.yaml -t --gpus 0, --scale_lr False --start_from_pretrained --logdir /path/to/storelogs  --seed 3 --component 3 --ensemble_name bootstrapped --overwrite_data_root ./imagenet_data
CUDA_VISIBLE_DEVICES=0 python main.py --base configs/latent-diffusion/cin256-v2.yaml -t --gpus 0, --scale_lr False --start_from_pretrained --logdir /path/to/storelogs  --seed 4 --component 4 --ensemble_name bootstrapped --overwrite_data_root ./imagenet_data
```

## Sample

Create Ensemble:

```
python save_ensemble.py --path /path/to/storelogs --ensemble_name bootstrapped --ensemble_comps 5
```

Sample Ensemble:

```
python sample_model.py --path /path/to/ensemble
```

Sample Rollout:

```
python sample_model_rollout.py --path /path/to/ensemble
```

Get Uncertainty:

```
python get_unc_synsets.py --path /path/to/ensemble
```

Get Uncertainty per Pixel:

```
python get_unc_pixels.py --path /path/to/ensemble
```
