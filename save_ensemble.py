import re
import gc
import pickle
import argparse
import torch
import os
from collections import Counter
from omegaconf import OmegaConf

from hurry.filesize import size

from ldm.util import instantiate_from_config




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
    checkpoints = os.listdir(os.path.join(path, 'checkpoints'))
    checkpoints.sort()
    #last_epoch = checkpoints[-2]
    #model_path = os.path.join(path, f'checkpoints/{last_epoch}')
    last = checkpoints[-1]
    model_path = os.path.join(path, f'checkpoints/{last}')
    model = load_model_from_config(config, model_path)
    return model

def load_ensemble(path, numb_comps):
    ensemble_model_config = OmegaConf.load('/home/nwaftp23/latent-diffusion/configs/latent-diffusion/cin_ensemble.yaml')
    ensemble_model_config.model.params.unet_config.params['ensemble_size'] = numb_comps
    ensemble_model_config.model.params.cond_stage_config.params['ensemble_size'] = numb_comps
    ensemble_model = instantiate_from_config(ensemble_model_config.model)
    ensemble_model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))
    ensemble_model.cuda()
    ensemble_model.eval()
    return ensemble_model
    

def save_ensemble(path, ensemble_name, numb_comps):
    model_dirs = os.listdir(path)
    model_dirs = [md for md in model_dirs if re.search(ensemble_name+'[0-9]', md)]
    #model_dirs = [md for md in model_dirs if ensemble_name in md]
    model_dirs.sort()
    model_dirs = model_dirs[:numb_comps]
    #model_dirs = model_dirs[:1]
    ensemble_model_config = OmegaConf.load('/home/nwaftp23/latent-diffusion/configs/latent-diffusion/cin_ensemble.yaml') 
    ensemble_model_config.model.params.unet_config.params['ensemble_size'] = numb_comps
    ensemble_model_config.model.params.cond_stage_config.params['ensemble_size'] = numb_comps
    ensemble_model = instantiate_from_config(ensemble_model_config.model)
    ensemble_model_state_dict = ensemble_model.state_dict()
    for i, comp in enumerate(model_dirs):
        model_comp = get_model(os.path.join(path, comp))
        model_comp_state_dict = model_comp.state_dict()
        if i == 0:
            keys_2_replace = [k for k, v in model_comp_state_dict.items()]
            keys_2_replace = [i for i in keys_2_replace if not i.startswith('first_stage_model')]
            keys_2_replace = (set(keys_2_replace)-
                set([#'model.diffusion_model.out.0.weight', 'model.diffusion_model.out.2.bias', 
                #'model.diffusion_model.out.2.weight', 'model.diffusion_model.out.0.bias',
                #"model.diffusion_model.time_embed.2.bias", 
                #"model.diffusion_model.time_embed.0.weight", 
                #"model.diffusion_model.time_embed.0.bias", 
                #"model.diffusion_model.time_embed.2.weight",
                'cond_stage_model.embedding.weight',]))
                #'cond_stage_model.embedding.lora_A', 
                #'cond_stage_model.embedding.lora_B']))
            replacement_dict = {k: model_comp_state_dict[k] for k in keys_2_replace}
            ensemble_model_state_dict.update(replacement_dict)
        keys_2_replace = [#f'model.diffusion_model.ensembles.{i}.0.weight', 
            #f'model.diffusion_model.ensembles.{i}.0.bias', 
            #f'model.diffusion_model.ensembles.{i}.2.weight', 
            #f'model.diffusion_model.ensembles.{i}.2.bias',
            #f"model.diffusion_model.time_embed_ensemble.{i}.2.bias",
            #f"model.diffusion_model.time_embed_ensemble.{i}.0.weight",
            #f"model.diffusion_model.time_embed_ensemble.{i}.0.bias",
            #f"model.diffusion_model.time_embed_ensemble.{i}.2.weight",
            f'cond_stage_model_ensemble.{i}.embedding.weight',]
            #f'cond_stage_model_ensemble.{i}.embedding.lora_A',
            #f'cond_stage_model_ensemble.{i}.embedding.lora_B']
        comp_keys = [#'model.diffusion_model.out.0.weight', 'model.diffusion_model.out.0.bias',
                #'model.diffusion_model.out.2.weight', 'model.diffusion_model.out.2.bias',
                #"model.diffusion_model.time_embed.2.bias",
                #"model.diffusion_model.time_embed.0.weight",
                #"model.diffusion_model.time_embed.0.bias",
                #"model.diffusion_model.time_embed.2.weight",
                'cond_stage_model.embedding.weight',]
                #'cond_stage_model.embedding.lora_A',
                #'cond_stage_model.embedding.lora_B']
        replacement_dict = {k: model_comp_state_dict[ck].clone() for 
            k, ck in zip(keys_2_replace, comp_keys)}
        ensemble_model_state_dict.update(replacement_dict)
        del model_comp
        torch.cuda.empty_cache()
        gc.collect()
        print(f'gpu usage:{size(torch.cuda.memory_allocated())}')
    ensemble_model.load_state_dict(ensemble_model_state_dict)
    ensemble_dir = os.path.join(path, ensemble_name+f'_imagenet_{numb_comps}')
    torch.save(ensemble_model.state_dict(), os.path.join(ensemble_dir, 'model.pth'))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get Uncertainty Per Synset')
    parser.add_argument('--path', type=str, help='path to model', required=True)
    parser.add_argument('--ensemble_name', type=str, help='which ensemble to load', default='bootstrapped')
    parser.add_argument('--ensemble_comps', type=int, help='number of ensemble comps', default=5)
    args = parser.parse_args()
    #seed_everything(42)
    
    save_dir = os.path.join(args.path, f'{args.ensemble_name}_imagenet_{args.ensemble_comps}')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_ensemble(args.path, args.ensemble_name, args.ensemble_comps)
