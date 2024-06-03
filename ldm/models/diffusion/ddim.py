"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like

from uncertainty_estimation.uncertainty_estimators import pairwise_exp

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
            to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', 
            to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', 
            to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', 
            to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               ensemble_comp=-3,
               return_distribution=False,
               return_unc=False,
               branch_split=1000,
               all_ucs=[],
               all_cs=[],
               ensemble_comps=[],
               unc_per_pixel=False,
               ensemble_size = 5,
               certain_comps = [],
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, uncertainty, intermediates, dist_mat = self.ddim_sampling(conditioning, size,
                                                            callback=callback,
                                                            img_callback=img_callback,
                                                            quantize_denoised=quantize_x0,
                                                            mask=mask, x0=x0,
                                                            ddim_use_original_steps=False,
                                                            noise_dropout=noise_dropout,
                                                            temperature=temperature,
                                                            score_corrector=score_corrector,
                                                            corrector_kwargs=corrector_kwargs,
                                                            x_T=x_T,
                                                            log_every_t=log_every_t,
                                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                                            unconditional_conditioning=unconditional_conditioning,
                                                            ensemble_comp=ensemble_comp,
                                                            return_distribution= return_distribution,
                                                            return_unc=return_unc,
                                                            branch_split=branch_split,
                                                            all_ucs=all_ucs,
                                                            all_cs=all_cs,
                                                            ensemble_comps=ensemble_comps,
                                                            unc_per_pixel=unc_per_pixel,
                                                            ensemble_size=ensemble_size,
                                                            certain_comps=certain_comps
                                                            )
        return samples, uncertainty, intermediates, dist_mat

    
    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=10,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, 
                      ensemble_comp=-3, return_distribution=False, return_unc=False,
                      branch_split=1000, all_ucs=[], all_cs=[], ensemble_comps=[], unc_per_pixel=False, 
                      ensemble_size = 5, certain_comps= []):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        if not return_distribution:
            intermediates = {'x_inter': [img], 'pred_x0': [img]}
        else:
            intermediates = {'x_inter': [img], 'pred_x0': [img], 
                'mean':[torch.zeros(shape, device=device)], 'std':[torch.ones(shape, device=device)]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)
        
        unc={}
        unc_wass = []
        unc_kl = []
        unc_bhatt = []
        if certain_comps:
            certain_unc={}
            certain_unc_wass = []
            certain_unc_kl = []
            certain_unc_bhatt = []
            uncertain_unc={}
            uncertain_unc_wass = []
            uncertain_unc_kl = []
            uncertain_unc_bhatt = []
            unc_wass = {} 
            unc_kl = {}
            unc_bhatt = {}
        dist_mat = 0
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            branch_time = i >= branch_split
            outs = self.p_sample_ddim(img, cond, ts, index=index, 
                                      use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, 
                                      temperature=temperature, noise_dropout=noise_dropout, 
                                      score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      ensemble_comp=ensemble_comp, 
                                      return_distribution=return_distribution,
                                      branch=branch_time,
                                      all_ucs=all_ucs,
                                      all_cs=all_cs,
                                      ensemble_comps=ensemble_comps,
                                      ensemble_size = ensemble_size
                                      )
            es = ensemble_size
            #if not ensemble_comps: 
                #es = self.model.model.diffusion_model.ensemble_size
            #else:
                #es = len(ensemble_comps)
            if branch_time and return_unc:
                mus = outs[-2]
                sigs = outs[-1]
                if (sigs==0).all():
                    ## Remove last five components from ensemble
                    paide, dist_mat = pairwise_exp(mus[:,:5,:,:,:], 0, 'Wass_0', 5, nth_diff_step=i, 
                        unc_per_pixel=unc_per_pixel, certain_comps=certain_comps)
                    ##
                    #paide, dist_mat = pairwise_exp(mus, 0, 'Wass_0', es, nth_diff_step=i, 
                    #    unc_per_pixel=unc_per_pixel, certain_comps=certain_comps)
                    if certain_comps:
                        certain_unc_wass.append(paide['certain'])
                        uncertain_unc_wass.append(paide['uncertain'])
                        unc_wass['certain'] = certain_unc_wass
                        unc_wass['uncertain'] = uncertain_unc_wass
                    else:
                        unc_wass.append(paide)
                    unc['Wass'] = unc_wass
                else:
                    sigs = sigs.unsqueeze(0).repeat(mus.shape[0], 1, mus.shape[2], mus.shape[3], mus.shape[4])
                    paide_wass, dist_mat_wass = pairwise_exp(mus, sigs, 'Wass', es, unc_per_pixel=unc_per_pixel, 
                        certain_comps=certain_comps)
                    if certain_comps:
                        certain_unc_wass.append(paide_wass['certain'])
                        uncertain_unc_wass.append(paide_wass['uncertain'])
                        unc_wass['certain'] = certain_unc_wass
                        unc_wass['uncertain'] = uncertain_unc_wass
                    else:
                        unc_wass.append(paide_wass)
                    unc['Wass'] = unc_wass
                    paide_kl, dist_mat_kl = pairwise_exp(mus, sigs, 'KL', es, unc_per_pixel=unc_per_pixel,
                        certain_comps=certain_comps)
                    if certain_comps:
                        certain_unc_kl.append(paide_kl['certain'])
                        uncertain_unc_kl.append(paide_kl['uncertain'])
                        unc_kl['certain'] = certain_unc_kl
                        unc_kl['uncertain'] = uncertain_unc_kl
                    else:
                        unc_kl.append(paide_kl)
                    unc['KL'] = unc_kl
                    paide_bhatt, dist_mat_bhatt = pairwise_exp(mus, sigs, 'Bhatt', es, unc_per_pixel=unc_per_pixel,
                        certain_comps=certain_comps)
                    if certain_comps:
                        certain_unc_bhatt.append(paide_bhatt['certain'])
                        uncertain_unc_bhatt.append(paide_bhatt['uncertain'])
                        unc_bhatt['certain'] = certain_unc_bhatt
                        unc_bhatt['uncertain'] = uncertain_unc_bhatt
                    else:
                        unc_bhatt.append(paide_bhatt)
                    unc['Bhatt'] = unc_bhatt
                    dist_mat = torch.stack([paide_wass, paide_kl, paide_bhatt])
            if not return_distribution:
                img, pred_x0 = outs
            else: 
                img, pred_x0, mean, std = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)
            
            if index % log_every_t == 0 or index == total_steps - 1:
                #tqdm.write(f'{index}')
                intermediates['x_inter'].append(img)
                #if not return_distribution:
                #    intermediates['x_inter'].append(img)
                #    intermediates['pred_x0'].append(pred_x0)
                #else:
                #    intermediates['x_inter'].append(img)
                #    intermediates['pred_x0'].append(pred_x0)
                #    intermediates['mean'].append(mean)
                #    intermediates['std'].append(std)

        return img, unc, intermediates, dist_mat

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, 
                      quantize_denoised=False, temperature=1., noise_dropout=0., 
                      score_corrector=None, corrector_kwargs=None, unconditional_guidance_scale=1., 
                      unconditional_conditioning=None, ensemble_comp=-3, return_distribution=False, 
                      branch=False, all_ucs=[], all_cs=[], ensemble_comps=[], ensemble_size=5):
        b, *_, device = *x.shape, x.device
        if len(x.shape) == 5:
            b, device = x.shape[1], x.device
        es = ensemble_size
        #if not ensemble_comps:  
        #    es = self.model.model.diffusion_model.ensemble_size
        #else:
        #    es = len(ensemble_comps)
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            if branch:
                e_ts = []
                if len(x.shape) == 5:
                    for i in range(es):
                        c = all_cs[i]
                        x_in = x[i,:,:,:,:]
                        if ensemble_comps:
                            e_t = ensemble_comps[i].apply_model(x_in, t, c, )
                        else:
                            e_t = self.model.apply_model(x_in, t, c, 
                                ensemble_comp=i)
                        e_ts.append(e_t)
                else:
                    for i in range(es):
                        c = all_cs[i]
                        if ensemble_comps:
                            e_t = ensemble_comps[i].apply_model(x, t, c, )
                        else:
                            e_t = self.model.apply_model(x, t, c, 
                                ensemble_comp=i)
                        e_ts.append(e_t)
                e_t = torch.stack(e_ts)
            else:
                e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            if branch:
                e_ts = []
                e_t_unconds = []
                if len(x.shape) == 5:
                    for i in range(es):
                        c_in = torch.cat([all_ucs[i], all_cs[i]])
                        x_in = torch.cat([x[i,:,:,:,:]] * 2)
                        if ensemble_comps:
                            e_t_uncond, e_t = ensemble_comps[i].apply_model(x_in, t_in, c_in,).chunk(2)
                        else:
                            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in, 
                                ensemble_comp=i).chunk(2)
                        e_ts.append(e_t)
                        e_t_unconds.append(e_t_uncond)
                else:
                    for i in range(es):
                        c_in = torch.cat([all_ucs[i], all_cs[i]])
                        if ensemble_comps:
                            e_t_uncond, e_t = ensemble_comps[i].apply_model(x_in, t_in, c_in,).chunk(2)
                        else:
                            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in, 
                                ensemble_comp=i).chunk(2)
                        e_ts.append(e_t)
                        e_t_unconds.append(e_t_uncond)
                e_t = torch.stack(e_ts)
                e_t_uncond = torch.stack(e_t_unconds)
            else:
                c_in = torch.cat([unconditional_conditioning, c])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in, 
                    ensemble_comp=ensemble_comp).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            #tqdm.write(f'{e_t.shape}')
        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        if not return_distribution:
            return x_prev, pred_x0
        else:
            mean = a_prev.sqrt() * pred_x0 + dir_xt
            std = sigma_t
            if branch and (t==self.ddim_timesteps[0]).all().item():
                tqdm.write(f'mean diff in latent space: {torch.abs((mean[0,:,:,:,:]-mean[1,:,:,:,:])).mean().item()}')
            return x_prev, pred_x0, mean, std 


