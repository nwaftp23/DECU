from tqdm import tqdm
import numpy as np
import torch



def kl_div(mus, sigs):
    mus = mus.type(torch.float64)
    sigs = sigs.type(torch.float64)
    Sigs = sigs**2
    mus = mus.reshape(mus.shape[0], mus.shape[1], -1)
    Sigs = Sigs.reshape(mus.shape[0], mus.shape[1], -1)
    sigs = sigs.reshape(mus.shape[0], mus.shape[1], -1)
    tr_term = (Sigs[:,None,:,:]*(Sigs**-1)).sum(3)
    det_term = torch.log((Sigs/Sigs[:,None,:,:]).prod(3))
    quad_term = torch.einsum('ijkl->ijk',(mus - mus[:,None,:,:])**2/Sigs)
    return .5 * (tr_term + det_term + quad_term - mus.shape[2])

def bhatt_div(mus, sigs):
    mus = mus.type(torch.float64)
    sigs = sigs.type(torch.float64)
    Sigs = sigs**2
    mus = mus.reshape(mus.shape[0], mus.shape[1], -1)
    Sigs = Sigs.reshape(mus.shape[0], mus.shape[1], -1)
    sigs = sigs.reshape(mus.shape[0], mus.shape[1], -1)
    mean_sig = (Sigs[:,None,:,:]+Sigs)/2
    quad_term = torch.einsum('ijkl->ijk',(mus[:,None,:,:] - mus)**2/mean_sig)
    log_term = torch.log(((mean_sig)/
            torch.sqrt(Sigs[:,None]*Sigs)).prod(3))
    return ((1/8)*quad_term+(1/2)*log_term)


def wasserstein_dist(mus, sigs):
    mus = mus.type(torch.float64)
    sigs = sigs.type(torch.float64)
    Sigs = sigs**2
    mus = mus.reshape(mus.shape[0], mus.shape[1], -1)
    Sigs = Sigs.reshape(mus.shape[0], mus.shape[1], -1)
    sigs = sigs.reshape(mus.shape[0], mus.shape[1], -1)
    quad_term = torch.einsum('ijkl->ijk',(mus[:,None,:,:] - mus)**2)
    tr_term = (Sigs[:,None,:,:]+Sigs-
            2*torch.sqrt(sigs[:,None,:,:]*Sigs*sigs[:,None,:,:])).sum(3)
    return quad_term+tr_term


def wasserstein_dist_zero_std(mus, unc_per_pixel=False):
    mus = mus.type(torch.float64)
    if unc_per_pixel:
        quad_term = torch.einsum('ijklmn->ijklmn',(mus[:,:,None,:,:,:] - mus)**2)
    else:
        mus = mus.reshape(mus.shape[0], mus.shape[1], -1)
        quad_term = torch.einsum('ijkl->ijk',(mus[:,None,:,:] - mus)**2)
    return quad_term

def pairwise_exp(mus, sigs, measure, numb_comp, device='cuda', nth_diff_step=0, unc_per_pixel=False, certain_comps=[]):
    if measure == 'KL':
        dist = kl_div(mus, sigs)
    elif measure == 'Bhatt':
        dist = bhatt_div(mus, sigs)
    elif measure == 'Wass':
        dist = wasserstein_dist(mus, sigs)
    elif measure == 'Wass_0':
        dist = wasserstein_dist_zero_std(mus, unc_per_pixel=unc_per_pixel)
    if certain_comps:
        all_comps = [i for i in range(numb_comp)]
        uncertain_comps = list(set(all_comps)-set(certain_comps))
        certain_dist = 0 
        certain_pairwise_dist = None 
        if len(certain_comps)>1:
            certain_mus = mus[certain_comps]
            if type(sigs) != int:
                certain_sigs = sigs[certain_comps]
            if measure == 'KL':
                certain_dist = kl_div(certain_mus, certain_sigs)
            elif measure == 'Bhatt':
                certain_dist = bhatt_div(certain_mus, certain_sigs)
            elif measure == 'Wass':
                certain_dist = wasserstein_dist(certain_mus, certain_sigs)
            elif measure == 'Wass_0':
                certain_dist = wasserstein_dist_zero_std(certain_mus, unc_per_pixel=unc_per_pixel)
            certain_weight = torch.tensor([1/len(certain_comps)]).to(device).type(torch.float64)
            certain_pairwise_dist = torch.log(certain_weight)+certain_weight*torch.log(torch.exp(-certain_dist).sum(1)).sum(0)
            certain_pairwise_dist = -certain_pairwise_dist
        uncertain_dist = 0 
        uncertain_pairwise_dist = None
        if len(uncertain_comps)>1: 
            uncertain_mus = mus[uncertain_comps]
            if type(sigs) != int:
                uncertain_sigs = sigs[certain_comps]
            if measure == 'KL':
                uncertain_dist = kl_div(uncertain_mus, uncertain_sigs)
            elif measure == 'Bhatt':
                uncertain_dist = bhatt_div(uncertain_mus, uncertain_sigs)
            elif measure == 'Wass':
                uncertain_dist = wasserstein_dist(uncertain_mus, uncertain_sigs)
            elif measure == 'Wass_0':
                uncertain_dist = wasserstein_dist_zero_std(uncertain_mus, unc_per_pixel=unc_per_pixel)
            uncertain_weight = torch.tensor([1/len(uncertain_comps)]).to(device).type(torch.float64)
            uncertain_pairwise_dist = torch.log(uncertain_weight)+uncertain_weight*torch.log(torch.exp(-uncertain_dist).sum(1)).sum(0)
            uncertain_pairwise_dist = -uncertain_pairwise_dist
        pairwise_dist = {'certain': certain_pairwise_dist, 'uncertain': uncertain_pairwise_dist}
    else:
        weight = torch.tensor([1/numb_comp]).to(device).type(torch.float64)
        ## Different weights
        #pairwise_dist = (torch.log((torch.exp(-dist)*weight).sum(1))*weight).sum(0)
        ## Same weights
        pairwise_dist = torch.log(weight)+weight*torch.log(torch.exp(-dist).sum(1)).sum(0)
        ##NOTE Logsumexp trick
        #const = 1
        #pairwise_dist_2 = torch.log(weight)+weight*(const+torch.log(torch.exp(-dist-const).sum(1))).sum(0)
        ##ALSO weight is same for each comp
        #if nth_diff_step ==199:
        #    import pdb; pdb.set_trace()
        #tqdm.write(f'{-pairwise_dist.mean().item()}')
        pairwise_dist = -pairwise_dist
    return pairwise_dist, dist
