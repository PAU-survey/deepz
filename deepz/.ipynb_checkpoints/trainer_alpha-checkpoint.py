#!/usr/bin/env python
# encoding: UTF8

# Created at 

# =============================================================================
# DOCS
# =============================================================================

"""trainer_alpha
"""

# =============================================================================
# IMPORTS
# =============================================================================

from IPython.core import debugger as ipdb
import time
import numpy as np
import pandas as pd
from itertools import chain

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

import utils

def smoother(log_pz, kappa=0.15):
    """Functionality for smoothing the p(z)."""
    
    pz = torch.exp(log_pz.detach())
    pz = pz / pz.sum(1)[:,None]

    csum = pz.cumsum(dim=1)
    pz_std = ((0.16 < csum) & (csum < 0.84)).sum(1)
    pz_std = (0.5*pz_std.float()).clamp(0.002, 0.02)
    X = torch.normal(0, kappa*pz_std.float())
    bz_rand = (X / 0.001).round().type(torch.long)
    
    return bz_rand


def mask_entries(fmes, Nexp=1):
    """Mask single exposure entries, removing a fixed number of
       exposure, but keeping at least one.
    """

    # Remove a fixed number of exposures.
    R = torch.rand(size=fmes.shape, device=fmes.device)

    # So we only remove actual measurements.
    ismes = (fmes != 0).type(torch.float)
    R = R * ismes

    torm = R == R.max(2)[0][:,:,None]

    tofewexp = (ismes.sum(2) <= Nexp)
    torm[tofewexp] = 0

    mask = (~torm) & (fmes != 0)
    mask = mask.to(torch.float)

    return mask

def mask_alpha(fmes, isnan, alpha, keep_last):

    inp = alpha*torch.ones_like(fmes)
    mask_rand = torch.bernoulli(inp)
    ismes = (fmes != 0).type(torch.float)

    # Code for having at least one measurement. Could possible
    # be more elegant, but it was not easy to find a good
    # solution.
    if keep_last:
        R = torch.rand(size=fmes.shape, device=fmes.device)
        selone = (R == R.max(2)[0][:,:,None])

        missing = (0 == (mask_rand*ismes).sum(2))
        tokeep = selone*missing[:,:,None]
        tokeep = tokeep.to(torch.float)

        mask = mask_rand*ismes + tokeep
        assert (mask < 2).all(), 'Internal error. Someone broke the code' 
        assert not (mask.sum(2) == 0).any(), 'Internal error.'
    else:
        mask = mask_rand*ismes 

    return mask



def get_coadd(flux, fmes, vinv, isnan, alpha=0.80, rm=True, Nexp=0, 
              keep_last=False):
    """Get the coadded fluxes, using only a fraction of the
       individual exposures.
    """

    assert not Nexp, 'Deleted option'   
    inp = alpha*torch.ones_like(fmes)
#    mask = (~isnan).type(torch.float)
#    mask *= torch.bernoulli(inp)
    mask = mask_alpha(fmes, isnan, alpha, keep_last)


#    ipdb.set_trace()



    weight = vinv * mask
    norm = weight.sum(2)
    coadd = (weight*fmes).sum(2) / norm
    #coadd_err = torch.sqrt(1./weight.sum(2))
    
    coadd = torch.cat([coadd, flux[:,40:]], 1)
    touse = ~(norm == 0).any(1)    

    if rm:
        coadd = coadd[touse]

    return coadd, touse

def train(optimizer, N, enc, dec, net_pz, train_dl, test_dl, use_mdn, alpha, Nexp, keep_last):
    loss_function = nn.CrossEntropyLoss()
    for i in range(N):
#        print('i', i)
        L = []
        
        enc.train()
        dec.train()
        net_pz.train()
        
        t1 = time.time()
        for Bflux, Bfmes, Bvinv, Bisnan, Bzbin in train_dl:
            if len(Bflux) < 10:
                continue

            optimizer.zero_grad()
            Bcoadd, touse = get_coadd(Bflux, Bfmes, Bvinv, Bisnan, alpha=alpha, Nexp=Nexp, \
                                      keep_last=keep_last)
            Bzbin = Bzbin[touse]
           
            # Testing training augmentation.            
            feat = enc(Bcoadd)
            Binput = torch.cat([Bcoadd, feat], 1)
            
            if not use_mdn:
                log_pz = net_pz(Binput)
                bz_rand = smoother(log_pz)
                Bzbin = Bzbin.cuda() + bz_rand.cuda()
                Bzbin = Bzbin.clamp(0, 2099)
                # Disable the smoother to simpler compare the results.
                loss = loss_function(log_pz, Bzbin.cuda())
            else:
                Bzbin = Bzbin.cuda()
                
                Bz = 0.001*Bzbin.type(torch.float)
#                log_pz, _ = net_pz.loss(Binput, Bz)
#                bz_rand = smoother(log_pz)
                
                #Bzbin = Bzbin + bz_rand.cuda()
                #Bzbin = Bzbin.clamp(0, 2099)
#                Bz = 0.001*Bzbin.type(torch.float)
                
                _, loss = net_pz.loss(Binput, Bz)
                
                #loss = loss_function(log_pz, Bzbin.cuda())
                
            loss.backward()
            optimizer.step()    
            L.append(loss.item())

        #print('time', time.time() - t1)
        if i == 0 or i % 20:
            continue
            
        loss_train = sum(L) / len(L)
        L = []
        dxL = []
        
        enc.eval()
        dec.eval()
        net_pz.eval()

        t2 = time.time()
        for Bflux, Bfmes, Bvinv, Bisnan, Bzbin in test_dl:
            Bflux = Bflux.cuda()
            feat = enc(Bflux)
            
            Bcoadd, touse = get_coadd(Bflux, Bfmes, Bvinv, Bisnan, alpha=1)
            Binput = torch.cat([Bcoadd, feat], 1)
            
            
            pred = net_pz(Binput)
            if not use_mdn:
                loss = loss_function(pred, Bzbin.cuda())
            else:
                Bz = 0.001*Bzbin.type(torch.float)
                log_pz, loss = net_pz.loss(Binput, Bz)
                
            L.append(loss.item())

            zbt = 0.001*pred.argmax(1).float()
            zbt = pd.Series(zbt.cpu().numpy())
            zst = 0.001*Bzbin.float()
            zst = pd.Series(zst.cpu().numpy())
            dx_part = (zbt - zst) / (1+zst)
            dxL.append(dx_part)

        dxt = pd.concat(dxL, ignore_index=True)
        sig68 = 0.5*(dxt.quantile(0.84) - dxt.quantile(0.16))
        loss_test = sum(L) / len(L)

        print('time eval', time.time() - t2)
        outl = (dxt.abs() > 0.02).mean()
        poutl = 100*outl
        print(f'{i}, {loss_train:.2f}, {loss_test:.2f}, sig68: {sig68:.5f}, outl:{poutl:.2f}')
      
