#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
import time
import numpy as np
import pandas as pd
from itertools import chain

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

import utils

def mask_entries(fmes, Nexp=1):
    """Mask single exposure entries, removing a fixed number of
       exposure, but keeping at least one. This procedure is
       discussed in the paper.

       :param fmes: {tensor} The individual flux measurements.
       :param Nexp: {int} Minimum number of exposures.
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
    """Mask single exposure entries.
       :param fmes: {tensor} The individual flux measurements.
       :param isnan: NOT USED.
       :param alpha: {float} Fraction of individual exposures used.
       :param keep_last: {bool} If keeping at least one measurement.
    """

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

       :param flux: {tensor} Coadded fluxes.
       :param fmes: {tensor} Individual flux measurements.
       :param vinv: {tensor} Inverse variance.
       :param isnan: {tensor} Used?
       :param alpha: {float} Fraction of the individual measurements used.
       :param rm: {bool} If removing entries without all narrow bands.
       :param Nexp: {int} DELETED option.
       :param keep_last: {bool} Keep the last flux measurement.
    """

    assert not Nexp, 'Deleted option'   
    inp = alpha*torch.ones_like(fmes)
    mask = mask_alpha(fmes, isnan, alpha, keep_last)


    weight = vinv * mask
    norm = weight.sum(2)
    coadd = (weight*fmes).sum(2) / norm
    
    coadd = torch.cat([coadd, flux[:,40:]], 1)
    touse = ~(norm == 0).any(1)    

    if rm:
        coadd = coadd[touse]

    return coadd, touse

def get_coadd_allexp(flux, fmes, vinv, isnan, rm=True):
    """Coadd using all the exposures. Simplified version to avoid potential
       problems in random selections.
    """

    mask = (~isnan).type(torch.float)

    weight = vinv * mask
    norm = weight.sum(2)
    coadd = (weight*fmes).sum(2) / norm

    coadd = torch.cat([coadd, flux[:,40:]], 1)
    touse = ~(norm == 0).any(1)

    if rm:
        coadd = coadd[touse]

    return coadd, touse

def train(optimizer, N, enc, dec, net_pz, train_dl, test_dl, alpha, Nexp, keep_last):
    """Train the network.
       :param optimizer: {object} PyTorch optimizer.
       :param N: {int} Number of epochs.
       :param enc: {object} Encoder network.
       :param dec: {object} Decoder network.
       :param net_pz: {object} Network for predicting the redshift.
       :param train_dl: {object} Training data loader.
       :param test_dl: {object} Test data loader.
       :param alpha: {float} Fraction of the individual measurements used.
       :param rm: {bool} If removing entries without all narrow bands.
       :param Nexp: {int} Used?
       :param keep_last: {bool} Keep the last flux measurement.
    """
 
    loss_function = nn.CrossEntropyLoss()
    for i in range(N):
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
            
            Bzbin = Bzbin.cuda()
            Bz = 0.001*Bzbin.type(torch.float)
            _, loss = net_pz.loss(Binput, Bz)
                
                
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
      
