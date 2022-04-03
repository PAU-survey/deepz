#!/usr/bin/env python
# encoding: UTF8

# Copyright (C) 2022 Martin B. Eriksen
# This file is part of Deepz <https://github.com/PAU-survey/deepz>.
#
# Deepz is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Deepz is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Deepz.  If not, see <http://www.gnu.org/licenses/>.


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



def get_coadd(flux, fmes, vinv, isnan, alpha=0.80, rm=True, keep_last=False):
    """Get the coadded fluxes, using only a fraction of the
       individual exposures.

       :param flux: {tensor} Coadded fluxes.
       :param fmes: {tensor} Individual flux measurements.
       :param vinv: {tensor} Inverse variance.
       :param isnan: {tensor} Used?
       :param alpha: {float} Fraction of the individual measurements used.
       :param rm: {bool} If removing entries without all narrow bands.
       :param keep_last: {bool} Keep the last flux measurement.
    """

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

def train(optimizer, N, net, train_dl, test_dl, alpha, keep_last):
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
       :param keep_last: {bool} Keep the last flux measurement.
    """
 
    loss_function = nn.CrossEntropyLoss()
    for i in range(N):
        L = []
       
        net.train() 
        
        for Bflux, Bfmes, Bvinv, Bisnan, Bzs in train_dl:
            if len(Bflux) < 10:
                continue

            optimizer.zero_grad()
            Bcoadd, touse = get_coadd(Bflux, Bfmes, Bvinv, Bisnan, alpha=alpha, \
                                      keep_last=keep_last)
            Bzs = Bzs[touse]
           
            loss = net.loss(Bcoadd, Bcoadd, Bzs)
                
                            
            loss.backward()
            optimizer.step()    
            L.append(loss.item())

        if i == 0 or i % 20:
            continue
            
        loss_train = sum(L) / len(L)
        L = []
        dxL = []
       
        net.eval() 

        t2 = time.time()
        for Bflux, Bfmes, Bvinv, Bisnan, Bzs in test_dl:
            
            Bcoadd, touse = get_coadd(Bflux, Bfmes, Bvinv, Bisnan, alpha=1)
            pred,_,loss = net.pred_recon_loss(Bcoadd, Bcoadd, Bzs)
                
            L.append(loss.item())

            zbt = 0.001*pred.argmax(1).float()
            zbt = pd.Series(zbt.cpu().numpy())
            zst = Bzs
            zst = pd.Series(zst.cpu().numpy())
            dx_part = (zbt - zst) / (1+zst)
            dxL.append(dx_part)

        dxt = pd.concat(dxL, ignore_index=True)
        sig68 = 0.5*(dxt.quantile(0.84) - dxt.quantile(0.16))
        loss_test = sum(L) / len(L)

        outl = (dxt.abs() > 0.02).mean()
        poutl = 100*outl
        print('time eval', time.time() - t2)
        print(f'{i}, {loss_train:.2f}, {loss_test:.2f}, sig68: {sig68:.5f}, outl:{poutl:.2f}')
