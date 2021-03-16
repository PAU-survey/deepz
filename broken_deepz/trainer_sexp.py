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

def get_coadd(flux, fmes, vinv, isnan, alpha=0.80, rm=True):
    """Get the coadded fluxes, using only a fraction of the
       individual exposures.
    """
    
    inp = alpha*torch.ones_like(fmes)
    mask = (~isnan).type(torch.float)
    mask *= torch.bernoulli(inp)

    weight = vinv * mask
    norm = weight.sum(2)
    coadd = (weight*fmes).sum(2) / norm
    #coadd_err = torch.sqrt(1./weight.sum(2))
    
    coadd = torch.cat([coadd, flux[:,40:]], 1)
    touse = ~(norm == 0).any(1)    

    if rm:
        coadd = coadd[touse]

    # For debugging...
    if np.isnan(coadd.cpu().numpy()).sum():
        fpr = np.isnan(coadd.cpu().numpy()).sum(1).nonzero()[0][0]
        orig_ind = touse.nonzero().flatten()[fpr].item()
        band = np.isnan(coadd[fpr].cpu().numpy()).nonzero()[0][0]


        ipdb.set_trace()

    return coadd, touse

def train(optimizer, N, enc, dec, net_pz, train_dl, test_dl, use_mdn, alpha):
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

            Bcoadd, touse = get_coadd(Bflux, Bfmes, Bvinv, Bisnan, alpha)
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
            
            Bcoadd, touse = get_coadd(Bflux, Bfmes, Bvinv, Bisnan, 1)
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
      
