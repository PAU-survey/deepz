#!/usr/bin/env python
# encoding: UTF8

# Pretrain the network on simulations. Later the trained network
# would be the start for training on data.

import os
import sys
import time

from pathlib import Path
import fire
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from itertools import chain
from IPython.core import debugger

import torch
from torch import nn
from torch import nn, optim
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, DataLoader

import networks

def load_sims(path_sims, broad_bands):
    """Load the FSPS simulations used for training.
       :param path_sims: {path} Path to the simulation directory.
       :param broad_bands: {list} List of broad bands.
    """

    path_sims = Path(path_sims)
    mags_df = pd.read_parquet(str(path_sims / 'mags.parquet'))
    params_df = pd.read_parquet(str(path_sims / 'params.parquet'))

    # Bands to select for the training...
    NB = ['nb{}'.format(x) for x in 455+10*np.arange(40)]
    BB = broad_bands
    bands = NB + BB
    SN = torch.tensor(len(NB)*[10] + len(BB)*[35], dtype=torch.float).cuda()

    col = mags_df.values - mags_df.values[:,-2][:,None]
    col = pd.DataFrame(col, index=mags_df.index, columns=mags_df.columns)

    # A small fraction has very extreme colors. Cutting
    # these away..
    tosel = ~(10 < col.abs()).any(axis=1)
    col = col.loc[tosel, bands]

    # Convert to fluxes.
    flux = torch.tensor(10**(-0.4*col.values)).float()
    flux_err = flux/ SN.cpu()[None,:]

    # And then select the parameters.
    params = torch.tensor(params_df.loc[tosel].values).float()
    zind = list(params_df.columns).index('zred')
    zbin = (params[:, zind]/0.001).round().type(torch.long)

    return flux, flux_err, zbin


def to_dl(flux, flux_err, zbin):
    """Convert input tensors to dataloaders.

       :param flux: {tensor} Galaxy fluxes.
       :param flux_err: {tensor} Galaxy flux errors.
       :param zbin: {tensor} Redshift bin.
    """

    ds = TensorDataset(flux, flux_err, zbin)
    Ngal = len(ds)

    Ntest = int(0.002*Ngal)
    Ntrain = Ngal - Ntest

    train_ds, test_ds = random_split(ds, (Ntrain, Ntest)) 
    train_dl = DataLoader(train_ds, batch_size=500, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=100)

    return train_dl, test_dl


def train_sim(train_dl, test_dl, Nbands):
    """Train on the simulations.
       :param train_dl: {object} Trainer data loader.
       :param test_dl: {object} Test data loader.
    """

    net = networks.Deepz(Nbands).cuda()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    for i in range(10):
        L = []
       
        net.train()
     
        t1 = time.time()
        for Bflux, Bflux_err, Bzbin in train_dl:
            t3 = time.time()
            optimizer.zero_grad()
            
            Bflux = Bflux.cuda()
            Bflux_err = Bflux_err.cuda()
            #Bflux_err = (Bflux / SN[None,:]).cuda()
            
            noise = Bflux_err * torch.randn(Bflux.shape).cuda()
            Bflux = Bflux + noise
            
            Bzbin = Bzbin.cuda()
            
            Bz = 0.001*Bzbin.type(torch.float)

            pred, recon, loss_pz = net.pred_recon_loss(Bflux, Bflux, Bz)
            loss_recon = (recon - Bflux).abs() / Bflux_err
            loss_recon = loss_recon.mean()

            loss = loss_pz + loss_recon

            loss.backward()
            optimizer.step()    
            L.append(loss.item())
            
        print('time train', time.time() - t1)
        loss_train = sum(L) / len(L)
        L = []
        dxL = []

        net.eval()

        print('starting test eval')
        for Bflux, Bflux_err, Bzbin in test_dl:
            Bflux = Bflux.cuda()
            Bflux_err = Bflux_err.cuda()
            
            # For some reason we did *not* include the reconstruction loss here.
            Bzbin = Bzbin.cuda()
            Bz = 0.001*Bzbin.type(torch.float)

            pred, recon, loss_pz = net.pred_recon_loss(Bflux, Bflux, Bz)

            loss_recon = (recon - Bflux).abs() / Bflux_err
            loss_recon = loss_recon.mean()
            loss = loss_pz + loss_recon
            
            L.append(loss.item())
            
            zbt = 0.001*pred.argmax(1).float()
            zbt = pd.Series(zbt.cpu().numpy())
            zs = 0.001*Bzbin.float()
            zs = pd.Series(zs.cpu().numpy())
            dx_part = (zbt - zs) / (1+zs)
            dxL.append(dx_part)
        
        dxt = pd.concat(dxL, ignore_index=True)
        sig68 = 0.5*(dxt.quantile(0.84) - dxt.quantile(0.16))
        loss_test = sum(L) / len(L)

        assert not np.isnan(loss_train), 'Found NaN when training. Try again.'
        outl = (dxt.abs() > 0.02).mean()
        print(i, loss_train, loss_test, 'sig68', sig68, 'outl', outl)

    return net

def parse_bb(broad_bands):
    """Parse the broad band input string.
       :param broad_bands: {object} Which broad bands to use.
    """

    if isinstance(broad_bands, list):
        pass 
    elif isinstance(broad_bands, tuple):
        broad_bands = list(broad_bands)
    elif broad_bands.lower() == 'cosmos':
        broad_bands = ['cfht_u', 'subaru_b', 'subaru_v', 'subaru_r', 'subaru_i', 'subaru_z']
    elif bb.lower() == 'cfht':
        # Just need to be specified.
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    return broad_bands

def train(path_sims, model_dir, pretrain_label, broad_bands):
    """Pretrain the model on simulations.
       :param path_sims: {path} Directory with the simulations.
       :param model_dir: {path} Directory where to store the models.
       :param pretrain_label: {str} 
       :param broad_bands: {str, list} Broad bands to use (or cosmos, cfht).
    """

    broad_bands = parse_bb(broad_bands)

    output_path = Path(model_dir) / 'pretrain_{pretrain_label}.pt'
    if output_path.exists():
        print('Output file aready exists:', output_path)
        return

    flux, flux_err, zbin = load_sims(path_sims, broad_bands)
    Nbands = flux.shape[1]

    train_dl, test_dl = to_dl(flux, flux_err, zbin)
    net = train_sim(train_dl, test_dl, Nbands)

    print('Storing model to:', output_path)
    torch.save(net.state_dict(), output_path)

if __name__ == '__main__':
    fire.Fire(train)
