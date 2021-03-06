#!/usr/bin/env python
# encoding: UTF8

# For doing the pretraining on simulations.
import sys
import os

import time
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
sys.path.append('var')
from itertools import chain

import torch
from torch import nn
from torch import nn, optim
from torch.utils.data.dataset import random_split
from torch.utils.data import TensorDataset, DataLoader

import arch_mdn
import var_networks

# The input simulations.
path_in = Path('/nfs/astro/eriksen/deepz/sims/v9/')
mags_df = pd.read_parquet(str(path_in / 'mags.parquet'))
params_df = pd.read_parquet(str(path_in / 'params.parquet'))



# Bands to select for the training...
NB = ['nb{}'.format(x) for x in 455+10*np.arange(40)]
#BB = ['cfht_u', 'subaru_b', 'subaru_v', 'subaru_r', 'subaru_i', 'subaru_z']
BB = ['cfht_u', 'cfht_g', 'cfht_r', 'cfht_i', 'cfht_z']
bands = NB + BB
SN = torch.tensor(len(NB)*[10] + len(BB)*[35], dtype=torch.float).cuda()

col = mags_df.values - mags_df.values[:,-2][:,None]
print('type', type(col))
col = pd.DataFrame(col, index=mags_df.index, columns=mags_df.columns)

# A small fraction has very extreme colors. Cutting
# these away..
tosel = ~(10 < col.abs()).any(axis=1)
#tosel = pd.Series(True, index=col.index)
assert not set(bands) - set(col.columns)
col = col.loc[tosel, bands]

# Convert to fluxes.
flux = torch.tensor(10**(-0.4*col.values)).float()
flux_err = flux/ SN.cpu()[None,:]

# And then select the parameters.
params = torch.tensor(params_df.loc[tosel].values).float()
zind = list(params_df.columns).index('zred')
zbin = (params[:, zind]/0.001).round().type(torch.long)



ds = TensorDataset(params, zbin, flux, flux_err)
Ngal = len(ds)

Ntest = int(0.002*Ngal)
Ntrain = Ngal - Ntest

train_ds, test_ds = random_split(ds, (Ntrain, Ntest)) 
train_dl = DataLoader(train_ds, batch_size=500, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=100)



# If using a mixure density network.
use_mdn = False
use_mdn = True

# Different networks.
Nfeat = 10
Nl = 5
Nbands = len(bands)
enc = var_networks.Encoder(Nfeat=Nfeat, Nl=Nl, Nbands=Nbands).cuda()
dec = var_networks.Decoder(Nfeat=Nfeat, Nl=Nl, Nbands=Nbands).cuda()

if use_mdn:
    net_pz = arch_mdn.MDNNetwork(len(bands)+Nfeat).cuda()
else:
    net_pz = arch_mdn.Network(len(bands)+Nfeat).cuda()

loss_function = nn.CrossEntropyLoss()

def xloss_function(net, pred, target):        
    loss = loss_function(pred, Bzbin.cuda())
    
    return loss


# In[ ]:


#optimizer = optim.Adam(net.parameters(), lr=1e-4)
params_chain = chain(enc.parameters(), dec.parameters(), net_pz.parameters())
optimizer = optim.Adam(params_chain, lr=1e-3)

for i in range(10):
    L = []
    enc.train()
    dec.train()
    net_pz.train()
    
    t1 = time.time()
    for Bparams, Bzbin, Bflux, Bflux_err in train_dl:
        t3 = time.time()
        optimizer.zero_grad()
        
        # Testint doing the 
#        ampl = 1+0.05*torch.randn(len(Bflux))
#        Bflux = ampl[:,None]*Bflux.mean(1)[:,None] + Bflux # HACK!!!
        Bflux = Bflux.cuda()
        Bflux_err = (Bflux / SN[None,:]).cuda()
        
        noise = Bflux_err * torch.randn(Bflux.shape).cuda()
        Bflux = Bflux + noise
        feat = enc(Bflux)
        Xinp = torch.cat([Bflux, feat], 1) # In train
         #Bflux + noise)
        
        #bz_rand = smoother(log_pz)
        #Bzbin = Bzbin.cuda() + bz_rand.cuda()
        #Bzbin = Bzbin.clamp(0, 2099)
        if use_mdn:
            Bzbin = Bzbin.cuda()
            
            Bz = 0.001*Bzbin.type(torch.float)
            _, loss_pz = net_pz.loss(Xinp, Bz)  
        else:
            log_pz = net_pz(Xinp)
            loss_pz = loss_function(log_pz, Bzbin.cuda())     
        
        # And then
        loss_recon = (dec(feat) - Bflux).abs() / Bflux_err
        loss_recon = loss_recon.mean()
        
        loss = loss_pz + loss_recon
        loss.backward()
        optimizer.step()    
        L.append(loss.item())
        
        #print('T', time.time() - t3)
    
    print('time train', time.time() - t1)
    loss_train = sum(L) / len(L)
    L = []
    dxL = []
    enc.eval()
    dec.eval()
    net_pz.eval()

    print('starting test eval')
    for Bparams, Bzbin, Bflux, Bflux_err in test_dl:
        Bflux = Bflux.cuda()
        feat = enc(Bflux)
        Xinp = torch.cat([Bflux, feat], 1) # In test.
        pred = net_pz(Xinp.cuda())
        
        # For some reason we did *not* include the reconstruction loss here.
        if use_mdn:
            Bzbin = Bzbin.cuda()
            Bz = 0.001*Bzbin.type(torch.float)
            _, loss = net_pz.loss(Xinp, Bz)    
        else:
            log_pz = pred
            loss_pz = loss_function(log_pz, Bzbin.cuda())
            
        
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


# Here we use a different version than in the normal 
version = 1
sim = 'fsps'
pztype = 'mdn' if use_mdn else 'normal'
#output_dir = Path('/nfs/astro/eriksen/deepz/redux/pretrain')
output_dir = Path('/cephfs/pic.es/astro/scratch/eriksen/deepz_wide')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

    
path_base = str(output_dir / (f'v{version}'+'_{}_'+f'{pztype}.pt'))
torch.save(enc.state_dict(), path_base.format('enc'))
torch.save(dec.state_dict(), path_base.format('dec'))
torch.save(net_pz.state_dict(), path_base.format('pz'))
