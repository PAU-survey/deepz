#!/usr/bin/env python
# encoding: UTF8

# The main photo-z code.

from IPython.core import debugger as ipdb
import os
import time
import numpy as np
import pandas as pd
from itertools import chain
import os
import sys

import torch
from torch import optim, nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt

#import arch_mdn
#import paus_data
import utils

import paus_sexp as paus_data
import trainer_sexp
import var_networks

pretrain = True #False
#pretrain = False

alpha = 0.8

# For the CFHTls fields.
NB = ['pau_nb{}'.format(x) for x in 455+10*np.arange(40)]
BB = [f'cfht_{x}' for x in 'ugriz']
bands = NB + BB

field = 'W3'
flux, flux_err, fmes, vinv, isnan, zbin, ref_id = paus_data.paus(bands, field, test_bb='cfht_i')


version = 1
output_dir = Path('/cephfs/pic.es/astro/scratch/eriksen/deepz_wide/models') / str(version)

catnr = 0 if len(sys.argv) == 1 else int(sys.argv[1])
inds_all = np.loadtxt('/nfs/astro/eriksen/deepz/inds/inds_large_v1.txt')

inds = inds_all[catnr][:len(flux)]

def get_loaders(ifold):
    """Get loaders for specific fold."""
    
    def sub(ix):
        ds = TensorDataset(flux[ix].cuda(), fmes[ix].cuda(), vinv[ix].cuda(),                            isnan[ix].cuda(), zbin[ix].cuda())
        
        return ds
    
    ix_train = torch.ByteTensor(1*(inds != ifold))
    ix_test = torch.ByteTensor(1*(inds == ifold))
    
    ds_train = sub(ix_train)
    ds_test = sub(ix_test)


    #train_dl = DataLoader(ds_train, batch_size=500, shuffle=True)
    train_dl = DataLoader(ds_train, batch_size=100, shuffle=True)
    test_dl = DataLoader(ds_test, batch_size=100)
    
    return train_dl, test_dl, zbin[ix_test]


def train(ifold, use_mdn):
    part = 'mdn' if use_mdn else 'normal'

    # COSMOS run..
#    path_base = '/nfs/astro/eriksen/deepz/redux/pretrain/v3_{}_'+part+'.pt'

    path_base = '/cephfs/pic.es/astro/scratch/eriksen/deepz_wide/pretrain/v1_{}_'+part+'.pt'

    enc, dec, net_pz = utils.get_nets(path_base, use_mdn, pretrain, Nbands=len(bands))
    train_dl, test_dl, _ = get_loaders(ifold)
    K = (enc, dec, net_pz, train_dl, test_dl, use_mdn, alpha)

    def params():
        return chain(enc.parameters(), dec.parameters(), net_pz.parameters())
   
    wd = 1e-4
    if False: #True: #False: #False: #False: #True: #False: #True: #True: #False: #True: #False: #False: #True: #False: #True: #pretrain:
        optimizer = optim.Adam(params(), lr=1e-3, weight_decay=wd)
        trainer_sexp.train(optimizer, 100, *K)

    print('main train function...')
    optimizer = optim.Adam(params(), lr=1e-4, weight_decay=wd)
    trainer_sexp.train(optimizer, 200, *K)
    
    optimizer = optim.Adam(params(), lr=1e-5, weight_decay=wd)
    trainer_sexp.train(optimizer, 200, *K)

    optimizer = optim.Adam(params(), lr=1e-6, weight_decay=wd)
    trainer_sexp.train(optimizer, 200, *K)
    
    return enc, dec, net_pz

# In[8]:


def pz_fold(ifold, use_mdn):
    # Load network..
    #model_dir = Path('/nfs/astro/eriksen/deepz/encmodels_data')
    #path = str(model_dir/ f'{ifold}.pt')
    
    # Loading the networks...
    net_base_path = output_dir / ('{}_'+str(catnr) + '_' +str(ifold)+'.pt')
    enc, dec, net_pz = utils.get_nets(str(net_base_path), use_mdn, Nbands=len(bands))
    enc.eval(), dec.eval(), net_pz.eval()
    
    #net = arch.Network(flux.shape[1]).cuda()
    #net.load_state_dict(torch.load(path))
    #net = net.eval()
    _, test_dl, zbin_test = get_loaders(ifold)
    
    # OK, this needs some improvement...
    L = []
    
    for Bflux, Bfmes, Bvinv, Bisnan, Bzbin in test_dl:
        Bcoadd, touse = trainer_sexp.get_coadd(Bflux, Bfmes, Bvinv, Bisnan, 1)
        assert touse.all()
            
        # Testing training augmentation.            
        feat = enc(Bcoadd)
        Binput = torch.cat([Bcoadd, feat], 1)
        pred = net_pz(Binput)
        
        zb_part = 0.001*pred.argmax(1).type(torch.float)
        L.append(zb_part)

    zb_fold = torch.cat(L).detach().cpu().numpy()
    zs_fold = 0.001*zbin_test.type(torch.float)


    refid_fold = ref_id.numpy()[inds == ifold]

    D = {'zs': zs_fold, 'zb': zb_fold, 'ref_id': refid_fold}
    
    part = pd.DataFrame(D)
    part['ifold'] = ifold
    #part = np.vstack([zs_fold, zb_fold]).T

    return part


def train_all(use_mdn):
    """Train all the folds."""
    for ifold in range(5):
        print('Running for:', ifold)
        t1 = time.time()
        enc, dec, net_pz = train(ifold, use_mdn)
        enc.eval()
        dec.eval()
        net_pz.train()
        
        print('time', time.time() - t1)
        
        #path = str(model_dir/ f'{ifold}.pt')
        torch.save(enc.state_dict(), str(output_dir/ f'enc_{catnr}_{ifold}.pt'))
        torch.save(dec.state_dict(), str(output_dir/ f'dec_{catnr}_{ifold}.pt'))
        torch.save(net_pz.state_dict(), str(output_dir/ f'netpz_{catnr}_{ifold}.pt'))

def photoz_all(use_mdn):
    
    L = []
    for ifold in range(5):
        t1 = time.time()
        L.append(pz_fold(ifold, use_mdn))
        print('time', time.time() - t1)
        
    df = pd.concat(L)
    
    return df


# In[16]:
#model_dir = Path('models/v7')
# Where we store the models based on the data...
#version = 3
#sim = 'fsps'


import trainer_sexp

use_mdn = True
train_all(use_mdn)

pz = photoz_all(use_mdn)
pz['dx'] = (pz.zb - pz.zs) / (1 + pz.zs)

sig68 = 0.5*(pz.dx.quantile(0.84) - pz.dx.quantile(0.16))
print('sig68', sig68)

# pz.to_csv('/cephfs/pic.es/astro/scratch/eriksen/cat/w3_test.csv')
ipdb.set_trace()

cat_out = str(output_dir / f'pzcat_{catnr}_mdn.csv') #'/nfs/pic.es/user/e/eriksen/papers/deepz/sims/cats/pzcat_v65_mdn.csv'
pz.to_csv(cat_out)
