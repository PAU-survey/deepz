#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created at 

# =============================================================================
# DOCS
# =============================================================================

"""train_data
"""

# =============================================================================
# IMPORTS
# =============================================================================

from IPython.core import debugger as ipdb
import os
import time
import numpy as np
import pandas as pd
from itertools import chain
import os
import sys

#sys.path.append('..')
#sys.path.append('../var')
sys.path.append('../code')

import torch
#assert torch.__version__.startswith('1.0'), 'For some reason the code fails badly on newer PyTorch versions.'


from torch import optim
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt

# import arch_mdn
# import paus_data
import utils

import paus_sexp as paus_data

# pretrain = True #False
"""Constants
"""
#: alpha
alpha = 0.8

#: train set size
Ntrain = 8000 #'all'

#: train set size
Ntrain = 100 #'all'

#: verpretrain
verpretrain = 3

import paus_sexp as paus_data
flux, flux_err, fmes, vinv, isnan, zbin, ref_id = paus_data.paus()


#model_dir = Path('/nfs/astro/eriksen/deepz/encmodels_data')

version = 2
output_dir = Path('/cephfs/pic.es/astro/scratch/eriksen/deepz/redux') / str(version)

# Other values collided with importing the code in a notebook...
catnr = 0 #if len(sys.argv) == 1 else int(sys.argv[1])
inds_all = np.loadtxt('/cephfs/pic.es/astro/scratch/eriksen/deepz/inds/inds_large_v1.txt')


def get_loaders(ifold, inds):
    """Get loaders for specific fold."""
    
    def sub(ix):
        ds = TensorDataset(flux[ix].cuda(), fmes[ix].cuda(), vinv[ix].cuda(), isnan[ix].cuda(), zbin[ix].cuda())
        
        return ds
    
    ix_train = torch.ByteTensor(1*(inds != ifold))
    ix_test = torch.ByteTensor(1*(inds == ifold))

    # Here we deterministically remove galaxies.
    if not Ntrain == 'all':
        # You *don't* need to use a different number because of the folds.
        Nsel = int(Ntrain)
        
        ix_train[Nsel:] = 0
        

    ds_train = sub(ix_train)
    ds_test = sub(ix_test)

    train_dl = DataLoader(ds_train, batch_size=500, shuffle=True)
    test_dl = DataLoader(ds_test, batch_size=100)
    
    return train_dl, test_dl, zbin[ix_test]


import trainer_alpha

def train(ifold, **config):
    """Train the networks for one fold.
    """
    
    verpretrain = config['verpretrain']
    pretrain = config['pretrain']

    part = 'mdn' if use_mdn else 'normal'
#    path_base = '/nfs/astro/eriksen/deepz/encmodels/v16_{}_'+part+'.pt'
    #path_base = f'/nfs/astro/eriksen/deepz/redux/5/pretrain_fsps_{part}'+'_{}.pt'
    
#    path_base = '/nfs/astro/eriksen/deepz/encmodels/v23_{}_'+part+'.pt'
#    path_base = '/nfs/astro/eriksen/deepz/encmodels/v16_{}_'+part+'.pt'
#/nfs/astro/eriksen/deepz/redux/pretrain/1

    # Where to find the pretrained files.
    path_base = f'/cephfs/pic.es/astro/scratch/eriksen/deepz/redux/pretrain/v{verpretrain}'+'_{}_'+part+'.pt'

    inds = inds_all[config['catnr']][:len(flux)]
    
    enc, dec, net_pz = utils.get_nets(path_base, use_mdn, pretrain)
    train_dl, test_dl, _ = get_loaders(ifold, inds)
    K = (enc, dec, net_pz, train_dl, test_dl, use_mdn, config['alpha'], config['Nexp'], \
         config['keep_last'])

    def params():
        return chain(enc.parameters(), dec.parameters(), net_pz.parameters())
   
    wd = 1e-4
    if True: #False: #False: #False: #True: #False: #True: #True: #False: #True: #False: #False: #True: #False: #True: #pretrain:
        optimizer = optim.Adam(params(), lr=1e-3, weight_decay=wd)
        trainer_alpha.train(optimizer, 100, *K)

    print('main train function...')
    optimizer = optim.Adam(params(), lr=1e-4, weight_decay=wd)
    trainer_alpha.train(optimizer, 200, *K)
    
    optimizer = optim.Adam(params(), lr=1e-5, weight_decay=wd)
    trainer_alpha.train(optimizer, 200, *K)

    optimizer = optim.Adam(params(), lr=1e-6, weight_decay=wd)
    trainer_alpha.train(optimizer, 200, *K)
    
    return enc, dec, net_pz

# In[8]:


def pz_fold(ifold, inds, out_fmt, use_mdn):
    """Estimate the photo-z for one fold."""
    
    # Load network..
    #model_dir = Path('/nfs/astro/eriksen/deepz/encmodels_data')
    #path = str(model_dir/ f'{ifold}.pt')
    
    # Loading the networks...
    net_base_path = out_fmt.format(ifold=ifold, net='{}')
    enc, dec, net_pz = utils.get_nets(str(net_base_path), use_mdn)
    enc.eval(), dec.eval(), net_pz.eval()
    
    _, test_dl, zbin_test = get_loaders(ifold, inds)

   
    assert isinstance(inds, torch.Tensor), 'This is required...'
 
    # OK, this needs some improvement...
    L = []
    for Bflux, Bfmes, Bvinv, Bisnan, Bzbin in test_dl:
        Bcoadd, touse = trainer_sexp.get_coadd(Bflux, Bfmes, Bvinv, Bisnan, alpha=1)
        assert touse.all()
            
        # Testing training augmentation.            
        feat = enc(Bcoadd)
        Binput = torch.cat([Bcoadd, feat], 1)
        pred = net_pz(Binput)
        
        zb_part = 0.001*pred.argmax(1).type(torch.float)
        L.append(zb_part)

    zb_fold = torch.cat(L).detach().cpu().numpy()
    zs_fold = 0.001*zbin_test.type(torch.float)

    refid_fold = ref_id[inds == ifold]
    D = {'zs': zs_fold, 'zb': zb_fold, 'ref_id': refid_fold}
    
    part = pd.DataFrame(D)
    part['ifold'] = ifold
    #part = np.vstack([zs_fold, zb_fold]).T

    return part


def train_all(**config):
    """Train all the folds."""
   
    out_fmt = config['out_fmt']
    for ifold in range(5):
        test_path = str(out_fmt.format(net='enc', ifold=ifold))
        if os.path.exists(test_path):
            print('Alread run, skipping:', test_path)
            continue
            
        print('Running for:', ifold)
        t1 = time.time()
        enc, dec, net_pz = train(ifold, **config)
        enc.eval()
        dec.eval()
        net_pz.train()
        
        print('time', time.time() - t1)
       
        #path = str(model_dir/ f'{ifold}.pt')
        torch.save(enc.state_dict(), str(out_fmt.format(net='enc', ifold=ifold)))
        torch.save(dec.state_dict(), str(out_fmt.format(net='dec', ifold=ifold)))
        torch.save(net_pz.state_dict(),  str(out_fmt.format(net='netpz', ifold=ifold)))
                   
def photoz_all(**config):
    """Run the photo-z for all folds."""
                   
    L = []
    inds = inds_all[config['catnr']][:len(flux)]

    inds = torch.Tensor(inds) # Inds_all should be a tensor in the first place.
    for ifold in range(5):
        L.append(pz_fold(ifold, inds, config['out_fmt'], config['use_mdn']))
        
    df = pd.concat(L)
    df = df.set_index('ref_id')

    return df


# In[16]:

import trainer_sexp

#model_dir = Path('models/v7')
# Where we store the models based on the data...
version = 9
sim = 'fsps'

def gen_conf():
    for catnr in range(10):
        for keep_last in [True]:
            for alpha in [0.8]:
#[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                yield catnr, keep_last, alpha

# This code for running different configurations has been moved out into a notebook.
if True: #True: #False: #False: #False: #True: #False:
    #verpretrain = 3
    #Ntrain = 8000
    #catnr = 0

    use_mdn = True
    model_dir = Path('/cephfs/pic.es/astro/scratch/eriksen/deepz/redux/train') / str(version)

    verpretrain = 8
    Ntrain = 'all'
    keep_last = False

    label = 'march11'

    for catnr, keep_last, alpha in gen_conf():
           #for Ntrain in ['all']:
        pretrain = False if verpretrain == 'no' else True
        config = {'verpretrain': verpretrain, 'Ntrain': Ntrain, 'catnr': catnr, 'use_mdn': use_mdn,
                  'Ntrain': Ntrain, 'pretrain': pretrain, 'keep_last': keep_last}

        config['Nexp'] = 0
        config['alpha'] = alpha
 
            #    config['output_dir'] = output_dir
#        out_fmt = 'pre{verpretrain}_alpha{alpha}_keep{keep_last}_catnr{catnr}'.format(**config)
        out_fmt = '{net}_'+label+'_ifold{ifold}.pt'
        out_fmt = str(model_dir / out_fmt)

        config['out_fmt'] = out_fmt

        print('To store at:')
        print(out_fmt)

        train_all(**config) 

        pz = photoz_all(**config)
        pz['dx'] = (pz.zb - pz.zs) / (1 + pz.zs)

        sig68 = 0.5*(pz.dx.quantile(0.84) - pz.dx.quantile(0.16))
        print('keep_last', keep_last, 'alpha', alpha, 'sig68', sig68)

        fname = f'{label}'+'_catnr{catnr}.csv'.format(**config)
        path_out = model_dir / fname

        pz.to_csv(path_out) 

        # By now we only want to run one catalogue.

        break
    #cat_out = str(output_dir / f'pzcat_{catnr}_mdn.csv') #'/nfs/pic.es/user/e/eriksen/papers/deepz/sims/cats/pzcat_v65_mdn.csv'
    #pz.to_csv(cat_out)
