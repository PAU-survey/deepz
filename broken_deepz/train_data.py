#!/usr/bin/env python
# coding: utf-8

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


from torch import optim, nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt

#import arch_mdn
#import paus_data
import utils

import paus_sexp as paus_data

#pretrain = True #False
alpha = 0.8
#Ntrain = 8000 #'all'
Ntrain = 'all' #100 #'all'
verpretrain = 3

field = 'W3'
field = 'cosmos'

if field == 'cosmos':
    # Just because of using some old files.
    NB = ['NB{}'.format(x) for x in 455+10*np.arange(40)]

    BB = ['cfht_u', 'subaru_B', 'subaru_V', 'subaru_r', 'subaru_i', 'subaru_z']
    test_bb = 'subaru_i'
elif field == 'W3':
    NB = ['pau_nb{}'.format(x) for x in 455+10*np.arange(40)]
    BB = [f'cfht_{x}' for x in 'ugriz']
    test_bb = 'cfht_i' 

bands = NB + BB

flux, flux_err, fmes, vinv, isnan, zbin, ref_id = paus_data.paus(bands, field, test_bb=test_bb)


#model_dir = Path('/nfs/astro/eriksen/deepz/encmodels_data')
version = 7
#output_dir = Path('/cephfs/pic.es/astro/scratch/eriksen/deepz/models') / str(version)
output_dir = Path('/cephfs/pic.es/astro/scratch/eriksen/deepz/redux') / str(version)


# Other values collided with importing the code in a notebook...
catnr = 0 #if len(sys.argv) == 1 else int(sys.argv[1])
inds_all = np.loadtxt('/cephfs/pic.es/astro/scratch/eriksen/deepz/inds/inds_large_v1.txt')



# In[6]:
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

    train_dl = DataLoader(ds_train, batch_size=100, shuffle=True)
    test_dl = DataLoader(ds_test, batch_size=100)
    
    return train_dl, test_dl, zbin[ix_test]


# In[7]:
import trainer_alpha
import var_networks

def train(ifold, **config):
    """Train the networks for one fold."""
    
    verpretrain = config['verpretrain']
    pretrain = config['pretrain']

    part = 'mdn' if use_mdn else 'normal'
    # Where to find the pretrained files.
    path_base = f'/cephfs/pic.es/astro/scratch/eriksen/deepz/redux/pretrain/v{verpretrain}'+'_{}_'+part+'.pt'

#    path_base = '/cephfs/pic.es/astro/scratch/eriksen/deepz_wide/pretrain/v1_{}_'+part+'.pt'

    inds = inds_all[config['catnr']][:len(flux)]
    
    enc, dec, net_pz = utils.get_nets(path_base, use_mdn, pretrain, Nbands=len(bands))
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
    enc, dec, net_pz = utils.get_nets(str(net_base_path), use_mdn, Nbands=len(bands))
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
version = 7
#sim = 'fsps'

#model_dir = Path('/nfs/astro/eriksen/deepz/redux/train') / str(version)
model_dir = Path('/cephfs/pic.es/astro/scratch/eriksen/deepz/models') / str(version)

pretrain = True #False if verpretrain == 'no' else True
use_mdn = True
keep_last = True

config = {'verpretrain': verpretrain, 'Ntrain': Ntrain, 'catnr': catnr, 'use_mdn': use_mdn,
          'Ntrain': Ntrain, 'pretrain': pretrain, 'keep_last': keep_last}

config['Nexp'] = 0
config['alpha'] = alpha

    #    config['output_dir'] = output_dir
out_fmt = 'pre{verpretrain}_alpha{alpha}_keep{keep_last}_catnr{catnr}'.format(**config)
out_fmt = '{net}_'+out_fmt+'_ifold{ifold}.pt'
out_fmt = str(model_dir / out_fmt)

config['out_fmt'] = out_fmt

print('To store at:')
print(out_fmt)

train_all(**config) 

pz = photoz_all(**config)
pz['dx'] = (pz.zb - pz.zs) / (1 + pz.zs)

sig68 = 0.5*(pz.dx.quantile(0.84) - pz.dx.quantile(0.16))
print('keep_last', keep_last, 'alpha', alpha, 'sig68', sig68)

#ipdb.set_tace()
