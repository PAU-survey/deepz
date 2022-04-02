#!/usr/bin/env python
# encoding: UTF8

# Train the network on observed PAUS data.
from IPython.core import debugger as ipdb
import os
import time
import numpy as np
import pandas as pd
from itertools import chain
import os
import sys

import torch
assert torch.__version__.startswith('1.0'), 'For some reason the code fails badly on newer PyTorch versions.'

from torch import optim, nn
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt

import paus_data
import trainer
import utils



flux, flux_err, fmes, vinv, isnan, zs, ref_id = paus_data.paus()

# Other values collided with importing the code in a notebook...
catnr = 0 #if len(sys.argv) == 1 else int(sys.argv[1])
inds_all = np.loadtxt('/data/astro/scratch/eriksen/deepz/inds/inds_large_v1.txt')


def get_loaders(ifold, inds):
    """Create data loaders for a specific fold.
       :param ifold: {int} Which fold to use.
       :param inds: {array} Ifold for each galaxy.
    """
    
    def sub(ix):
        """Select subset.
           :param ix: {array} Array indices to use.
        """
        ds = TensorDataset(flux[ix].cuda(), fmes[ix].cuda(), vinv[ix].cuda(), isnan[ix].cuda(), zs[ix].cuda())
        
        return ds
    
    ix_train = torch.ByteTensor(1*(inds != ifold))
    ix_test = torch.ByteTensor(1*(inds == ifold))

    ds_train = sub(ix_train)
    ds_test = sub(ix_test)

    train_dl = DataLoader(ds_train, batch_size=500, shuffle=True)
    test_dl = DataLoader(ds_test, batch_size=100)
    
    return train_dl, test_dl, zs[ix_test]


def train(ifold, **config):
    """Train the networks for one fold.
       :param config: {dict} Dictionary with the configuration.
    """
    
    verpretrain = config['verpretrain']
    pretrain = config['pretrain']

    # Where to find the pretrained files.
    path_base = f'/data/astro/scratch/eriksen/deepz/redux/pretrain/v{verpretrain}'+'_{}.pt'

    inds = inds_all[config['catnr']][:len(flux)]
    
    enc, dec, net_pz = utils.get_nets(path_base, pretrain)
    train_dl, test_dl, _ = get_loaders(ifold, inds)
    K = (enc, dec, net_pz, train_dl, test_dl, config['alpha'], config['keep_last'])

    def params():
        return chain(enc.parameters(), dec.parameters(), net_pz.parameters())
   
    wd = 1e-4
    if True: # Since I tested this so many times.
        optimizer = optim.Adam(params(), lr=1e-3, weight_decay=wd)
        trainer.train(optimizer, 100, *K)

    print('main train function...')
    optimizer = optim.Adam(params(), lr=1e-4, weight_decay=wd)
    trainer.train(optimizer, 200, *K)
    
    optimizer = optim.Adam(params(), lr=1e-5, weight_decay=wd)
    trainer.train(optimizer, 200, *K)

    optimizer = optim.Adam(params(), lr=1e-6, weight_decay=wd)
    trainer.train(optimizer, 200, *K)
    
    return enc, dec, net_pz


def pz_fold(ifold, inds, out_fmt):
    """Estimate the photo-z for one fold.
       :param ifold: {int} Which ifold to use.
       :param inds: {array} Indices to use.
       :param out_fmt: {str} Format of output path.
    """
    
    # Loading the networks...
    net_base_path = out_fmt.format(ifold=ifold, net='{}')
    enc, dec, net_pz = utils.get_nets(str(net_base_path))
    enc.eval(), dec.eval(), net_pz.eval()
    
    _, test_dl, zs_test = get_loaders(ifold, inds)

    assert isinstance(inds, torch.Tensor), 'This is required...'
 
    L = []
    for Bflux, Bfmes, Bvinv, Bisnan, Bzs in test_dl:
        Bcoadd, touse = trainer.get_coadd_allexp(Bflux, Bfmes, Bvinv, Bisnan)
        assert touse.all()
            
        feat = enc(Bcoadd)
        Binput = torch.cat([Bcoadd, feat], 1)
        pred = net_pz(Binput)
        
        zb_part = 0.001*pred.argmax(1).type(torch.float)
        L.append(zb_part)

    zb_fold = torch.cat(L).detach().cpu().numpy()
    zs_fold = zs_test

    refid_fold = ref_id[inds == ifold]
    D = {'zs': zs_fold, 'zb': zb_fold, 'ref_id': refid_fold}
    
    part = pd.DataFrame(D)
    part['ifold'] = ifold

    return part


def train_all(**config):
    """Train all the folds.
       :param config: {dict} Configuration dictionary.

    """
   
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
      
        print('where getting stored...', str(out_fmt.format(net='enc', ifold=ifold))) 
        torch.save(enc.state_dict(), str(out_fmt.format(net='enc', ifold=ifold)))
        torch.save(dec.state_dict(), str(out_fmt.format(net='dec', ifold=ifold)))
        torch.save(net_pz.state_dict(),  str(out_fmt.format(net='netpz', ifold=ifold)))
                   
def make_catalogue(catnr, out_fmt):
    """Run the photo-z for all folds.
       :param catnr: {int} Which of the indexes to use per fold.
       :param out_fmt: {str} FIX THIS!
    """
                   
    L = []
    inds = inds_all[catnr][:len(flux)]

    inds = torch.Tensor(inds) # Inds_all should be a tensor in the first place.
    for ifold in range(5):
        L.append(pz_fold(ifold, inds, out_fmt))
        
    df = pd.concat(L)
    df = df.set_index('ref_id')

    return df


def photoz_all(out_fmt, verpretrain, catnr=0, pretrain=True, alpha=0.8, keep_last=True):
    """Train the networks and return the catalogs.
       :param out_fmt: {str} Where to store the models.
       :param verpretrain: {int} Version of the pretrained network.
       :param catnr: {int} Which of the indexes to use per fold.
       :param pretrain: {bool} If using a pretrained network.
       :param alpha: {float} Fraction of measurements used when training.
       :param keep_last: {bool} Keeping at least one measurement per band.
    """

    # This part still needs to be cleaned.
    config = {'out_fmt': out_fmt, 'verpretrain': verpretrain, 'catnr': catnr, 'pretrain': pretrain,
              'alpha': alpha, 'keep_last': keep_last}

    train_all(**config)
    pz = make_catalogue(catnr, out_fmt)
    pz['dx'] = (pz.zb - pz.zs) / (1 + pz.zs)

    sig68 = 0.5*(pz.dx.quantile(0.84) - pz.dx.quantile(0.16))
    print('sig68:', sig68)

    return pz



version = 13
label = 'april02'
verpretrain = 8

model_dir = Path('/data/astro/scratch/eriksen/deepz/redux/train') / str(version)
out_fmt = '{net}_'+label+'_ifold{ifold}.pt'
out_fmt = str(model_dir / out_fmt)

pz = photoz_all(out_fmt, verpretrain)
