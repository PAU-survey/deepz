#!/usr/bin/env python
# encoding: UTF8

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

import numpy as np
import pandas as pd
from itertools import chain

import sys
import time
import trainer_alpha
import trainer_sexp
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

import os
from pathlib import Path
import paus_data

from matplotlib import pyplot as plt
import utils

import local_settings
import loaders

#sys.path.append('../code')


def train(ifold, data, **config):
    """Train the networks for one fold.

    :param ifold: id of a sample

    :type ifold: tensor
    
    :param data: Paus data
    
    :type data: tensors
    
    :param Ntrain: train size
    
    :type Ntrain: int

    :param config: This parameter has several conf of the train.

    :type config: dictionary

    :return: networks

    :rtype:

    """
    
    #: data
    flux, flux_err, fmes, vinv, isnan, zbin, ref_id = data
    
    #: Loading the pretrained network.
    #: /redux/pretrain/v
    pretrain_v_path = local_settings.pretrain_v
    verpretrain = config['verpretrain']
    part = 'mdn' if config['use_mdn'] else 'normal'
    path_base = pretrain_v_path + f'{verpretrain}' + '_{}_' + part + '.pt'
    enc, dec, net_pz = utils.get_nets(path_base, config['use_mdn'], config['pretrain'])
    
    #: Indices of the selected sources with flow information and 
    inds_all = np.loadtxt(local_settings.inds_large_v1_path)
    inds = inds_all[config['catnr']][:len(flux)]
    
    #: Train and test Samples (sample_Kfold, mask, data, train set size)
    train_dl, test_dl, _ = loaders.get_loaders(ifold, inds, data, config['Ntrain'])

    #: Model, samples, config
    K = (enc, dec, net_pz, train_dl, test_dl, config['use_mdn'], config['alpha'], config['Nexp'], \
         config['keep_last'])

    #: Networks parameters
    def params():
        return chain(enc.parameters(), dec.parameters(), net_pz.parameters())
   
    #:wd
    wd = 1e-4
    
    #: trainer_alpha.train with the better hyperparameters, optimizer=params, lrm wd), N?, K=Model, samples, config
    if True: 
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


def pz_fold(ifold, inds, data, **config):
    """Estimate the photo-z for one fold.
       to predict photometric redshifts receives both the 
       encoded latent variables and the original input flux 
       ratios. 

    :param ifold: id of a sample

    :type ifold: tensor
    
    :param inds: Indices of the selected sources with flux information an
    
    :type inds: tensor
    
    :param data: Paus data
    
    :type data: tensors

    :param config: This parameter has several conf of the train.

    :type config: dictionary

    :return: Dataframe with photo and spec redshifts and id

    :rtype: DataFrame

    """

    #: data
    flux, flux_err, fmes, vinv, isnan, zbin, ref_id = data
    
    # Loading the networks...
    net_base_path = config['out_fmt'].format(ifold=ifold, net='{}')
    enc, dec, net_pz = utils.get_nets(str(net_base_path), config['use_mdn'])
    enc.eval(), dec.eval(), net_pz.eval()

    # Loading test data
    _, test_dl, zbin_test = loaders.get_loaders(ifold, inds, data, config['Ntrain'])

    assert isinstance(inds, torch.Tensor)
 
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

    
    #refid_fold = ref_id[1*(inds == ifold).type(torch.bool)]
    refid_fold = ref_id[inds == ifold]
    print(zb_fold.shape, zs_fold.shape, refid_fold.shape)
    D = {'zs': zs_fold, 'zb': zb_fold, 'ref_id': refid_fold}
    
    part = pd.DataFrame(D)
    part['ifold'] = ifold

    return part


def train_all(data, **config):
    """Train all the folds and saves the results.
    
    :param config: This parameter has several conf of the train.
    :type config: dictionary
    """
   
    out_fmt = config['out_fmt']
    for ifold in range(5):
        test_path = str(out_fmt.format(net='enc', ifold=ifold))
        if os.path.exists(test_path):
            print('Alread run, skipping:', test_path)
            continue
            
        print('Running for:', ifold)
        t1 = time.time()
        enc, dec, net_pz = train(ifold, data, **config)
        enc.eval()
        dec.eval()
        net_pz.train()
        
        print('time', time.time() - t1)
       
        #path = str(model_dir/ f'{ifold}.pt')
        torch.save(enc.state_dict(), str(out_fmt.format(net='enc', ifold=ifold)))
        torch.save(dec.state_dict(), str(out_fmt.format(net='dec', ifold=ifold)))
        torch.save(net_pz.state_dict(),  str(out_fmt.format(net='netpz', ifold=ifold)))
        print(torch.save(net_pz.state_dict(),  str(out_fmt.format(net='netpz', ifold=ifold))))


def photoz_all(inds_all, data, **config):
    """Run the photo-z for all folds.
    
    :param inds_all: 

    :type inds_all:
    
    :param data:
    
    :type data:
    
    :param config: This parameter has several conf of the train.

    :type config: dictionary
    
    :return: Dataframe with  photo and spec redshifts and id

    :rtype: DataFrame
    """
    
    #: data
    flux, flux_err, fmes, vinv, isnan, zbin, ref_id = data
                   
    L = []
    inds = inds_all[config['catnr']][:len(flux)]

    inds = torch.Tensor(inds) # Inds_all should be a tensor in the first place.
    
    for ifold in range(5):
        L.append(pz_fold(ifold, inds, data, **config))
        
    df = pd.concat(L)
    df = df.set_index('ref_id')

    return df
