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
import torch
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

import os
from pathlib import Path
import paus_data

import local_settings
import loaders
import train_funct

import user_settings


# train_funct.train(ifold, data, Ntrain, **config)
# train_funct.pz_fold(ifold, inds, data, Ntrain, **config)
# train_funct.train_all(data, Ntrain, **config) 
# train_funct.photoz_all(inds_all, data, **config)

def gen_conf():
    for catnr in range(10):
        for keep_last in [True]:
            for alpha in [0.8]:
                yield catnr, keep_last, alpha
                
"""Config.
"""
version = 9
verpretrain = 8
#Ntrain = 'all'
catnr = 0 #if len(sys.argv) == 1 else int(sys.argv[1])
keep_last = False
alpha = 0
label = 'march18'
label = 'pruebaVane'


"""Input
"""
if os.path.exists(user_settings.path):
    print('data in use', user_settings.path)
    df_cosmos = pd.read_csv(user_settings.cosmos_path, comment='#')
    df_galcat = pd.read_hdf(user_settings.galcat_path, 'cat')
    df_fa = pd.read_parquet(user_settings.indexp_path)
    data = paus_data.paus(True, df_galcat, df_fa, df_cosmos)
else:
    print('default')
    df_galcat = local_settings.galcat
    df_cosmos = local_settings.cosmos
    df_fa = local_settings.df_fa
    data = paus_data.paus(True, df_galcat, df_fa, df_cosmos)


# index
inds_all = local_settings.inds_all


if True:
    use_mdn = True
    model_dir = Path('network') / str(version)
    Ntrain = 'all'
    for catnr, keep_last, alpha in gen_conf():
        pretrain = False if verpretrain == 'no' else True
        config = {'verpretrain': verpretrain, 'Ntrain': Ntrain, 'catnr': catnr, 'use_mdn': use_mdn,
                  'Ntrain': Ntrain, 'pretrain': pretrain, 'keep_last': keep_last}

        config['Nexp'] = 0
        config['alpha'] = alpha
 
        out_fmt = '{net}_'+label+'_ifold{ifold}.pt'
        out_fmt = str(model_dir / out_fmt)

        config['out_fmt'] = out_fmt

        print('To store at:')
        print(out_fmt)

        train_funct.train_all(data, **config) 

        pz = train_funct.photoz_all(inds_all, data, **config)
        pz['dx'] = (pz.zb - pz.zs) / (1 + pz.zs)

        sig68 = 0.5*(pz.dx.quantile(0.84) - pz.dx.quantile(0.16))
        print('keep_last', keep_last, 'alpha', alpha, 'sig68', sig68)

        fname = f'{label}'+'_catnr{catnr}.csv'.format(**config)
        path_out = '../../output/pz.csv'

        pz.to_csv(path_out) 

        break
