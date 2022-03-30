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

import loaders
import train_funct

import fire


# train_funct.train(ifold, data, Ntrain, **config)
# train_funct.pz_fold(ifold, inds, data, Ntrain, **config)
# train_funct.train_all(data, Ntrain, **config) 
# train_funct.photoz_all(inds_all, data, **config)

"""Input
"""
def run_photoz(input_path='../../input/', output_path='../../output/', 
                coadd_cat_path='coadd_v8.h5', cosmos_path='cosmos.csv', fmes_path='fa_v8.pq'):
    """The Deepz code. 
    """

    print('data in use', input_path)
    df_cosmos = pd.read_csv(input_path+cosmos_path, comment='#')
    df_galcat = pd.read_hdf(input_path+coadd_cat_path, 'cat')
    df_fa = pd.read_parquet(input_path+fmes_path)
    data = paus_data.paus(True, df_galcat, df_fa, df_cosmos)

    print('Calulcate photo-z')

    #: inds_all
    inds_all = np.loadtxt('inds/inds_large_v1.txt')


    #: Path where the settings.py lives
    #PATH = os.path.abspath(os.path.dirname(__file__))
    #: Local path
    #path = os.path.join('')
    #D = {'photoz': '4199.csv', 'coadd': '4213.csv', 'cosmos': '/4378.csv'}
    #data_in_path = os.path.join(path, 'input')
    #cosmos = pd.read_csv(data_in_path + D['cosmos'], comment='#')
    #: input/lumus/coadd_v8.h5
    #galcat_path = os.path.join(path, 'input/lumus/coadd_v8.h5')
    #galcat = pd.read_hdf(galcat_path, 'cat')
    #: input/lumus/fa_v8.h5
    #indexp_path = os.path.join(path, 'input/lumus/fa_v8.pq')
    # columns: ref_id band flux flux_error nr
    #df_fa = pd.read_parquet(indexp_path)
    #: redux
    #redux_path = os.path.join(path, 'redux')
    # ndarray: Data read from the text file.
    #: inds/inds_large_v1.txt
    #inds_large_v1_path = os.path.join(path, 'inds/inds_large_v1.txt')
    #: inds_all
    #inds_all = np.loadtxt(inds_large_v1_path)
    #: /redux/pretrain/v
    #pretrain_v = os.path.join(path, 'redux/pretrain/v')
    #: Ouput
    #version = 2
    #output_dir = Path(redux_path) / str(version)

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

if __name__ == '__main__':
    fire.Fire(run_photoz)
