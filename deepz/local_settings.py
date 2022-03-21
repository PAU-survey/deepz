#!/usr/bin/env python
# encoding: UTF8

# Created at 2021-05-27T16:48:32.642811 by corral 0.3


# =============================================================================
# DOCS
# =============================================================================

"""Global configuration for deepz."""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# CONFIGURATIONS
# =============================================================================

#: Path where the settings.py lives
PATH = os.path.abspath(os.path.dirname(__file__))


#path = os.path.join('/', 'cephfs', 'pic.es', 'data', 'astro', 'scratch', 'eriksen', 'deepz')
# Local path
path = os.path.join('/data/astro/scratch/eriksen/deepz/')

# For training on data.

#: input 
D = {'photoz': '4199.csv', 'coadd': '4213.csv', 'cosmos': '/4378.csv'}
data_in_path = os.path.join(path, 'input')
cosmos = pd.read_csv(data_in_path + D['cosmos'], comment='#')

#: input/lumus/coadd_v8.h5
galcat_path = os.path.join(path, 'input/lumus/coadd_v8.h5 ')
galcat_path = '/data/astro/scratch/eriksen/deepz/input/lumus/coadd_v8.h5'
galcat = pd.read_hdf(galcat_path, 'cat')


#: input/lumus/fa_v8.h5
indexp_path = os.path.join(path, 'input/lumus/fa_v8.pq')
df_fa = pd.read_parquet(indexp_path)

#################
# config        #
#################

#: verpretrain: is the version pretrain
#verpretrain = config['verpretrain']

#: pretrain is pretrain, (boolean)
#pretrain = config['pretrain']

#: part
#part = 'mdn' if config['use_mdn'] else 'normal'

#config = {'verpretrain': verpretrain, 'Ntrain': Ntrain, 'catnr': catnr, 'use_mdn': use_mdn,
#                  'Ntrain': Ntrain, 'pretrain': pretrain, 'keep_last': keep_last}





#################
# train_data.py #
#################

#: redux
redux_path = os.path.join(path, 'redux')

# ndarray: Data read from the text file.
#: inds/inds_large_v1.txt
inds_large_v1_path = os.path.join(path, 'inds/inds_large_v1.txt')
#: inds_all
inds_all = np.loadtxt(inds_large_v1_path)

#: /redux/pretrain/v
pretrain_v = os.path.join(path, 'redux/pretrain/v')



#: Ouput
version = 2
output_dir = Path(redux_path) / str(version)
