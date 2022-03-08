#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created at 2021-05-27T16:48:32.642811 by corral 0.3


# =============================================================================
# DOCS
# =============================================================================

"""Global configuration for deepz."""


# =============================================================================
# IMPORTS
# =============================================================================

import os

# =============================================================================
# CONFIGURATIONS
# =============================================================================

#: Path where the settings.py lives
PATH = os.path.abspath(os.path.dirname(__file__))


#path = os.path.join('/', 'cephfs', 'pic.es', 'data', 'astro', 'scratch', 'eriksen', 'deepz')
# Local path
path = os.path.join('/', 'data', 'astro', 'scratch', 'idazaper', 'deepz')

##################
# paus_sexp.py   #
##################

#: input 
data_in_path = os.path.join(path, 'input')

#: input/lumus/coadd_v8.h5
galcat_path = os.path.join(path, 'input/lumus/coadd_v8.h5 ')

#: input/lumus/coadd_v8.h5
indexp_path = os.path.join(path, 'input/lumus/fa_v8.pq')

#################
# train_data.py #
#################

#: redux
redux_path = os.path.join(path, 'redux')

# ndarray: Data read from the text file.
#: inds/inds_large_v1.txt
inds_large_v1_path = os.path.join(path, 'inds/inds_large_v1.txt')

#: /redux/pretrain/v
pretrain_v_path = os.path.join(path, 'redux/pretrain/v')

