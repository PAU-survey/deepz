#!/usr/bin/env python
# encoding: UTF8

from IPython.core import debugger as ipdb
from pathlib import Path
import numpy as np
import pandas as pd
import torch

data_in = Path('/cephfs/pic.es/astro/scratch/eriksen/deepz/input')

NB = ['NB{}'.format(x) for x in 455+10*np.arange(40)]
BB = ['cfht_u', 'subaru_B', 'subaru_V', 'subaru_r', 'subaru_i', 'subaru_z']
bands = NB + BB

D = {'photoz': '4199.csv', 'coadd': '4213.csv', 'cosmos': '4378.csv'}


def get_cosmos(apply_cuts):
    """Load the COSMOS catalogue to have spectra."""

    cosmos = pd.read_csv(str(data_in / D['cosmos']), comment='#')
    cosmos = cosmos.set_index('paudm_id')
    cosmos = cosmos[0 < cosmos.r50]
    
    if apply_cuts:
        cosmos = cosmos[(3 <= cosmos.conf) & (cosmos.conf <= 5)]
        cosmos = cosmos[cosmos.i_auto < 22.5]

    return cosmos

def get_cfhtls(apply_cuts):
    """Load the CFHTls catalogue."""

    path = '/cephfs/pic.es/astro/scratch/eriksen/deepz_wide/input/deep2.h5'
    df = pd.read_hdf(path, 'cat')

    # It seems this catalogue already had z-spec quality cuts applied.
    if apply_cuts:
        df = df[df.magi < 22.5]

    return df


def get_indexp(inds_touse, NB_bands, field):
    # COSMOS field
    field = field.lower()
    if field == 'cosmos':
        path = Path('/cephfs/pic.es/astro/scratch/eriksen/deepz/input/fa/comb_v7.parquet')
    elif field == 'w3':
        # W3 field
        path = Path('/cephfs/pic.es/astro/scratch/eriksen/deepz_wide/input/fa/fa_w3deep2_v1.pq')

    df_fa = pd.read_parquet(path)
    df_fa = df_fa.set_index('ref_id')

    df_fa = df_fa.loc[inds_touse]

    # Way less than 1% exposures missed.
    df_fa = df_fa[df_fa.nr < 10]

    # This operation if faster going through xarrays..
    df_fa = df_fa.reset_index().set_index(['ref_id', 'band', 'nr'])

    X = df_fa.to_xarray()

    ipdb.set_trace()

    X = X.sel(band=NB_bands)
    
    return X.flux.values, X.flux_error.values

def paus(bands, field, apply_cuts=True, test_bb='subaru_i'):
    # COSMOS
    if field == 'cosmos':
        galcat_path = '/cephfs/pic.es/astro/scratch/eriksen/deepz/input/cosmos_pau_matched_v2.h5'
        galcat = pd.read_hdf(galcat_path, 'cat')
        galcat = galcat.rename(columns={'flux_err': 'flux_error'})
    elif field.lower() == 'w3':
        galcat_path = '/cephfs/pic.es/astro/scratch/eriksen/deepz_wide/input/w3_memba941_v3.h5'
        galcat = pd.read_hdf(galcat_path, 'cat')


    sub = galcat.loc[~np.isnan(galcat.flux[test_bb])]

    # Here we only use galaxies where all bands are observed.
    sub = sub.loc[~np.isnan(sub.flux[bands]).any(1)]

    # Overlap between the two catalogues ...
    if field == 'COSMOS':
        get_parent = get_cosmos
    else:
        get_parent = get_cfhtls


    parent_cat = get_parent(apply_cuts)

    touse = parent_cat.index & sub.index

    # Here we are not actually using the flux, but a flux
    # ratio..
    flux_df = sub.loc[touse]
    
    flux = flux_df.flux[bands].values
    flux_error = flux_df.flux_error[bands].values
    
    norm = flux[:, -2]
    flux = torch.Tensor(flux / norm[:, None])
    flux_error = torch.Tensor(flux_error / norm[:, None])

    # The individual exposures.
    fmes, emes = get_indexp(touse, NB, field)
    #fmes = np.nan_to_num(fmes, copy=False)
    #emes = np.nan_to_num(emes, copy=False)
    fmes = torch.Tensor(fmes / norm[:,None,None])
    emes = torch.Tensor(emes / norm[:,None,None])
    
    vinv = 1. / emes.pow(2)
    isnan = np.isnan(vinv)
    vinv[isnan] = 0
    fmes[isnan] = 0

    flux = torch.Tensor(flux)
    zbin = torch.tensor(parent_cat.loc[touse].zspec.values / 0.001).round().type(torch.long)

    print('# Galaxies', len(flux))
    assert len(flux) == len(zbin), 'Inconsistent number of galaxies'
    
    print('here..')
    
    # Test, this makes a difference when selecting with a PyTorch
    # uint8 tensor.
    #ref_id = torch.Tensor(flux_df.index.values)
   
    ref_id = torch.tensor(flux_df.index.values)

    # Hacking in a scaling relation to test...
    flux[:40] /= 0.625
    flux_error[:40] /= 0.625
    fmes /= 0.625 
    vinv *= 0.625**2

#    ipdb.set_trace()
 
    return flux, flux_error, fmes, vinv, isnan, zbin, ref_id
