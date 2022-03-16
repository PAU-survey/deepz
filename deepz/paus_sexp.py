#!/usr/bin/env python
# encoding: UTF8

from pathlib import Path
import numpy as np
import pandas as pd
import torch

data_in = Path('/data/astro/scratch/eriksen/deepz/input')

NB = ['NB{}'.format(x) for x in 455+10*np.arange(40)]
BB = ['cfht_u', 'subaru_B', 'subaru_V', 'subaru_r', 'subaru_i', 'subaru_z']
bands = NB + BB

D = {'photoz': '4199.csv', 'coadd': '4213.csv', 'cosmos': '4378.csv'}


def get_cosmos(apply_cuts):
    cosmos = pd.read_csv(str(data_in / D['cosmos']), comment='#')
    cosmos = cosmos.set_index('paudm_id')
    cosmos = cosmos[0 < cosmos.r50]
    
    if apply_cuts:
        cosmos = cosmos[(3 <= cosmos.conf) & (cosmos.conf <= 5)]
        cosmos = cosmos[cosmos.i_auto < 22.5]

    return cosmos

def get_indexp(inds_touse, NB_bands, indexp_path):
# Testing Lumus.
#    path = Path('/cephfs/pic.es/astro/scratch/eriksen/deepz/input/fa/comb_v7.parquet')

    df_fa = pd.read_parquet(indexp_path)
    df_fa = df_fa.set_index('ref_id')

    # Way less than 1% exposures missed.
    df_fa = df_fa[df_fa.nr < 10]

    # This operation if faster going through xarrays..
    df_fa = df_fa.reset_index().set_index(['ref_id', 'band', 'nr'])

    X = df_fa.to_xarray()
    
    X = X.sel(ref_id=inds_touse)
    X = X.sel(band=NB_bands)
    
    return X.flux.values, X.flux_error.values

def paus(apply_cuts=True):
# Testing Lumus.
#    galcat_path = '/cephfs/pic.es/astro/scratch/eriksen/deepz/input/cosmos_pau_matched_v2.h5'
    galcat_path = '/data/astro/scratch/eriksen/deepz/input/lumus/coadd_v8.h5'
    indexp_path = Path('/data/astro/scratch/eriksen/deepz/input/lumus/fa_v8.pq')

    galcat = pd.read_hdf(galcat_path, 'cat')

    sub = galcat.loc[~np.isnan(galcat.flux.subaru_i)]

    # Here we only use galaxies where all bands are observed.
    sub = sub.loc[~np.isnan(sub.flux[bands]).any(1)]

    # Overlap between the two catalogues ...
    cosmos = get_cosmos(apply_cuts)
    touse = cosmos.index & sub.index

    # Here we are not actually using the flux, but a flux
    # ratio..
    flux_df = sub.loc[touse]
    
    flux = flux_df.flux[bands].values
    flux_err = flux_df.flux_err[bands].values
    
    norm = flux[:, -2]
    flux = torch.Tensor(flux / norm[:, None])
    flux_err = torch.Tensor(flux_err / norm[:, None])

    # The individual exposures.
    fmes, emes = get_indexp(touse, NB, indexp_path)
    #fmes = np.nan_to_num(fmes, copy=False)
    #emes = np.nan_to_num(emes, copy=False)
    fmes = torch.Tensor(fmes / norm[:,None,None])
    emes = torch.Tensor(emes / norm[:,None,None])
    
    vinv = 1. / emes.pow(2)
    isnan = np.isnan(vinv)
    vinv[isnan] = 0
    fmes[isnan] = 0

    flux = torch.Tensor(flux)
    zbin = torch.tensor(cosmos.loc[touse].zspec.values / 0.001).round().type(torch.long)

    print('# Galaxies', len(flux))
    assert len(flux) == len(zbin), 'Inconsistent number of galaxies'
    
    print('here..')
    
    # Test, this makes a difference when selecting with a PyTorch
    # uint8 tensor.
    ref_id = torch.Tensor(flux_df.index.values)
   
    # Testing....
    
    #return flux, flux_err, fmes, vinv, isnan, zbin, ref_id
    return flux, flux_err, fmes, vinv, isnan, zbin, ref_id #, flux_df
