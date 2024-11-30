#!/usr/bin/env python
# encoding: UTF8

# Copyright (C) 2022 Martin B. Eriksen
# This file is part of Deepz <https://github.com/PAU-survey/deepz>.
#
# Deepz is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Deepz is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Deepz.  If not, see <http://www.gnu.org/licenses/>.

# The PAUS data in the COSMOS field.

from IPython.core import debugger as ipdb
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import torch

#from . import utils

data_in = Path('/data/astro/scratch/eriksen/deepz/input')
D = {'photoz': '4199.csv', 'coadd': '4213.csv', 'cosmos': '4378.csv'}


def get_cosmos(apply_cuts):
    """Return the COSMOS catalogue.
       :param apply_cuts: {bool} If applying a cut.
    """

    cosmos = pd.read_csv(str(data_in / D['cosmos']), comment='#')
    cosmos = cosmos.set_index('paudm_id')
    cosmos = cosmos[0 < cosmos.r50]
    
    if apply_cuts:
        cosmos = cosmos[(3 <= cosmos.conf) & (cosmos.conf <= 5)]
        cosmos = cosmos[cosmos.i_auto < 22.5]

    return cosmos

def to_array(df_fa):
    """Convert individual exposures dataframe into arry structure."""
    
    df_fa = df_fa.set_index('ref_id')

    # Way less than 1% exposures missed.
    df_fa = df_fa[df_fa.nr < 10]

    # This operation if faster going through xarrays..
    df_fa = df_fa.reset_index().set_index(['ref_id', 'band', 'nr'])

    X = df_fa.to_xarray()

    return X
    
def get_indexp(inds_touse, NB_bands, indexp_path):
    """Load the individual exposure data.
       :param inds_touse: {array} Reference IDs to use.
       :param NB_bands: {list} Narrow bands to use.
       :param indexp_path: {str} Path to the individual exposures file.
    """

    indexp_path = Path(indexp_path)
    suffix = indexp_path.suffix
    if suffix == '.nc':
        X = xr.open_dataset(indexp_path)
    elif indexp_path.suffix == '.pq':
        df_fa = pd.read_parquet(indexp_path)
        X = to_array(df_fa)
    elif suffix == '.csv':
        df_fa = pd.read_csv(indexp_path)
        X = to_array(df_fa)

    #return X
    
    X = X.sel(ref_id=inds_touse)
    X = X.sel(band=NB_bands)

    return X.flux.values, X.flux_error.values

def replace(bb, f, t):
    """Replace occurence of f by t.
       :param bb: {list} List of bands.
       :param f: {str} From string.
       :param t: {str} To string.
    """

    return [(t if x == f else x) for x in bb]

def paus_data(bb, field, d_data=None, path_galcat=None, path_indexp=None, apply_cuts=True, norm_band='flux_i'):
    """The PAUS data.
       :param bb: {list} List of the broad bands.
       :param apply_cuts: {bool} If applying the cuts.
       :param norm_band: {str} Band used for the normalization.
    """


    field = field.lower()
    assert field in ['cosmos', 'w1_w3']
    NB = ['NB{}'.format(x) for x in 455+10*np.arange(40)]

    if d_data:
        d_data = Path(d_data)
        path_galcat = d_data / f'{field}_train.pq'
        path_indexp = d_data / f'indexp_calib_{field}.nc'
    else:
        assert path_galcat and path_indexp, 'Either specify d_data or path_galcat and path_indexp.'

    # There are always some exceptions to handle.
    BB = replace(bb, 'subaru_b', 'subaru_B')
    BB = replace(BB, 'subaru_v', 'subaru_V')
    bands = NB + BB


    if field == 'w1_w3':
        galcat = pd.read_parquet(path_galcat)

        # Not quite sure why Vanessa avoided this galaxy, but
        # doing the same.
        galcat = galcat[galcat.ref_id != 20280229]
        touse = galcat.ref_id.values

        galcat = galcat.set_index('ref_id')
        flux_df = galcat.loc[touse]

        flux = flux_df[bands].values
        NBerr = [x.replace('NB', 'NBerr') for x in NB]
        BBerr = [x.replace('flux', 'fluxerr') for x in BB]

        flux_err = flux_df[NBerr+BBerr].values

    if field == 'cosmos':
        galcat_path = '/data/astro/scratch/eriksen/deepz/input/lumus/coadd_v8.h5'
        indexp_path = Path('/data/astro/scratch/eriksen/deepz/input/lumus/fa_v8.pq')
        galcat = pd.read_hdf(galcat_path, 'cat')
        sub = galcat.loc[~np.isnan(galcat.flux.subaru_i)]

        # Here we only use galaxies where all bands are observed.
        sub = sub.loc[~np.isnan(sub.flux[bands]).any(1)]

        # Overlap between the two catalogues ...
        cosmos = get_cosmos(apply_cuts)
        touse = cosmos.index.intersection(sub.index).values
        flux_df = sub.loc[touse]

        flux = flux_df.flux[bands].values
        flux_err = flux_df.flux_err[bands].values


    # Here we are not actually using the flux, but a flux
    # ratio..

#    norm_band = utils.norm_band(bb, norm_band)
    norm_ind = bands.index(norm_band)

#    debugger.set_trace()
    print('Normalization index', norm_ind) 
    norm = flux[:, norm_ind]
    flux = torch.Tensor(flux / norm[:, None])
    flux_err = torch.Tensor(flux_err / norm[:, None])

    # The individual exposures.
    fmes, emes = get_indexp(touse, NB, path_indexp)
    fmes = torch.Tensor(fmes / norm[:,None,None])
    emes = torch.Tensor(emes / norm[:,None,None])
    
    vinv = 1. / emes.pow(2)
    isnan = torch.isnan(vinv)
    vinv[isnan] = 0
    fmes[isnan] = 0

    flux = torch.Tensor(flux)

    # The spectroscopic redshift.
    #zs = torch.Tensor(cosmos.loc[touse].zspec.values)
    zs = torch.Tensor(galcat.zs.values)

    print('# Galaxies', len(flux))
    assert len(flux) == len(zs), 'Inconsistent number of galaxies'
    
    # Test, this makes a difference when selecting with a PyTorch
    # uint8 tensor.
    ref_id = torch.Tensor(flux_df.index.values)

    # So one don't need to remember the order in a tuple later.
    data = {'flux': flux, 'flux_err': flux_err, 'fmes': fmes, 
            'vinv': vinv, 'isnan': isnan, 'zs': zs, 'ref_id': ref_id}

    return data
 
#    return flux, flux_err, fmes, vinv, isnan, zs, ref_id
