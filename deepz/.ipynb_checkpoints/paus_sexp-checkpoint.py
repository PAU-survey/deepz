#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created at 

# =============================================================================
# DOCS
# =============================================================================

"""This paper focuses on the data from the Cosmological Evolution Survey 
`(COSMOS) <https://cosmos.astro.caltech.edu/>`_,  field1 where we have PAUS 
observations and there are abundant spectroscopic measurements. Our fiducial 
setup uses the Canada-FranceHawaii Telescope Lensing Survey (CFHTLenS) u-band 
and the B, V, r, i, z bands from the Subaru telescope as in 
Eriksen et al. (2019). As the spectroscopic catalogue, we use 8566 secure 
(3 ≤ CLASS ≤ 5) redshifts from the zCOSMOS DR3 survey (Lilly et al. 2009) that 
are observed with all 40 narrow bands.

The PAUS data are acquired at the William Herschel Telescope (WHT) with the 
PAUCam instrument and transferred to the Port d’Informaci Cientfica 
(PIC, Tonello et al. 2019). First the images are detrended in the nightly pipeline 
(Serrano et al. in prep). Our astrometry is relative to Gaia DR2 
(Brown et al. 2018), while the photometry is calibrated relative to the Sloan 
Digital Sky Survey (SDSS) by fitting the Pickles stellar templates 
(Pickles 1998) to the u, g, r, i, z broad bands from SDSS (Smith et al. 2002) and 
then predicting the expected fluxes in the narrow bands.
The final zero-points are determined by using the median star zero-point for each 
image. PAUS observes weak lensing fields (CFHTLenS: W1, W3 and W4) with deeper 
broad-band data from external surveys. PAUS uses forced photometry, assuming known 
galaxy positions, morphologies and sizes from external catalogues. The photometry 
code determines for each galaxy the radius needed to capture a fixed fraction of 
light, assuming the galaxy follows a Srsic profile convolved with a known
Point Spread Function (PSF). The algorithm uses apertures that measure 62.5% of the 
light, since this is considered statistically optimal. A given galaxy is observed 
several times (3-10) from different overlapping exposures. The coadded fluxes are 
produced using inverse variance weighting of the individual measurements.
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import local_settings

data_in = local_settings.data_in_path

NB = ['NB{}'.format(x) for x in 455+10*np.arange(40)]
BB = ['cfht_u', 'subaru_B', 'subaru_V', 'subaru_r', 'subaru_i', 'subaru_z']
bands = NB + BB

D = {'photoz': '4199.csv', 'coadd': '4213.csv', 'cosmos': '4378.csv'}


def get_cosmos(apply_cuts):
    """Preprocessing on cosmos data 

    :param apply_cuts: true if you want to apply constraints on conf and i_auto, 
        such that the cosmos data complies with: 3 <= conf <= 5 and i_auto < 22.5. 
        Otherwise, if False, then only apply constraint on column r50 of the cosmos 
        data, such that: r50 > 0.

    :type apply_cuts: bool

    :return: Preprocessed cosmos data.

    :rtype: Pandas DataFrame
    """
    cosmos = pd.read_csv(str(data_in / D['cosmos']), comment='#')
    cosmos = cosmos.set_index('paudm_id')
    cosmos = cosmos[0 < cosmos.r50]
    
    if apply_cuts:
        cosmos = cosmos[(3 <= cosmos.conf) & (cosmos.conf <= 5)]
        cosmos = cosmos[cosmos.i_auto < 22.5]

    return cosmos

def get_indexp(inds_touse, NB_bands, indexp_path):
    """Flow and flow error of the sources of interest.

    :param inds_touse: Common sources between COSMOS and Subaru.

    :type inds_touse: array

    :param NB_bands: List of the names of the 40 narrow bands to be used in the 
        studio. 

    :type NB_bands: list

    :param indexp_path: path
 
    :type indexp_path: str

    :return: Flux and flux error in the 40 narrow bands of the sources in common 
        between COSMOS and subaru catalogues with nr < 10.

    :rtype: arrays
    """
    # Testing Lumus.
    # path = Path('/cephfs/pic.es/astro/scratch/eriksen/deepz/input/fa/comb_v7.parquet')

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


galcat_path = local_settings.galcat_path
indexp_path = local_settings.indexp_path

def paus(apply_cuts, galcat_path, indexp_path):
    """PAUS data

    :param apply_cuts: true if you want to apply constraints on conf and i_auto, 
        such that the cosmos data complies with: 3 <= conf <= 5 and i_auto < 22.5. 
        Otherwise, if False, then only apply constraint on column r50 of the cosmos 
        data, such that: r50 > 0.

    :type apply_cuts: bool

    :param galcat_path: coadd_v8.h5

    :type galcat_path: str

    :param indexp_path: fa_v8.pq

    :type indexp_path: str

    :return: flux, flux_err, fmes, vinv, isnan, zbin, ref_id.

    :rtype: tensor, tensor, tensor, tensor, array, tensor, tensor
    """

    # Testing Lumus.
    # galcat_path = '/cephfs/pic.es/astro/scratch/eriksen/deepz/input/cosmos_pau_matched_v2.h5'
    # galcat_path = '/cephfs/pic.es/astro/scratch/eriksen/deepz/input/lumus/coadd_v8.h5'
    # indexp_path = Path('/cephfs/pic.es/astro/scratch/eriksen/deepz/input/lumus/fa_v8.pq')

    galcat = pd.read_hdf(galcat_path, 'cat')

    sub = galcat.loc[~np.isnan(galcat.flux.subaru_i)]
    # Here we only use galaxies where all bands are observed.
    sub = sub.loc[~np.isnan(sub.flux[bands]).any(1)]

    # Overlap between the two catalogues ...
    cosmos = get_cosmos(apply_cuts)
    touse = cosmos.index & sub.index

    # Here we are not actually using the flux, but a flux ratio..
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
    fmes = torch.Tensor(fmes / norm[:, None, None])
    emes = torch.Tensor(emes / norm[:, None, None])
    
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
