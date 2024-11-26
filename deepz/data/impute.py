#!/usr/bin/env python
# coding: utf-8

# Flux imputation and joining tables, including spec-z.

from pathlib import Path
import extlib
from IPython.core import debugger as ipdb

import numpy as np
import pandas as pd

def impute_bb_fit(cat):
    """Impute narrow bands using a fitting methods developed
       by Enrique Gaztanaga.
    """
    
    # Estimate factors.

    # convert GRI Broad Band magnitudes into PAU flux units:
    fg = 10**(0.4*(26-cat['mag_g_0']))
    fr = 10**(0.4*(26-cat['mag_r_0']))
    fi = 10**(0.4*(26-cat['mag_i_0']))

    # fit a quadratic curve scale = A lambda^2+ B lambda + C that passes 
    # through 3 points: (fg,fr,fi) at (lg,lr,li)
    # median wavelength of BB filters GRI:
    # To  CFHTLens
    lg = 460. # in nm
    lr = 620.
    li = 750.

    # inverse matrix M=(l**2 l 1)
    detM = lg*(li**2-lr**2) + lr*(lg**2-li**2) + li*(lr**2-lg**2)        

    A = (fg*(lr-li) + fr*(li-lg) + fi*(lg-lr)) / detM
    B = (fg*(li**2-lr**2) + fr*(lg**2-li**2) + fi*(lr**2-lg**2)) / detM
    C = (fg*(li*lr**2-lr*li**2) + fr*(lg*li**2-li*lg**2) + fi*(lr*lg**2-lg*lr**2)) / detM
    
    # Actually apply the correction. Only replace values with NaNs.
    lmbL = 455 + 10*np.arange(40)
    for lmb in lmbL:
        col = f'NB{lmb}'
        replace_val = A*lmb**2+B*lmb+C
        cat[col] = cat[col].fillna(replace_val)


def load_cfht(d_root, field):
    """Load the CFHT parent catalogue."""
    
    # Only load the required field.
    path_cfht = d_root / 'download' / 'cfhtlens.pq'
    field = field.upper()
    cfht = pd.read_parquet(path_cfht, filters=[('xfield', '=', field)])

    return cfht
    
def load_paus(d_root, memba_prod):
    """Load the PAUS coadd catalogue."""
    
    #Catalogue of Vanessa.
    #vpaus = pd.read_csv('/data/astro/scratch/idazaper/idazaper_recoverfiles/w1/paus_clean_NaN.csv')
    paus = pd.read_parquet(d_root / 'intermed' / f'paus_clean_NaN_memba{memba_prod}.pq')

    return paus


def combine_catalogs(cfht, paus, spec_cat):
    """Combine BB, NB and spec-z catalogues."""

    # Counts the number of NaNs.
    bands = [f'NB{x}' for x in 455 + 10*np.arange(40)]
    paus['nr_nans'] = paus[bands].isnull().sum(axis=1)
    
    # Merge the catalogs.
    comb = paus.merge(cfht, left_on='ref_id', right_on='paudm_id')
    
    # Clipping error to a small positive value to avoid dividing by zero.
    #for band in 'ugriz':
    #    comb[f'magerr_{band}'] = comb[f'magerr_{band}'].clip(0.001, np.inf)
        
    # Remove broad band extinctions in the catalogue.
    extlib.remove_bb_extcorr(comb)
    
    # Impute missing bands.
    impute_bb_fit(comb)
    
    # Merge with the spectroscopic catalogue.
    comb_with_zs = comb.merge(spec_cat, on='ref_id')

    # And the sample without spectra.
    comb_nospecz = comb[~comb.ref_id.isin(comb_with_zs.ref_id.values)]
    assert len(comb) == len(comb_nospecz) + len(comb_with_zs), 'Numbers adds up'

    return comb_with_zs, comb_nospecz

def load_combine(d_root, memba_prod, field):
    """Load and combine the catalogues."""

    cfht = load_cfht(d_root)
    paus = load_paus(d_root, memba_prod)
    spec_cat = load_specz()
    comb_with_zs, comb_nospecz = combine_catalogs(cfht, paus, spec_cat)

    return comb_with_zs, comb_nospecz
