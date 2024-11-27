#!/usr/bin/env python
# encoding: UTF8

from pathlib import Path
import numpy as np
import pandas as pd

from . import coadd
from . import download
from . import extlib
from . import specz

def load_cfht(d_root, field):
    """Load the CFHT parent catalogue."""

    # Only load the required field.
    d_root = Path(d_root)
    path_cfht = d_root / 'download' / 'cfhtlens.pq'
    field = field.upper()
    cfht = pd.read_parquet(path_cfht, filters=[('xfield', '=', field)])

    return cfht

def combine_catalogs(cfht=None, paus=None, specz=None):
    """Combine BB, NB and spec-z catalogues."""

    # Avoiding positional arguments. Current you need to specify
    # all catalogues, but this can change later.
    assert len(cfht), 'Need to provive the CFHT catalogue.'
    assert len(paus), 'Need to provive the PAUS catalogue.'
    assert len(specz), 'Need to provive the SPECZ catalogue.'

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

    comb = comb.merge(specz, on='ref_id', how='outer')
    comb['has_specz'] = ~np.isnan(comb.zs)

    return comb

def coadd_combine(d_root, memba_prod, field):
    """Combine the coadd, spec-z and parent catalogue."""
    
    # This is fast, so it can be run everytime if silent.
    download(d_root, debug=False, memba_prodL=[memba_prod])
    paus = coadd.load_downloaded(d_root, memba_prod)
    
    paus = coadd.change_format(paus)
    specz_cat = specz.specz(field)
    cfht = load_cfht(d_root, field)
    
    comb = combine_catalogs(cfht=cfht, paus=paus, specz=specz_cat)
    
    return comb



#  THE STEPS BELOW NEEDS TO BE MOVED.

#    # Impute missing bands.
#    impute_bb_fit(comb)
#
#    # Merge with the spectroscopic catalogue.
#    comb_with_zs = comb.merge(specz, on='ref_id')
#
#    # And the sample without spectra.
#    comb_nospecz = comb[~comb.ref_id.isin(comb_with_zs.ref_id.values)]
#    assert len(comb) == len(comb_nospecz) + len(comb_with_zs), 'Numbers adds up'
#
#    return comb_with_zs, comb_nospecz
