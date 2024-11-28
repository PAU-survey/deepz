#!/usr/bin/env python
# encoding: UTF8

from pathlib import Path
import os
import numpy as np
import pandas as pd

from . import coadds
from . import download
from . import extlib
from . import impute
from . import specz
from . import split_train_val

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
    download.download(d_root, debug=False, memba_prodL=[memba_prod])
    paus = coadds.load_downloaded(d_root, memba_prod)
    
    paus = coadds.change_format(paus)
    specz_cat = specz.specz(field)
    cfht = load_cfht(d_root, field)
    
    comb = combine_catalogs(cfht=cfht, paus=paus, specz=specz_cat)
    comb['field'] = field
 
    return comb
