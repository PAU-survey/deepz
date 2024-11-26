#!/usr/bin/env python
# encoding: UTF8

from pathlib import Path
import pandas as pd

def load_downloaded(d_root, memba_prod):
    """Load the downloaded catalogue."""

    d_root = Path(d_root)
    fname_out = f'coadd_memba{memba_prod}.pq'
    path = d_root / 'download' / fname_out

    cat = pd.read_parquet(path)

    return cat

def duplicate_removal(paus):
    """Remove perceived duplicates from the coadd files. This should not
       be needed.
    """
    
    senseless_duplication_removal = True
    
    # I don't think this makes sense. It looks for duplicates in (band, flux, flux_err). This does
    # not have to be unique and non-uniqueness is not an issue. Only a *very small* fraction 1e-6
    # is removed. Keeping this here so we can continue making an exact comparison.
    if senseless_duplication_removal:
        df1_without_nan = paus.copy()
        df1_without_nan.set_index("ref_id", inplace=True)
        df1_without_duplicates = df1_without_nan[~df1_without_nan.astype(str).duplicated()]
        df1_without_duplicates = df1_without_duplicates.reset_index()
    else:
        df1_without_duplicates = paus
    
    return df1_without_duplicates


def change_format(paus):
    """Change the format of the cataloge."""
    
    # Changing format, including renaming the flux columns.
    paus_flux = paus.pivot(index='ref_id', columns='band', values='flux')
    
    cat_tmp = paus.pivot(index='ref_id', columns='band')
    flux = cat_tmp.flux
    flux_error = cat_tmp.flux_error
    flux_error.columns = flux_error.columns.str.replace('NB', 'NBerr')
    
    paus_all = pd.concat([flux, flux_error], axis=1)
    paus_all = paus_all.reset_index()

    return paus_all

def remove_minus99(cat):
    """Remove galaxies containing -99 in the BB magnitudes."""

    # This cut was in 4_Mask&Cuts in Vanessas pipeline.
    cols = [f'mag_{band}' for band in 'ugriz']
    cat_out = cat[(cat[cols] > -99).all(axis=1)]

    return cat_out
