#!/usr/bin/env python
# encoding: UTF8

import pandas as pd

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

# Vanessa was storing the galaxies needing imputation or not separately.
