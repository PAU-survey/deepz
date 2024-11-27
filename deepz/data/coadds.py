#!/usr/bin/env python
# encoding: UTF8

import os
from pathlib import Path
import pandas as pd

from . import combine
from . import impute
from . import split_train_val

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


def store_coadds(d_root, coadd_label):
    """Estimate and store the coadds in files."""

    # For separating different tests in directories.
    d_out = d_root / 'intermed' / coadd_label
    os.makedirs(d_out, exist_ok=True)

    # We split into training and validation *before* doing the imputation. If using
    # an imputation using training, like a KNN, there is a certain risk information
    # correlated to the test set labels enters into the training through the inputation.
    # Better safe than sorry.
    train_hasnan_path = d_out / 'w1_w3_train_hasnan.pq'
    val_hasnan_path = d_out / 'w1_w3_val_hasnan.pq'

    #train_hasnan_path = d_out / 'w1_w3_train_hasnan.pq'
    #val_hasnan_path = d_out / 'w1_w3_val_hasnan.pq'

    if not (train_hasnan_path.exists() and val_hasnan_path.exists()):
        print('Before coadd...')
        coadd_w1 = combine.coadd_combine(d_root, 1015, 'w1')
        coadd_w3 = combine.coadd_combine(d_root, 1012, 'w3')
        coadd = pd.concat([coadd_w1, coadd_w3])
        print('After coadd...')

        coadd_train_hasnan, coadd_val_hasnan = split_train_val.split_existing(coadd)

        coadd_train_hasnan.to_parquet(train_hasnan_path)
        coadd_val_hasnan.to_parquet(val_hasnan_path)

        # Impute coadd values. Store to file.
        coadd_train = impute.impute(coadd_train_hasnan)
        coadd_val = impute.impute(coadd_val_hasnan)

        coadd_train.to_parquet(d_out / 'w1_w3_train.pq')
        coadd_val.to_parquet(d_out / 'w1_w3_val.pq')
