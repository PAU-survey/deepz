#!/usr/bin/env python
# encoding: UTF8

# Code for splitting the catalogue in train/validation sets.

from pathlib import Path
import pandas as pd

def load_ref_id(path):
    """Load ref_id from stored catalogue."""
    
    xcat = pd.read_csv(path, usecols=['ref_id'])
    ref_id = xcat.ref_id.astype(int).values

    return ref_id 

def split_existing(coadd):
    """Split catalogue by existing catalogue."""
    
    # Needs to be implemented also for COSMOS and G09.
    d_in = Path('/data/astro/scratch/idazaper/idazaper_recoverfiles/w1_w3/')
    ref_id_train = load_ref_id(d_in / 'train_E1_Complemento.csv')
    ref_id_val = load_ref_id(d_in / 'val_E1_Complemento.csv')

    # The coadd catalogue.
    coadd = coadd.set_index('ref_id')
    coadd_train = coadd.loc[ref_id_train].reset_index()
    coadd_val = coadd.loc[ref_id_val].reset_index()

    return coadd_train, coadd_val
