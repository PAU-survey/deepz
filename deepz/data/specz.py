#!/usr/bin/env python
# encoding: UTF8

# This only contains one spectroscopic catalogue. This catalogue
# should be extended if adding more catalogues.

import pandas as pd

def load_specz():
    """Load the spectroscopic catalogue."""

    # Files Vanessa sent, coming from David. 
    spec_cat = pd.read_csv('/data/astro/scratch/idazaper/idazaper_recoverfiles/w1/VIPERS_plus_DES_total.csv')
    _spec_cat_label = pd.read_csv('/data/astro/scratch/idazaper/idazaper_recoverfiles/w1/W1_validation_sample.csv')

    # Only adding label from the second file. Rename to be compatible with Vanessas code.
    spec_cat = spec_cat.merge(_spec_cat_label[['ref_id', 'SOURCE']], on = 'ref_id')
    spec_cat = spec_cat.rename(columns={'SOURCE': 'catalogue'})

    return spec_cat
