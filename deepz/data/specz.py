#!/usr/bin/env python
# encoding: UTF8

# This only contains one spectroscopic catalogue. This catalogue
# should be extended if adding more catalogues.

import pandas as pd


def specz_w1():
    """Spectroscopic catalogue in W1 field."""

    # Files Vanessa sent, coming from David. 
    spec_cat = pd.read_csv('/data/astro/scratch/idazaper/idazaper_recoverfiles/w1/VIPERS_plus_DES_total.csv')
    _spec_cat_label = pd.read_csv('/data/astro/scratch/idazaper/idazaper_recoverfiles/w1/W1_validation_sample.csv')

    # Only adding label from the second file. Rename to be compatible with Vanessas code.
    spec_cat = spec_cat.merge(_spec_cat_label[['ref_id', 'SOURCE']], on = 'ref_id')
    spec_cat = spec_cat.rename(columns={'SOURCE': 'catalogue'})

    return spec_cat

def specz_w3():
    """Spectroscopic redshift in the W3 field."""
    
    specz_cat = pd.read_csv('/data/astro/scratch/idazaper/w3/coadd_w3.csv')[['ref_id', 'zs', 'zb_bb', 'zb_bcnz']]
    specz_label = pd.read_csv('/data/astro/scratch/idazaper/w3/W3_validation_sample.csv')
    specz_cat = specz_cat.merge(specz_label[['ref_id', 'SOURCE']])

    return specz_cat


def specz(field):
    """Return spec-z of a given field."""

    # Just to avoid if-statements elsewhere in the pipeline.
    fD = {'w1': specz_w1, 'w3': specz_w3}
    res = fD[field.lower()]()

    return res
