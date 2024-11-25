#!/usr/bin/env python
# encoding: UTF8

# Transformations for the individual exposures.

from pathlib import Path
import numpy as np
import dask
import dask.dataframe as dd

def calibrate(df):
    """Calibrate the individual fluxes."""
    
    # Here the zero-points are really multiplicative factors.
    e1 = df['flux_uncal']
    e2 = df['zp']
    
    v1 = df['flux_error_uncal']**2
    v2 = df['zp_error']**2
    
    df['flux'] = e1 * e2
    df['flux_error'] = np.sqrt(v1*v2 + v1*e2**2 + v2*e1**2)


def transform(memba_prod):
    """Calibrate and add the exposure number."""

    path_out = Path(f'/data/aai/common/eriksen/reprod/tmp/indexp_calib_memba{memba_prod}.pq')
    if path_out.exists():
        print('Already transformed:', memba_prod)
        return
    else:
        print('Transforming:', memba_prod)
       
    # The P2P method caused an error. Sticking with the task based scheduling for now.
    dask.config.set({"dataframe.shuffle.method": "tasks"})
 
    df = dd.read_parquet(f'/data/aai/common/eriksen/reprod/download/fa_memba{memba_prod}.pq')
    
    # Zero-point calibration.
    calibrate(df)
    
    # Precalculate a number used to store the individual exposures. 
    df = df[['ref_id', 'band', 'flux', 'flux_error']]
    df['nr'] = df.groupby(['ref_id', 'band']).cumcount()
    
    df.to_parquet(path_out)

