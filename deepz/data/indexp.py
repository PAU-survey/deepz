#!/usr/bin/env python
# encoding: UTF8

# Transformations for the individual exposures.

from pathlib import Path
import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd

from IPython.core import debugger as ipdb

def calibrate(df):
    """Calibrate the individual fluxes."""
    
    # Here the zero-points are really multiplicative factors.
    e1 = df['flux_uncal']
    e2 = df['zp']
    
    v1 = df['flux_error_uncal']**2
    v2 = df['zp_error']**2
    
    df['flux'] = e1 * e2
    df['flux_error'] = np.sqrt(v1*v2 + v1*e2**2 + v2*e1**2)


def store_indexp_memba(d_root, coadd_label, memba_prod):
    """Calibrate and add the exposure number."""

    d_root = Path(d_root)
    path_out = d_root / 'intermed' / coadd_label / f'indexp_calib_memba{memba_prod}.pq'

    if path_out.exists():
        print('Already transformed:', memba_prod)
        return path_out
    else:
        print('Transforming:', memba_prod)
       
    # The P2P method caused an error. Sticking with the task based scheduling for now.
    dask.config.set({"dataframe.shuffle.method": "tasks"})
 
    df = dd.read_parquet(d_root / 'download' / f'fa_memba{memba_prod}.pq')
    
    # Zero-point calibration.
    calibrate(df)
    
    # Precalculate a number used to store the individual exposures. 
    df = df[['ref_id', 'band', 'flux', 'flux_error']]
    df['nr'] = df.groupby(['ref_id', 'band']).cumcount()
    
    df.to_parquet(path_out)

    return path_out

def to_xarray(df_fa):
    """Convert to xarray."""
    df_fa = df_fa.set_index('ref_id')

    # Way less than 1% exposures missed.
    df_fa = df_fa[df_fa.nr < 10]

    # This operation if faster going through xarrays..
    df_fa = df_fa.reset_index().set_index(['ref_id', 'band', 'nr'])
    X = df_fa.to_xarray()

    return X

def convert_to_netcdf(path_in):
    """Convert output to xarray datastructure and store as a netcdf file.."""
   
    path_out = path_in.with_suffix('.nc')
    if path_out.exists():
        return

    print('Storing as netcdf')
    df_fa = pd.read_parquet(path_in)    
    X = to_xarray(df_fa)
    X.to_netcdf(path_out) 

def store_indexp(d_root, coadd_label):
    for memba_prod in [1012, 1015]:
        path_out = store_indexp_memba(d_root, coadd_label, memba_prod) 
        convert_to_netcdf(path_out)
