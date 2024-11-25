#!/usr/bin/env python
# coding: utf-8

# Download catalogs from the PAUdm database. The data downloaded as CSV using 
#psql and then converted to parquet files using Dask.
# 
# - Downloads the forced aperture measurements together with the calibration 
# information. No calibration is applied in this step.
# No calibration is applied.


import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
from IPython.core import debugger as ipdb

from matplotlib import pyplot as plt

# Forces a single process. Otherwise it starts too many threads and gets
# killed.
dask.config.set(scheduler='single-threaded')


def get_passwd():
    """Get the PAUdm password."""

    # Assumes the password is stored in the location below.
    path_pwfile = '~/paudm_pw_readonly'
    path_pwfile = os.path.expanduser(path_pwfile)
    assert Path(path_pwfile).exists(), f'Please store the PAUdm readonly password in: {path_pwfile}'
    
    pw = open(path_pwfile).read().strip()
    
    return pw

def find_psql():
    """Find absolute path to installed psql."""
    
    # Test if utility is on the path.
    psql_path = shutil.which('psql')
    if psql_path:
        return psql_path
    
    # Alternatively, if in the current environment.
    psql_path = Path(sys.executable).parent / 'psql'
    if psql_path.exists():
        return str(psql_path)
    
    # If not found.
    raise FileNotFoundError('Missing psql binary path')


def download_cat(path_out, sql, dtype={}):
    """Download and store catalogue as parquet file."""
    
    assert path_out.suffix == '.pq', 'Only storing to parquet files.'
    if path_out.exists():
        print('Path exists:', path_out)
        return
    else:
        print('Downloading:', path_out)
    
    env = {'PGPASSWORD': get_passwd()}
    psql_path = find_psql()

    # This part could be better. For some reason it is using too much memory.
    # I suspect this is because passing the file object to subprocess.
    t1 = time.time()

    # Set delete=True to automatically delete temporary files.
    with tempfile.NamedTemporaryFile(dir=path_out.parent, delete=True) as temp_file:
        print('Tmp file:', temp_file.name)

        # Dumps the table to a temporary file.
        command = [psql_path, '-Ureadonly', '-hdb.pau.pic.es', 'dm', '-c', sql, '--csv']
        subprocess.run(command, env=env, stdout=temp_file.file)
    
        # Convert to a Parquet file.
        df = dd.read_csv(temp_file.name, dtype=dtype).reset_index(drop=True)
        df.to_parquet(path_out)

    print(f'Time downloading:', time.time() - t1)

def fa_sql(memba_prod):
    """SQL for downloading forced aperture plus calibration."""
    
    sql = f"""
    SELECT production_id, fa.image_id, image.filter AS band, ref_id, flux AS flux_uncal, 
           flux_error AS flux_error_uncal, annulus_ellipticity, zp, zp_error
    FROM forced_aperture AS fa
    JOIN image ON image.id = fa.image_id
    JOIN image_zp ON image_zp.image_id = image.id
    WHERE fa.production_id = {memba_prod}
    AND image_zp.calib_method = 'MBE2.1_xsl'
    AND image_zp.phot_method_id = 2
    AND fa.flag = 0
    """

    return sql


def coadd_sql(memba_prod):
    """SQL for downloading the coadds."""
    
    sql_coadd = f"""
    SELECT *
    FROM forced_aperture_coadd
    WHERE production_id = {memba_prod}
    AND run = 1.0
    """

    return sql_coadd


def download(d_root, memba_prodL=[1012, 1015, 1057]):
    """Download the different catalogs needed."""

    # The productions used in the PAUS data release.

    d_root = Path(d_root)

    # Downloading the CFHT catalogue.
    sql_cfht = """
        SELECT left(field, 2) as xfield, paudm_id, alpha_j2000, delta_j2000, z_b,\
        mag_u, mag_g, mag_r, mag_i, mag_y, mag_z, extinction_u, extinction_g, extinction_r, extinction_i, extinction_y,
        extinction_z, magerr_u, magerr_g, magerr_r, magerr_i, magerr_y, magerr_z, mask, lp_mi, lp_mr, star_flag, class_star\
                 FROM cfhtlens
    """

    os.makedirs(d_root / 'download', exist_ok=True)

    # Specifying dtype here. Needed depending on the order of the rows.
    dtype={'extinction_y': 'float64',
           'mag_y': 'float64',
           'magerr_y': 'float64'}
    path_out = d_root / 'download' / 'cfhtlens.pq'
    download_cat(path_out, sql_cfht, dtype=dtype)

    # Downloading forced aperture catalogues.
    for prod_id in memba_prodL:
        sql_memba = fa_sql(prod_id)
        fname_out = f'fa_memba{prod_id}.pq'
        path_out = d_root / 'download' / fname_out
        download_cat(path_out, sql_memba)


    # Downloading forced aperture catalogues.
    for prod_id in memba_prodL:
        sql_memba = coadd_sql(prod_id)
        fname_out = f'coadd_memba{prod_id}.pq'
        path_out = d_root / 'download' / fname_out

        download_cat(path_out, sql_memba)
