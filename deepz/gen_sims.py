#!/usr/bin/env python
# encoding: UTF8

# Copyright (C) 2022 Martin B. Eriksen
# This file is part of Deepz <https://github.com/PAU-survey/deepz>.
#
# Deepz is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Deepz is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Deepz.  If not, see <http://www.gnu.org/licenses/>.

# Generate FSPS simulations to pretrain the simulations.

import os
import dask
import dask.config
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import numpy as np
import pandas as pd
from pathlib import Path
from time import sleep
from dask.diagnostics import ProgressBar
from astropy.cosmology import Planck15 
import fsps

bands = fsps.filters.list_filters()

def empty_gen(params):
    """Generate an empty set of data.
       :param params: {dataframe} Per galaxy parameters.
    """
   
    os.environ['SPS_HOME'] = '/nfs/astro/eriksen/source/fsps'
 
    # This hard coding should actually not be needed..
    L = ['zred', 'logzsol', 'tau', 'const', 'sf_start', 'dust2']
    mags = np.zeros((len(params), len(bands)))
    
    import fsps
    pop = fsps.StellarPopulation(zcontinuous=1, add_neb_emission=True, sfh=1, dust_type=2)
    for i, (_, row) in enumerate(params.iterrows()):    
        # Note, here we have modified fsps to not panick..
        for key in L:
            pop.params[key] = row[key]
        
        mags[i] = pop.get_mags(tage=row.tage)
    
    return pd.DataFrame(mags, columns=bands, index=params.index)


# Parameters defining the ranges. Based on numbers from Cigal.
ranges = {\
'zred': [0, 1.5],
'logzsol': [-0.5, 0.2],
#'tage': [1, 13.7],
'gas_logu': [-4., 1.],
'tau': [0.1, 12],
'const': [0, 0.25],
'dust2': [0, 0.6]}


def gen_params(Ngal):
    """Generate the distribution of parameters.
       :param Ngal: {int} Numbers of galaxies to generate.
       :returns: Galaxy parameters.
    """

    print('starting...')
    # Input parameters.
    D = {}
    for key, (low,high) in ranges.items():
        D[key] = np.random.uniform(low, high, size=Ngal)

    # Following:
    # https://github.com/dfm/python-fsps/issues/68
    # I have connected the age and redshift using the cosmology.
    D['tage'] = np.array(Planck15.age(D['zred']))
    D['sf_start'] = np.random.random(Ngal)*D['tage'] #-0.1)
 
    # For consistency, the gas phase metallicity should be set to
    # the same value as for the galaxy (FSPS documentation).
    D['gas_logz'] = D['logzsol'] 
    print('finished random numbers...')

    df1 = pd.DataFrame(D)

    return df1

def make_sim(df1, out_path):
    """Create the simulations.
       :param df1: {dataframe} Galaxy parameters.
       :param out_path: {path} Output path.
    """

    df2 = dd.from_pandas(df1, chunksize=10000)
    df2.to_parquet(str(out_path / 'params.parquet'))

    meta = [(x, float) for x in bands]
    G = df2.map_partitions(empty_gen, meta=meta)

    print('Staring parallel..')
    with ProgressBar():
        G.to_parquet(str(out_path / 'mags.parquet'))


Ngal = int(1e6)
out_path = Path('/nfs/astro/eriksen/deepz/sims/v9')

if __name__ == '__main__':
    df1 = gen_params(Ngal)

    client = Client('tcp://193.109.175.131:44657')
    make_sim(df1, out_path)
