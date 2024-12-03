#!/usr/bin/env python
# coding: utf-8

# Flux imputation and joining tables, including spec-z.

from pathlib import Path
from . import extlib
from IPython.core import debugger as ipdb

import numpy as np
import pandas as pd

def impute_qfit(cat):
    """Impute narrow bands using a fitting methods developed
       by Enrique Gaztanaga.
    """

    # It's easier working with a copy than modifications inplace.
    cat = cat.copy() 

    # convert GRI Broad Band magnitudes into PAU flux units:
    fg = 10**(0.4*(26-cat['mag_g_0']))
    fr = 10**(0.4*(26-cat['mag_r_0']))
    fi = 10**(0.4*(26-cat['mag_i_0']))

    # fit a quadratic curve scale = A lambda^2+ B lambda + C that passes 
    # through 3 points: (fg,fr,fi) at (lg,lr,li)
    # median wavelength of BB filters GRI:
    # To  CFHTLens
    lg = 460. # in nm
    lr = 620.
    li = 750.

    # inverse matrix M=(l**2 l 1)
    detM = lg*(li**2-lr**2) + lr*(lg**2-li**2) + li*(lr**2-lg**2)        

    A = (fg*(lr-li) + fr*(li-lg) + fi*(lg-lr)) / detM
    B = (fg*(li**2-lr**2) + fr*(lg**2-li**2) + fi*(lr**2-lg**2)) / detM
    C = (fg*(li*lr**2-lr*li**2) + fr*(lg*li**2-li*lg**2) + fi*(lr*lg**2-lg*lr**2)) / detM
    
    # Actually apply the correction. Only replace values with NaNs.
    lmbL = 455 + 10*np.arange(40)
    for lmb in lmbL:
        col = f'NB{lmb}'
        replace_val = A*lmb**2+B*lmb+C
        cat[col] = cat[col].fillna(replace_val)

    return cat

def impute_knn(cat):
    """Imputation using a KNN method."""

    # Here you would need to implement the KNN method.
    raise NotImplementedError('To be implemented.')

def impute(cat, method='qfit'):

    fD = {'qfit': impute_qfit, 'knn': impute_knn}
    assert method in fD, f'No such imputation method: {method}'

    res = fD[method](cat)

    return res
