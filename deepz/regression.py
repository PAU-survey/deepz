#!/usr/bin/env python
# encoding: UTF8

import pandas as pd
from matplotlib import pyplot as plt

def w1w3_fig6(cat):
    """Comparing with Vanessas W1+W3 catalogue corresponding to Fig.6 in the paper."""

    # Directly looking at the stored files. The comparison here is only meant to be used
    # for some time.
    van = pd.read_csv('/data/astro/scratch/idazaper/w1_U_w3_U_w2/val_E1_W1uW3_znn_E1_Com.csv')
    van['ref_id'] = van.ref_id.astype(int)
    van = van.rename(columns={'z_nn': 'z_van'})
    
    comb = cat.merge(van, on='ref_id')
    comb['dx_martin'] = (comb.z_nn - comb.zs) / (1 + comb.zs)
    comb['dx_van'] = (comb.z_van - comb.zs) / (1 + comb.zs)
    
    gr = comb.groupby(pd.qcut(comb.mag_i, 10))
    xvals = gr.mag_i.mean()
    
    y_martin = 0.5*(gr.dx_martin.quantile(0.84) - gr.dx_martin.quantile(0.16))
    y_van = 0.5*(gr.dx_van.quantile(0.84) - gr.dx_van.quantile(0.16))
    
    plt.plot(xvals, y_martin, label='Reproduced')
    plt.plot(xvals, y_van, ls='--', label='Fig.6 (Daza et al.)')
    plt.xlabel('$i$-band mag', size=14)
 
    plt.legend()
