#!/usr/bin/env python
# encoding: UTF8

# Methods for handling extinction. Only one method so far, but
# we should include the code to correct for galaxy extinction
# here.

def remove_bb_extcorr(cat):
    """Remove the broad-band extinction and convert to fluxes."""
    
    for band in 'ugriyz':
        mag_0 = cat[f'mag_{band}'] + cat[f'extinction_{band}']
        flux = 10**(-0.4*(mag_0 - 26))
        SNR = 1. / (10**(0.4*cat[f'magerr_{band}']) - 1.)
        fluxerr = flux / SNR
        
        cat[f'mag_{band}_0'] = mag_0
        cat[f'flux_{band}'] = flux
        cat[f'SNm_{band}'] = SNR
        cat[f'fluxerr_{band}'] = fluxerr
