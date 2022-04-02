#!/usr/bin/env python
# encoding: UTF8

# Various utils.

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import networks

def path_pretrain(model_dir, label):
    """Path to the pretrained model.
       :param model_dir: {str} Model directory.
       :param label: {str} Label describing the pretrained model.
    """

    path = Path(model_dir) / f'pretrain_{label}.pt'

    return path

def path_model(model_dir, label, catnr, ifold):
    """Path to the model trained on data.
       :param model_dir: {str} Model directory.
       :param label: {str} Label describing the model.
       :param catnr: {int} Catalogue number.
       :param ifold: {int} Fold number.
    """

    path = Path(model_dir) / f'ondata_{label}_cat{catnr}_ifold{ifold}.pt'

    return path

def broad_bands(bb):
    """Parse the broad band input string.
       :param bb: {object} Which broad bands to use.
    """

    if isinstance(bb, list):
        pass
    elif isinstance(bb, tuple):
        bb = list(bb)
    elif bb.lower() == 'cosmos':
        bb = ['cfht_u', 'subaru_b', 'subaru_v', 'subaru_r', 'subaru_i', 'subaru_z']
    elif bb.lower() == 'cfht':
        bb = ['cfht_u', 'cfht_g', 'cfht_r', 'cfht_i', 'cfht_z']
    else:
        raise NotImplementedError()

    return bb 

def norm_band(bb, norm):
    """Determine the band for normalization.
       :param norm: {str} Band for normalization.
    """

    # Reasonable default!
    if not norm:
        ibands = [x for x in bb if x.endswith('_i')]
        assert len(ibands) == 1, 'No unique iband: {}'.format(ibands)
        norm = ibands[0]

    return norm


def get_loaders(ifold, inds, data):
    """Get loaders for specific fold.
       :param ifold: {int} The ifold to use.
       :param inds: {array} Indices to use.
       :param data: {tuple} All the data to use.
    """
   
    raise NotImplementedError('Fisk')
 
    flux, flux_err, fmes, vinv, isnan, zs, ref_id = data
    
    def sub(ix):
        ds = TensorDataset(flux[ix].cuda(), fmes[ix].cuda(), vinv[ix].cuda(), \
                           isnan[ix].cuda(), zs[ix].cuda())
        
        return ds
    
    ix_train = torch.ByteTensor(1*(inds != ifold))
    ix_test = torch.ByteTensor(1*(inds == ifold))
    
    ds_train = sub(ix_train)
    ds_test = sub(ix_test)

    train_dl = DataLoader(ds_train, batch_size=500, shuffle=True)
    test_dl = DataLoader(ds_test, batch_size=100)
    
    return train_dl, test_dl, zs[ix_test]
