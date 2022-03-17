#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created at 

# =============================================================================
# DOCS
# =============================================================================

"""load of data
"""

# =============================================================================
# IMPORTS
# =============================================================================

import torch
from torch.utils.data import TensorDataset, DataLoader


# Use
# train_dl, test_dl, zbin_ix_test = loaders.get_loaders(ifold, inds, data, Ntrain)

def get_loaders(ifold, inds, data, Ntrain):
    """Get loaders for specific fold.

        First, it generates two tensors with the indices of the 
        sources to be selected for the train and test sets. In 
        particular, the tensor of indices of the train set is 
        equal to zero when it exceeds the size of Ntrain.

        Then, using DataLoader, a PyTorch class that is a Python 
        interable, map-style data sets are entered to define the 
        train and test sets based on the index tensors and batch size 
        (batch_size_train=500 and batch_size_test=100), in particular, 
        the train set is given the plus parameter shuffle=True.

    :param ifold: ?

    :type ifold: ?

    :param inds: Indices of the selected sources with flux information and 
                 ¿¿¿config['catnr']???.

    :type inds: tensor

    :param data: paus data

    :dtype data: tensors

    :return: Tensors train_dl, test_dl, zbin[ix_test]

    :rtype: PyTorch tensors
    """

    flux, flux_err, fmes, vinv, isnan, zbin, ref_id = data

    def sub(ix, flux,  fmes, vinv, isnan, zbin):
        """Generate a sample dataset.

        :param ix: Index tensor.

        :type ix: tensor of type 8-bit integer

        :return: Dataset with information of flows.

        :rtype: map-style dataset
        """
        ds = TensorDataset(flux[ix].cuda(), fmes[ix].cuda(), vinv[ix].cuda(), isnan[ix].cuda(), zbin[ix].cuda())
        
        return ds
    
    #ix_train = 1*(inds != ifold).type(torch.bool)
    #ix_test = 1*(inds == ifold).type(torch.bool)
    ix_train = torch.ByteTensor(1*(inds != ifold))
    ix_test = torch.ByteTensor(1*(inds == ifold))

    # Here we deterministically remove galaxies.
    if not Ntrain == 'all':
        # You *don't* need to use a different number because of the folds.
        Nsel = int(Ntrain)
        
        ix_train[Nsel:] = 0
        
    # sub-samples
    ds_train = sub(ix_train, flux, fmes, vinv, isnan, zbin)
    ds_test = sub(ix_test, flux, fmes, vinv, isnan, zbin)

    # the bath size would have to be an input get_loaders 
    # ¿hace cross-validation with batch_size?
    train_dl = DataLoader(ds_train, batch_size=500, shuffle=True) 
    test_dl = DataLoader(ds_test, batch_size=100)
    
    return train_dl, test_dl, zbin[ix_test]

