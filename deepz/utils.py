#!/usr/bin/env python
# encoding: UTF8

import sys
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import networks



def get_nets(path_base, use_mdn, pretrain=True, Nbands=46):
    """Initialize networks."""
    
    Nfeat = 10
    Nl = 5
    kwds = {'Nfeat': 10, 'Nl': 5, 'Nbands': Nbands}
    enc = networks.Encoder(**kwds).cuda()
    dec = networks.Decoder(**kwds).cuda()
   
    if use_mdn:
        net_pz = networks.MDNNetwork(Nbands+Nfeat).cuda()
    else:
        net_pz = networks.Network(Nbands+Nfeat).cuda()

    # And then loading the pretrained network.
    if pretrain:
        enc.load_state_dict(torch.load(path_base.format('enc')))
        dec.load_state_dict(torch.load(path_base.format('dec')))
        net_pz.load_state_dict(torch.load(path_base.format('netpz')))

    net_pz.train()
    #enc.eval()
    #dec.eval()
    
    return enc, dec, net_pz


def get_loaders(ifold, inds, data):
    """Get loaders for specific fold."""
    
    flux, flux_err, fmes, vinv, isnan, zbin, ref_id = data
    
    def sub(ix):
        ds = TensorDataset(flux[ix].cuda(), fmes[ix].cuda(), vinv[ix].cuda(), \
                           isnan[ix].cuda(), zbin[ix].cuda())
        
        return ds
    
    ix_train = torch.ByteTensor(1*(inds != ifold))
    ix_test = torch.ByteTensor(1*(inds == ifold))
    
    ds_train = sub(ix_train)
    ds_test = sub(ix_test)


    train_dl = DataLoader(ds_train, batch_size=500, shuffle=True)
    test_dl = DataLoader(ds_test, batch_size=100)
    
    return train_dl, test_dl, zbin[ix_test]


def all_inds():
    """Load a matrix with all the indices."""

#    index_path = '/nfs/pic.es/user/e/eriksen/papers/deepz/redux/inds_large_v1.txt'
    index_path = '/nfs/astro/eriksen/deepz/inds/inds_large_v1.txt'
    inds_all = np.loadtxt(index_path)
    inds_all = torch.from_numpy(inds_all)
    
    return inds_all
