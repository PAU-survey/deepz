#!/usr/bin/env python
# encoding: UTF8

import numpy as np
import torch
import var_networks

# My hackish way of forcing this...
#path_backup = sys.path.copy()
#sys.path = ['/nfs/astro/eriksen/play_deepz/sims/flag']
import arch_mdn
##sys.path = path_backup

print('loaded path', arch_mdn.__file__)
#import ranker

from torch.utils.data import TensorDataset, DataLoader

def get_nets(path_base, use_mdn, pretrain=True, Nbands=46):
    """Initialize networks.? si se inicializa por qué hay un 
    parámetro pretrain?

    :param path_base: ?

    :type path_base: ?

    :param use_mdn: Is a boolean. If is True, ...?
    
    :type use_mdm: boolean

    :param pretrain: Boolean, if is True then the networks have been 
    trained.

    :type pretrain:

    :param Nbands: The total number of bands is 46 by default.

    :type Nbands: int
    
    
    """
    
    #: Nfeat is 
    Nfeat = 10

    #: Nl is 
    Nl = 5

    #: kwds
    kwds = {'Nfeat': 10, 'Nl': 5, 'Nbands': Nbands}

    #: enc encoder
    enc = var_networks.Encoder(**kwds).cuda()

    #: dec, is decoder
    dec = var_networks.Decoder(**kwds).cuda()
   
    if use_mdn:
        net_pz = arch_mdn.MDNNetwork(Nbands+Nfeat).cuda()
    else:
        net_pz = arch_mdn.Network(Nbands+Nfeat).cuda()

    # And then loading the pretrained network.
    if pretrain:
        enc.load_state_dict(torch.load(path_base.format('enc')))
        dec.load_state_dict(torch.load(path_base.format('dec')))
        net_pz.load_state_dict(torch.load(path_base.format('netpz')))

    net_pz.train()
    #enc.eval()
    #dec.eval()
    
    return enc, dec, net_pz



def all_inds():
    """Load a matrix with all the indices."""

#    index_path = '/nfs/pic.es/user/e/eriksen/papers/deepz/redux/inds_large_v1.txt'
    index_path = '/nfs/astro/eriksen/deepz/inds/inds_large_v1.txt'
    inds_all = np.loadtxt(index_path)
    inds_all = torch.from_numpy(inds_all)
    
    return inds_all
