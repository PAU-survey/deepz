#!/usr/bin/env python
# encoding: UTF8

# Train the network on observed PAUS data.
from IPython.core import debugger as ipdb
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import chain
from IPython.core import debugger

import torch

# Later versions works slightly different with the galaxy selection. The results will
# therefore 
assert torch.__version__.startswith('1.0'), 'For some reason the code fails badly on newer PyTorch versions.'

import fire
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt

import paus_data
import networks
import trainer
import utils

def get_loaders(data, ifold, inds):
    """Create data loaders for a specific fold.
       :param ifold: {int} Which fold to use.
       :param inds: {array} Ifold for each galaxy.
    """

    # There might be a better way of doing this. PyTorch does support
    # dataloaders returning dictionaries with tensors.
 
    def sub(ix):
        """Select subset.
           :param ix: {array} Array indices to use.
        """

        ds = TensorDataset(data['flux'][ix].cuda(), data['fmes'][ix].cuda(), \
                           data['vinv'][ix].cuda(), data['isnan'][ix].cuda(), \
                           data['zs'][ix].cuda())
        
        return ds
    
    ix_train = torch.ByteTensor(1*(inds != ifold))
    ix_test = torch.ByteTensor(1*(inds == ifold))

    ds_train = sub(ix_train)
    ds_test = sub(ix_test)

    train_dl = DataLoader(ds_train, batch_size=500, shuffle=True)
    test_dl = DataLoader(ds_test, batch_size=100)
    
    return train_dl, test_dl, data['zs'][ix_test]


def train(data, inds_split, ifold, **config):
    """Train the networks for one fold.
       :param data: {dict} Flux data and redshifts for training.
       :param inds_split: {tensor} Indices explaining how to split in folds.
       :param ifold: {int} Which fold to train.
       :param config: {dict} Dictionary with the configuration.
    """

    inds = inds_split[:len(data['flux'])]

    Nbands = 40 + len(config['bb'])
    net = networks.Deepz(Nbands).cuda()
    path_pretrain = utils.path_pretrain(config['model_dir'], config['pretrain_label'])

    if config['pretrain']:
        print('Loading pretrain:', path_pretrain)
        assert path_pretrain.exists()
        net.load_state_dict(torch.load(path_pretrain))

    train_dl, test_dl, _ = get_loaders(data, ifold, inds)
    K = (net, train_dl, test_dl, config['alpha'], config['keep_last'])

    def params():
        return net.parameters()
   
    wd = 1e-4
    if True: # Since I tested this so many times.
        optimizer = optim.Adam(params(), lr=1e-3, weight_decay=wd)
        trainer.train(optimizer, 100, *K)

    print('main train function...')
    optimizer = optim.Adam(params(), lr=1e-4, weight_decay=wd)
    trainer.train(optimizer, 200, *K)
    
    optimizer = optim.Adam(params(), lr=1e-5, weight_decay=wd)
    trainer.train(optimizer, 200, *K)

    optimizer = optim.Adam(params(), lr=1e-6, weight_decay=wd)
    trainer.train(optimizer, 200, *K)
    
    return net

def train_all(data, inds_split, **config):
    """Train all the folds.
       :param data: {dict} Flux data and redshifts for training.
       :param inds_split: {tensor} Indices explaining how to split in folds.
       :param config: {dict} Configuration dictionary.
    """
 
    C = config
    Nfolds = int(inds_split.max() + 1)
    for ifold in range(Nfolds):

        model_path = utils.path_model(C['model_dir'], C['model_label'], C['catnr'], ifold)
        if model_path.exists():
            print('Alread trained:', ifold)
            continue

        print('Running for:', ifold)
        print('storing to:', model_path)
        net = train(data, inds_split, ifold, **config)
        torch.save(net.state_dict(), model_path)


def pz_fold(catnr, ifold, inds, model_dir, model_label, bb):
    """Estimate the photo-z for one fold.
       :param catnr: {int} Catalogue number.
       :param ifold: {int} Which ifold to use.
       :param inds: {array} Indices to use.
       :param model_dir: {path} Model directory.
       :param model_label: {str} Label describing the model.
       :param bb: {list} List of broad bands to use.
    """

#    debugger.set_trace()
 
    # Here we should not hard-code the number of bands.
    Nbands = 40 + len(bb)
    net = networks.Deepz(Nbands=Nbands).cuda()
    net.eval()
 
    path_model = utils.path_model(model_dir, model_label, catnr, ifold)
    net.load_state_dict(torch.load(path_model))
    
    _, test_dl, zs_test = get_loaders(ifold, inds)

    assert isinstance(inds, torch.Tensor), 'This is required...'
 
    L = []
    for Bflux, Bfmes, Bvinv, Bisnan, Bzs in test_dl:
        Bcoadd, touse = trainer.get_coadd_allexp(Bflux, Bfmes, Bvinv, Bisnan)
        assert touse.all()
            
        with torch.no_grad():
            pred = net(Bcoadd, Bcoadd)

        
        zb_part = 0.001*pred.argmax(1).type(torch.float)
        L.append(zb_part)

    zb_fold = torch.cat(L).cpu().numpy()
    zs_fold = zs_test

    refid_fold = ref_id[inds == ifold]
    D = {'zs': zs_fold, 'zb': zb_fold, 'ref_id': refid_fold}
    
    part = pd.DataFrame(D)
    part['ifold'] = ifold

    return part

                   
def make_catalogue(catnr, model_dir, model_label, bb):
    """Run the photo-z for all folds.
       :param catnr: {int} Which of the indexes to use per fold.
       :param model_dir: {path} Directory where the models are stored.
    """
                   
    L = []
    inds = inds_all[catnr][:len(flux)]

    inds = torch.Tensor(inds) # Inds_all should be a tensor in the first place.
    for ifold in range(5):
        L.append(pz_fold(catnr, ifold, inds, model_dir, model_label, bb))
        
    df = pd.concat(L)
    df = df.set_index('ref_id')

    return df


def photoz_all(model_dir, pretrain_label, model_label, bb, inds_path, catnr=0, pretrain=True, alpha=0.8, keep_last=True):
    """Train the networks and return the catalogs.
       :param model_dir: {str} Directory to store models.
       :param pretrain_label: {str} Label to describe the pretrained model.
       :param model_label: {str} Label to describe the final model.
       :param bb: {str} Broad bands.
       :param inds_path: {path} Path to file containing indices to use.
       :param catnr: {int} Which of the indexes to use per fold.
       :param pretrain: {bool} If using a pretrained network.
       :param alpha: {float} Fraction of measurements used when training.
       :param keep_last: {bool} Keeping at least one measurement per band.
    """

    bb = utils.broad_bands(bb)
    config = {'model_dir': model_dir, 'pretrain_label': pretrain_label, 'model_label': model_label,
              'bb': bb, 'catnr': catnr, 'pretrain': pretrain, 'alpha': alpha, 'keep_last': keep_last}


    # Indices determining the splitting in folds. We could be generating these on the
    # fly if not being specified by the user.
    inds_split = np.loadtxt(inds_path)[config['catnr']]
    data = paus_data.paus(bb)

    train_all(data, inds_split, **config)
    pz = make_catalogue(catnr, model_dir, model_label, bb)
    pz['dx'] = (pz.zb - pz.zs) / (1 + pz.zs)

    sig68 = 0.5*(pz.dx.quantile(0.84) - pz.dx.quantile(0.16))
    print('sig68:', sig68)

    # Not strictly needed, but might make life simpler.
    path_pzcat = Path(model_dir) / f'pzcat_{model_label}_cat{catnr}.csv'
    if not path_pzcat.exists():
        print('Storing catalogs in:', path_pzcat)
        pz.to_csv(path_pzcat)

    return pz

if __name__ == '__main__':
    fire.Fire(photoz_all)
