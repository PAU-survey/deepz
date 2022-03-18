#!/usr/bin/env python
# encoding: UTF8
#
# Experience with a MCMC approach over an already trained network.

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import time
from torch.utils.data import TensorDataset, DataLoader
from IPython.core import debugger as ipdb

import utils

Ntrain = 'all'

import trainer_sexp
import paus_data
flux, flux_err, fmes, vinv, isnan, zbin, ref_id = paus_data.paus()

catnr = 0 #if len(sys.argv) == 1 else int(sys.argv[1])
inds_all = np.loadtxt('/cephfs/pic.es/astro/scratch/eriksen/deepz/inds/inds_large_v1.txt')


def gen_config():
    catnr = 0
    keep_last = True
    alpha = 0.8

    verpretrain = 8
    Ntrain = 'all'
    keep_last = False

    label = 'march11'
    label = 'memba_march16'

    version = 9
    use_mdn = True
    model_dir = Path('/cephfs/pic.es/astro/scratch/eriksen/deepz/redux/train') / str(version)

    pretrain = False if verpretrain == 'no' else True
    config = {'verpretrain': verpretrain, 'Ntrain': Ntrain, 'catnr': catnr, 'use_mdn': use_mdn,
              'Ntrain': Ntrain, 'pretrain': pretrain, 'keep_last': keep_last}

    config['Nexp'] = 0
    config['alpha'] = alpha

    out_fmt = '{net}_'+label+'_ifold{ifold}.pt'
    out_fmt = str(model_dir / out_fmt)

    config['out_fmt'] = out_fmt

    return config

# In[6]:
def get_loaders(ifold, inds):
    """Get loaders for specific fold."""

    def sub(ix):
        ds = TensorDataset(flux[ix].cuda(), flux_err[ix].cuda(), fmes[ix].cuda(), vinv[ix].cuda(), isnan[ix].cuda(), zbin[ix].cuda())

        return ds

    ix_train = torch.ByteTensor(1*(inds != ifold))
    ix_test = torch.ByteTensor(1*(inds == ifold))

    # Here we deterministically remove galaxies.
    if not Ntrain == 'all':
        # You *don't* need to use a different number because of the folds.
        Nsel = int(Ntrain)

        ix_train[Nsel:] = 0


    ds_train = sub(ix_train)
    ds_test = sub(ix_test)

    train_dl = DataLoader(ds_train, batch_size=500, shuffle=True)
    test_dl = DataLoader(ds_test, batch_size=100)

    return train_dl, test_dl, zbin[ix_test]

def pz_fold(ifold, inds, out_fmt, use_mdn, err_frac=0.):
    """Estimate the photo-z for one fold."""

    # Load network..
    #model_dir = Path('/nfs/astro/eriksen/deepz/encmodels_data')
    #path = str(model_dir/ f'{ifold}.pt')

    # Loading the networks...
    net_base_path = out_fmt.format(ifold=ifold, net='{}')
    enc, dec, net_pz = utils.get_nets(str(net_base_path), use_mdn)
    enc.eval(), dec.eval(), net_pz.eval()

    _, test_dl, zbin_test = get_loaders(ifold, inds)


    assert isinstance(inds, torch.Tensor), 'This is required...'

    # OK, this needs some improvement...
    L = []

    for Bflux, Bflux_err, Bfmes, Bvinv, Bisnan, Bzbin in test_dl:
        Bcoadd, Bcoadd_err, touse = trainer_sexp.get_coadd(Bflux, Bflux_err, Bfmes, Bvinv, Bisnan, alpha=1)
        assert touse.all()

        K = []
        for i in range(100):
            xBcoadd = torch.normal(mean=Bcoadd, std=err_frac*Bcoadd_err)

            feat = enc(xBcoadd)
            Binput = torch.cat([xBcoadd, feat], 1)
            pred = net_pz(Binput).detach().cpu()

            K.append(pred)

#        ipdb.set_trace()
        #pred = np.mean([x.numpy() for x in K], axis=0)
        pred = np.sum([x.numpy() for x in K], axis=0)

        pred = torch.Tensor(pred)


        zb_part = 0.001*pred.argmax(1).type(torch.float)
        L.append(zb_part)

    zb_fold = torch.cat(L).detach().cpu().numpy()
    zs_fold = 0.001*zbin_test.type(torch.float)

    refid_fold = ref_id[inds == ifold]
    D = {'zs': zs_fold, 'zb': zb_fold, 'ref_id': refid_fold}

    part = pd.DataFrame(D)
    part['ifold'] = ifold
    #part = np.vstack([zs_fold, zb_fold]).T

    return part

def photoz_all(**config):
    """Run the photo-z for all folds."""

    L = []
    inds = inds_all[config['catnr']][:len(flux)]

    inds = torch.Tensor(inds) # Inds_all should be a tensor in the first place.
    for ifold in range(5):
        L.append(pz_fold(ifold, inds, config['out_fmt'], config['use_mdn'], config['err_frac']))

    df = pd.concat(L)
    df = df.set_index('ref_id')

    return df

config = gen_config()


#print('To store at:')
#print(out_fmt)

#train_all(**config)

for err_frac in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]:
    config['err_frac'] = err_frac
    t1 = time.time()
    pz = photoz_all(**config)
    pz['dx'] = (pz.zb - pz.zs) / (1 + pz.zs)

    sig68 = 0.5*(pz.dx.quantile(0.84) - pz.dx.quantile(0.16))
    #print('keep_last', keep_last, 'alpha', alpha, 'sig68', sig68)
    print(err_frac, sig68)

print('time eval', time.time() - t1)
