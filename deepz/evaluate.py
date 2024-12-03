#!/usr/bin/env python
# encoding: UTF8

import pandas as pd
import torch

from . import networks
from . import utils

def gen_photoz(cat_val, model_path, bands, norm_band='flux_i'):
    """Generate the photometric redshift."""
    
    # Normalize the input data.
    inputbandsAllbands_i = cat_val[bands].values 
    inputbandsAllbands_i = torch.Tensor(inputbandsAllbands_i)
#    norm_band = utils.norm_band(BB, norm_band)
    norm_ind = bands.index(norm_band)
    norm = inputbandsAllbands_i[:, norm_ind]
    inputsbands = torch.Tensor(inputbandsAllbands_i / norm[:, None]).cuda()
    
    # Load the network.
    Nbands = 45
    net = networks.Deepz(Nbands).cuda()
    net.load_state_dict(torch.load(model_path, weights_only=False))
    net = net.eval()
    
    # Photometric redshift prediction.
    print('inputbands_NB', inputsbands.shape)
    pred = net(inputsbands, inputsbands)
    zp_part = 0.001*pred.argmax(1).type(torch.float)
    zp_fold = zp_part.cpu().numpy()

    # Directly adding the zs and mag_i. Not pure, but user friendly.
    df_z = pd.DataFrame({'z_nn': zp_fold, 'ref_id': cat_val.ref_id,
                         'mag_i': cat_val.mag_i, 'zs': cat_val.zs})

    return df_z

