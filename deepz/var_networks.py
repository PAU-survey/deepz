#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created at 

# =============================================================================
# DOCS
# =============================================================================

"""Autoencoder: a type of unsupervised neural network whose intent is to reduce 
    noise and extract features without knowing the redshift, making it possible 
    to train it with a larger dataset. We input the flux ratios by dividing on 
    the i-band flux. In the first step, the encoder maps raw information into a 
    lower dimensionality feature space, whereas the second step attempts to map 
    it to the original input data in the original dimensions.
"""

# =============================================================================
# IMPORTS
# =============================================================================

from torch import nn

class Encoder(nn.Module):
    """Encoder network.
    objects belonging to this class have the following attributes:

    - Nfeat:
    
    - Nl: Number of linear layers

    - Nbands: Number of narrow and broad bands
    
    
    """
    
    def __init__(self, Nfeat=10, Nl=10, Nbands=46):
        super().__init__()
        
        Nw = 250
        fr = 0.01
        L = [nn.Linear(Nbands, Nw)]
        
        for i in range(Nl):
            L += [nn.BatchNorm1d(Nw), nn.Linear(Nw, Nw), nn.Dropout(fr), nn.ReLU()]
            
        self.bulk = nn.Sequential(*L)
        self.last = nn.Linear(Nw, Nfeat)
        
    def forward(self, x):
        x = self.bulk(x)
        x = self.last(x)
        
        return x
    
class Decoder(nn.Module):
    """Decoder network.
    """
    
    def __init__(self, Nfeat=10, Nl=10, Nbands=46):
        super().__init__()
        
        Nw = 250
        fr = 0.01
        L = [nn.Linear(Nfeat, Nw)]
        
        for i in range(Nl):
            L += [nn.BatchNorm1d(Nw), nn.Linear(Nw,Nw), nn.Dropout(fr), nn.ReLU()]
            
        self.bulk = nn.Sequential(*L)
        self.last = nn.Linear(Nw, Nbands)
        
    def forward(self, x):
        x = self.bulk(x)
        x = self.last(x)
        
        return x