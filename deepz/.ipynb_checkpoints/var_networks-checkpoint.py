#!/usr/bin/env python
# encoding: UTF8

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
    Puede ser que el autoencoder no sea el reflejo del 
    encoder?. Hacer dibujo.
    
    
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