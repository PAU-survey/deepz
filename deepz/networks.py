#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Created at 

# =============================================================================
# DOCS
# =============================================================================

"""arc_mdn
"""

# =============================================================================
# IMPORTS
# =============================================================================

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

class Network(nn.Module):
    """[Summary]

    :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
    :type [ParamName]: [ParamType](, optional)

    :raises [ErrorType]: [ErrorDescription]

    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """

    def __init__(self, Nbands, fr=0.2):
        super().__init__()
        
        n2 = 600
        n2 = 400

        n3 = 250
        self.zp = nn.BatchNorm1d(Nbands)
        self.lin1 = nn.Sequential(\
                    nn.Linear(Nbands, n2), nn.Dropout(fr), nn.ReLU())

        self.lin2 = nn.Sequential(nn.BatchNorm1d(n2), \
                    nn.Linear(n2, n3), nn.Dropout(fr), nn.ReLU())
        
        L = []
        for i in range(13): # was 13
            layer = nn.Sequential(nn.BatchNorm1d(n3), \
                    nn.Linear(n3, n3), nn.Dropout(fr), nn.ReLU()) 
            L.append(layer)
            
        self.lin3 = nn.Sequential(*L)
        
        self.lin4 = nn.BatchNorm1d(n3)
        self.last = nn.Linear(n3, 2100)
        
    def forward(self, x):
        x = self.zp(x)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)

        x = self.lin4(x)
        x = self.last(x)
        
        return x
    
class MDNNetwork(nn.Module):
    """We feed the galaxy flux ratios and the autoencoder features 
       into the photo-z network. Here the layers follow the same 
       structure as the autoencoder, but with 1 per cent dropout 
       after all linear layers. This network is a mixture density 
       network and describe the redshift distribution as a linear 
       mixture of 10 normal distributions.

    :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
    :type [ParamName]: [ParamType](, optional)

    :raises [ErrorType]: [ErrorDescription]

    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """

    def __init__(self, Nbands, fr=0.02):
        super().__init__()
        
        n2 = 600
        n2 = 400

        n3 = 250
        zp = nn.BatchNorm1d(Nbands)
        lin1 = nn.Sequential(\
               nn.Linear(Nbands, n2), nn.Dropout(fr), nn.ReLU())

        lin2 = nn.Sequential(nn.BatchNorm1d(n2), \
                nn.Linear(n2, n3), nn.Dropout(fr), nn.ReLU())
        
        L = []
        for i in range(10): # was 13
            layer = nn.Sequential(nn.BatchNorm1d(n3), \
                    nn.Linear(n3, n3), nn.Dropout(fr), nn.ReLU()) 
            L.append(layer)
            
        for i in range(3): # was 13
            layer = nn.Sequential(nn.BatchNorm1d(n3), \
                    nn.Linear(n3, n3), nn.ReLU()) 
            L.append(layer)
            
        lin3 = nn.Sequential(*L)
        
        lin4 = nn.Sequential(nn.BatchNorm1d(n3), nn.ReLU())
        
        self.net = nn.Sequential(zp, lin1, lin2, lin3, lin4)

        n_gaussian = 10
        self.lin_logalpha = nn.Linear(250, n_gaussian)
        self.lin_mu = nn.Linear(250, n_gaussian)
        self.lin_logvar = nn.Linear(250, n_gaussian) # Not changing name because of pretraining...
    
    def get_dist(self, X):
        X = self.net(X)
      
        logalpha = self.lin_logalpha(X)
        logalpha = logalpha - torch.logsumexp(logalpha, 1)[:,None]
        
        mu = self.lin_mu(X) #.abs()
        logsig = self.lin_logvar(X)
       
        # Testing scaling the sigma..
        #sig = torch.exp(logsig)
        #sig = sig + 0.001 #:*1.15
        #logsig = torch.log(sig)
    
        return logalpha, mu, logsig
    
    def forward(self, X):
        # This way of using the forward is for not having to modify the
        # test code everywhere.
      
        #fac = 2
        fac = 1
        z = torch.linspace(0, 2.1, fac*2100).cuda()
        
        logalpha, mu, logsig = self.get_dist(X)
        sig = torch.exp(logsig)
        
        
        log_prob = logalpha[:,None,:] - 0.5*(z[None,:,None] - mu[:,None,:]).pow(2) / sig[:,None,:].pow(2) - logsig[:,None,:]
        log_prob = torch.logsumexp(log_prob, 2)

        prob = torch.exp(log_prob)
        prob = prob / prob.sum(1)[:,None]

        
        return prob    
        
    def loss(self, X, y):
        """Evaluates the logarithmic probability."""

        logalpha, mu, logsig = self.get_dist(X)
        y = y.unsqueeze(1)
        sig = torch.exp(logsig)
        log_prob = logalpha - 0.5*(y - mu).pow(2) / sig.pow(2) - logsig
        
        log_prob = torch.logsumexp(log_prob, 1)
        loss = -log_prob.mean()
  

        # For testing....
        z = torch.linspace(0, 2.1, 2100).cuda()
        
        sig = torch.exp(logsig)
        log_prob = logalpha[:,None,:] - 0.5*(z[None,:,None] - mu[:,None,:]).pow(2) / sig[:,None,:].pow(2) - logsig[:,None,:]
        log_prob = torch.logsumexp(log_prob, 2)
    
    
        return log_prob, loss
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
