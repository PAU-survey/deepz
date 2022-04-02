#!/usr/bin/env python
# encoding: UTF8

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

class MDNNetwork(nn.Module):
    """Mixture density network."""

    def __init__(self, Ninput, fr=0.02):
        """Initalize the network.
           :param Ninput: {int} Number of inputs, #Bands+#Features.
           :param fr: {float} Dropout fraction.
        """

        super().__init__()
        
        n2 = 600
        n2 = 400

        n3 = 250
        zp = nn.BatchNorm1d(Ninput)
        lin1 = nn.Sequential(\
               nn.Linear(Ninput, n2), nn.Dropout(fr), nn.ReLU())

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
        """Get the parameters describing the MDN.
           :param X: {tensor} Network input.
        """

        X = self.net(X)
      
        logalpha = self.lin_logalpha(X)
        logalpha = logalpha - torch.logsumexp(logalpha, 1)[:,None]
        
        mu = self.lin_mu(X)
        logsig = self.lin_logvar(X)
    
        return logalpha, mu, logsig
    
    def forward(self, X):
        """Get the probability function evaluated on a grid.
           :param X: {tensor} Network input.
        """

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
        """Evaluates the logarithmic probability.
           :param X: {tensor} Network input.
           :param y: {tensor} Spectroscopic redshift (label).
        """

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
    
class Encoder(nn.Module):
    """Encoder network."""
    
    def __init__(self, Nfeat=10, Nl=10, Nbands=46):
        """Initialize the network.
           :param Nfeat: {int} Feature space dimention.
           :param Nl: {int} Number of layers.
           :param Nbands: {int} Number of input bands.
        """

        super().__init__()
        
        Nw = 250
        fr = 0.01
        L = [nn.Linear(Nbands, Nw)]
        
        for i in range(Nl):
            L += [nn.BatchNorm1d(Nw), nn.Linear(Nw, Nw), nn.Dropout(fr), nn.ReLU()]
            
        self.bulk = nn.Sequential(*L)
        self.last = nn.Linear(Nw, Nfeat)
        
    def forward(self, x):
        """Encode flux ratios to features.
          :param x: {tensor} Flux ratios.
        """

        x = self.bulk(x)
        x = self.last(x)
        
        return x
    
class Decoder(nn.Module):
    """Decoder network."""
    
    def __init__(self, Nfeat=10, Nl=10, Nbands=46):
        """Initialize the network.
           :param Nfeat: {int} Feature space dimention.
           :param Nl: {int} Number of layers.
           :param Nbands: {int} Number of input bands.
        """

        super().__init__()
        
        Nw = 250
        fr = 0.01
        L = [nn.Linear(Nfeat, Nw)]
        
        for i in range(Nl):
            L += [nn.BatchNorm1d(Nw), nn.Linear(Nw,Nw), nn.Dropout(fr), nn.ReLU()]
            
        self.bulk = nn.Sequential(*L)
        self.last = nn.Linear(Nw, Nbands)
        
    def forward(self, x):
        """Decode features to flux ratios.
           :param x: {tensor} Features.
        """

        x = self.bulk(x)
        x = self.last(x)
        
        return x

class Deepz(nn.Module):
    """The Deepz network."""

    def __init__(self, Nbands, Nfeat=10, Nl=5):
        """Initialize network
           :param Nbands: {int} Number of input bands.
           :param Nl: {int} Number of layers in encoder/decoder.
        """
        super().__init__()

        # Setting Nl=5. I think we used 5 in the paper.
        self.enc = Encoder(Nl=Nl, Nbands=Nbands)
        self.dec = Decoder(Nl=Nl, Nbands=Nbands)
        self.mdn = MDNNetwork(Nbands+Nfeat)

    def forward(self, flux, coadd):
        """Predict the p(z) for each galaxy.
           :param flux: {tensor} All the coadded fluxes.
           :param coadd: {tensor} Newly constructed coadds.
        """

        feat = self.enc(flux)
        Binput = torch.cat([coadd, feat], 1)
        pred = self.mdn(Binput)

        return pred

    def loss(self, flux, coadd, zs):
        """Evaluate the MDN loss.
           :param flux: {tensor} All the coadded fluxes.
           :param coadd: {tensor} Newly constructed coadds.
           :param zs: {tensor} Spectroscopic redshift.
        """

        feat = self.enc(flux)
        Binput = torch.cat([coadd, feat], 1)

        _, loss = self.mdn.loss(Binput, zs)

        return loss

    def pred_recon_loss(self, flux, coadd, zs):
        """Predict the p(z) and return the loss.
           :param flux: {tensor} All the coadded fluxes.
           :param coadd: {tensor} Newly constructed coadds.
           :param zs: {tensor} Spectroscopic redshift.
        """

        feat = self.enc(flux)
        Binput = torch.cat([coadd, feat], 1)
        pred = self.mdn(Binput)

        _, loss = self.mdn.loss(Binput, zs)

        recon = self.dec(feat)

        return pred, recon, loss
