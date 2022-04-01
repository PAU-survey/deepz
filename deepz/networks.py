#!/usr/bin/env python
# encoding: UTF8

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

class MDNNetwork(nn.Module):
    """Mixture density network."""

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
    
    
    
class Conv(nn.Module):
    """Convolutional neural network."""
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.LeakyReLU(0.1), \
                              nn.MaxPool1d(2))

        self.conv2 = nn.Sequential(nn.Conv1d(32, 64, 3, padding=1), nn.LeakyReLU(0.1), \
                              nn.MaxPool1d(2))

        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, 3, padding=1), nn.LeakyReLU(0.1), \
                              nn.MaxPool1d(2))

        self.conv = nn.Sequential(self.conv1, self.conv2, self.conv3)
        
        # Testing reducing the number of neurons here...
        self.dens = nn.Linear(128*5, 128)
        
    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        x = x.view(len(x), -1)
        
        #x = self.dens(x.view(len(x), -1))
        
        return x


class Encoder(nn.Module):
    """Encoder network."""
    
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
    """Decoder network."""
    
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
