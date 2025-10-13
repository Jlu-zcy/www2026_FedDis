from typing import Any
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from models.layers import STConvBlock, cal_cheb_polynomial,cal_laplacian

class CLUB(nn.Module):  

    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()       
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)        
        sample_size = x_samples.shape[0]
        random_index = torch.randperm(sample_size).long()        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def loglikeli(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

class ST_encoder(nn.Module):
    def __init__(self, num_nodes, d_input, d_output, Ks, Kt, blocks, input_window, drop_prob, device):
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_dim = d_output
        self.output_dim = d_output
        self.Ks = Ks
        self.Kt = Kt
        self.blocks = blocks
        self.input_window = input_window
        self.output_window = 12
        self.drop_prob = drop_prob


        self.blocks[0][0] = self.feature_dim
        if self.input_window - len(self.blocks) * 2 * (self.Kt - 1) <= 0:
            raise ValueError('Input_window must bigger than 4*(Kt-1) for 2 STConvBlock'
                             ' have 4 kt-kernel convolutional layer.')
        self.device = device
        self.input_conv=nn.Conv2d(d_input, d_output, 1)

        self.st_conv1 = STConvBlock(self.Ks, self.Kt, self.num_nodes,
                                    self.blocks[0], self.drop_prob, self.device)
        self.st_conv2 = STConvBlock(self.Ks, self.Kt, self.num_nodes,
                                    self.blocks[1], self.drop_prob, self.device)

    def forward(self, x ,graph):       
        lap_mx = cal_laplacian(graph)
        Lk = cal_cheb_polynomial(lap_mx, self.Ks)
        x=self.input_conv(x)
        x_st1 = self.st_conv1(x,Lk)  
        x_st2 = self.st_conv2(x_st1,Lk)  
        return x_st2

    def variant_encode(self,x,graph):
        x=self.input_conv(x)
        x_st1 = self.st_conv1(x, graph)  
        x_st2 = self.st_conv2(x_st1, graph)  
        return x_st2

class Configs(object):
    
    def __init__(self,config={}):
        self.config = config

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise KeyError('{} is not in the config'.format(key))

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return key in self.config