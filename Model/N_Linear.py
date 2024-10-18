# -*- coding: utf-8 -*-
"""
Created on Wed May 24 02:56:30 2023

@author: RYU

original from:
https://github.com/cure-lab/LTSF-Linear/tree/main/models

arranged by:
https://today-1.tistory.com/60

"""
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#%%

class LTSF_NLinear(torch.nn.Module):
    def __init__(self, window_size, forcast_size, individual, feature_size):
        super(LTSF_NLinear, self).__init__()
        self.window_size = window_size
        self.forcast_size = forcast_size
        self.individual = individual
        self.channels = feature_size
        
        if self.individual:
            self.Linear = torch.nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(torch.nn.Linear(self.window_size, self.forcast_size))
                self.Linear.weight = nn.Parameter((1/self.window_size)*torch.ones([self.forcast_size,self.window_size]))
        else:
            self.Linear = torch.nn.Linear(self.window_size, self.forcast_size)
            self.Linear.weight = nn.Parameter((1/self.window_size)*torch.ones([self.forcast_size,self.window_size]))
            
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.forcast_size, x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
            
        x = x + seq_last # [Batch, Output length, Channel]
        return x  # [Batch, Output length, Channel]

