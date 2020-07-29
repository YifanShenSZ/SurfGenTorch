'''
A feedforward neural network to reduce dimensionality

Each irreducible has its own dimensionality reduction network
'''

from typing import List

import torch
import torch.nn

import SSAIC

class DimRedNet(torch.nn.Module):
    # max_depth <= 0 means unlimited
    def __init__(self, irred:int, max_depth=0) -> None:
        super(DimRedNet, self).__init__()
        assert -1 < irred < SSAIC.NIrred
        # Determine depth
        depth = SSAIC.NSAIC_per_irred[irred] - 1
        if max_depth > 0 and depth > max_depth: depth = max_depth
        # Fully connected layer to reduce dimensionality
        self.fc = torch.nn.ModuleList()
        for i in range(depth):
            self.fc.append(torch.nn.Linear(
                SSAIC.NSAIC_per_irred[irred] - i,
                SSAIC.NSAIC_per_irred[irred] - i - 1,
                bias=False))
        # Fully connected layer to inverse the reduction
        self.fc_inv = torch.nn.ModuleList()
        for i in range(depth):
            self.fc_inv.append(torch.nn.Linear(
                SSAIC.NSAIC_per_irred[irred] - i - 1,
                SSAIC.NSAIC_per_irred[irred] - i,
                bias=False))
    
    #def reduce(self, x:torch.Tensor) -> torch.Tensor:
    #    y = x.clone()
    #    # Reduce dimensionality
    #    for layer in self.fc:
    #        y = layer(y)
    #        y = torch.tanh(y)
    #    return y

    #def inverse(self, x:torch.Tensor) -> torch.Tensor:
    #    y = x.clone()
    #    # Inverse the reduction
    #    for layer in reversed(self.fc_inv):
    #        y = layer(y)
    #        y = torch.tanh(y)
    #    return y

    # For pretraining
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        y = x.clone()
        # Reduce dimensionality
        for layer in self.fc:
            y = layer(y)
            y = torch.tanh(y)
        # Inverse the reduction
        for layer in reversed(self.fc_inv):
            y = torch.tanh(y)
            y = layer(y)
        return y

# The dimensionality reduction networks of each irreducible
net_list = None

def define_DimRed():
    global net_list
    net_list = []

def reduce(x:List[torch.Tensor]) -> List[torch.Tensor]:
    y = x.copy()
    for i in range(SSAIC.NIrred):
        y[i] = net_list[i].reduce(y[i])
    return y

def inverse(x:List[torch.Tensor]) -> List[torch.Tensor]:
    y = x.copy()
    for i in range(SSAIC.NIrred):
        y[i] = net_list[i].inverse(y[i])
    return y