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
    def __init__(self, init_dim:int, max_depth=0) -> None:
        super(DimRedNet, self).__init__()
        # Determine depth
        depth = init_dim - 1
        if max_depth > 0 and depth > max_depth: depth = max_depth
        # Fully connected layer to reduce dimensionality
        self.fc = torch.nn.ModuleList()
        for i in range(depth):
            self.fc.append(torch.nn.Linear(
                init_dim - i, init_dim - i - 1,
                bias=False))
        # Fully connected layer to inverse the reduction
        self.fc_inv = torch.nn.ModuleList()
        for i in range(depth):
            self.fc_inv.append(torch.nn.Linear(
                init_dim - i - 1, init_dim - i,
                bias=False))

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

    def freeze(self, freeze:int) -> None:
        assert freeze < len(self.fc), "All layers are frozen, so nothing to train"
        for i in range(freeze):
            self.fc[i].weight.requires_grad = False
            self.fc_inv[i].weight.requires_grad = False

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