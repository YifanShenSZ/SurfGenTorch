'''
To load and process ab initio data

The prerequisite of any fit is data
Specifically, for Hd construction data comes from ab initio

The ab initio data will be classified into regular or degenerate,
based on energy gap and degeneracy threshold

In addition, geometries can be extracted alone to feed pretraining
'''

from pathlib import Path
from typing import List
import numpy
import torch
import PythonLibrary.LinearAlgebra as PLLA
import symmetry

DegThresh = 0.0001

class GeomLoader:
    def __init__(self, cartdim:int, intdim:int):
        self.cartgeom = numpy.empty(cartdim)
        self.intgeom  = numpy.empty(intdim)

class geom:
    def __init__(self, loader:GeomLoader):
        self.SAIgeom  = symmetry.symmetrize(loader.)

class DataLoader:
    def __init__(self, cartdim:int, intdim:int, NStates:int):
        GeomLoader.__init__(cartdim, intdim)
        self.BT     = numpy.empty((cartdim, intdim))
        self.energy = numpy.empty(NStates)
        self.dH     = numpy.empty((NStates, NStates, cartdim))

class RegularData:
    def __init__(self, loader:DataLoader):
        self.intgeom = torch.tensor(loader.intgeom, requires_grad=True)
        self.BT      = torch.tensor(loader.BT)
        self.energy  = torch.tensor(loader.energy)
        self.dH      = torch.tensor(loader.dH)

class DegenerateData:
    def __init__(self, loader:DataLoader):
        self.intgeom = torch.tensor(loader.intgeom, requires_grad=True)
        self.BT      = torch.tensor(loader.BT)
        # Diagonalize ▽H . ▽H
        dHdH = PLLA.matdotmul(loader.dH, loader.dH)
        eigval, eigvec = dHdH.symeig(True, True)
        dHdH = eigvec.transpose(0, 1)
        # Transform H and dH
        self.H  = dHdH.mm(loader.energy.diag().mm(eigvec))
        self.dH = PLLA.UT_A3_U(dHdH, loader.dH, eigvec)

class DataSet(torch.utils.data.Dataset):
    def __init__(self, example:List):
        self.example = example
    
    # Override the __len__ method to infer the size of the data set
    def __len__(self):
        return len(self.example)
    # Override the __getitem__ method to load custom data
    def __getitem__(self, index):
        return self.example[index]

def read_GeomSet(data_set:List[Path], origin:torch.Tensor, intdim:int)