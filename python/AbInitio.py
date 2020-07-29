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

import FortranLibrary as FL
import PythonLibrary.LinearAlgebra as PLLA

import SSAIC

DegThresh = 0.0001

class GeomLoader:
    def __init__(self, cartdim:int, intdim:int) -> None:
        self.cartgeom = numpy.empty(cartdim)
        self.intgeom  = numpy.empty(intdim)

    def cart2int(self) -> None:
        FL.InternalCoordinate(self.cartgeom, self.intgeom)

class geom:
    def __init__(self, loader:GeomLoader) -> None:
        self.SAIgeom = SSAIC.compute_SSAIC(loader.intgeom)

class DataLoader(GeomLoader):
    def __init__(self, cartdim:int, intdim:int, NStates:int) -> None:
        GeomLoader.__init__(cartdim, intdim)
        self.BT     = numpy.empty((cartdim, intdim))
        self.energy = numpy.empty(NStates)
        self.dH     = numpy.empty((NStates, NStates, cartdim))

class RegularData:
    def __init__(self, loader:DataLoader) -> None:
        self.intgeom = torch.tensor(loader.intgeom, requires_grad=True)
        self.BT      = torch.tensor(loader.BT)
        self.energy  = torch.tensor(loader.energy)
        self.dH      = torch.tensor(loader.dH)

class DegenerateData:
    def __init__(self, loader:DataLoader) -> None:
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
    def __init__(self, example:List) -> None:
        self.example = example
    
    # Override the __len__ method to infer the size of the data set
    def __len__(self) -> int:
        return len(self.example)
    # Override the __getitem__ method to load custom data
    def __getitem__(self, index):
        return self.example[index]

def collate(batch): return batch

def read_GeomSet(data_set:List[Path]) -> DataSet:
    # Data set loader
    geom_list = []
    # Read data set files
    for set in data_set:
        NData = len(open(set/"energy.data").readlines())
        # Raw data loader
        RawGeomLoader = []
        # Cartesian geometry
        with open(set/"geom.data", 'r') as f:
            for i in range(NData):
                RawGeomLoader.append(GeomLoader(SSAIC.cartdim, SSAIC.intdim))
                for j in range(int(SSAIC.cartdim / 3)):
                    strs = f.readline().split()
                    RawGeomLoader[i].cartgeom[int(3*j)  ] = strs[1]
                    RawGeomLoader[i].cartgeom[int(3*j)+1] = strs[2]
                    RawGeomLoader[i].cartgeom[int(3*j)+2] = strs[3]
        # Process raw data
        for raw in RawGeomLoader:
            # Modify raw data
            raw.cart2int()
            # Insert to data set loader
            geom_list.append(geom(raw))
    # Create DataSet with data set loader
    GeomSet = DataSet(geom_list)
    return GeomSet