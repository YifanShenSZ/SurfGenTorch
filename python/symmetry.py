'''
Symmetry adapted and scaled internal coordinate

The internal coordinates have to be nondimensionalized first:
    for length, ic = (ic - origin) / origin
    for angle , ic = ic

A scaled internal coordinate is defined by:
    if self = scaler: sic = pi * erf(ic)
    else            : sic = ic * exp(-alpha * scaler)
'''

from pathlib import Path
from typing import List
import re
import numpy
import numpy.linalg
import torch

# Number of irreducible representations
NIrred = 1
# A matrix, as usual product table
product_table = []
# symmetry_adaptation[i][j] contains the definition of
# j-th symmetry adapted internal coordinate in i-th irreducible
symmetry_adaptation = []

# Symmetry Adapted Internal Coordinate Definition
class SymAdIntCoordDef:
    def __init__(self):
        self.coeff  = []
        self.IC     = []
        self.scaler = []
        self.alpha  = []

def define_symmetry(symmetry_file:Path) -> None:
    with open(symmetry_file, 'r') as f:
        # Number of irreducible representations
        f.readline()
        NIrred = int(f.readline())
        # Product table
        f.readline()
        product_table = []
        for _ in range(NIrred):
            temp = f.readline().split()
            for _ in range(len(temp)): temp[_] = int(temp[_]) - 1
            product_table.append(temp)
        # Number of symmetry adapted coordinates per irreducible
        f.readline()
        NSAIC_per_irred = f.readline().split()
        for _ in range(len(NSAIC_per_irred)): NSAIC_per_irred[_] = int(NSAIC_per_irred[_])
        # Coordinates of irreducibles
        f.readline()
        symmetry_adaptation = []
        for _ in range(NIrred):
            SAICD_list = [None] * NSAIC_per_irred[_]
            SAICD_index = -1
            while True:
                line = f.readline()
                if not line: break
                strs = line.split()
                if not re.match('-?\d+', strs[0]): break
                if re.match('^\d+$', strs[0]):
                    SAICD_index += 1
                    SAICD_list[SAICD_index] = SymAdIntCoordDef()
                    del strs[0]
                SAICD_list[SAICD_index].coeff.append(float(strs[0]))
                SAICD_list[SAICD_index].IC   .append(int(strs[1])-1)
                if len(strs) > 2:
                    SAICD_list[SAICD_index].scaler.append(int(strs[2])-1)
                    SAICD_list[SAICD_index].alpha .append(float(strs[3]))
            # Normalize linear combination coefficients
            for SAICD in SAICD_list:
                norm = numpy.linalg.norm(numpy.array(SAICD.coeff))
                for _ in range(len(SAICD.coeff)): SAICD.coeff[_] /= norm
            symmetry_adaptation.append(SAICD_list)

def symmetrize(intgeom:torch.Tensor, origin=None, grad=False) -> List:
    SAIgeom = []
    for SAICD_list in symmetry_adaptation:
        irred_geom = intgeom.new_zeros(len(SAICD_list), requires_grad=grad)
        for i in SAICD_list:
            SAICD = SAICD_list[i]
            if len(SAICD.scaler) > 0:
                for j in range(len(SAICD.coeff)):
                    irred_geom[i] += SAICD.coeff[j] * intgeom[SAICD.IC[j]] \
                        * torch.exp(-SAICD.alpha[j] * (intgeom[SAICD.scaler[j]] - origin[SAICD.scaler[j]])/origin[SAICD.scaler[j]])
            else:
                for j in range(len(SAICD.coeff)):
                    irred_geom[i] += SAICD.coeff[j] * intgeom[SAICD.IC[j]]
        SAIgeom.append(irred_geom)
    return SAIgeom