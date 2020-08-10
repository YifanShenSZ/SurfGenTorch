'''
Scaled and symmetry adapted internal coordinate (SSAIC)

The procedure of this module:
    1. Define internal coordinates
    2. Nondimensionalize the internal coordinates:
       for length, ic = (ic - origin) / origin
       for angle , ic =  ic - origin
    3. Scale the dimensionless internal coordinates:
       if no scaler      : ic = ic
       elif scaler = self: ic = pi * erf(ic)
       else              : ic = ic * exp(-alpha * scaler)
    4. Symmetry adapted linear combinate the scaled dimensionless internal coordinates
'''

from pathlib import Path
from typing import List
import re
import numpy
import numpy.linalg
import scipy.special
import torch

import FortranLibrary as FL
import PythonLibrary.io as PLio

# Symmetry adapted linear combination
class SymmAdaptLinComb:
    def __init__(self):
        self.coeff    = []
        self.IntCoord = []

# Internal coordinate dimension, not necessarily = cartdim - 6 or 5
intdim = None
# Fortran-Library internal coordinate definition
IntCoordDef = None
# Cartesian coordinate dimension
cartdim = None
# Internal coordinate origin
origin = None
# Internal coordinates who are scaled by themselves
self_scaling = None
# other_scaling[i][0] is scaled by [i][1] with alpha = [i][2]
other_scaling = None
# Number of irreducible representations
NIrred = None
# A matrix (2nd order List), as usual product table
product_table = None
# Number of symmetry adapted internal coordinates per irreducible
NSAIC_per_irred = None
# symmetry_adaptation[i][j] contains the definition of
# j-th symmetry adapted internal coordinate in i-th irreducible
symmetry_adaptation = None

def define_SSAIC(format:str, IntCoordDef_file:Path, origin_file:Path, ScaleSymm_file:Path) -> None:
    # Internal coordinate
    global intdim, IntCoordDef, cartdim, origin, self_scaling, other_scaling, NIrred, product_table, NSAIC_per_irred, symmetry_adaptation
    intdim, IntCoordDef = FL.FetchInternalCoordinateDefinition(format, file=IntCoordDef_file)
    FL.DefineInternalCoordinate(format, file=IntCoordDef_file)
    print("Number of internal coordinates: %d" % intdim)
    # Origin
    if format == 'Columbus7':
        NAtoms, _, _, r, _ = PLio.read_geom_Columbus7(origin_file)
    else:
        NAtoms, _, r = PLio.read_geom_xyz(origin_file)
        r *= 1.8897261339212517
    cartdim = 3 * NAtoms
    origin = numpy.empty(intdim)
    FL.InternalCoordinate(r, origin)
    # Scale and symmetry
    with open(ScaleSymm_file, 'r') as f:
        # Internal coordinates who are scaled by themselves
        f.readline()
        self_scaling = []
        while True:
            line = f.readline().strip()
            if not re.match('^\d+$', line): break
            self_scaling.append(int(line)-1)
        # Internal coordinates who are scaled by others
        other_scaling = []
        while True:
            strs = f.readline().split()
            if not re.match('^\d+$', strs[0]): break
            other_scaling.append((int(strs[0])-1, int(strs[1])-1, float(strs[2])))
        # Number of irreducible representations
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
        # Symmetry adapted linear combinations of each irreducible
        f.readline()
        symmetry_adaptation = []
        for _ in range(NIrred):
            SALC_list = []
            SALC_index = -1
            while True:
                line = f.readline()
                if not line: break
                strs = line.split()
                if not re.match('-?\d+', strs[0]): break
                if re.match('^\d+$', strs[0]):
                    SALC_index += 1
                    SALC_list.append(SymmAdaptLinComb())
                    del strs[0]
                SALC_list[SALC_index].coeff   .append(float(strs[0]))
                SALC_list[SALC_index].IntCoord.append(int(strs[1])-1)
            # Normalize linear combination coefficients
            for SALC in SALC_list:
                norm = numpy.linalg.norm(numpy.array(SALC.coeff))
                for _ in range(len(SALC.coeff)): SALC.coeff[_] /= norm
            symmetry_adaptation.append(SALC_list)

def compute_SSAIC(q) -> List:
    SSAgeom = []
    if isinstance(q, numpy.ndarray):
        # Nondimensionalize
        q -= origin
        for i in range(q.shape[0]):
            if IntCoordDef[i].motion[0].type == 'stretching':
                q[i] /= origin[i]       
        # Scale
        for scaling in other_scaling:
            q[scaling[0]] *= numpy.exp(-scaling[2] * q[scaling[1]])
        for scaling in self_scaling:
            q[scaling] = numpy.pi * scipy.special.erf(q[scaling])
        # Symmetrize
        temp = torch.as_tensor(q)
        for SALC_list in symmetry_adaptation:
            irred_geom = temp.new_zeros(len(SALC_list))
            for i in range(irred_geom.size(0)):
                SALC = SALC_list[i]
                for j in range(len(SALC.coeff)):
                    irred_geom[i] += SALC.coeff[j] * q[SALC.IntCoord[j]]
            SSAgeom.append(irred_geom)
    else:
        # Nondimensionalize
        work = q.clone()
        work -= origin
        for i in range(work.size(0)):
            if IntCoordDef[i].motion[0].type == 'stretching':
                work[i] /= origin[i]
        # Scale
        for scaling in other_scaling:
            work[scaling[0]] *= numpy.exp(-scaling[2] * work[scaling[1]])
        for scaling in self_scaling:
            work[scaling] = numpy.pi * scipy.special.erf(work[scaling])
        # Symmetrize
        for SALC_list in symmetry_adaptation:
            irred_geom = work.new_zeros(len(SALC_list))
            for i in range(irred_geom.size(0)):
                SALC = SALC_list[i]
                for j in range(len(SALC.coeff)):
                    irred_geom[i] += SALC.coeff[j] * work[SALC.IntCoord[j]]
            SSAgeom.append(irred_geom)
    return SSAgeom

if __name__ == "__main__": print(__doc__)