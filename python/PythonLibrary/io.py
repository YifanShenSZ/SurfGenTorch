'''
This is one of the basics of Tool-Collection

Input/Output routines
'''

from pathlib import Path
from typing import List
import math
import numpy

# Read a vector from file (1 number/line)
def read_vector_1(file:Path) -> numpy.ndarray:
    with open(file,'r') as f: lines = f.readlines()
    size = len(lines); v = numpy.empty(size)
    for i in range(size): v[i] = float(lines[i])
    return v

# Read a vector from file (3 numbers/line)
def read_vector_3(file:Path, size:int = 0) -> numpy.ndarray:
    with open(file,'r') as f: lines = f.readlines()
    if size == 0:
        n = len(lines)
        v = numpy.empty(int(3*n))
        for i in range(n):
            temp = lines[i].split()
            v[3*i  ] = float(temp[0])
            v[3*i+1] = float(temp[1])
            v[3*i+2] = float(temp[2])
    else:
        v = numpy.empty(size)
        n = len(lines); m = size % 3
        for i in range(n-1):
            temp = lines[i].split()
            v[3*i  ] = float(temp[0])
            v[3*i+1] = float(temp[1])
            v[3*i+2] = float(temp[2])
        if m == 0:
            temp = lines[n-1].split()
            v[size-3] = float(temp[0])
            v[size-2] = float(temp[1])
            v[size-1] = float(temp[2])
        elif m == 1:
            v[size-1] = float(lines[n-1])
        else: # 2
            temp = lines[n-1].split()
            v[size-2] = float(temp[0])
            v[size-1] = float(temp[1])
    return v
# Inverse to read_vector_3
def write_vector_3(file:Path, vector:List) -> None:
    size = len(vector)
    with open(file,'a') as f:
        n = math.ceil(int(size/3)); m = size % 3
        for i in range(n-1):
            print(('%20.14f%20.14f%20.14f')%(vector[3*i],vector[3*i+1],vector[3*i+2]), file=f)
        if m == 0:
            print(('%20.14f%20.14f%20.14f')%(vector[size-3],vector[size-2],vector[size-1]), file=f)
        elif m == 1:
            print('%20.14f'%vector[size-1], file=f)
        else: # 2
            print(('%20.14f%20.14f')%(vector[size-2],vector[size-1]), file=f)

# Read 2 vectors from file (2 numbers/line), each column is a vector
def read_2vectors(file:Path) -> numpy.ndarray:
    with open(file,'r') as f: lines = f.readlines()
    size = len(lines); x = numpy.empty(size); y = numpy.empty(size)
    for i in range(size):
        temp = lines[i].split()
        x[i] = float(temp[0]); y[i] = float(temp[1])
    return x, y

# Read a matrix from file
def read_matrix(file:Path) -> numpy.ndarray:
    with open(file,'r') as f: lines = f.readlines()
    temp = lines[0].split()
    size0 = len(lines); size1 = len(temp)
    m = numpy.empty((size0, size1))
    for i in range(size0):
        temp = lines[i].split()
        for j in range(size1): m[i, j] = float(temp[j])
    return m

# Read Columbus7 geometry file, return:
#     NAtoms (number of atoms)
#     symbol (element symbol of each atom)
#     number (element number of each atom)
#     r      (Cartesian coordinate in Bohr)
#     mass   (mass of each atom in atomic mass unit)
def read_geom_Columbus7(GeomFile:Path) -> (int, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray):
    with open(GeomFile,'r') as f: lines = f.readlines()
    NAtoms = len(lines)
    symbol = numpy.empty(NAtoms,dtype=str)
    number = numpy.empty(NAtoms)
    r      = numpy.empty(int(3*NAtoms))
    mass   = numpy.empty(NAtoms)
    for i in range(NAtoms):
        temp = lines[i].split()
        symbol[i] = temp[0].strip()
        number[i] = float(temp[1])
        r[3*i  ]  = float(temp[2])
        r[3*i+1]  = float(temp[3])
        r[3*i+2]  = float(temp[4])
        mass[i]   = float(temp[5])
    return NAtoms, symbol, number, r, mass
# Inverse to read_geom_Columbus7
def write_geom_Columbus7(GeomFile:Path, NAtoms:int, symbol:List, number:List, r:List, mass:List) -> None:
    with open(GeomFile,'a') as f:
        for i in range(NAtoms):
            print((' %-2s  %5.1f%14.8f%14.8f%14.8f%14.8f')%\
                (symbol[i],number[i],r[3*i],r[3*i+1],r[3*i+2],mass[i]),file=f)

# Read xyz file, return:
#     NAtoms (number of atoms)
#     symbol (element symbol of each atom)
#     r      (Cartesian coordinate in angstrom)
def read_geom_xyz(GeomFile:Path) -> (int, numpy.ndarray, numpy.ndarray):
    with open(GeomFile,'r') as f: lines = f.readlines()
    NAtoms = int(lines[0])
    symbol = numpy.empty(NAtoms,dtype=str)
    r      = numpy.empty(int(3*NAtoms))
    for i in range(NAtoms):
        temp = lines[i+2].split()
        symbol[i] = temp[0].strip()
        r[3*i  ]  = float(temp[1])
        r[3*i+1]  = float(temp[2])
        r[3*i+2]  = float(temp[3])
    return NAtoms, symbol, r
# Inverse to read_geom_xyz
def write_geom_xyz(GeomFile:Path, NAtoms:int, symbol:List, r:List) -> None:
    with open(GeomFile,'a') as f:
        print(NAtoms,file=f)
        print(file=f)
        for i in range(NAtoms): print(('%2s%20.14f%20.14f%20.14f')%\
            (symbol[i],r[3*i],r[3*i+1],r[3*i+2]),file=f)

# Read an internal coordinate geometry from file (1 or 3 numbers/line)
def read_geom_int(GeomFile:Path, intdim:int) -> numpy.ndarray:
    with open(GeomFile,'r') as f: lines = f.readlines()
    if len(lines) == intdim:
        geom = read_vector_1(GeomFile)
    else:
        geom = read_vector_3(GeomFile, intdim)
    return geom
# Inverse to read_geom_int
def write_geom_int(GeomFile:Path, geom:List) -> None:
    write_vector_3(GeomFile, geom)

# Read a mass from file (1 or 3 numbers/line)
def read_mass(MassFile:Path, NAtoms:int) -> numpy.ndarray:
    with open(MassFile,'r') as f: lines = f.readlines()
    if len(lines) == NAtoms:
        mass = read_vector_1(MassFile)
    else:
        mass = read_vector_3(MassFile, NAtoms)
    return mass

# Read a Cartesian coordinate gradient from file (3 numbers/line)
def read_grad_cart(GradFile:Path) -> numpy.ndarray:
    grad = read_vector_3(GradFile)
    return grad
# Inverse to read_grad_cart
def write_grad_cart(GradFile:Path, grad:numpy.ndarray) -> None:
    write_vector_3(GradFile, grad)

# Read an internal coordinate gradient from file (1 or 3 numbers/line)
def read_grad_int(GradFile:Path, intdim:int) -> numpy.ndarray:
    with open(GradFile,'r') as f: lines = f.readlines()
    if len(lines) == intdim:
        grad = read_vector_1(GradFile)
    else:
        grad = read_vector_3(GradFile, intdim)
    return grad
# Inverse to read_grad_int
def write_grad_int(GradFile:Path, grad:List) -> None:
    write_vector_3(GradFile, grad)

if __name__ == "__main__": print(__doc__)