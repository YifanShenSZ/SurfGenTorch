# No gradient is provided for now

from typing import List
import numpy
import torch

# vector a & b, return a b
def vector_direct_product(a:numpy.ndarray, b:numpy.ndarray) -> numpy.ndarray:
    m = a.shape[0]; n = b.shape[0]; c = numpy.empty((m,n))
    for i in range(m):
        for j in range(n): c[i,j] = a[i] * b[j]
    return c

# Matrix dot multiplication for 3rd-order tensor A and B
# A.size(2) == B.size(2), A.size(1) == B.size(0)
# result_ij = A_ikm * B_kjm
def matdotmul(A:torch.Tensor, B:torch.Tensor) -> torch.Tensor:
    result = A.new_zeros((A.size(0), B.size(1)))
    for i in range(result.size(0)):
        for j in range(result.size(1)):
            for k in range(B.size(0)):
                result[i,j] += A[i,k,:].dot(B[k,j,:])
    return result

# Unitary transformation for 3rd-order tensor A
# result_ijm = U^T_ia * A_abm * U_bj
def UT_A3_U(UT:torch.Tensor, A:torch.Tensor, U:torch.Tensor) -> torch.Tensor:
    N = U.size(0)
    # work_ibm = U^T_ia * A_abm
    work = A.new_zeros(A.sizes())
    for i in range(N):
        for b in range(N):
            for a in range(N):
                work[i,b,:] += UT[i,a] * A[a,b,:]
    # result_ijm = work_ibm * U_bj
    result = A.new_zeros(A.sizes())
    for i in range(N):
        for j in range(N):
            for b in range(N):
                result[i,j,:] += work[i,b,:] * U[b,j]
    return result