// Vector & matrix & high order tensor operation for libtorch tensor

#ifndef torch_LinearAlgebra_hpp
#define torch_LinearAlgebra_hpp

#include <torch/torch.h>

namespace torch_LinearAlgebra {

// Matrix dot multiplication for 3rd-order tensor A and B
// A.size(2) == B.size(2), A.size(1) == B.size(0)
// result_ij = A_ikm * B_kjm
inline at::Tensor matdotmul(const at::Tensor & A, const at::Tensor & B) {
    at::Tensor result = at::zeros({A.size(0), B.size(1)}, A.options());
    for (int i = 0; i < result.size(0); i++)
    for (int j = 0; j < result.size(1); j++)
    for (int k = 0; k < B.size(0); k++)
    result[i][j] += A[i][k].dot(B[k][j]);
    return result;
}
inline void matdotmul(const at::Tensor & A, const at::Tensor & B, at::Tensor & result) {
    result.fill_(0.0);
    for (int i = 0; i < result.size(0); i++)
    for (int j = 0; j < result.size(1); j++)
    for (int k = 0; k < B.size(0); k++)
    result[i][j] += A[i][k].dot(B[k][j]);
}

// Unitary transformation for 3rd-order tensor A
// result_ijm = U^T_ia * A_abm * U_bj
inline at::Tensor UT_A3_U(const at::Tensor & UT, const at::Tensor & A, const at::Tensor & U) {
    int N = U.size(0);
    // work_ibm = U^T_ia * A_abm
    at::Tensor work = at::zeros(A.sizes(), A.options());
    for (int i = 0; i < N; i++)
    for (int b = 0; b < N; b++)
    for (int a = 0; a < N; a++)
    work[i][b] += UT[i][a] * A[a][b];
    // result_ijm = work_ibm * U_bj
    at::Tensor result = at::zeros(A.sizes(), A.options());
    for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
    for (int b = 0; b < N; b++)
    result[i][j] += work[i][b] * U[b][j];
    return result;
}
inline at::Tensor UT_A3_U(const at::Tensor & A, const at::Tensor & U) {
    int N = U.size(0);
    at::Tensor UT = U.transpose(0,1);
    // work_ibm = U^T_ia * A_abm
    at::Tensor work = at::zeros(A.sizes(), A.options());
    for (int i = 0; i < N; i++)
    for (int b = 0; b < N; b++)
    for (int a = 0; a < N; a++)
    work[i][b] += UT[i][a] * A[a][b];
    // result_ijm = work_ibm * U_bj
    at::Tensor result = at::zeros(A.sizes(), A.options());
    for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
    for (int b = 0; b < N; b++)
    result[i][j] += work[i][b] * U[b][j];
    return result;
}
// On exit A harvests the result
inline void UT_A3_U(const at::Tensor & UT, at::Tensor & A, const at::Tensor & U) {
    int N = U.size(0);
    // work_ibm = U^T_ia * A_abm
    at::Tensor work = at::zeros(A.sizes(), A.options());
    for (int i = 0; i < N; i++)
    for (int b = 0; b < N; b++)
    for (int a = 0; a < N; a++)
    work[i][b] += UT[i][a] * A[a][b];
    // result_ijm = work_ibm * U_bj
    A.fill_(0.0);
    for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
    for (int b = 0; b < N; b++)
    A[i][j] += work[i][b] * U[b][j];
}
inline void UT_A3_U(at::Tensor & A, const at::Tensor & U) {
    int N = U.size(0);
    at::Tensor UT = U.transpose(0,1);
    // work_ibm = U^T_ia * A_abm
    at::Tensor work = at::zeros(A.sizes(), A.options());
    for (int i = 0; i < N; i++)
    for (int b = 0; b < N; b++)
    for (int a = 0; a < N; a++)
    work[i][b] += UT[i][a] * A[a][b];
    // result_ijm = work_ibm * U_bj
    A.fill_(0.0);
    for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
    for (int b = 0; b < N; b++)
    A[i][j] += work[i][b] * U[b][j];
}

} // namespace torch_LinearAlgebra

#endif