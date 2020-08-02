// Vector & matrix & high order tensor operation
// for std::vector and libtorch tensor

#include <torch/torch.h>

namespace CL { namespace LA {

// 2-norm of vector x
double norm2(const std::vector<double> & x) {
    double norm = x[0] * x[0];
    for (size_t i = 1; i < x.size(); i++) norm += x[i] * x[i];
    return std::sqrt(norm);
}

// 1-norm of the network parameter gradient
double NetGradNorm(const std::vector<at::Tensor> & parameters) {
    double norm = 0.0;
    for (auto & p : parameters) norm += p.grad().norm(1).item<double>();
    return norm;
}

// Matrix dot multiplication for 3rd-order tensor A and B
// A.size(2) == B.size(2), A.size(1) == B.size(0)
// result_ij = A_ikm * B_kjm
at::Tensor matdotmul(const at::Tensor & A, const at::Tensor & B) {
    at::Tensor result = at::zeros({A.size(0), B.size(1)}, A.options());
    for (int i = 0; i < result.size(0); i++)
    for (int j = 0; j < result.size(1); j++)
    for (int k = 0; k < B.size(0); k++)
    result[i][j] += A[i][k].dot(B[k][j]);
    return result;
}
void matdotmul(const at::Tensor & A, const at::Tensor & B, at::Tensor & result) {
    result.fill_(0.0);
    for (int i = 0; i < result.size(0); i++)
    for (int j = 0; j < result.size(1); j++)
    for (int k = 0; k < B.size(0); k++)
    result[i][j] += A[i][k].dot(B[k][j]);
}

// Unitary transformation for 3rd-order tensor A
// result_ijm = U^T_ia * A_abm * U_bj
at::Tensor UT_A3_U(const at::Tensor & UT, const at::Tensor & A, const at::Tensor & U) {
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
at::Tensor UT_A3_U(const at::Tensor & A, const at::Tensor & U) {
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
void UT_A3_U(const at::Tensor & UT, at::Tensor & A, const at::Tensor & U) {
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
void UT_A3_U(at::Tensor & A, const at::Tensor & U) {
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

} // namespace LA
} // namespace CL