/*
Additional linear algebra for libtorch tensor

Nomenclature (following LAPACK):
    ge  = general
    sy  = real symmetric
    asy = anti symmetric
    po  = real symmetric positive definite
Only use upper triangle of sy & po, strictly upper triangle of asy, otherwise specified

Symmetric high order tensor definition:
    3rd-order tensor: A_ijk = A_jik
*/

#include <torch/torch.h>

namespace CL { namespace TS { namespace LA {
    // Matrix dot multiplication for 3rd-order tensor A and B
    // A.size(2) == B.size(2), A.size(1) == B.size(0)
    // result_ij = A_ikm * B_kjm
    at::Tensor ge3matdotmul(const at::Tensor & A, const at::Tensor & B) {
        at::Tensor result = A.new_zeros({A.size(0), B.size(1)});
        for (int i = 0; i < result.size(0); i++)
        for (int j = 0; j < result.size(1); j++)
        for (int k = 0; k < B.size(0); k++)
        result[i][j] += A[i][k].dot(B[k][j]);
        return result;
    }
    void ge3matdotmul(const at::Tensor & A, const at::Tensor & B, at::Tensor & result) {
        result.fill_(0.0);
        for (int i = 0; i < result.size(0); i++)
        for (int j = 0; j < result.size(1); j++)
        for (int k = 0; k < B.size(0); k++)
        result[i][j] += A[i][k].dot(B[k][j]);
    }
    // For symmetric A and B
    at::Tensor sy3matdotmul(const at::Tensor & A, const at::Tensor & B) {
        at::Tensor result = A.new_zeros({A.size(0), B.size(1)});
        for (int i = 0; i < result.size(0); i++) {
            for (int j = 0; j < i; j++) {
                for (int k = 0; k < j; k++) result[i][j] += A[k][i].dot(B[k][j]);
                for (int k = j; k < i; k++) result[i][j] += A[k][i].dot(B[j][k]);
                for (int k = i; k < B.size(0); k++) result[i][j] += A[i][k].dot(B[j][k]);
            }
            for (int j = i; j < result.size(1); j++) {
                for (int k = 0; k < i; k++) result[i][j] += A[k][i].dot(B[k][j]);
                for (int k = i; k < j; k++) result[i][j] += A[i][k].dot(B[k][j]);
                for (int k = j; k < B.size(0); k++) result[i][j] += A[i][k].dot(B[j][k]);
            }
        }
        return result;
    }
    void sy3matdotmul(const at::Tensor & A, const at::Tensor & B, at::Tensor & result) {
        result.fill_(0.0);
        for (int i = 0; i < result.size(0); i++) {
            for (int j = 0; j < i; j++) {
                for (int k = 0; k < j; k++) result[i][j] += A[k][i].dot(B[k][j]);
                for (int k = j; k < i; k++) result[i][j] += A[k][i].dot(B[j][k]);
                for (int k = i; k < B.size(0); k++) result[i][j] += A[i][k].dot(B[j][k]);
            }
            for (int j = i; j < result.size(1); j++) {
                for (int k = 0; k < i; k++) result[i][j] += A[k][i].dot(B[k][j]);
                for (int k = i; k < j; k++) result[i][j] += A[i][k].dot(B[k][j]);
                for (int k = j; k < B.size(0); k++) result[i][j] += A[i][k].dot(B[j][k]);
            }
        }
    }

    // Unitary transformation for symmetric 3rd-order tensor A
    // result_ijm = U^T_ia * A_abm * U_bj
    at::Tensor UT_A3_U(const at::Tensor & UT, const at::Tensor & A, const at::Tensor & U) {
        int N = U.size(0);
        // work_ibm = U^T_ia * A_abm
        at::Tensor work = A.new_zeros(A.sizes());
        for (int i = 0; i < N; i++)
        for (int b = 0; b < N; b++) {
            for (int a = 0; a < b; a++) work[i][b] += UT[i][a] * A[a][b];
            for (int a = b; a < N; a++) work[i][b] += UT[i][a] * A[b][a];
        }
        // result_ijm = work_ibm * U_bj
        at::Tensor result = A.new_zeros(A.sizes());
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
        at::Tensor work = A.new_zeros(A.sizes());
        for (int i = 0; i < N; i++)
        for (int b = 0; b < N; b++) {
            for (int a = 0; a < b; a++) work[i][b] += UT[i][a] * A[a][b];
            for (int a = b; a < N; a++) work[i][b] += UT[i][a] * A[b][a];
        }
        // result_ijm = work_ibm * U_bj
        at::Tensor result = at::zeros(A.sizes(), A.options());
        for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        for (int b = 0; b < N; b++)
        result[i][j] += work[i][b] * U[b][j];
        return result;
    }
    // On exit A harvests the result
    void UT_A3_U_InPlace(const at::Tensor & UT, at::Tensor & A, const at::Tensor & U) {
        int N = U.size(0);
        // work_ibm = U^T_ia * A_abm
        at::Tensor work = A.new_zeros(A.sizes());
        for (int i = 0; i < N; i++)
        for (int b = 0; b < N; b++) {
            for (int a = 0; a < b; a++) work[i][b] += UT[i][a] * A[a][b];
            for (int a = b; a < N; a++) work[i][b] += UT[i][a] * A[b][a];
        }
        // result_ijm = work_ibm * U_bj
        A.fill_(0.0);
        for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        for (int b = 0; b < N; b++)
        A[i][j] += work[i][b] * U[b][j];
    }
    void UT_A3_U_InPlace(at::Tensor & A, const at::Tensor & U) {
        int N = U.size(0);
        at::Tensor UT = U.transpose(0,1);
        // work_ibm = U^T_ia * A_abm
        at::Tensor work = A.new_zeros(A.sizes());
        for (int i = 0; i < N; i++)
        for (int b = 0; b < N; b++) {
            for (int a = 0; a < b; a++) work[i][b] += UT[i][a] * A[a][b];
            for (int a = b; a < N; a++) work[i][b] += UT[i][a] * A[b][a];
        }
        // result_ijm = work_ibm * U_bj
        A.fill_(0.0);
        for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        for (int b = 0; b < N; b++)
        A[i][j] += work[i][b] * U[b][j];
    }
} // namespace TS
} // namespace LA
} // namespace CL