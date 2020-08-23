// Support for libtorch

#include <torch/torch.h>

namespace CL { namespace TS {

// Number of trainable network parameters
size_t NParameters(const std::vector<at::Tensor> & parameters) {
    size_t N = 0;
    for (auto & p : parameters) if (p.requires_grad()) N += p.numel();
    return N;
}

// 1-norm of the network parameter gradient
double NetGradNorm(const std::vector<at::Tensor> & parameters) {
    double norm = 0.0;
    for (auto & p : parameters) norm += p.grad().norm(1).item<double>();
    return norm;
}

namespace LA {
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
    void UT_A3_U_InPlace(const at::Tensor & UT, at::Tensor & A, const at::Tensor & U) {
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
    void UT_A3_U_InPlace(at::Tensor & A, const at::Tensor & U) {
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

namespace chemistry {
    bool check_degeneracy(const double & threshold, const at::Tensor & energy) {
        bool deg = false;
        for (size_t i = 0; i < energy.numel() - 1; i++) {
            if (energy[i+1].item<double>() - energy[i].item<double>() < threshold) {
                deg = true;
                break;
            }
        }
        return deg;
    }

    // Matrix off-diagonal elements do not have determinate phase, because
    // the eigenvectors defining a representation have indeterminate phase difference
    // This module will fix the phase of NStates x NStates matrices
    size_t NStates;
    // There are 2^(NStates-1) possibilities in total,
    // where the user input matrix indicates the base case
    // and is excluded from trying,
    // so phase_possibility.size() = 2^(NStates-1) - 1
    // phase_possibility[i] contains one possibility of the phases of NStates states
    // where true means -, false means +,
    // the phase of the last state is always arbitrarily assigned to +,
    // so phase_possibility[i].size() == NStates-1
    std::vector<std::vector<bool>> phase_possibility;

    void initialize_phase_fixing(const size_t & NStates_) {
        // NStates
        NStates = NStates_;
        // phase_possibility
        phase_possibility.clear();
        // Unchanged case is exculded
        phase_possibility.resize(1 << (NStates-1) - 1);
        for (auto & phase : phase_possibility) phase.resize(NStates-1);
        phase_possibility[0][0] = true;
        for (size_t i = 1; i < NStates-1; i++)
        phase_possibility[0][i] = false;
        for (size_t i = 1; i < phase_possibility.size(); i++) {
            for (size_t j = 0; j < NStates-1; j++)
            phase_possibility[i][j] = phase_possibility[i-1][j];
            size_t count = 0;
            while(phase_possibility[i][count]) {
                phase_possibility[i][count] = false;
                count++;
            }
            phase_possibility[i][count] = true;
        }
    }

    // Fix M by minimizing || M - ref ||_F^2
    // return || M - ref ||_F^2
    at::Tensor fix(at::Tensor & M, const at::Tensor & ref) {
        at::Tensor diff = (M - ref).pow(2).sum();
        int phase_min = -1;
        // Try out phase possibilities to determine
        // the smallest difference and the corresponding phase
        for (int phase = 0; phase < phase_possibility.size(); phase++) {
            at::Tensor M_temp = M.clone();
            for (size_t i = 0; i < NStates; i++) {
                for (size_t j = i+1; j < NStates-1; j++)
                if (phase_possibility[phase][i] != phase_possibility[phase][j])
                M_temp[i][j] = -M_temp[i][j];
                if (phase_possibility[phase][i])
                M_temp[i][NStates-1] = -M_temp[i][NStates-1];
            }
            at::Tensor diff_temp = (M_temp - ref).pow(2).sum();
            if (diff_temp.item<double>() < diff.item<double>()) {
                diff = diff_temp;
                phase_min = phase;
            }
        }
        // Modify M if the best phase is different from the input
        if (phase_min > -1) {
            for (size_t i = 0; i < NStates; i++) {
                for (size_t j = i+1; j < NStates-1; j++)
                if (phase_possibility[phase_min][i] != phase_possibility[phase_min][j])
                M[i][j] = -M[i][j];
                if (phase_possibility[phase_min][i])
                M[i][NStates-1] = -M[i][NStates-1];
            }
        }
        return diff;
    }

    // Fix M1 and M2 by minimizing weight * || M1 - ref1 ||_F^2 + || M2 - ref2 ||_F^2
    // return weight * || M1 - ref1 ||_F^2 + || M2 - ref2 ||_F^2
    at::Tensor fix2(at::Tensor & M1, at::Tensor & M2, const at::Tensor & ref1, const at::Tensor & ref2, const double & weight) {
        at::Tensor diff = weight * (M1 - ref1).pow(2).sum() + (M2 - ref2).pow(2).sum();
        int phase_min = -1;
        // Try out phase possibilities to determine
        // the smallest difference and the corresponding phase
        for (int phase = 0; phase < phase_possibility.size(); phase++) {
            at::Tensor M1_temp = M1.clone();
            at::Tensor M2_temp = M2.clone();
            for (size_t i = 0; i < NStates; i++) {
                for (size_t j = i+1; j < NStates-1; j++)
                if (phase_possibility[phase][i] != phase_possibility[phase][j]) {
                    M1_temp[i][j] = -M1_temp[i][j];
                    M2_temp[i][j] = -M2_temp[i][j];
                }
                if (phase_possibility[phase][i]) {
                    M1_temp[i][NStates-1] = -M1_temp[i][NStates-1];
                    M2_temp[i][NStates-1] = -M2_temp[i][NStates-1];
                }
            }
            at::Tensor diff_temp = weight * (M1_temp - ref1).pow(2).sum() + (M2_temp - ref2).pow(2).sum();
            if (diff_temp.item<double>() < diff.item<double>()) {
                diff = diff_temp;
                phase_min = phase;
            }
        }
        // Modify M1 and M2 if the best phase is different from the input
        if (phase_min > -1) {
            for (size_t i = 0; i < NStates; i++) {
                for (size_t j = i+1; j < NStates-1; j++)
                if (phase_possibility[phase_min][i] != phase_possibility[phase_min][j]) {
                    M1[i][j] = -M1[i][j];
                    M2[i][j] = -M2[i][j];
                }
                if (phase_possibility[phase_min][i]) {
                    M1[i][NStates-1] = -M1[i][NStates-1];
                    M2[i][NStates-1] = -M2[i][NStates-1];
                }
            }
        }
        return diff;
    }
} // namespace chemistry

} // namespace TS
} // namespace CL