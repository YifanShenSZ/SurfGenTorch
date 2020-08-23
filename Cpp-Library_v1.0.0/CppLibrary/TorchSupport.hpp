// Support for libtorch

#ifndef TorchSupport_hpp
#define TorchSupport_hpp

#include <torch/torch.h>

namespace CL { namespace TS {

// Number of trainable network parameters
size_t NParameters(const std::vector<at::Tensor> & parameters);

// 1-norm of the network parameter gradient
double NetGradNorm(const std::vector<at::Tensor> & parameters);

namespace LA {
    // Matrix dot multiplication for 3rd-order tensor A and B
    // A.size(2) == B.size(2), A.size(1) == B.size(0)
    // result_ij = A_ikm * B_kjm
    at::Tensor matdotmul(const at::Tensor & A, const at::Tensor & B);
    void matdotmul(const at::Tensor & A, const at::Tensor & B, at::Tensor & result);

    // Unitary transformation for 3rd-order tensor A
    // result_ijm = U^T_ia * A_abm * U_bj
    at::Tensor UT_A3_U(const at::Tensor & UT, const at::Tensor & A, const at::Tensor & U);
    at::Tensor UT_A3_U(const at::Tensor & A, const at::Tensor & U);
    // On exit A harvests the result
    void UT_A3_U_InPlace(const at::Tensor & UT, at::Tensor & A, const at::Tensor & U);
    void UT_A3_U_InPlace(at::Tensor & A, const at::Tensor & U);
} // namespace LA

namespace chemistry {
    bool check_degeneracy(const double & threshold, const at::Tensor & energy);

    // Matrix off-diagonal elements do not have determinate phase, because
    // the eigenvectors defining a representation have indeterminate phase difference
    void initialize_phase_fixing(const size_t & NStates_);
    // Fix M by minimizing || M - ref ||_F^2
    // return || M - ref ||_F^2
    at::Tensor fix(at::Tensor & M, const at::Tensor & ref);
    // Fix M1 and M2 by minimizing weight * || M1 - ref1 ||_F^2 + || M2 - ref2 ||_F^2
    // return weight * || M1 - ref1 ||_F^2 + || M2 - ref2 ||_F^2
    at::Tensor fix2(at::Tensor & M1, at::Tensor & M2, const at::Tensor & ref1, const at::Tensor & ref2, const double & weight);
} // namespace chemistry

} // namespace TS
} // namespace CL

#endif