// Support for libtorch

#ifndef TorchSupport_hpp
#define TorchSupport_hpp

#include <torch/torch.h>

namespace CL { namespace TS {

// Number of trainable network parameters
size_t NParameters(const std::vector<at::Tensor> & parameters);

// 1-norm of the network parameter gradient
double NetGradNorm(const std::vector<at::Tensor> & parameters);

// Copy the data in A to B
void copy(const at::Tensor & A, const at::Tensor & B);

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
    void UT_A3_U(const at::Tensor & UT, at::Tensor & A, const at::Tensor & U);
    void UT_A3_U(at::Tensor & A, const at::Tensor & U);
} // namespace LA

} // namespace TS
} // namespace CL

#endif