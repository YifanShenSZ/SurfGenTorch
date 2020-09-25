/*
A feedforward neural network to reduce dimensionality

To maintain symmetry:
    1. Each irreducible owns an autoencoder, whose inputs are the SSAICs belonging to this irreducible
    2. Only the totally symmetric irreducible has bias
    3. The activation functions are odd, except for the totally symmetric irreducible
*/

#ifndef DimRed_hpp
#define DimRed_hpp

#include <torch/torch.h>

namespace DimRed {

struct Net : torch::nn::Module {
    std::vector<torch::nn::Linear *> fc, fc_inv;

    Net();
    // Totally symmetric irreducible additionally has const term (bias)
    // max_depth < 0 means unlimited
    Net(const size_t & init_dim, const bool & totally_symmetric, const int64_t & max_depth = -1);
    ~Net();

    at::Tensor reduce(const at::Tensor & x);
    at::Tensor inverse(const at::Tensor & x);

    // For pretraining
    at::Tensor forward(const at::Tensor & x);
    void copy(const std::shared_ptr<Net> & net);
    void warmstart(const std::string & chk, const int64_t & chk_depth);
    void freeze(const size_t & freeze);
};

// The 0th irreducible is assumed to be totally symmetric
void define_DimRed(const std::string & DimRed_in);

std::vector<at::Tensor> reduce(const std::vector<at::Tensor> & x);

std::vector<at::Tensor> inverse(const std::vector<at::Tensor> & x);

} // namespace DimRed

#endif