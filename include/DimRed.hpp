/*
A feedforward neural network to reduce dimensionality

To maintain symmetry, the inputs must belong to a same irreducible
*/

#ifndef DimRed_hpp
#define DimRed_hpp

#include <torch/torch.h>

namespace DimRed {

struct Net : torch::nn::Module {
    std::vector<torch::nn::Linear *> fc, fc_inv;

    Net();
    // max_depth == 0 means unlimited
    Net(const size_t & init_dim, const size_t & max_depth = 0);
    ~Net();

    at::Tensor reduce(const at::Tensor & x);
    at::Tensor inverse(const at::Tensor & x);

    // For pretraining
    at::Tensor forward(const at::Tensor & x);
    void copy(const std::shared_ptr<Net> & net);
    void warmstart(const std::string & chk, const size_t & chk_depth);
    void freeze(const size_t & freeze);
};

void define_DimRed(const std::string & DimRed_in);

std::vector<at::Tensor> reduce(const std::vector<at::Tensor> & x);

std::vector<at::Tensor> inverse(const std::vector<at::Tensor> & x);

} // namespace DimRed

#endif