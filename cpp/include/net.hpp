/*
A feedforward neural network to reduce dimensionality

Each irreducible has its own dimensionality reduction network
*/

#ifndef net_hpp
#define net_hpp

#include <torch/torch.h>

namespace DimRed {

struct Net : torch::nn::Module {
    std::vector<torch::nn::Linear *> fc, fc_inv;

    // max_depth == 0 means unlimited
    Net(const size_t & irred, const size_t & max_depth = 0);

    // Compute the reduced dimension
    //torch::Tensor reduce(const torch::Tensor & x);
    
    // Inverse reduction
    //torch::Tensor inverse(const torch::Tensor & x);

    // For pretraining
    torch::Tensor forward(const at::Tensor & x);
};

} // namespace DimRed

#endif