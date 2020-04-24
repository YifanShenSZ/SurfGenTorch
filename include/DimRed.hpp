// A feedforward neural network to reduce dimensionality

#ifndef DimRed_hpp
#define DimRed_hpp

#include <torch/torch.h>

namespace DimRed {

struct net : torch::nn::Module {
    std::vector<std::vector<torch::nn::Linear*>> fc;

    net(const std::vector<size_t> & dim_per_irred);

    torch::Tensor forward(torch::Tensor x);
};

} // namespace DimRed

#endif