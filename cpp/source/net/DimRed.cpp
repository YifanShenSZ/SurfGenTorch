// A feedforward neural network to reduce dimensionality

#include <torch/torch.h>

#include "../../include/SSAIC.hpp"
#include "../../include/net.hpp"

namespace DimRed {

// max_depth == 0 means unlimited
Net::Net(const size_t & irred, const size_t & max_depth) {
    // Determine depth
    size_t depth = SSAIC::NSAIC_per_irred[irred] - 1;
    if (max_depth > 0 && depth > max_depth) depth = max_depth;
    // Fully connected layer to reduce dimensionality
    fc.resize(depth);
    for (size_t i = 0; i < depth; i++) {
        fc[i] = new torch::nn::Linear{nullptr};
        * fc[i] = register_module("fc-"+std::to_string(i),
            torch::nn::Linear(torch::nn::LinearOptions(
                SSAIC::NSAIC_per_irred[irred] - i,
                SSAIC::NSAIC_per_irred[irred] - i - 1)
            .bias(false)));
    }
    // Fully connected layer to inverse the reduction
    fc_inv.resize(depth);
    for (size_t i = 0; i < depth; i++) {
        fc_inv[i] = new torch::nn::Linear{nullptr};
        * fc_inv[i] = register_module("fc_inv-"+std::to_string(i),
            torch::nn::Linear(torch::nn::LinearOptions(
                SSAIC::NSAIC_per_irred[irred] - i - 1,
                SSAIC::NSAIC_per_irred[irred] - i)
            .bias(false)));
    }
}

// For pretraining
torch::Tensor Net::forward(const at::Tensor & x) {
    torch::Tensor y = x.clone();
    // Reduce dimensionality
    for (auto & layer : this->fc) {
        y = (*layer)->forward(y);
        y = torch::tanh(y);
    }
    // Inverse the reduction
    for (auto layer = this->fc_inv.rbegin();
        layer != this->fc_inv.rend(); ++layer) {
        y = torch::tanh(y);
        y = (**layer)->forward(y);
    }
    return y;
}

} // namespace DimRed