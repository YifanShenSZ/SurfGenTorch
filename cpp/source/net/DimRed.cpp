// A feedforward neural network to reduce dimensionality

#include <torch/torch.h>

#include "../../Cpp-Library_v1.0.0/TorchSupport.hpp"

#include "../../include/SSAIC.hpp"
#include "../../include/net.hpp"

namespace DimRed {

// max_depth == 0 means unlimited
Net::Net(const size_t & init_dim, const size_t & max_depth) {
    // Determine depth
    size_t depth = init_dim - 1;
    if (max_depth > 0 && depth > max_depth) depth = max_depth;
    // Fully connected layer to reduce dimensionality
    fc.resize(depth);
    for (size_t i = 0; i < depth; i++) {
        fc[i] = new torch::nn::Linear{nullptr};
        * fc[i] = register_module("fc-"+std::to_string(i),
            torch::nn::Linear(torch::nn::LinearOptions(
            init_dim - i, init_dim - i - 1)
            .bias(false)));
    }
    // Fully connected layer to inverse the reduction
    fc_inv.resize(depth);
    for (size_t i = 0; i < depth; i++) {
        fc_inv[i] = new torch::nn::Linear{nullptr};
        * fc_inv[i] = register_module("fc_inv-"+std::to_string(i),
            torch::nn::Linear(torch::nn::LinearOptions(
            init_dim - i - 1, init_dim - i)
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

void Net::warmstart(const std::string & chk, const size_t & chk_depth) {
    //size_t init_dim = (*fc[0])->options.in_features();
    //auto warm_net = std::make_shared<Net>(init_dim, chk_depth_);
    auto warm_net = std::make_shared<Net>((*fc[0])->options.in_features(), chk_depth);
    warm_net->to(torch::kFloat64);
    torch::load(warm_net, chk);
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < (fc.size() < warm_net->fc.size() ? fc.size() : warm_net->fc.size()); i++) {
        CL::TS::copy((*fc    [i])->weight, (*(warm_net->fc    [i]))->weight);
        CL::TS::copy((*fc_inv[i])->weight, (*(warm_net->fc_inv[i]))->weight);
    }
    warm_net.reset();
}

// 
void Net::freeze(const size_t & freeze) {
    
}

} // namespace DimRed