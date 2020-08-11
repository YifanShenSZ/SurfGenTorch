// A feedforward neural network to reduce dimensionality

#include <torch/torch.h>

#include <CppLibrary/TorchSupport.hpp>

#include "SSAIC.hpp"
#include "net.hpp"

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
    for (auto & layer : fc) {
        y = (*layer)->forward(y);
        y = torch::tanh(y);
    }
    // Inverse the reduction
    for (auto layer = fc_inv.rbegin();
        layer != fc_inv.rend(); ++layer) {
        y = torch::tanh(y);
        y = (**layer)->forward(y);
    }
    return y;
}

void Net::copy(const std::shared_ptr<Net> & net) {
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < (fc.size() < net->fc.size() ? fc.size() : net->fc.size()); i++) {
        CL::TS::copy((*(net->fc    [i]))->weight, (*fc    [i])->weight);
        CL::TS::copy((*(net->fc_inv[i]))->weight, (*fc_inv[i])->weight);
    }
}

void Net::warmstart(const std::string & chk, const size_t & chk_depth) {
    auto warm_net = std::make_shared<Net>((*fc[0])->options.in_features(), chk_depth);
    warm_net->to(torch::kFloat64);
    torch::load(warm_net, chk);
    this->copy(warm_net);
    warm_net.reset();
}

void Net::freeze(const size_t & freeze) {
    assert(("All layers are frozen, so nothing to train", freeze < fc.size()));
    for (size_t i = 0; i < freeze; i++) {
        (*fc    [i])->weight.set_requires_grad(false);
        (*fc_inv[i])->weight.set_requires_grad(false);
    }
}

} // namespace DimRed