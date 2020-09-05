/*
A feedforward neural network to reduce dimensionality

The 0th irreducible is assumed to be totally symmetric

To maintain symmetry:
    1. the inputs of a network must belong to a same irreducible
    2. the activation functions must be odd (except for the totally symmetric irreducible)
    3. only the totally symmetric irreducible can have bias
*/

#include <torch/torch.h>

#include <CppLibrary/utility.hpp>
#include <CppLibrary/TorchSupport.hpp>

#include "DimRed.hpp"

namespace DimRed {

Net::Net() {}
// Totally symmetric irreducible additionally has const term (bias)
// max_depth == 0 means unlimited
Net::Net(const size_t & init_dim, const bool & totally_symmetric, const size_t & max_depth) {
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
            .bias(totally_symmetric)));
    }
    // Fully connected layer to inverse the reduction
    fc_inv.resize(depth);
    for (size_t i = 0; i < depth; i++) {
        fc_inv[i] = new torch::nn::Linear{nullptr};
        * fc_inv[i] = register_module("fc_inv-"+std::to_string(i),
            torch::nn::Linear(torch::nn::LinearOptions(
            init_dim - i - 1, init_dim - i)
            .bias(totally_symmetric)));
    }
}
Net::~Net() {}
at::Tensor Net::reduce(const at::Tensor & x) {
    at::Tensor y = x.clone();
    for (auto & layer : fc) {
        y = (*layer)->forward(y);
        y = torch::tanh(y);
    }
    return y;
}
at::Tensor Net::inverse(const at::Tensor & x) {
    at::Tensor y = x.clone();
    for (auto layer = fc_inv.rbegin(); layer != fc_inv.rend(); ++layer) {
        y = torch::tanh(y);
        y = (**layer)->forward(y);
    }
    return y;
}
// For pretraining
at::Tensor Net::forward(const at::Tensor & x) {
    at::Tensor y = x.clone();
    // Reduce dimensionality
    for (auto & layer : fc) {
        y = (*layer)->forward(y);
        y = torch::tanh(y);
    }
    // Inverse the reduction
    for (auto layer = fc_inv.rbegin(); layer != fc_inv.rend(); ++layer) {
        y = torch::tanh(y);
        y = (**layer)->forward(y);
    }
    return y;
}
void Net::copy(const std::shared_ptr<Net> & net) {
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < (fc.size() < net->fc.size() ? fc.size() : net->fc.size()); i++) {
        std::memcpy((*fc[i])->weight.data_ptr<double>(),
                    (*(net->fc[i]))->weight.data_ptr<double>(),
                    (*fc[i])->weight.numel() * sizeof(double));
        if ((*fc[i])->options.bias())
        std::memcpy((*fc[i])->bias.data_ptr<double>(),
                    (*(net->fc[i]))->bias.data_ptr<double>(),
                    (*fc[i])->bias.numel() * sizeof(double));
        std::memcpy((*fc_inv[i])->weight.data_ptr<double>(),
                    (*(net->fc_inv[i]))->weight.data_ptr<double>(),
                    (*fc_inv[i])->weight.numel() * sizeof(double));
        if ((*fc_inv[i])->options.bias())
        std::memcpy((*fc_inv[i])->bias.data_ptr<double>(),
                    (*(net->fc_inv[i]))->bias.data_ptr<double>(),
                    (*fc_inv[i])->bias.numel() * sizeof(double));
    }
}
void Net::warmstart(const std::string & chk, const size_t & chk_depth) {
    auto warm_net = std::make_shared<Net>((*fc[0])->options.in_features(), (*fc[0])->options.bias(), chk_depth);
    warm_net->to(torch::kFloat64);
    torch::load(warm_net, chk);
    this->copy(warm_net);
    warm_net.reset();
}
void Net::freeze(const size_t & freeze) {
    for (size_t i = 0; i < freeze; i++) {
        (*fc[i])->weight.set_requires_grad(false);
        (*fc[i])->bias  .set_requires_grad(false);
        (*fc_inv[i])->weight.set_requires_grad(false);
        (*fc_inv[i])->bias  .set_requires_grad(false);
    }
}

// Each irreducible owns a network
std::vector<std::shared_ptr<Net>> nets;

void define_DimRed(const std::string & DimRed_in) {
    size_t NIrred;
    std::ifstream ifs; ifs.open(DimRed_in);
        std::string line;
        std::vector<std::string> strs;
        // Initial dimension of each network
        std::vector<std::string> init_dims;
        std::getline(ifs, line);
        std::getline(ifs, line); CL::utility::split(line, init_dims);
        NIrred = init_dims.size();
        // Network parameters
        std::vector<std::string> net_pars(NIrred);
        std::vector<size_t> net_depths(NIrred);
        std::getline(ifs, line);
        for (size_t i = 0; i < NIrred; i++) {
            std::getline(ifs, line);
            strs = CL::utility::split(line);
            net_pars[i] = strs[0];
            if (strs.size() > 1) net_depths[i] = std::stoul(strs[1]);
            else net_depths[i] = 0;
        }
    ifs.close();
    // Initialize networks
    nets.resize(NIrred);
    for (size_t i = 0; i < NIrred; i++) {
        nets[i] = std::make_shared<Net>(std::stoul(init_dims[i]), i == 0, net_depths[i]);
        nets[i]->to(torch::kFloat64);
        torch::load(nets[i], net_pars[i]);
        nets[i]->eval();
    }
}

std::vector<at::Tensor> reduce(const std::vector<at::Tensor> & x) {
    std::vector<at::Tensor> y(x.size());
    for (size_t i = 0; i < x.size(); i++) y[i] = nets[i]->reduce(x[i]);
    return y;
}

std::vector<at::Tensor> inverse(const std::vector<at::Tensor> & x) {
    std::vector<at::Tensor> y(x.size());
    for (size_t i = 0; i < x.size(); i++) y[i] = nets[i]->inverse(x[i]);
    return y;
}

} // namespace DimRed