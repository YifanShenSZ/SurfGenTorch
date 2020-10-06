/*
A feedforward neural network to reduce dimensionality

To maintain symmetry:
    1. Each irreducible owns an autoencoder, whose inputs are the SSAICs belonging to this irreducible
    2. Only the totally symmetric irreducible has bias
    3. The activation functions are odd, except for the totally symmetric irreducible
*/

#include <torch/torch.h>

#include <CppLibrary/utility.hpp>
#include <CppLibrary/TorchSupport.hpp>

#include "DimRed.hpp"

namespace DimRed {

Net::Net() {}
// The dimensions of `fc` are determined by `dims`
// `fc_inv` has the mirror structure to `fc`
Net::Net(const std::vector<size_t> dims, const bool & totally_symmetric) {
    fc.resize(dims.size() - 1);
    fc_inv.resize(fc.size());
    for (size_t i = 0; i < fc.size(); i++) {
        fc[i] = new torch::nn::Linear{nullptr};
        * fc[i] = register_module("fc-"+std::to_string(i),
            torch::nn::Linear(torch::nn::LinearOptions(
            dims[i], dims[i + 1])
            .bias(totally_symmetric)));
        fc_inv[i] = new torch::nn::Linear{nullptr};
        * fc_inv[i] = register_module("fc_inv-"+std::to_string(i),
            torch::nn::Linear(torch::nn::LinearOptions(
            dims[i + 1], dims[i])
            .bias(totally_symmetric)));
    }
}
// Same structure to net
Net::Net(const std::shared_ptr<Net> & net) {
    fc.resize(net->fc.size());
    for (size_t i = 0; i < fc.size(); i++) {
        fc[i] = new torch::nn::Linear{nullptr};
        * fc[i] = register_module("fc-"+std::to_string(i), torch::nn::Linear((*(net->fc[i]))->options));
    }
    fc_inv.resize(net->fc_inv.size());
    for (size_t i = 0; i < fc_inv.size(); i++) {
        fc_inv[i] = new torch::nn::Linear{nullptr};
        * fc_inv[i] = register_module("fc_inv-"+std::to_string(i), torch::nn::Linear((*(net->fc_inv[i]))->options));
    }
}
Net::~Net() {}
at::Tensor Net::reduce(const at::Tensor & x) {
    at::Tensor y = x;
    for (auto & layer : fc) {
        y = (*layer)->forward(y);
        y = torch::tanh(y);
    }
    return y;
}
at::Tensor Net::inverse(const at::Tensor & x) {
    at::Tensor y = x;
    for (auto layer = fc_inv.rbegin(); layer != fc_inv.rend(); ++layer) {
        y = torch::tanh(y);
        y = (**layer)->forward(y);
    }
    return y;
}
at::Tensor Net::forward(const at::Tensor & x) {
    at::Tensor y = x;
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
// Copy the parameters from net
void Net::copy(const std::shared_ptr<Net> & net) {
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < std::min(fc.size(), net->fc.size()); i++) {
        std::memcpy((*      fc[i]) ->weight.data_ptr<double>(),
                    (*(net->fc[i]))->weight.data_ptr<double>(),
                    (*      fc[i]) ->weight.numel() * sizeof(double));
        if ((*fc[i])->options.bias())
        std::memcpy((*      fc[i]) ->bias.data_ptr<double>(),
                    (*(net->fc[i]))->bias.data_ptr<double>(),
                    (*      fc[i]) ->bias.numel() * sizeof(double));
    }
    for (size_t i = 0; i < std::min(fc_inv.size(), net->fc_inv.size()); i++) {
        std::memcpy((*      fc_inv[i]) ->weight.data_ptr<double>(),
                    (*(net->fc_inv[i]))->weight.data_ptr<double>(),
                    (*      fc_inv[i]) ->weight.numel() * sizeof(double));
        if ((*fc_inv[i])->options.bias())
        std::memcpy((*      fc_inv[i]) ->bias.data_ptr<double>(),
                    (*(net->fc_inv[i]))->bias.data_ptr<double>(),
                    (*      fc_inv[i]) ->bias.numel() * sizeof(double));
    }
}
// Warmstart from checkpoint
void Net::warmstart(const std::string & chk, const std::vector<size_t> chk_dims) {
    auto warm_net = std::make_shared<Net>(chk_dims, (*fc[0])->options.bias());
    warm_net->to(torch::kFloat64);
    torch::load(warm_net, chk);
    this->copy(warm_net);
    this->cold = false;
    warm_net.reset();
}
// Freeze the leading `freeze` layers in fc and fc_inv
void Net::freeze(const size_t & freeze) {
    for (size_t i = 0; i < std::min(freeze, fc.size()); i++) {
        (*fc[i])->weight.set_requires_grad(false);
        (*fc[i])->bias  .set_requires_grad(false);
    }
    for (size_t i = 0; i < std::min(freeze, fc_inv.size()); i++) {
        (*fc_inv[i])->weight.set_requires_grad(false);
        (*fc_inv[i])->bias  .set_requires_grad(false);
    }
}
// Freeze fc
void Net::freeze_reduction() {
    for (auto & layer : fc) {
        (*layer)->weight.set_requires_grad(false);
        (*layer)->bias  .set_requires_grad(false);
    }
}
// Freeze fc_inv
void Net::freeze_inverse() {
    for (auto & layer : fc_inv) {
        (*layer)->weight.set_requires_grad(false);
        (*layer)->bias  .set_requires_grad(false);
    }
}

// Each irreducible owns a network
std::vector<std::shared_ptr<Net>> nets;

// Define the dimensionality reduction and set to training mode
void define_DimRed_train(const std::string & DimRed_in) {
    std::ifstream ifs; ifs.open(DimRed_in);
        std::string line;
        std::vector<std::string> strs;
        // Number of irreducibles
        std::getline(ifs, line);
        std::getline(ifs, line);
        size_t NIrred = std::stoul(line);
        // Network structure
        std::vector<std::vector<size_t>> dimss(NIrred);
        std::getline(ifs, line);
        for (size_t j = 0; j < NIrred; j++) {
            std::getline(ifs, line); CL::utility::split(line, strs);
            auto & dims = dimss[j];
            dims.resize(strs.size());
            for (size_t i = 0; i < dims.size(); i++) dims[i] = std::stoul(strs[i]);
        }
        // Network parameters to warmstart from
        std::vector<std::string> net_pars;
        std::vector<std::vector<size_t>> chk_dimss;
        std::getline(ifs, line);
        if (ifs.good()) { // Checkpoint is present
            net_pars.resize(NIrred);
            for (auto & par : net_pars) {
                std::getline(ifs, par);
                CL::utility::trim(par);
            }
            std::getline(ifs, line);
            if (ifs.good()) { // The checkpoint network has different structure
                chk_dimss.resize(NIrred);
                for (size_t j = 0; j < NIrred; j++) {
                    std::getline(ifs, line); CL::utility::split(line, strs);
                    auto & dims = chk_dimss[j];
                    dims.resize(strs.size());
                    for (size_t i = 0; i < dims.size(); i++) dims[i] = std::stoul(strs[i]);
                }
            }
        }
    ifs.close();
    // Initialize networks
    nets.resize(NIrred);
    for (size_t i = 0; i < NIrred; i++) {
        nets[i] = std::make_shared<Net>(dimss[i], i == 0);
        nets[i]->to(torch::kFloat64);
        if (! net_pars.empty()) {
            if (chk_dimss.empty()) nets[i]->warmstart(net_pars[i], dimss[i]);
            else                   nets[i]->warmstart(net_pars[i], chk_dimss[i]);
        }
        nets[i]->train();
    }
}

// Define the dimensionality reduction and set to evaluation mode
void define_DimRed(const std::string & DimRed_in) {
    std::ifstream ifs; ifs.open(DimRed_in);
        std::string line;
        std::vector<std::string> strs;
        // Number of irreducibles
        std::getline(ifs, line);
        std::getline(ifs, line);
        size_t NIrred = std::stoul(line);
        // Network structure
        std::vector<std::vector<size_t>> dimss(NIrred);
        std::getline(ifs, line);
        for (size_t j = 0; j < NIrred; j++) {
            std::getline(ifs, line); CL::utility::split(line, strs);
            auto & dims = dimss[j];
            dims.resize(strs.size());
            for (size_t i = 0; i < dims.size(); i++) dims[i] = std::stoul(strs[i]);
        }
        // Network parameters
        std::vector<std::string> net_pars(NIrred);
        std::getline(ifs, line);
        for (auto & par : net_pars) {
            std::getline(ifs, par);
            CL::utility::trim(par);
        }
    ifs.close();
    // Initialize networks
    nets.resize(NIrred);
    for (size_t i = 0; i < NIrred; i++) {
        nets[i] = std::make_shared<Net>(dimss[i], i == 0);
        nets[i]->to(torch::kFloat64);
        torch::load(nets[i], net_pars[i]);
        nets[i]->eval(); nets[i]->freeze(-1);
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