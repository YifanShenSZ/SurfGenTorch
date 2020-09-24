/*
A feedforward neural network to fit a matrix element of an observable
based on the symmetry adapted internal coordinates (SAIC)

To maintain symmetry:
    1. The input layer takes the symmetry adapted polynomials of SAICs
    2. Only the totally symmetric irreducible has bias
    3. The activation functions are odd, except for the totally symmetric irreducible
*/

#include <regex>
#include <torch/torch.h>

#include <CppLibrary/utility.hpp>

#include "observable_net.hpp"

namespace ON {

// The general form for a matrix element of an observable
Net::Net() {}
// Totally symmetric irreducible additionally has const term (bias)
// max_depth < 0 means unlimited
Net::Net(const size_t & init_dim, const bool & totally_symmetric, const int64_t & max_depth) {
    // Determine depth
    size_t depth = std::max((int)init_dim - 2, 0);
    if (max_depth >= 0 && depth > max_depth) depth = max_depth;
    // The starting layers gradually reduce dimensionality
    fc.resize(depth);
    for (size_t i = 0; i < depth; i++) {
        fc[i] = new torch::nn::Linear{nullptr};
        * fc[i] = register_module("fc-"+std::to_string(i),
            torch::nn::Linear(torch::nn::LinearOptions(
            init_dim - i, init_dim - i - 1)
            .bias(totally_symmetric)));
    }
    // The tail layer reduces to scalar
    tail = register_module("tail",
        torch::nn::Linear(torch::nn::LinearOptions(
        init_dim - depth, 1)
        .bias(totally_symmetric)));
}
Net::~Net() {}
at::Tensor Net::forward(const at::Tensor & x) {
    at::Tensor y = x.clone();
    // The starting layers gradually reduce dimensionality
    for (auto & layer : fc) {
        y = (*layer)->forward(y);
        y = torch::tanh(y);
    }
    // The tail layer reduces to scalar
    y = tail->forward(y);
    return y[0];
}
// For training
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
    }
    if (tail->weight.numel() == net->tail->weight.numel()) {
        std::memcpy(tail->weight.data_ptr<double>(),
                    net->tail->weight.data_ptr<double>(),
                    tail->weight.numel() * sizeof(double));
        if (tail->options.bias())
        std::memcpy(tail->bias.data_ptr<double>(),
                    net->tail->bias.data_ptr<double>(),
                    tail->bias.numel() * sizeof(double));
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
        (*fc[i])->bias.set_requires_grad(false);
    }
}

// Polynomial numbering rule
PolNumbRul::PolNumbRul() {}
// For example, the input line of a 2nd order term made up by 
// the 1st coordinate in the 2nd irreducible and 
// the 3rd coordinate in the 4th irreducible is:
//     2    2,1    4,3
PolNumbRul::PolNumbRul(const std::vector<std::string> & input_line) {
    size_t order = std::stoul(input_line[0]);
    irred.resize(order);
    coord.resize(order);
    for (size_t i = 0; i < order; i++) {
        std::vector<std::string> irred_coord = CL::utility::split(input_line[i+1], ',');
        irred[i] = std::stoul(irred_coord[0]) - 1;
        coord[i] = std::stoul(irred_coord[1]) - 1;
    }
}
PolNumbRul::~PolNumbRul() {}

// Polynomial numbering rule, PNR[i][j] (the j-th polynomial in i-th irreducible)
// = product of PNR[i][j].coord[k]-th coordinate in PNR[i][j].irred[k]-th irreducible
std::vector<std::vector<PolNumbRul>> PNR;

// Symmetry adapted polynomials serve as the input layer,
// so we have to define the polynomial numbering rule in global variable PNR
void define_PNR(const std::string & input_layer_in) {
    std::ifstream ifs; ifs.open(input_layer_in);
        std::string line;
        std::vector<std::string> strs;
        std::getline(ifs, line);
        while (true) {
            PNR.push_back({});
            auto & irred = PNR[PNR.size() - 1];
            while (true) {
                std::getline(ifs, line);
                if (! ifs.good()) break;
                CL::utility::split(line, strs);
                if (! std::regex_match(strs[0], std::regex("\\d+"))) break;
                irred.push_back(PolNumbRul(strs));
            }
            if (! ifs.good()) break;
        }
    ifs.close();
}

// Compute the symmetry adapted polynomials from monomials
std::vector<at::Tensor> input_layer(const std::vector<at::Tensor> & x) {
    std::vector<at::Tensor> input_layer(PNR.size());
    for (size_t irred = 0; irred < PNR.size(); irred++) {
        input_layer[irred] = at::empty(PNR[irred].size(), at::TensorOptions().dtype(torch::kFloat64));
        for (size_t i = 0; i < PNR[irred].size(); i++) {
            input_layer[irred][i] = 1.0;
            for (size_t j = 0; j < PNR[irred][i].coord.size(); j++)
            input_layer[irred][i] *= x[PNR[irred][i].irred[j]][PNR[irred][i].coord[j]];
        }
    }
    return input_layer;
}

} // namespace ON