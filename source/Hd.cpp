// A feedforward neural network to compute diabatic Hamiltonian (Hd)

#include <regex>
#include <torch/torch.h>

#include <CppLibrary/utility.hpp>
#include <CppLibrary/TorchSupport.hpp>

#include "Hd.hpp"

namespace Hd {

Net::Net() {}
// Totally symmetric irreducible additionally has const term (bias)
// max_depth == 0 means unlimited
Net::Net(const size_t & init_dim, const bool & totally_symmetric, const size_t & max_depth) {
    // Determine depth
    size_t depth = init_dim - 1;
    if (max_depth > 0 && depth > max_depth) depth = max_depth;
    // The starting layers gradually reduce dimensionality
    fc.resize(depth);
    for (size_t i = 0; i < depth - 1; i++) {
        fc[i] = new torch::nn::Linear{nullptr};
        * fc[i] = register_module("fc-"+std::to_string(i),
            torch::nn::Linear(torch::nn::LinearOptions(
            init_dim - i, init_dim - i - 1)
            .bias(totally_symmetric)));
    }
    // The final layer reduces to scalar
    size_t i = depth - 1;
    fc[i] = new torch::nn::Linear{nullptr};
    * fc[i] = register_module("fc-"+std::to_string(i),
        torch::nn::Linear(torch::nn::LinearOptions(
        init_dim - i, 1)
        .bias(totally_symmetric)));
}
Net::~Net() {}
at::Tensor Net::forward(const at::Tensor & x) {
    at::Tensor y = x.clone();
    for (auto & layer : fc) {
        y = (*layer)->forward(y);
        y = torch::tanh(y);
    }
    return y[0];
}
// For training
void Net::copy(const std::shared_ptr<Net> & net) {
    torch::NoGradGuard no_grad;
    for (size_t i = 0; i < (fc.size() < net->fc.size() ? fc.size() : net->fc.size()); i++) {
        std::memcpy((*fc[i])->weight.data_ptr<double>(),
                (*(net->fc[i]))->weight.data_ptr<double>(),
                (*fc[i])->weight.numel() * sizeof(double));
        std::memcpy((*fc[i])->bias.data_ptr<double>(),
                (*(net->fc[i]))->bias.data_ptr<double>(),
                (*fc[i])->bias.numel() * sizeof(double));
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
        (*fc[i])->weight.set_requires_grad(false);
        (*fc[i])->bias.set_requires_grad(false);
    }
}

// Number of irreducible representations
size_t NIrred;
// Number of electronic states
int NStates;
// Symmetry of Hd elements
size_t ** Hd_symm;
// Each Hd element owns a network
std::vector<std::vector<std::shared_ptr<Net>>> nets;

// Symmetry adapted polynomials serve as the input layer
namespace input {
    // Polynomial numbering rule
    struct PolNumbRul {
        std::vector<size_t> irred, coord;

        PolNumbRul() {}
        // For example, the input line of a 2nd order term made up by 
        // the 1st coordinate in the 2nd irreducible and 
        // the 3rd coordinate in the 4th irreducible is:
        //     2    2,1    4,3
        PolNumbRul(const std::vector<std::string> & input_line) {
            size_t order = std::stoul(input_line[0]);
            irred.resize(order);
            coord.resize(order);
            for (size_t i = 0; i < order; i++) {
                std::vector<std::string> irred_coord = CL::utility::split(input_line[i+1], ',');
                irred[i] = std::stoul(irred_coord[0]) - 1;
                coord[i] = std::stoul(irred_coord[1]) - 1;
            }
        }
        ~PolNumbRul() {}
    };

    // Polynomial numbering rule, PNR[i][j] (the j-th polynomial in i-th irreducible)
    // = product of PNR[i][j].coord[k]-th coordinate in PNR[i][j].irred[k]-th irreducible
    std::vector<std::vector<PolNumbRul>> PNR;

    // Return number of input neurons per irreducible
    std::vector<size_t> prepare_PNR(const std::string & Hd_input_layer_in) {
        PNR.resize(NIrred);
        std::ifstream ifs; ifs.open(Hd_input_layer_in);
            std::string line;
            std::vector<std::string> strs;
            std::getline(ifs, line);
            for (auto & irred : PNR) {
                while (true) {
                    std::getline(ifs, line);
                    if (! ifs.good()) break;
                    CL::utility::split(line, strs);
                    if (! std::regex_match(strs[0], std::regex("\\d+"))) break;
                    irred.push_back(PolNumbRul(strs));
                }
            }
        ifs.close();
        std::vector<size_t> NInput_per_irred(NIrred);
        for (size_t i = 0; i < NIrred; i++) NInput_per_irred[i] = PNR[i].size();
        return NInput_per_irred;
    }

    std::vector<at::Tensor> input_layer(const std::vector<at::Tensor> & x) {
        std::vector<at::Tensor> input_layer(NIrred);
        for (size_t irred = 0; irred < NIrred; irred++) {
            input_layer[irred] = at::empty(PNR[irred].size(), at::TensorOptions().dtype(torch::kFloat64));
            for (size_t i = 0; i < PNR[irred].size(); i++) {
                input_layer[irred][i] = 1.0;
                for (size_t j = 0; j < PNR[irred][i].coord.size(); j++)
                input_layer[irred][i] *= x[PNR[irred][i].irred[j]][PNR[irred][i].coord[j]];
            }
        }
        return input_layer;
    }
} // namespace input

void define_Hd(const std::string & Hd_in) {
    std::ifstream ifs; ifs.open(Hd_in);
        std::string line;
        std::vector<std::string> strs;
        // Number of electronic states
        std::getline(ifs, line);
        std::getline(ifs, line);
        NStates = std::stoul(line);
        // Symmetry of Hd elements
        std::getline(ifs, line);
        CL::utility::CreateArray(Hd_symm, NStates, NStates);
        for (size_t i = 0; i < NStates; i++) {
            std::getline(ifs, line); CL::utility::split(line, strs);
            for (size_t j = 0; j < NStates; j++)
            Hd_symm[i][j] = std::stoul(strs[j]) - 1;
        }
        // Input layer specification file
        std::string Hd_input_layer_in;
        std::getline(ifs, line);
        std::getline(ifs, Hd_input_layer_in);
        CL::utility::trim(Hd_input_layer_in);
        // Network parameters
        std::vector<std::string> net_pars(NStates*(NStates-1)/2);
        std::getline(ifs, line);
        for (auto & p : net_pars) {
            std::getline(ifs, p);
            CL::utility::trim(p);
        }
    ifs.close();
    // Number of irreducible representations
    NIrred = 0;
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = 0; j < NStates; j++)
    NIrred = Hd_symm[i][j] > NIrred ? Hd_symm[i][j] : NIrred;
    NIrred++;
    // Polynomial numbering rule
    std::vector<size_t> NInput_per_irred = input::prepare_PNR(Hd_input_layer_in);
    // Initialize networks
    nets.resize(NStates);
    size_t index = 0;
    for (size_t i = 0; i < NStates; i++) {
        nets[i].resize(NStates);
        for (size_t j = i; j < NStates; j++) {
            nets[i][j] = std::make_shared<Net>(NInput_per_irred[Hd_symm[i][j]],
                Hd::Hd_symm[i][j] == 0);
            nets[i][j]->to(torch::kFloat64);
            torch::load(nets[i][j], net_pars[index]);
            nets[i][j]->eval();
            index++;
        }
    }
}

at::Tensor compute_Hd(const std::vector<at::Tensor> & x) {
    // Determine input layer
    std::vector<at::Tensor> input_layer = input::input_layer(x);
    // Compute upper triangle
    at::Tensor Hd = x[0].new_empty({NStates, NStates});
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    Hd[i][j] = nets[i][j]->forward(input_layer[Hd_symm[i][j]]);
    return Hd;
}

at::Tensor compute_Hd_from_input_layer(const std::vector<at::Tensor> & input_layer) {
    // Compute upper triangle
    at::Tensor Hd = input_layer[0].new_empty({NStates, NStates});
    for (size_t i = 0; i < NStates; i++)
    for (size_t j = i; j < NStates; j++)
    Hd[i][j] = nets[i][j]->forward(input_layer[Hd_symm[i][j]]);
    return Hd;
}

} // namespace Hd