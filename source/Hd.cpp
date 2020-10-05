/*
A feedforward neural network to fit diabatic Hamiltonian (Hd)
*/

#include <torch/torch.h>

#include <CppLibrary/utility.hpp>

#include "observable_net.hpp"
#include "Hd.hpp"

namespace Hd {

// Number of electronic states
int NStates;
// Symmetry of Hd elements
size_t ** symmetry;
// Each Hd element owns a bunch of networks
std::vector<std::vector<std::vector<std::shared_ptr<ON::Net>>>> nets;

// Define the diabatic Hamiltonian and set to training mode
void define_Hd_train(const std::string & Hd_in) {
    std::ifstream ifs; ifs.open(Hd_in);
        std::string line;
        std::vector<std::string> strs;
        // Number of electronic states
        std::getline(ifs, line);
        std::getline(ifs, line);
        NStates = std::stoul(line);
        // Symmetry of Hd elements
        std::getline(ifs, line);
        CL::utility::CreateArray(symmetry, NStates, NStates);
        for (int i = 0; i < NStates; i++) {
            std::getline(ifs, line); CL::utility::split(line, strs);
            for (int j = 0; j < NStates; j++)
            symmetry[i][j] = std::stoul(strs[j]) - 1;
        }
        size_t NNets = NStates;
        for (size_t i = 0; i < NStates - 1; i++)
        for (size_t j = i + 1; j < NStates; j++)
        NNets += ON::PNR[symmetry[i][j]].size();
        // Network structure
        std::vector<std::vector<size_t>> dimss(NNets);
        std::getline(ifs, line);
        for (size_t j = 0; j < NNets; j++) {
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
            net_pars.resize(NNets);
            for (auto & par : net_pars) {
                std::getline(ifs, par);
                CL::utility::trim(par);
            }
            std::getline(ifs, line);
            if (ifs.good()) { // The checkpoint network has different structure
                chk_dimss.resize(NNets);
                for (size_t j = 0; j < NNets; j++) {
                    std::getline(ifs, line); CL::utility::split(line, strs);
                    auto & dims = dimss[j];
                    dims.resize(strs.size());
                    for (size_t i = 0; i < dims.size(); i++) dims[i] = std::stoul(strs[i]);
                }
            }
        }
    ifs.close();
    // Initialize networks
    nets.resize(NStates);
    size_t count = 0;
    for (int i = 0; i < NStates; i++) {
        nets[i].resize(NStates);
        nets[i][i].resize(1);
        auto & net = nets[i][i][0];
        net = std::make_shared<ON::Net>(dimss[count], true);
        net->to(torch::kFloat64);
        if (! net_pars.empty()) {
            if (chk_dimss.empty()) net->warmstart(net_pars[count], dimss[count]);
            else                   net->warmstart(net_pars[count], chk_dimss[count]);
        }
        net->train();
        count++;
        for (int j = i + 1; j < NStates; j++) {
            nets[i][j].resize(ON::PNR[symmetry[i][j]].size());
            for (auto & net : nets[i][j]) {
                net = std::make_shared<ON::Net>(dimss[count], true);
                net->to(torch::kFloat64);
                if (! net_pars.empty()) {
                    if (chk_dimss.empty()) net->warmstart(net_pars[count], dimss[count]);
                    else                   net->warmstart(net_pars[count], chk_dimss[count]);
                }
                net->train();
                count++;
            }
        }
    }
}

// Define the diabatic Hamiltonian and set to evaluation mode
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
        CL::utility::CreateArray(symmetry, NStates, NStates);
        for (int i = 0; i < NStates; i++) {
            std::getline(ifs, line); CL::utility::split(line, strs);
            for (int j = 0; j < NStates; j++)
            symmetry[i][j] = std::stoul(strs[j]) - 1;
        }
        size_t NNets = NStates;
        for (size_t i = 0; i < NStates - 1; i++)
        for (size_t j = i + 1; j < NStates; j++)
        NNets += ON::PNR[symmetry[i][j]].size();
        // Network structure
        std::vector<std::vector<size_t>> dimss(NNets);
        std::getline(ifs, line);
        std::getline(ifs, line); CL::utility::split(line, strs);
        for (size_t j = 0; j < NNets; j++) {
            std::getline(ifs, line); CL::utility::split(line, strs);
            auto & dims = dimss[j];
            dims.resize(strs.size());
            for (size_t i = 0; i < dims.size(); i++) dims[i] = std::stoul(strs[i]);
        }
        // Network parameters
        std::vector<std::string> net_pars(NNets);
        std::getline(ifs, line);
        for (auto & par : net_pars) {
            std::getline(ifs, par);
            CL::utility::trim(par);
        }
    ifs.close();
    // Initialize networks
    nets.resize(NStates);
    size_t count = 0;
    for (int i = 0; i < NStates; i++) {
        nets[i].resize(NStates);
        nets[i][i].resize(1);
        auto & net = nets[i][i][0];
        net = std::make_shared<ON::Net>(dimss[count], true);
        net->to(torch::kFloat64);
        torch::load(net, net_pars[count]);
        net->eval(); net->freeze(-1);
        count++;
        for (int j = i + 1; j < NStates; j++) {
            nets[i][j].resize(ON::PNR[symmetry[i][j]].size());
            for (auto & net : nets[i][j]) {
                net = std::make_shared<ON::Net>(dimss[count], true);
                net->to(torch::kFloat64);
                torch::load(net, net_pars[count]);
                net->eval(); net->freeze(-1);
                count++;
            }
        }
    }
}

// Input:  input layer
// Output: diabatic Hamiltonian
at::Tensor compute_Hd(const std::vector<at::Tensor> & x) {
    at::Tensor H = x[0].new_empty({NStates, NStates});
    for (int i = 0; i < NStates; i++) {
        H[i][i] = nets[i][i][0]->forward(x[0]);
        for (int j = i + 1; j < NStates; j++) {
            auto & irred = x[symmetry[i][j]];
            at::Tensor net_outputs = irred.new_empty(irred.sizes());
            for (int k = 0; k < net_outputs.size(0); k++) net_outputs[k] = nets[i][j][k]->forward(x[0]);
            H[i][j] = net_outputs.dot(irred);
        }
    }
    return H;
}

} // namespace Hd