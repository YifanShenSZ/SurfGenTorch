/*
A feedforward neural network to fit diabatic Hamiltonian (Hd)
*/

#include <torch/torch.h>

#include <CppLibrary/utility.hpp>

#include "observable_net.hpp"
#include "Hd.hpp"

namespace Hd {

Net::Net() {}
Net::Net(const size_t & init_dim, const bool & totally_symmetric, const int64_t & max_depth) : ON::Net(init_dim, totally_symmetric, max_depth) {}
Net::~Net() {}

// Number of electronic states
int NStates;
// Symmetry of Hd elements
size_t ** symmetry;
// Each Hd element owns a network
std::vector<std::vector<std::shared_ptr<Net>>> nets;

// The 0th irreducible is assumed to be totally symmetric
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
        // Network parameters
        std::vector<std::string> net_pars((NStates+1)*NStates/2);
        std::getline(ifs, line);
        for (auto & p : net_pars) {
            std::getline(ifs, p);
            CL::utility::trim(p);
        }
    ifs.close();
    // Initialize networks
    nets.resize(NStates);
    size_t index = 0;
    for (int i = 0; i < NStates; i++) {
        nets[i].resize(NStates);
        for (int j = i; j < NStates; j++) {
            nets[i][j] = std::make_shared<Net>(ON::PNR[symmetry[i][j]].size(), symmetry[i][j] == 0);
            nets[i][j]->to(torch::kFloat64);
            torch::load(nets[i][j], net_pars[index]);
            nets[i][j]->eval();
            index++;
        }
    }
}

// Input:  input layer
// Output: Hd
at::Tensor compute_Hd(const std::vector<at::Tensor> & x) {
    at::Tensor Hd = x[0].new_empty({NStates, NStates});
    for (int i = 0; i < NStates; i++)
    for (int j = i; j < NStates; j++)
    Hd[i][j] = nets[i][j]->forward(x[symmetry[i][j]]);
    return Hd;
}

} // namespace Hd