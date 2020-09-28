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
// Each Hd element owns a bunch of networks
std::vector<std::vector<std::vector<std::shared_ptr<Net>>>> nets;

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
        std::vector<std::string> net_pars;
        std::vector<int64_t> net_depths;
        std::getline(ifs, line);
        while (true) {
            std::getline(ifs, line);
            if (! ifs.good()) break;
            strs = CL::utility::split(line);
            net_pars.push_back(strs[0]);
            if (strs.size() > 1) net_depths.push_back(std::stoi(strs[1]));
            else                 net_depths.push_back(-1);
        }
    ifs.close();
    // Initialize networks
    nets.resize(NStates);
    size_t count = 0;
    for (int i = 0; i < NStates; i++) {
        assert((count < net_pars.size(), "Insufficient network parameter files"));
        nets[i].resize(NStates);
        nets[i][i].resize(1);
        nets[i][i][0] = std::make_shared<Net>(ON::PNR[0].size(), true, net_depths[count]);
        nets[i][i][0]->to(torch::kFloat64);
        torch::load(nets[i][i][0], net_pars[count]);
        nets[i][i][0]->eval();
        count++;
        for (int j = i + 1; j < NStates; j++) {
            nets[i][j].resize(ON::PNR[symmetry[i][j]].size());
            for (auto & net : nets[i][j]) {
                assert((count < net_pars.size(), "Insufficient network parameter files"));
                net = std::make_shared<Net>(ON::PNR[0].size(), true, net_depths[count]);
                net->to(torch::kFloat64);
                torch::load(net, net_pars[count]);
                net->eval();
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