#ifndef net_hpp
#define net_hpp

#include <torch/torch.h>

#include "AbInitio.hpp"

// A feedforward neural network to reduce dimensionality
// Each irreducible has its own dimensionality reduction network
namespace DimRed {

struct Net : torch::nn::Module {
    std::vector<torch::nn::Linear *> fc, fc_inv;

    // max_depth == 0 means unlimited
    Net(const size_t & irred, const size_t & max_depth = 0);

    // Compute the reduced dimension
    //torch::Tensor reduce(const torch::Tensor & x);
    
    // Inverse reduction
    //torch::Tensor inverse(const torch::Tensor & x);

    // For pretraining
    torch::Tensor forward(const at::Tensor & x);
};

double RMSD(const size_t & irred, const std::shared_ptr<Net> & net, const std::vector<AbInitio::geom*> & geom_set);

void pretrain(const size_t & irred, const size_t & max_depth,
const std::vector<std::string> & data_set,
const std::string & chk="null", const std::string & opt="Adam", const size_t & epoch=1000);

} // namespace DimRed

#endif