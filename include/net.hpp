#ifndef net_hpp
#define net_hpp

#include <torch/torch.h>

// A feedforward neural network to reduce dimensionality
namespace DimRed {

struct Net : torch::nn::Module {
    std::vector<size_t> dim;
    std::vector<std::vector<torch::nn::Linear*>> fc, fc_inv;

    // max_depth == 0 means unlimited
    Net(const std::vector<size_t> & _dim, const size_t & max_depth = 0);

    // Compute the reduced dimension
    torch::Tensor reduce(const torch::Tensor & x);
    
    // Inverse reduction
    torch::Tensor inverse(const torch::Tensor & x);

    // For pretraining
    torch::Tensor forward(const torch::Tensor & x);
};

void pretrain(const at::Tensor & origin, const int & intdim, 
const std::vector<size_t> & symmetry, 
const std::vector<std::string> & data_set, 
const std::string & data_type);

} // namespace DimRed

#endif