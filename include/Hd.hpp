/*
A feedforward neural network to fit diabatic Hamiltonian (Hd)
*/

#ifndef Hd_hpp
#define Hd_hpp

#include <torch/torch.h>

#include "observable_net.hpp"

namespace Hd {

struct Net : ON::Net {
    Net();
    Net(const size_t & init_dim, const bool & totally_symmetric, const int64_t & max_depth = -1);
    ~Net();
};

// Number of electronic states
extern int NStates;
// Symmetry of Hd elements
extern size_t ** symmetry;
// Each Hd element owns a network
extern std::vector<std::vector<std::vector<std::shared_ptr<Net>>>> nets;

// The 0th irreducible is assumed to be totally symmetric
void define_Hd(const std::string & Hd_in);

// Input:  input layer
// Output: Hd
at::Tensor compute_Hd(const std::vector<at::Tensor> & x);

} // namespace Hd

#endif