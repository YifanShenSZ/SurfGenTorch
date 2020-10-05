/*
A feedforward neural network to fit diabatic Hamiltonian (Hd)
*/

#ifndef Hd_hpp
#define Hd_hpp

#include <torch/torch.h>

#include "observable_net.hpp"

namespace Hd {

// Number of electronic states
extern int NStates;
// Symmetry of Hd elements
extern size_t ** symmetry;
// Each Hd element owns a network
extern std::vector<std::vector<std::vector<std::shared_ptr<ON::Net>>>> nets;

// Define the diabatic Hamiltonian and set to training mode
void define_Hd_train(const std::string & Hd_in);

// Define the diabatic Hamiltonian and set to evaluation mode
void define_Hd(const std::string & Hd_in);

// Input:  input layer
// Output: Hd
at::Tensor compute_Hd(const std::vector<at::Tensor> & x);

} // namespace Hd

#endif