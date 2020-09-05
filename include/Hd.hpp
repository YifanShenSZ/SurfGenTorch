/*
A feedforward neural network to compute diabatic Hamiltonian (Hd)

The 0th irreducible is assumed to be totally symmetric
*/

#ifndef Hd_hpp
#define Hd_hpp

#include <torch/torch.h>

namespace Hd {

struct Net : torch::nn::Module {
    std::vector<torch::nn::Linear *> fc;

    Net();
    // Totally symmetric irreducible additionally has const term (bias)
    // max_depth == 0 means unlimited
    Net(const size_t & init_dim, const bool & totally_symmetric, const size_t & max_depth = 0);
    ~Net();

    at::Tensor forward(const at::Tensor & x);

    // For training
    void copy(const std::shared_ptr<Net> & net);
    void warmstart(const std::string & chk, const size_t & chk_depth);
    void freeze(const size_t & freeze);
};

// Number of irreducible representations
extern size_t NIrred;
// Number of electronic states
extern int NStates;
// Symmetry of Hd elements
extern size_t ** symmetry;
// Each Hd element owns a network
extern std::vector<std::vector<std::shared_ptr<Net>>> nets;

// Symmetry adapted polynomials serve as the input layer
namespace input {
    // Return number of input neurons per irreducible
    std::vector<size_t> prepare_PNR(const std::string & Hd_input_layer_in);

    std::vector<at::Tensor> input_layer(const std::vector<at::Tensor> & x);
} // namespace input

void define_Hd(const std::string & Hd_in);

// Input:  input layer
// Output: Hd
at::Tensor compute_Hd(const std::vector<at::Tensor> & x);

} // namespace Hd

#endif