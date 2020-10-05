/*
A feedforward neural network to fit a matrix element of an observable
based on the symmetry adapted internal coordinates (SAIC)

To maintain symmetry:
    1. The input layer takes the symmetry adapted polynomials of SAICs
    2. Only the totally symmetric irreducible has bias
    3. The activation functions are odd, except for the totally symmetric irreducible
*/

#ifndef observable_net_hpp
#define observable_net_hpp

#include <torch/torch.h>

namespace ON {

// The general form for a matrix element of an observable
// Only the totally symmetric irreducible has bias
struct Net : torch::nn::Module {
    // Fully connected layers to gradually reduce dimensionality
    std::vector<torch::nn::Linear *> fc;
    // A dot product to reduce to scalar
    torch::nn::Linear tail{nullptr};
    // A status flag: whether the network has been warmstarted or not
    bool cold = true;

    Net();
    // The dimensions of `fc` are determined by `dims`
    Net(const std::vector<size_t> dims, const bool & totally_symmetric);
    // Same structure to net
    Net(const std::shared_ptr<Net> & net);
    ~Net();

    at::Tensor forward(const at::Tensor & x);

    // Copy the parameters from net
    void copy(const std::shared_ptr<Net> & net);
    // Warmstart from checkpoint
    void warmstart(const std::string & chk, const std::vector<size_t> chk_dims);
    // Freeze the leading `freeze` layers in fc
    void freeze(const size_t & freeze);
};

// Polynomial numbering rule
struct PolNumbRul {
    std::vector<size_t> irred, coord;

    PolNumbRul();
    // For example, the input line of a 2nd order term made up by 
    // the 1st coordinate in the 2nd irreducible and 
    // the 3rd coordinate in the 4th irreducible is:
    //     2    2,1    4,3
    PolNumbRul(const std::vector<std::string> & input_line);
    ~PolNumbRul();
};

// Polynomial numbering rule, PNR[i][j] (the j-th polynomial in i-th irreducible)
// = product of PNR[i][j].coord[k]-th coordinate in PNR[i][j].irred[k]-th irreducible
extern std::vector<std::vector<PolNumbRul>> PNR;

// Symmetry adapted polynomials serve as the input layer,
// so we have to define the polynomial numbering rule in global variable PNR
void define_PNR(const std::string & input_layer_in);

// Compute the symmetry adapted polynomials from monomials
std::vector<at::Tensor> input_layer(const std::vector<at::Tensor> & x);

} // namespace ON

#endif