/*
A feedforward neural network to reduce dimensionality

To maintain symmetry:
    1. Each irreducible owns an autoencoder, whose inputs are the SSAICs belonging to this irreducible
    2. Only the totally symmetric irreducible has bias
    3. The activation functions are odd, except for the totally symmetric irreducible
*/

#ifndef DimRed_hpp
#define DimRed_hpp

#include <torch/torch.h>

namespace DimRed {

// Only the totally symmetric irreducible has bias
struct Net : torch::nn::Module {
    // Fully connected layer to reduce dimensionality
    std::vector<torch::nn::Linear *> fc;
    // Fully connected layer to inverse the reduction
    std::vector<torch::nn::Linear *> fc_inv;
    // A status flag: whether the network has been warmstarted or not
    bool cold = true;

    Net();
    // The dimensions of `fc` are determined by `dims`
    // `fc_inv` has the mirror structure to `fc`
    Net(const std::vector<size_t> dims, const bool & totally_symmetric);
    // Same structure to net
    Net(const std::shared_ptr<Net> & net);
    ~Net();

    at::Tensor reduce(const at::Tensor & x);
    at::Tensor inverse(const at::Tensor & x);
    at::Tensor forward(const at::Tensor & x);

    // Copy the parameters from net
    void copy(const std::shared_ptr<Net> & net);
    // Warmstart from checkpoint
    void warmstart(const std::string & chk, const std::vector<size_t> chk_dims);
    // Freeze the leading `freeze` layers in fc and fc_inv
    void freeze(const size_t & freeze);
    // Freeze fc
    void freeze_reduction();
    // Freeze fc_inv
    void freeze_inverse();
};

// Each irreducible owns a network
extern std::vector<std::shared_ptr<Net>> nets;

// Define the dimensionality reduction and set to training mode
void define_DimRed_train(const std::string & DimRed_in);

// Define the dimensionality reduction and set to evaluation mode
void define_DimRed(const std::string & DimRed_in);

std::vector<at::Tensor> reduce(const std::vector<at::Tensor> & x);

std::vector<at::Tensor> inverse(const std::vector<at::Tensor> & x);

} // namespace DimRed

#endif