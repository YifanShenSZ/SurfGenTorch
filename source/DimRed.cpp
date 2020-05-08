// A feedforward neural network to reduce dimensionality

#include <torch/torch.h>
#include "../include/net.hpp"

namespace DimRed {

// max_depth == 0 means unlimited
Net::Net(const std::vector<size_t> & _dim, const size_t & max_depth) {
    dim.resize(_dim.size());
    dim = _dim;

    std::vector<size_t> depth = dim;
    if (max_depth == 0) for (size_t & d : depth) d -= 1;
    else for (size_t & d : depth) {
        d -= 1;
        d = d < max_depth ? d : max_depth;
    }

    fc.resize(dim.size());
    for (size_t iirred = 0; iirred < dim.size(); iirred++) {
        fc[iirred].resize(depth[iirred]);
        for (size_t idim = 0; idim < depth[iirred]; idim++) {
            fc[iirred][idim] = new torch::nn::Linear{nullptr};
            * fc[iirred][idim] = register_module(
            "fc-"+std::to_string(iirred)+"-"+std::to_string(idim),
            torch::nn::Linear(torch::nn::LinearOptions(dim[iirred]-idim, dim[iirred]-idim-1)
            .bias(false)));
    } }

    fc_inv.resize(dim.size());
    for (size_t iirred = 0; iirred < dim.size(); iirred++) {
        fc_inv[iirred].resize(depth[iirred]);
        for (size_t idim = 0; idim < depth[iirred]; idim++) {
            fc_inv[iirred][idim] = new torch::nn::Linear{nullptr};
            * fc_inv[iirred][idim] = register_module(
            "fc_inv-"+std::to_string(iirred)+"-"+std::to_string(idim),
            torch::nn::Linear(torch::nn::LinearOptions(idim+dim[iirred]-depth[iirred], idim+dim[iirred]-depth[iirred]+1)
            .bias(false)));
    } }

    // torch::NoGradGuard no_grad;
    // for (at::Tensor & p : this->parameters()) p *= 10.0;
}

// Compute the reduced dimension
torch::Tensor Net::reduce(const torch::Tensor & x) {
    size_t count = 0;
    std::vector<torch::Tensor> x_per_irred(dim.size());
    for (size_t iirred = 0; iirred < dim.size(); iirred++) {
        // Initialize an irreducible
        x_per_irred[iirred] = torch::empty(dim[iirred], x.options());
        for (size_t idim = 0; idim < dim[iirred]; idim++) {
            x_per_irred[iirred][idim] = x[count];
            count++;
        }
        // Reduce dimensionality
        for (auto & layer : fc[iirred]) {
            x_per_irred[iirred] = (*layer)->forward(x_per_irred[iirred]);
            x_per_irred[iirred] = torch::tanh(x_per_irred[iirred]);
        }
    }
    // Output the reduced dimensions
    torch::Tensor y = torch::empty(dim.size(), x.options());
    for (size_t iirred = 0; iirred < dim.size(); iirred++) y[iirred] = x_per_irred[iirred];
    return y;
}

// Inverse reduction
torch::Tensor Net::inverse(const torch::Tensor & x) {
    size_t count = 0;
    std::vector<torch::Tensor> x_per_irred(dim.size());
    for (size_t iirred = 0; iirred < dim.size(); iirred++) {
        // Initialize an irreducible
        x_per_irred[iirred] = x[count]; count++;
        // Inverse the reduction
        for (auto & layer : fc_inv[iirred]) {
            x_per_irred[iirred] = (*layer)->forward(x_per_irred[iirred]);
            x_per_irred[iirred] = torch::tanh(x_per_irred[iirred]);
        }
    }
    // Output the original dimension
    count = 0;
    torch::Tensor y = torch::empty(x.sizes(), x.options());
    for (size_t iirred = 0; iirred < dim.size(); iirred++)
    for (size_t idim = 0; idim < dim[iirred]; idim++) {
        y[count] = x_per_irred[iirred][idim];
        count++;
    }
    return y;
}

// For pretraining
torch::Tensor Net::forward(const torch::Tensor & x) {
    size_t count = 0;
    std::vector<torch::Tensor> x_per_irred(dim.size());
    for (size_t iirred = 0; iirred < dim.size(); iirred++) {
        // Initialize an irreducible
        x_per_irred[iirred] = torch::empty(dim[iirred], x.options());
        for (size_t idim = 0; idim < dim[iirred]; idim++) {
            x_per_irred[iirred][idim] = x[count];
            count++;
        }
        // Reduce dimensionality
        for (auto & layer : fc[iirred]) {
            x_per_irred[iirred] = (*layer)->forward(x_per_irred[iirred]);
            
            x_per_irred[iirred] = torch::tanh(x_per_irred[iirred]);
            // x_per_irred[iirred] = torch::nn::functional::tanhshrink(x_per_irred[iirred]);
        }
        // Inverse the reduction
        for (auto & layer : fc_inv[iirred]) {
            x_per_irred[iirred] = (*layer)->forward(x_per_irred[iirred]);
            
            x_per_irred[iirred] = torch::tanh(x_per_irred[iirred]);
            // x_per_irred[iirred] = torch::nn::functional::tanhshrink(x_per_irred[iirred]);
        }
    }
    // Output the original dimension
    count = 0;
    torch::Tensor y = torch::empty(x.sizes(), x.options());
    for (size_t iirred = 0; iirred < dim.size(); iirred++)
    for (size_t idim = 0; idim < dim[iirred]; idim++) {
        y[count] = x_per_irred[iirred][idim];
        count++;
    }
    return y;
}
//torch::Tensor Net::loss()

} // namespace DimRed