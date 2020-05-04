#include <torch/torch.h>
#include "../include/DimRed.hpp"

namespace DimRed {

Net::Net(const std::vector<size_t> & dim_per_irred) {
    dim_per_irred_.resize(dim_per_irred.size());
    dim_per_irred_ = dim_per_irred;

    fc.resize(dim_per_irred_.size());
    fc_inv.resize(dim_per_irred_.size());
    for (size_t iirred = 0; iirred < dim_per_irred_.size(); iirred++) {
    fc[iirred].resize(dim_per_irred_[iirred]-1);
    fc_inv[iirred].resize(dim_per_irred_[iirred]-1);
    for (size_t idim = 0; idim < dim_per_irred_[iirred]-1; idim++) {
        fc[iirred][idim] = new torch::nn::Linear{nullptr};
        * fc[iirred][idim] = register_module(
        "fc-"+std::to_string(iirred)+"-"+std::to_string(idim),
        torch::nn::Linear(torch::nn::LinearOptions(dim_per_irred_[iirred]-idim, dim_per_irred_[iirred]-idim-1).bias(false)));
        fc_inv[iirred][idim] = new torch::nn::Linear{nullptr};
        * fc_inv[iirred][idim] = register_module(
        "fc_inv-"+std::to_string(iirred)+"-"+std::to_string(idim),
        torch::nn::Linear(torch::nn::LinearOptions(idim+1, idim+2).bias(false)));
    } }
}

torch::Tensor Net::forward(const torch::Tensor & x) {
    size_t count = 0;
    std::vector<torch::Tensor> x_per_irred(dim_per_irred_.size());
    for (size_t iirred = 0; iirred < dim_per_irred_.size(); iirred++) {
        // Initialize an irreducible
        x_per_irred[iirred] = torch::empty(dim_per_irred_[iirred], x.options());
        for (size_t idim = 0; idim < dim_per_irred_[iirred]; idim++) {
            x_per_irred[iirred][idim] = x[count];
            count++;
        }
        // Reduce dimensionality
        for (size_t idim = 0; idim < dim_per_irred_[iirred]-1; idim++) {
            x_per_irred[iirred] = (*fc[iirred][idim])->forward(x_per_irred[iirred]);
            x_per_irred[iirred] = torch::tanh(x_per_irred[iirred]);
        }
        // Inverse the reduction
        for (size_t idim = 0; idim < dim_per_irred_[iirred]-1; idim++) {
            x_per_irred[iirred] = (*fc_inv[iirred][idim])->forward(x_per_irred[iirred]);
            x_per_irred[iirred] = torch::tanh(x_per_irred[iirred]);
        }
    }
    // Output the original dimension
    count = 0;
    torch::Tensor y = torch::empty(x.sizes(), x.options());
    for (size_t iirred = 0; iirred < dim_per_irred_.size(); iirred++)
    for (size_t idim = 0; idim < dim_per_irred_[iirred]; idim++) {
        y[count] = x_per_irred[iirred][idim];
        count++;
    }
    return y;
}

torch::Tensor Net::reduce(const torch::Tensor & x) {
    size_t count = 0;
    std::vector<torch::Tensor> x_per_irred(dim_per_irred_.size());
    for (size_t iirred = 0; iirred < dim_per_irred_.size(); iirred++) {
        // Initialize an irreducible
        x_per_irred[iirred] = torch::empty(dim_per_irred_[iirred], x.options());
        for (size_t idim = 0; idim < dim_per_irred_[iirred]; idim++) {
            x_per_irred[iirred][idim] = x[count];
            count++;
        }
        // Reduce dimensionality
        for (size_t idim = 0; idim < dim_per_irred_[iirred]-1; idim++) {
            x_per_irred[iirred][idim] = (*fc[iirred][idim])->forward(x_per_irred[iirred][idim]);
            x_per_irred[iirred][idim] = torch::tanh(x_per_irred[iirred][idim]);
        }
    }
    // Output the reduced dimensions
    torch::Tensor y = torch::empty(dim_per_irred_.size(), x.options());
    for (size_t iirred = 0; iirred < dim_per_irred_.size(); iirred++) y[iirred] = x_per_irred[iirred][0];
    return y;
}

} // namespace DimRed