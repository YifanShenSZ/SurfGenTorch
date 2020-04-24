#include <torch/torch.h>
#include "../include/DimRed.hpp"

DimRed::net::net(const std::vector<size_t> & dim_per_irred) {
    for (size_t iirred = 0; iirred < dim_per_irred.size(); iirred++) {
    fc[iirred].resize(dim_per_irred[iirred]-1);
    for (size_t idim = 0; idim < dim_per_irred[iirred]-1; idim++) {
        fc[iirred][idim] = new torch::nn::Linear{nullptr};
        * fc[iirred][idim] = register_module(
        "fc-"+std::to_string(iirred)+"-"+std::to_string(idim),
        torch::nn::Linear(dim_per_irred[iirred]-idim, dim_per_irred[iirred]-idim-1));
    } }
}

torch::Tensor DimRed::net::forward(torch::Tensor x) {
    return x;
}

//struct Net : torch::nn::Module {
//    // Implement the Net's algorithm.
//    torch::Tensor forward(torch::Tensor x) {
//        // Use one of many tensor manipulation functions.
//        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
//        x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
//        x = torch::relu(fc2->forward(x));
//        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
//        return x;
//    }
//};