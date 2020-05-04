// A feedforward neural network to reduce dimensionality

#ifndef DimRed_hpp
#define DimRed_hpp

#include <torch/torch.h>

namespace DimRed {

struct Net : torch::nn::Module {
    std::vector<size_t> dim_per_irred_;
    std::vector<std::vector<torch::nn::Linear*>> fc, fc_inv;

    Net(const std::vector<size_t> & dim_per_irred);

    // For pretraining
    torch::Tensor forward(const torch::Tensor & x);

    // Compute the reduced dimension
    torch::Tensor reduce(const torch::Tensor & x);
    // its inverse
    torch::Tensor inverse(const torch::Tensor & x);
};

template <typename Tl, typename Tn> void pretrain(
Tl & geom_loader, Tn net, const size_t & batch_size,
const float & learning_rate = 0.01,
const size_t & epoch = 1000, const size_t & follow = 1) {
    torch::optim::SGD optimizer(net->parameters(), learning_rate);
    for (size_t iepoch = 0; iepoch < epoch; iepoch++) {
        torch::Tensor loss;
        std::vector<torch::Tensor> prediction(batch_size), deviation(batch_size);
        for (auto & batch : * geom_loader) {
            optimizer.zero_grad();
            #pragma omp parallel for
            for (size_t i = 0; i < batch.size(); i++) {
                prediction[i] = net->forward(batch[i]->intgeom());
                deviation[i] = torch::mse_loss(prediction[i], batch[i]->intgeom(), at::Reduction::Sum);
            }
            loss = deviation[0];
            for (size_t i = 1; i < batch.size(); i++) loss += deviation[i];
            loss.backward();

            optimizer.step();
        }
        if (iepoch % follow == 0) {
            std::cout << "Epoch: " << iepoch << " | Loss: " << loss.item<float>() << '\n';
            torch::save(net, "pretrain.pt");
        }
    }
}

} // namespace DimRed

#endif