#include <torch/torch.h>
#include "../Cpp-Library_v1.0.0/torch_LinearAlgebra.hpp"
#include "../include/AbInitio.hpp"
#include "../include/net.hpp"

void DimRed::pretrain(const at::Tensor & origin, const int & intdim, 
const std::vector<size_t> & symmetry, 
const std::vector<std::string> & data_set, 
const std::string & data_type) {
    if (data_type == "double") {
        AbInitio::DataSet<AbInitio::geom<double>> * GeomSet;
        AbInitio::read_GeomSet(data_set, origin, intdim, GeomSet);
        std::cout << "Number of geometries = " << GeomSet->size_int() << '\n';

        size_t batch_size = GeomSet->size_int();
        std::cout << "batch size = " << batch_size << '\n';
        auto geom_loader = torch::data::make_data_loader(* GeomSet, batch_size);

        size_t max_depth = * std::max_element(symmetry.begin(), symmetry.end()) - 1;
        for (size_t idepth = 1; idepth <= max_depth; idepth++) {
            auto net = std::make_shared<DimRed::Net>(symmetry, idepth);
            net->to(torch::kFloat64);
            
            float learning_rate = 0.01;
            size_t epoch = 10000;
            size_t follow = 100;

            torch::optim::SGD optimizer(net->parameters(), learning_rate);
            // torch::optim::LBFGS optimizer(net->parameters(), learning_rate);
            
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
                    std::cout << "Epoch = " << iepoch
                              << ", Loss = " << loss.item<float>()
                              << ", ||Grad|| = " << torch_LinearAlgebra::NetGradNorm(net->parameters()) << '\n';
                    torch::save(net, "pretrain.pt");
                }
            }
        }
    } else {
        // Not implemented
    }
}