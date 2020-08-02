#include <torch/torch.h>
#include "../../Cpp-Library_v1.0.0/LinearAlgebra.hpp"
#include "../../include/AbInitio.hpp"
#include "../../include/net.hpp"

namespace DimRed {

void pretrain(const size_t & irred, const size_t & max_depth,
const std::vector<std::string> & data_set,
const std::string & chk, const std::string & opt, const size_t & epoch) {
    std::cout << "Start pretraining\n";

    auto net = std::make_shared<Net>(irred, max_depth);
    net->to(torch::kFloat64);

    auto * GeomSet = AbInitio::read_GeomSet(data_set);
    std::cout << "Number of geometries = " << GeomSet->size_int() << '\n';

    auto geom_loader = torch::data::make_data_loader(* GeomSet, 1);
    std::cout << "batch size = " << geom_loader->options().batch_size << '\n';

    torch::optim::Adam optimizer(net->parameters(), 0.01);

    size_t follow = epoch / 10;
    for (size_t iepoch = 0; iepoch < epoch; iepoch++) {
        for (auto & batch : * geom_loader) {
            optimizer.zero_grad();
            torch::Tensor loss = torch::zeros(1, at::TensorOptions().dtype(torch::kFloat64));
            for (auto & data : batch) {
                loss += torch::mse_loss(net->forward(data->SAIgeom[irred]), data->SAIgeom[irred],
                    at::Reduction::Sum);
            }
            loss.backward();
            optimizer.step();
        }
        if (iepoch % follow == 0) {
            std::cout << "epoch = " << iepoch + 1
                      << ", RMSD = " << RMSD(irred, net, GeomSet->example())
                      << ", grad = " << CL::LA::NetGradNorm(net->parameters()) << '\n';
            torch::save(net, "pretrain.pt");
        }
    }
}

} // namespace DimRed