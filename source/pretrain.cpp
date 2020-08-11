#include <torch/torch.h>

#include <CppLibrary/TorchSupport.hpp>

#include "SSAIC.hpp"
#include "AbInitio.hpp"
#include "net.hpp"

namespace DimRed {

namespace FLopt {
    void initialize(const std::shared_ptr<Net> & net_,
        const size_t & irred_, const size_t & freeze_,
        const std::vector<AbInitio::geom *> & GeomSet_);
    void optimize(const std::string & opt, const size_t & epoch);
} // namespace FLopt

double RMSD(const size_t & irred, const std::shared_ptr<Net> & net, const std::vector<AbInitio::geom*> & GeomSet) {
    double e = 0.0;
    torch::NoGradGuard no_grad;
    for (auto & geom : GeomSet) {
        e += torch::mse_loss(net->forward(geom->SAIgeom[irred]), geom->SAIgeom[irred],
             at::Reduction::Sum).item<double>();
    }
    e /= (double)GeomSet.size();
    e /= (double)SSAIC::NSAIC_per_irred[irred];
    return std::sqrt(e);
}

void pretrain(const size_t & irred, const size_t & max_depth, const size_t & freeze,
const std::vector<std::string> & data_set,
const std::vector<std::string> & chk, const size_t & chk_depth,
const std::string & opt, const size_t & epoch,
const size_t & batch_size, const double & learning_rate) {
    std::cout << "Start pretraining\n";
    // Initialize network
    auto net = std::make_shared<Net>(SSAIC::NSAIC_per_irred[irred], max_depth);
    net->to(torch::kFloat64);
    if (! chk.empty()) net->warmstart(chk[0], chk_depth);
    net->freeze(freeze);
    std::cout << "Number of trainable parameters = " << CL::TS::NParameters(net->parameters()) << '\n';
    // Read geometry set
    auto * GeomSet = AbInitio::read_GeomSet(data_set);
    std::cout << "Number of geometries = " << GeomSet->size_int() << '\n';
    if (opt == "Adam" || opt == "SGD") {
        // Create geometry set loader
        auto geom_loader = torch::data::make_data_loader(* GeomSet,
            torch::data::DataLoaderOptions(batch_size).drop_last(true));
        std::cout << "batch size = " << geom_loader->options().batch_size << '\n';
        if (opt == "Adam") {
            // Create optimizer
            torch::optim::Adam optimizer(net->parameters(), learning_rate);
            if (chk.size() > 1 && max_depth == chk_depth) torch::load(optimizer, chk[1]);
            // Start training
            size_t follow = epoch / 10;
            for (size_t iepoch = 0; iepoch < epoch; iepoch++) {
                for (auto & batch : * geom_loader) {
                    optimizer.zero_grad();
                    torch::Tensor loss = torch::zeros(1, at::TensorOptions().dtype(torch::kFloat64));
                    for (auto & data : batch) {
                        loss += torch::mse_loss(
                            net->forward(data->SAIgeom[irred]), data->SAIgeom[irred],
                            at::Reduction::Sum);
                    }
                    loss.backward();
                    optimizer.step();
                }
                if (iepoch % follow == follow - 1) {
                    std::cout << "epoch = " << iepoch + 1
                              << ", RMSD = " << RMSD(irred, net, GeomSet->example()) << '\n';
                    torch::save(net, "pretrain_net_"+std::to_string(iepoch+1)+".pt");
                    torch::save(optimizer, "pretrain_optimizer_"+std::to_string(iepoch+1)+".pt");
                }
            }
        }
        else {
            // Create optimizer
            torch::optim::SGD optimizer(net->parameters(),
                torch::optim::SGDOptions(learning_rate)
                .momentum(0.9)
                .nesterov(true));
            if (chk.size() > 1 && max_depth == chk_depth) torch::load(optimizer, chk[1]);
            // Start training
            size_t follow = epoch / 10;
            for (size_t iepoch = 0; iepoch < epoch; iepoch++) {
                for (auto & batch : * geom_loader) {
                    optimizer.zero_grad();
                    torch::Tensor loss = torch::zeros(1, at::TensorOptions().dtype(torch::kFloat64));
                    for (auto & data : batch) {
                        loss += torch::mse_loss(
                            net->forward(data->SAIgeom[irred]), data->SAIgeom[irred],
                            at::Reduction::Sum);
                    }
                    loss.backward();
                    optimizer.step();
                }
                if (iepoch % follow == follow - 1) {
                    std::cout << "epoch = " << iepoch + 1
                              << ", RMSD = " << RMSD(irred, net, GeomSet->example()) << '\n';
                    torch::save(net, "pretrain_net_"+std::to_string(iepoch+1)+".pt");
                    torch::save(optimizer, "pretrain_optimizer_"+std::to_string(iepoch+1)+".pt");
                }
            }
        }
    }
    else {
        FLopt::initialize(net, irred, freeze, GeomSet->example());
        FLopt::optimize(opt, epoch);
        std::cout << "RMSD = " << RMSD(irred, net, GeomSet->example()) << '\n';
        torch::save(net, "pretrain_net.pt");
    }
}

} // namespace DimRed