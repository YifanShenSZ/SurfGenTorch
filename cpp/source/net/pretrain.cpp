#include <torch/torch.h>

#include "../../Cpp-Library_v1.0.0/torch.hpp"

#include "../../include/pretrain.hpp"

namespace DimRed {

void warmstart(const size_t & irred, std::shared_ptr<Net> & net,
const std::string & chk, const size_t & chk_depth_) {
    auto warm_net = std::make_shared<Net>(irred, chk_depth_);
    size_t net_depth = net->fc.size();
    size_t chk_depth = warm_net->fc.size();
    if (net_depth == chk_depth) torch::load(net, chk);
    else {
        warm_net->to(torch::kFloat64);
        torch::load(warm_net, chk);
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < (net_depth < chk_depth ? net_depth : chk_depth); i++) {
            for (size_t j = 0; j < net->parameters()[i].numel(); j++) {
                net->parameters()[i].data_ptr<double>()[j] =
                    warm_net->parameters()[i].data_ptr<double>()[j];
                net->parameters()[i + net_depth].data_ptr<double>()[j] =
                    warm_net->parameters()[i + chk_depth].data_ptr<double>()[j];
            }
        }
    }
    warm_net.reset();
}

void pretrain(const size_t & irred, const size_t & max_depth,
const std::vector<std::string> & data_set,
const std::vector<std::string> & chk, const size_t & chk_depth,
const std::string & opt, const size_t & epoch) {
    std::cout << "Start pretraining\n";
    auto net = std::make_shared<Net>(irred, max_depth);
    net->to(torch::kFloat64);
    auto * GeomSet = AbInitio::read_GeomSet(data_set);
    std::cout << "Number of geometries = " << GeomSet->size_int() << '\n';
    if (opt == "Adam") {
        auto geom_loader = torch::data::make_data_loader(* GeomSet, 1);
        std::cout << "batch size = " << geom_loader->options().batch_size << '\n';
        torch::optim::Adam optimizer(net->parameters(), 0.01);
        if (! chk.empty()) {
            warmstart(irred, net, chk[0], chk_depth);
            if (chk.size() > 1) torch::load(optimizer, chk[1]);
        }
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
        if (! chk.empty()) warmstart(irred, net, chk[0], chk_depth);;
        FLopt::initialize(irred, net, GeomSet->example());
        FLopt::optimize(opt);
        std::cout << "RMSD = " << RMSD(irred, net, GeomSet->example()) << '\n';
        torch::save(net, "pretrain_net.pt");
    }
    
}

} // namespace DimRed