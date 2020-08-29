#include <omp.h>
#include <torch/torch.h>

#include <FortranLibrary.hpp>
#include <CppLibrary/TorchSupport.hpp>

#include "SSAIC.hpp"
#include "DimRed.hpp"
#include "AbInitio.hpp"

namespace pretrain {

// To make use of Fortran-Library nonlinear optimizers:
//     1. Map the network parameters to vector c
//     2. Compute residue and Jacobian
namespace FLopt {
    int OMP_NUM_THREADS;

    // The dimensionality reduction network (each thread owns a copy)
    std::vector<std::shared_ptr<DimRed::Net>> nets;

    // The irreducible to pretrain
    size_t irred;

    // Data set
    std::vector<AbInitio::geom *> GeomSet;
    // Divide data set for parallelism
    std::vector<size_t> chunk, start;

    // Push network parameters to parameter vector c
    void p2c(const std::shared_ptr<DimRed::Net> & net, double * c) {
        size_t count = 0;
        for (auto & p : net->parameters())
        if (p.requires_grad()) {
            std::memcpy(&(c[count]), p.data_ptr<double>(), p.numel() * sizeof(double));
            count += p.numel();
        }
    }
    // The other way round
    void c2p(const double * c, const std::shared_ptr<DimRed::Net> & net) {
        torch::NoGradGuard no_grad;
        size_t count = 0;
        for (auto & p : net->parameters())
        if (p.requires_grad()) {
            std::memcpy(p.data_ptr<double>(), &(c[count]), p.numel() * sizeof(double));
            count += p.numel();
        }
    }

    void net_zero_grad(const std::shared_ptr<DimRed::Net> & net) {
        for (auto & p : net->parameters())
        if (p.requires_grad() && p.grad().defined()) {
            p.grad().detach_();
            p.grad().zero_();
        };
    }

    void loss(double & l, const double * c, const int & Nc) {
        torch::NoGradGuard no_grad;
        std::vector<at::Tensor> loss(OMP_NUM_THREADS);
        #pragma omp parallel for
        for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
            c2p(c, nets[thread]);
            loss[thread] = at::zeros({}, at::TensorOptions().dtype(torch::kFloat64));
            for (size_t data = chunk[thread] - chunk[0]; data < chunk[thread]; data++) {
                loss[thread] += torch::mse_loss(
                    nets[thread]->forward(GeomSet[data]->SAIgeom[irred]), GeomSet[data]->SAIgeom[irred],
                    at::Reduction::Sum);
            }
        }
        l = 0.0;
        for (at::Tensor & piece : loss) l += piece.item<double>();
    }
    void grad(double * g, const double * c, const int & Nc) {
        std::vector<at::Tensor> loss(OMP_NUM_THREADS);
        #pragma omp parallel for
        for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
            auto & net = nets[thread];
            c2p(c, net);
            loss[thread] = at::zeros({}, at::TensorOptions().dtype(torch::kFloat64));
            for (size_t data = chunk[thread] - chunk[0]; data < chunk[thread]; data++) {
                loss[thread] += torch::mse_loss(
                    net->forward(GeomSet[data]->SAIgeom[irred]), GeomSet[data]->SAIgeom[irred],
                    at::Reduction::Sum);
            }
            net_zero_grad(net);
            loss[thread].backward();
        }
        // Push network gradients to g
        size_t count = 0;
        for (auto & p : nets[0]->parameters())
        if (p.requires_grad()) {
            std::memcpy(&(g[count]), p.grad().data_ptr<double>(), p.grad().numel() * sizeof(double));
            count += p.grad().numel();
        }
        for (int thread = 1; thread < OMP_NUM_THREADS; thread++) {
            size_t count = 0;
            for (auto & p : nets[thread]->parameters())
            if (p.requires_grad()) {
                double * pg = p.grad().data_ptr<double>();
                for (size_t i = 0; i < p.grad().numel(); i++) {
                    g[count] += pg[i];
                    count++;
                }
            }
        }
    }
    int loss_grad(double & l, double * g, const double * c, const int & Nc) {
        std::vector<at::Tensor> loss(OMP_NUM_THREADS);
        #pragma omp parallel for
        for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
            auto & net = nets[thread];
            c2p(c, net);
            loss[thread] = at::zeros({}, at::TensorOptions().dtype(torch::kFloat64));
            for (size_t data = chunk[thread] - chunk[0]; data < chunk[thread]; data++) {
                loss[thread] += torch::mse_loss(
                    net->forward(GeomSet[data]->SAIgeom[irred]), GeomSet[data]->SAIgeom[irred],
                    at::Reduction::Sum);
            }
            net_zero_grad(net);
            loss[thread].backward();
        }
        l = 0.0;
        for (at::Tensor & piece : loss) l += piece.item<double>();
        // Push network gradients to g
        size_t count = 0;
        for (auto & p : nets[0]->parameters())
        if (p.requires_grad()) {
            std::memcpy(&(g[count]), p.grad().data_ptr<double>(), p.grad().numel() * sizeof(double));
            count += p.grad().numel();
        }
        for (int thread = 1; thread < OMP_NUM_THREADS; thread++) {
            size_t count = 0;
            for (auto & p : nets[thread]->parameters())
            if (p.requires_grad()) {
                double * pg = p.grad().data_ptr<double>();
                for (size_t i = 0; i < p.grad().numel(); i++) {
                    g[count] += pg[i];
                    count++;
                }
            }
        }
        return 0;
    }

    void residue(double * r, const double * c, const int & NEq, const int & Nc) {
        torch::NoGradGuard no_grad;
        #pragma omp parallel for
        for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
            c2p(c, nets[thread]);
            size_t count = start[thread];
            for (size_t data = chunk[thread] - chunk[0]; data < chunk[thread]; data++) {
                at::Tensor r_tensor = nets[thread]->forward(GeomSet[data]->SAIgeom[irred]) - GeomSet[data]->SAIgeom[irred];
                std::memcpy(&(r[count]), r_tensor.data_ptr<double>(), r_tensor.numel() * sizeof(double));
                count += r_tensor.numel();
            }
        }
    }
    void Jacobian(double * JT, const double * c, const int & NEq, const int & Nc) {
        #pragma omp parallel for
        for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
            auto & net = nets[thread];
            c2p(c, net);
            size_t column = start[thread];
            for (size_t data = chunk[thread] - chunk[0]; data < chunk[thread]; data++) {
                at::Tensor r_tensor = net->forward(GeomSet[data]->SAIgeom[irred]);
                for (size_t el = 0; el < r_tensor.numel(); el++) {
                    net_zero_grad(net);
                    r_tensor[el].backward({}, true);
                    size_t row = 0;
                    for (auto & p : net->parameters())
                    if (p.requires_grad()) {
                        double * pg = p.grad().data_ptr<double>();
                        for (size_t i = 0; i < p.grad().numel(); i++) {
                            JT[row * NEq + column] = pg[i];
                            row++;
                        }
                    }
                    column++;
                }
            }
        }
    }

    void initialize(const std::shared_ptr<DimRed::Net> & net_,
    const size_t & irred_, const size_t & freeze_,
    const std::vector<AbInitio::geom *> & GeomSet_) {
        OMP_NUM_THREADS = omp_get_max_threads();
        std::cout << "The number of threads = " << OMP_NUM_THREADS << '\n';

        irred = irred_;

        nets.resize(OMP_NUM_THREADS);
        nets[0] = net_;
        for (int i = 1; i < OMP_NUM_THREADS; i++) {
            nets[i] = std::make_shared<DimRed::Net>(SSAIC::NSAIC_per_irred[irred], nets[0]->fc.size());
            nets[i]->to(torch::kFloat64);
            nets[i]->copy(nets[0]);
            nets[i]->freeze(freeze_);
        }

        GeomSet = GeomSet_;
        std::cout << "For parallelism, the number of data in use = "
                  << OMP_NUM_THREADS * (GeomSet.size() / OMP_NUM_THREADS) << '\n';
        chunk.resize(OMP_NUM_THREADS);
        start.resize(OMP_NUM_THREADS);
        chunk[0] = GeomSet.size() / OMP_NUM_THREADS;
        start[0] = 0;
        for (int i = 1; i < OMP_NUM_THREADS; i++) {
            chunk[i] = chunk[i-1] + chunk[0];
            start[i] = chunk[i-1] * SSAIC::NSAIC_per_irred[irred];
        }
    }

    void optimize(const std::string & opt, const size_t & epoch) {
        // Initialize
        int Nc = CL::TS::NParameters(nets[0]->parameters());
        std::cout << "There are " << Nc << " parameters to train\n";
        double * c = new double[Nc];
        p2c(nets[0], c);
        int NEq = SSAIC::NSAIC_per_irred[irred] * OMP_NUM_THREADS * (GeomSet.size() / OMP_NUM_THREADS);
        std::cout << "The data set corresponds to " << NEq << " least square equations" << std::endl;
        // Train
        if (opt == "CG") {
            FL::NO::ConjugateGradient(loss, grad, loss_grad, c, Nc, "DY", false, true, epoch);
        } else {
            FL::NO::TrustRegion(residue, Jacobian, c, NEq, Nc, true, epoch);
        }
        // Finish
        c2p(c, nets[0]);
        delete [] c;
    }
} // namespace FLopt

double RMSD(const size_t & irred, const std::shared_ptr<DimRed::Net> & net, const std::vector<AbInitio::geom *> & GeomSet) {
    double e = 0.0;
    torch::NoGradGuard no_grad;
    for (auto & geom : GeomSet) {
        e += torch::mse_loss(net->forward(geom->SAIgeom[irred]), geom->SAIgeom[irred],
             at::Reduction::Sum).item<double>();
    }
    e /= GeomSet.size() * SSAIC::NSAIC_per_irred[irred];
    return std::sqrt(e);
}

void pretrain(const size_t & irred, const size_t & max_depth, const size_t & freeze,
const std::vector<std::string> & data_set,
const std::vector<std::string> & chk, const size_t & chk_depth,
const std::string & opt, const size_t & epoch, const size_t & batch_size, const double & learning_rate) {
    std::cout << "Start pretraining\n";
    // Initialize network
    auto net = std::make_shared<DimRed::Net>(SSAIC::NSAIC_per_irred[irred], max_depth);
    net->to(torch::kFloat64);
    if (! chk.empty()) net->warmstart(chk[0], chk_depth);
    net->freeze(freeze);
    std::cout << "Number of trainable network parameters = " << CL::TS::NParameters(net->parameters()) << '\n';
    // Read geometry set
    auto * GeomSet = AbInitio::read_GeomSet(data_set);
    std::cout << "Number of geometries = " << GeomSet->size_int() << '\n';
    std::cout << "The initial guess gives:\n";
    std::cout << "RMSD = " << RMSD(irred, net, GeomSet->example()) << '\n';
    std::cout << std::endl;
    if (opt == "Adam" || opt == "SGD") {
        // Create geometry set loader
        auto geom_loader = torch::data::make_data_loader(* GeomSet,
            torch::data::DataLoaderOptions(batch_size).drop_last(true));
        std::cout << "batch size = " << batch_size << '\n';
        if (opt == "Adam") {
            // Create optimizer
            torch::optim::Adam optimizer(net->parameters(), learning_rate);
            if (chk.size() > 1 && max_depth == chk_depth) torch::load(optimizer, chk[1]);
            // Start training
            size_t follow = epoch / 10;
            for (size_t iepoch = 1; iepoch <= epoch; iepoch++) {
                for (auto & batch : * geom_loader) {
                    at::Tensor loss = at::zeros({}, at::TensorOptions().dtype(torch::kFloat64));
                    for (auto & data : batch) {
                        loss += torch::mse_loss(
                            net->forward(data->SAIgeom[irred]), data->SAIgeom[irred],
                            at::Reduction::Sum);
                    }
                    optimizer.zero_grad();
                    loss.backward();
                    optimizer.step();
                }
                if (iepoch % follow == 0) {
                    std::cout << "epoch = " << iepoch
                              << ", RMSD = " << RMSD(irred, net, GeomSet->example()) << '\n';
                    torch::save(net, "pretrain_"+std::to_string(iepoch)+".net");
                    torch::save(optimizer, "pretrain_"+std::to_string(iepoch)+".opt");
                }
            }
        }
        else {
            // Create optimizer
            torch::optim::SGD optimizer(net->parameters(),
                torch::optim::SGDOptions(learning_rate)
                .momentum(0.9).nesterov(true));
            if (chk.size() > 1 && max_depth == chk_depth) torch::load(optimizer, chk[1]);
            // Start training
            size_t follow = epoch / 10;
            for (size_t iepoch = 1; iepoch <= epoch; iepoch++) {
                for (auto & batch : * geom_loader) {
                    at::Tensor loss = at::zeros({}, at::TensorOptions().dtype(torch::kFloat64));
                    for (auto & data : batch) {
                        loss += torch::mse_loss(
                            net->forward(data->SAIgeom[irred]), data->SAIgeom[irred],
                            at::Reduction::Sum);
                    }
                    optimizer.zero_grad();
                    loss.backward();
                    optimizer.step();
                }
                if (iepoch % follow == 0) {
                    std::cout << "epoch = " << iepoch
                              << ", RMSD = " << RMSD(irred, net, GeomSet->example()) << '\n';
                    torch::save(net, "pretrain_"+std::to_string(iepoch)+".net");
                    torch::save(optimizer, "pretrain_"+std::to_string(iepoch)+".opt");
                }
            }
        }
    }
    else {
        FLopt::initialize(net, irred, freeze, GeomSet->example());
        FLopt::optimize(opt, epoch);
        std::cout << "RMSD = " << RMSD(irred, net, GeomSet->example()) << '\n';
        torch::save(net, "pretrain.net");
    }
}

} // namespace pretrain