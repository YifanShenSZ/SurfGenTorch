/*
To make use of Fortran-Library nonlinear optimizers:
    1. Map the network parameters to vector c
    2. Compute residue and Jacobian
*/

#include <omp.h>

#include <FortranLibrary.hpp>
#include <CppLibrary/TorchSupport.hpp>

#include "SSAIC.hpp"
#include "AbInitio.hpp"
#include "net.hpp"

namespace DimRed { namespace FLopt {

size_t OMP_NUM_THREADS;

// The dimensionality reduction network (each thread owns a copy)
std::vector<std::shared_ptr<Net>> nets;
// Number of trainable parameters (length of parameter vector c)
int Nc;
// parameter vector c
double * c;

// The irreducible to pretrain
size_t irred;

// data set
std::vector<AbInitio::geom *> GeomSet;
// Number of least square equations
int NEq;
// divide data set for parallelism
std::vector<size_t> chunk;

// work variable: a fake optimizer to zero_grad
std::vector<torch::optim::Adam *> optimizer;

// Push network parameters to c
void p2c(const std::shared_ptr<Net> & net) {
    size_t count = 0;
    for (auto & p : net->parameters())
    if (p.requires_grad()) {
        double * pp = p.data_ptr<double>();
        for (size_t i = 0; i < p.numel(); i++) {
            c[count] = pp[i];
            count++;
        }
    }
}
// Push c to network parameters
void c2p(const std::shared_ptr<Net> & net) {
    torch::NoGradGuard no_grad;
    size_t count = 0;
    for (auto & p : net->parameters())
    if (p.requires_grad()) {
        double * pp = p.data_ptr<double>();
        for (size_t i = 0; i < p.numel(); i++) {
            pp[i] = c[count];
            count++;
        }
    }
}

void loss(double & l, const double * c, const int & Nc) {
    torch::NoGradGuard no_grad;
    std::vector<at::Tensor> loss(OMP_NUM_THREADS);
    #pragma omp parallel for
    for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
        c2p(nets[thread]);
        loss[thread] = torch::zeros(1, at::TensorOptions().dtype(torch::kFloat64));
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
    std::vector<torch::Tensor> loss(OMP_NUM_THREADS);
    #pragma omp parallel for
    for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
        c2p(nets[thread]);
        loss[thread] = torch::zeros(1, at::TensorOptions().dtype(torch::kFloat64));
        for (size_t data = chunk[thread] - chunk[0]; data < chunk[thread]; data++) {
            loss[thread] += torch::mse_loss(
                nets[thread]->forward(GeomSet[data]->SAIgeom[irred]), GeomSet[data]->SAIgeom[irred],
                at::Reduction::Sum);
        }
        optimizer[thread]->zero_grad();
        loss[thread].backward();
    }
    // Push network gradients to g
    size_t count = 0;
    for (auto & p : nets[0]->parameters())
    if (p.requires_grad()) {
        double * pg = p.grad().data_ptr<double>();
        for (size_t i = 0; i < p.grad().numel(); i++) {
            g[count] = pg[i];
            count++;
        }
    }
    for (int thread = 1; thread < OMP_NUM_THREADS; thread++) {
        size_t count = 0;
        for (auto & p : nets[0]->parameters())
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
    std::vector<torch::Tensor> loss(OMP_NUM_THREADS);
    #pragma omp parallel for
    for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
        c2p(nets[thread]);
        loss[thread] = torch::zeros(1, at::TensorOptions().dtype(torch::kFloat64));
        for (size_t data = chunk[thread] - chunk[0]; data < chunk[thread]; data++) {
            loss[thread] += torch::mse_loss(
                nets[thread]->forward(GeomSet[data]->SAIgeom[irred]), GeomSet[data]->SAIgeom[irred],
                at::Reduction::Sum);
        }
        optimizer[thread]->zero_grad();
        loss[thread].backward();
    }
    l = 0.0;
    for (at::Tensor & piece : loss) l += piece.item<double>();
    // Push network gradients to g
    size_t count = 0;
    for (auto & p : nets[0]->parameters())
    if (p.requires_grad()) {
        double * pg = p.grad().data_ptr<double>();
        for (size_t i = 0; i < p.grad().numel(); i++) {
            g[count] = pg[i];
            count++;
        }
    }
    for (int thread = 1; thread < OMP_NUM_THREADS; thread++) {
        size_t count = 0;
        for (auto & p : nets[0]->parameters())
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
        c2p(nets[thread]);
        size_t count = (chunk[thread] - chunk[0]) * SSAIC::NSAIC_per_irred[irred];
        for (size_t data = chunk[thread] - chunk[0]; data < chunk[thread]; data++) {
            torch::Tensor r_tensor = nets[thread]->forward(GeomSet[data]->SAIgeom[irred]) - GeomSet[data]->SAIgeom[irred];
            double * pr = r_tensor.data_ptr<double>();
            for (size_t i = 0; i < r_tensor.numel(); i++) {
                r[count] = pr[i];
                count++;
            }
        }
    }
}
void Jacobian(double * JT, const double * c, const int & NEq, const int & Nc) {
    #pragma omp parallel for
    for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
        c2p(nets[thread]);
        size_t column = (chunk[thread] - chunk[0]) * SSAIC::NSAIC_per_irred[irred];
        for (size_t data = chunk[thread] - chunk[0]; data < chunk[thread]; data++) {
            torch::Tensor r_tensor = nets[thread]->forward(GeomSet[data]->SAIgeom[irred]);
            for (size_t el = 0; el < r_tensor.size(0); el++) {
                optimizer[thread]->zero_grad();
                r_tensor[el].backward({}, true);
                size_t row = 0;
                for (auto & p : nets[thread]->parameters())
                if (p.requires_grad()) {
                    double * pg = p.grad().data_ptr<double>();
                    for (size_t i = 0; i < p.grad().numel(); i++) {
                        JT[row*NEq+column] = pg[i];
                        row++;
                    }
                }
                column++;
            }
        }
    }
}

void initialize(const std::shared_ptr<Net> & net_,
const size_t & irred_, const size_t & freeze_,
const std::vector<AbInitio::geom *> & GeomSet_) {
    OMP_NUM_THREADS = omp_get_max_threads();

    irred = irred_;

    nets.resize(OMP_NUM_THREADS);
    nets[0] = net_;
    for (size_t i = 1; i < OMP_NUM_THREADS; i++) {
        nets[i] = std::make_shared<Net>(SSAIC::NSAIC_per_irred[irred], nets[0]->fc.size());
        nets[i]->to(torch::kFloat64);
        nets[i]->copy(nets[0]);
        nets[i]->freeze(freeze_);
    }
    Nc = CL::TS::NParameters(nets[0]->parameters());
    c = new double[Nc];
    p2c(nets[0]);

    GeomSet = GeomSet_;
    size_t NData = OMP_NUM_THREADS * (GeomSet.size() / OMP_NUM_THREADS);
    std::cout << "For parallelism, the number of data in use = " << NData << '\n';
    NEq = SSAIC::NSAIC_per_irred[irred] * NData;
    chunk.resize(OMP_NUM_THREADS);
    chunk[0] = GeomSet.size() / OMP_NUM_THREADS;
    for (size_t i = 1; i < OMP_NUM_THREADS; i++) chunk[i] = chunk[i-1] + chunk[0];

    optimizer.resize(OMP_NUM_THREADS);
    for (size_t i = 0; i < OMP_NUM_THREADS; i++)
    optimizer[i] = new torch::optim::Adam(nets[i]->parameters(), 0.01);
}

void optimize(const std::string & opt, const size_t & epoch) {
    if (opt == "CG") {
        FL::NO::ConjugateGradient(loss, grad, loss_grad, c, Nc, "DY", false, true, epoch);
    } else {
        FL::NO::TrustRegion(residue, Jacobian, c, NEq, Nc, true, epoch);
    }
    c2p(nets[0]);
    delete [] c;
}

} // namespace FLopt
} // namespace DimRed