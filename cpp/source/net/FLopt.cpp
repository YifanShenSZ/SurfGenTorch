/*
To make use of Fortran-Library nonlinear optimizers:
    1. Map the network parameters to vector c
    2. Compute residue and Jacobian
*/

#include <omp.h>

#include <FortranLibrary.hpp>
#include "../../Cpp-Library_v1.0.0/TorchSupport.hpp"

#include "../../include/SSAIC.hpp"
#include "../../include/pretrain.hpp"

namespace DimRed { namespace FLopt {

size_t OMP_NUM_THREADS;

// The irreducible to pretrain
size_t irred;

// The dimensionality reduction network
std::vector<std::shared_ptr<Net>> net;
// Number of trainable parameters (length of parameter vector c)
int Nc;
// parameter vector c
double * c;

// data set
std::vector<AbInitio::geom *> GeomSet;
// Number of data
size_t NData;
// Number of least square equations
int NEq;
// divide data set for parallelism
std::vector<size_t> chunk;

// work variable
// a fake optimizer to zero_grad
std::vector<torch::optim::Adam *> optimizer;
// residue, Jacobian (transpose), Hessian
double * r, * JT, * H;

// Push network parameters to c
void p2c(const std::shared_ptr<Net> & net) {
    size_t count = 0;
    for (auto & p : net->parameters()) {
        for (size_t i = 0; i < p.numel(); i++) {
            c[count] = p.data_ptr<double>()[i];
            count++;
        }
    }
}
// Push c to network parameters
void c2p(const std::shared_ptr<Net> & net) {
    torch::NoGradGuard no_grad;
    size_t count = 0;
    for (auto & p : net->parameters()) {
        for (size_t i = 0; i < p.numel(); i++) {
            p.data_ptr<double>()[i] = c[count];
            count++;
        }
    }
}

void loss(double & l, const double * c, const int & Nc) {
    torch::NoGradGuard no_grad;
    std::vector<at::Tensor> loss(OMP_NUM_THREADS);
    #pragma omp parallel for
    for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
        c2p(net[thread]);
        loss[thread] = torch::zeros(1, at::TensorOptions().dtype(torch::kFloat64));
        for (size_t data = chunk[thread] - chunk[0]; data < chunk[thread]; data++) {
            loss[thread] += torch::mse_loss(
                net[thread]->forward(GeomSet[data]->SAIgeom[irred]), GeomSet[data]->SAIgeom[irred],
                at::Reduction::Sum);
        }
    }
    l = 0.0;
    for (at::Tensor & piece : loss) l += piece.item<double>();
    l /= 2.0;
}

void residue(double * r, const double * c, const int & NEq, const int & Nc) {
    torch::NoGradGuard no_grad;
    #pragma omp parallel for
    for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
        c2p(net[thread]);
        size_t count = (chunk[thread] - chunk[0]) * SSAIC::NSAIC_per_irred[irred];
        for (size_t data = chunk[thread] - chunk[0]; data < chunk[thread]; data++) {
            torch::Tensor r_tensor = net[thread]->forward(GeomSet[data]->SAIgeom[irred]) - GeomSet[data]->SAIgeom[irred];
            for (size_t i = 0; i < r_tensor.numel(); i++) {
                r[count] = r_tensor.data_ptr<double>()[i];
                count++;
            }
        }
    }
}

void Jacobian(double * JT, const double * c, const int & NEq, const int & Nc) {
    #pragma omp parallel for
    for (int thread = 0; thread < OMP_NUM_THREADS; thread++) {
        c2p(net[thread]);
        size_t column = (chunk[thread] - chunk[0]) * SSAIC::NSAIC_per_irred[irred];
        for (size_t data = chunk[thread] - chunk[0]; data < chunk[thread]; data++) {
            torch::Tensor r_tensor = net[thread]->forward(GeomSet[data]->SAIgeom[irred]);
            for (size_t j = 0; j < r_tensor.size(0); j++) {
                optimizer[thread]->zero_grad();
                r_tensor[j].backward({}, true);
                size_t row = 0;
                for (auto & p : net[thread]->parameters())
                for (size_t i = 0; i < p.numel(); i++) {
                    JT[row*NEq+column] = p.grad().data_ptr<double>()[i];
                    row++;
                }
                column++;
            }
        }
    }
}

void initialize(const size_t & irred_, const std::shared_ptr<Net> & net_, const std::vector<AbInitio::geom *> & GeomSet_) {
    OMP_NUM_THREADS = omp_get_max_threads();

    irred = irred_;

    net.resize(OMP_NUM_THREADS);
    net[0] = net_;
    for (size_t i = 1; i < OMP_NUM_THREADS; i++) {
        net[i] = std::make_shared<Net>(SSAIC::NSAIC_per_irred[irred], net[0]->fc.size());
        net[i]->to(torch::kFloat64);
    }
    Nc = CL::TS::NParameters(net[0]->parameters());
    c = new double[Nc];
    p2c(net[0]);

    GeomSet = GeomSet_;
    NData = OMP_NUM_THREADS * (GeomSet.size() / OMP_NUM_THREADS);
    std::cout << "For parallelism, the number of geometries in use is " << NData << '\n';
    NEq = SSAIC::NSAIC_per_irred[irred] * NData;
    chunk.resize(OMP_NUM_THREADS);
    chunk[0] = GeomSet.size() / OMP_NUM_THREADS;
    for (size_t i = 1; i < OMP_NUM_THREADS; i++) chunk[i] = chunk[i-1] + chunk[0];
    
    optimizer.resize(OMP_NUM_THREADS);
    for (size_t i = 0; i < OMP_NUM_THREADS; i++)
    optimizer[i] = new torch::optim::Adam(net[i]->parameters(), 0.01);
}

void optimize(const std::string & opt) {
    if (opt == "CG") {
        std::cout << "Not implemented\n";
    } else {
        r  = new double[NEq     ];
        JT = new double[NEq * Nc];
        FL::NO::TrustRegion(residue, Jacobian, c, NEq, Nc);
        delete [] r;
        delete [] JT;
    }
    c2p(net[0]);
    delete [] c;
}

} // namespace FLopt
} // namespace DimRed