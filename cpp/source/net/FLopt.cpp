/*
To make use of Fortran-Library nonlinear optimizers:
    1. Map the network parameters to vector c
    2. Compute residue and Jacobian

The parameters-c mapping is implemented by letting them share memory
*/

#include <FortranLibrary.hpp>
#include "../../Cpp-Library_v1.0.0/torch.hpp"

#include "../../include/SSAIC.hpp"
#include "../../include/pretrain.hpp"

namespace DimRed { namespace FLopt {

// The irreducible to pretrain
size_t irred;

// The dimensionality reduction network
std::shared_ptr<Net> net;
// Number of trainable parameters (length of parameter vector c)
int Nc;
// parameter c
double * c;

// data set
std::vector<AbInitio::geom *> GeomSet;
// Number of data
int NData;

// work variable: residue and Jacobian (transpose)
double * r, * JT;
// work variable: a fake optimizer to zero_grad
torch::optim::Adam * optimizer;

void initialize(const size_t & irred_, const std::shared_ptr<Net> & net_, const std::vector<AbInitio::geom *> & GeomSet_) {
    irred = irred_;
    
    net = net_;
    Nc = CL::torch::NParameters(net->parameters());
    c = new double[Nc];

    GeomSet = GeomSet_;
    NData = SSAIC::NSAIC_per_irred[irred] * GeomSet.size();    

    r  = new double[NData     ];
    JT = new double[NData * Nc];
    optimizer = new torch::optim::Adam(net->parameters(), 0.01);
}

void optimize(const std::string & opt) {
    p2c();
    if (opt == "CG") {
        std::cout << "Not implemented\n";
    } else {
        FL::NO::TrustRegion(residue, Jacobian, c, NData, Nc);
    }
    c2p();
}

void finish() {
    delete [] c;
    delete [] r;
    delete [] JT;
}

void p2c() {
    size_t count = 0;
    for (auto & p : net->parameters()) {
        for (size_t i = 0; i < p.numel(); i++) {
            c[count] = p.data_ptr<double>()[i];
            count++;
        }
    }
}

void c2p() {
    torch::NoGradGuard no_grad;
    size_t count = 0;
    for (auto & p : net->parameters()) {
        for (size_t i = 0; i < p.numel(); i++) {
            p.data_ptr<double>()[i] = c[count];
            count++;
        }
    }
}

void residue(double * r, const double * c, const int & NData, const int & Nc) {
    c2p();
    torch::NoGradGuard no_grad;
    size_t count = 0;
    for (auto & data : GeomSet) {
        torch::Tensor r_tensor = net->forward(data->SAIgeom[irred]) - data->SAIgeom[irred];
        for (size_t i = 0; i < r_tensor.numel(); i++) {
            r[count] = r_tensor.data_ptr<double>()[i];
            count++;
        }
    }
}

void Jacobian(double * JT, const double * c, const int & NData, const int & Nc) {
    c2p();
    size_t row, column;
    column = 0;
    for (auto & data : GeomSet) {
        torch::Tensor r_tensor = net->forward(data->SAIgeom[irred]);
        for (size_t j = 0; j < r_tensor.size(0) - 1; j++) {
            optimizer->zero_grad();
            r_tensor[j].backward({}, true);
            row = 0;
            for (auto & p : net->parameters())
            for (size_t i = 0; i < p.numel(); i++) {
                JT[row*NData+column] = p.grad().data_ptr<double>()[i];
                row++;
            }
            column++;
        }
        optimizer->zero_grad();
        r_tensor[r_tensor.size(0) - 1].backward();
        row = 0;
        for (auto & p : net->parameters())
        for (size_t i = 0; i < p.numel(); i++) {
            JT[row*NData+column] = p.grad().data_ptr<double>()[i];
            row++;
        }
        column++;
    }
}

} // namespace FLopt
} // namespace DimRed