#include <torch/torch.h>

#include <FortranLibrary.hpp>

#include <libSGT.hpp>
#include "basic.hpp"
using namespace basic;

namespace sad {
    // Adiabatic gradient wrapper
    void g(double * g, const double * q, const int & M, const int & N) {
        at::Tensor r = at::empty(cartdim, at::TensorOptions().dtype(torch::kFloat64));
        FL::GT::CartesianCoordinate(q, r.data_ptr<double>(), intdim, cartdim, init_geom, DefID);
        at::Tensor energy, dH;
        std::tie(energy, dH) = compute_energy_dH(r);
        double * qtemp = new double[intdim];
        FL::GT::Cartesian2Internal(r.data_ptr<double>(), dH[state][state].data_ptr<double>(),
            qtemp, g, cartdim, intdim, 1, DefID);
        delete [] qtemp;
    }

    void search_sad(bool diabatic, std::string opt) {
        // FL::NO::TrustRegion(g, );
    }
} // namespace sad