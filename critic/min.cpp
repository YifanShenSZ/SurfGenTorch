#include <torch/torch.h>

#include <FortranLibrary.hpp>

#include <libSGT.hpp>
#include "basic.hpp"
using namespace basic;

namespace min {
    // Adiabatic energy and gradient wrapper
    void e(double & e, const double * q, const int & dim) {
        at::Tensor r = at::empty(cartdim, at::TensorOptions().dtype(torch::kFloat64));
        FL::GT::CartesianCoordinate(q, r.data_ptr<double>(), intdim, cartdim, init_geom, DefID);
        at::Tensor energy = compute_energy(r);
        e = energy[state].item<double>();
    }
    void g(double * g, const double * q, const int & dim) {
        at::Tensor r = at::empty(cartdim, at::TensorOptions().dtype(torch::kFloat64));
        FL::GT::CartesianCoordinate(q, r.data_ptr<double>(), intdim, cartdim, init_geom, DefID);
        at::Tensor energy, dH;
        std::tie(energy, dH) = compute_energy_dH(r);
        double * qtemp = new double[intdim];
        FL::GT::Cartesian2Internal(r.data_ptr<double>(), dH[state][state].data_ptr<double>(),
            qtemp, g, cartdim, intdim, 1, DefID);
        delete [] qtemp;
    }
    int e_g(double & e, double * g, const double * q, const int & dim) {
        at::Tensor r = at::empty(cartdim, at::TensorOptions().dtype(torch::kFloat64));
        FL::GT::CartesianCoordinate(q, r.data_ptr<double>(), intdim, cartdim, init_geom, DefID);
        at::Tensor energy, dH;
        std::tie(energy, dH) = compute_energy_dH(r);
        e = energy[state].item<double>();
        double * qtemp = new double[intdim];
        FL::GT::Cartesian2Internal(r.data_ptr<double>(), dH[state][state].data_ptr<double>(),
            qtemp, g, cartdim, intdim, 1, DefID);
        delete [] qtemp;
        return 0;
    }

    // Diabatic "energy" and gradient wrapper
    void ed(double & ed, const double * q, const int & dim) {
        at::Tensor r = at::empty(cartdim, at::TensorOptions().dtype(torch::kFloat64));
        FL::GT::CartesianCoordinate(q, r.data_ptr<double>(), intdim, cartdim, init_geom, DefID);
        at::Tensor H = compute_Hd(r);
        ed = H[state][state].item<double>();
    }
    void gd(double * gd, const double * q, const int & dim) {
        at::Tensor r = at::empty(cartdim, at::TensorOptions().dtype(torch::kFloat64));
        FL::GT::CartesianCoordinate(q, r.data_ptr<double>(), intdim, cartdim, init_geom, DefID);
        at::Tensor H, dH;
        std::tie(H, dH) = compute_Hd_dHd(r);
        double * qtemp = new double[intdim];
        FL::GT::Cartesian2Internal(r.data_ptr<double>(), dH[state][state].data_ptr<double>(),
            qtemp, gd, cartdim, intdim, 1, DefID);
        delete [] qtemp;
    }
    int ed_gd(double & ed, double * gd, const double * q, const int & dim) {
        at::Tensor r = at::empty(cartdim, at::TensorOptions().dtype(torch::kFloat64));
        FL::GT::CartesianCoordinate(q, r.data_ptr<double>(), intdim, cartdim, init_geom, DefID);
        at::Tensor H, dH;
        std::tie(H, dH) = compute_Hd_dHd(r);
        ed = H[state][state].item<double>();
        double * qtemp = new double[intdim];
        FL::GT::Cartesian2Internal(r.data_ptr<double>(), dH[state][state].data_ptr<double>(),
            qtemp, gd, cartdim, intdim, 1, DefID);
        delete [] qtemp;
        return 0;
    }

    void search_min(bool diabatic, std::string opt) {
        if (opt == "CG") {
            FL::NO::ConjugateGradient(e, g, e_g, q, intdim);
        }
        else {
            // FL::NO::BFGS()
        }
    }
} // namespace min