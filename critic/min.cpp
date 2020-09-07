#include <torch/torch.h>

#include <FortranLibrary.hpp>

#include <libSGT.hpp>

namespace basic {
    extern int intdim, DefID;

    extern int cartdim;
    extern double * init_geom;

    extern size_t state;
} // namespace basic

namespace min {
    using namespace basic;

    // Adiabatic energy and gradient wrapper for Fortran-Library
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
        return 0;
    }

    // Diabatic "energy" and gradient wrapper for Fortran-Library
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
        return 0;
    }
} // namespace min