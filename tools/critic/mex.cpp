#include <torch/torch.h>

#include <FortranLibrary.hpp>

#include <libSGT.hpp>
#include "basic.hpp"
using namespace basic;

namespace mex {
    // Adiabatic energy, gradient, gap, gap gradient wrapper
    void e(double & e, const double * q, const int & N) {
        at::Tensor r = at::empty(cartdim, at::TensorOptions().dtype(torch::kFloat64));
        FL::GT::CartesianCoordinate(q, r.data_ptr<double>(), intdim, cartdim, init_geom, DefID);
        at::Tensor energy = compute_energy(r);
        e = energy[state].item<double>();
    }
    void g(double * g, const double * q, const int & N) {
        at::Tensor r = at::empty(cartdim, at::TensorOptions().dtype(torch::kFloat64));
        FL::GT::CartesianCoordinate(q, r.data_ptr<double>(), intdim, cartdim, init_geom, DefID);
        at::Tensor energy, dH;
        std::tie(energy, dH) = compute_energy_dH(r);
        double * qtemp = new double[intdim];
        FL::GT::Cartesian2Internal(r.data_ptr<double>(), dH[state][state].data_ptr<double>(),
            qtemp, g, cartdim, intdim, 1, DefID);
        delete [] qtemp;
    }
    int e_g(double & e, double * g, const double * q, const int & N) {
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
    void gap(double * gap, const double * q, const int & M, const int & N) {
        at::Tensor r = at::empty(cartdim, at::TensorOptions().dtype(torch::kFloat64));
        FL::GT::CartesianCoordinate(q, r.data_ptr<double>(), intdim, cartdim, init_geom, DefID);
        at::Tensor energy = compute_energy(r);
        gap[0] = (energy[state+1] - energy[state]).item<double>();
    }
    void gapgrad(double * gapgrad, const double * q, const int & M, const int & N) {
        at::Tensor r = at::empty(cartdim, at::TensorOptions().dtype(torch::kFloat64));
        FL::GT::CartesianCoordinate(q, r.data_ptr<double>(), intdim, cartdim, init_geom, DefID);
        at::Tensor energy, dH;
        std::tie(energy, dH) = compute_energy_dH(r);
        dH = dH.slice(0, state, state+2);
        dH = dH.slice(1, state, state+2);
        double * qtemp = new double[intdim];
        double * gtemp = new double[2*2*intdim];
        FL::GT::Cartesian2Internal(r.data_ptr<double>(), dH.data_ptr<double>(),
            qtemp, gtemp, cartdim, intdim, 2, DefID);
        for (size_t i = 0; i < intdim; i++) gapgrad[i] = gtemp[3 * intdim + i] - gtemp[i];
        delete [] qtemp;
        delete [] gtemp;
    }

    // Diabatic "energy", gradient, constraint (gap and off-diagonal), constraint gradient wrapper
    void ed(double & ed, const double * q, const int & N) {
        at::Tensor r = at::empty(cartdim, at::TensorOptions().dtype(torch::kFloat64));
        FL::GT::CartesianCoordinate(q, r.data_ptr<double>(), intdim, cartdim, init_geom, DefID);
        at::Tensor H = compute_Hd(r);
        ed = H[state][state].item<double>();
    }
    void gd(double * gd, const double * q, const int & N) {
        at::Tensor r = at::empty(cartdim, at::TensorOptions().dtype(torch::kFloat64));
        FL::GT::CartesianCoordinate(q, r.data_ptr<double>(), intdim, cartdim, init_geom, DefID);
        at::Tensor H, dH;
        std::tie(H, dH) = compute_Hd_dHd(r);
        double * qtemp = new double[intdim];
        FL::GT::Cartesian2Internal(r.data_ptr<double>(), dH[state][state].data_ptr<double>(),
            qtemp, gd, cartdim, intdim, 1, DefID);
        delete [] qtemp;
    }
    int ed_gd(double & ed, double * gd, const double * q, const int & N) {
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
    void c(double * c, const double * q, const int & M, const int & N) {
        at::Tensor r = at::empty(cartdim, at::TensorOptions().dtype(torch::kFloat64));
        FL::GT::CartesianCoordinate(q, r.data_ptr<double>(), intdim, cartdim, init_geom, DefID);
        at::Tensor H = compute_Hd(r);
        c[0] = (H[state+1][state+1] - H[state][state]).item<double>();
        c[1] = H[state][state+1].item<double>();
    }
    void cd(double * cd, const double * q, const int & M, const int & N) {
        at::Tensor r = at::empty(cartdim, at::TensorOptions().dtype(torch::kFloat64));
        FL::GT::CartesianCoordinate(q, r.data_ptr<double>(), intdim, cartdim, init_geom, DefID);
        at::Tensor H, dH;
        std::tie(H, dH) = compute_Hd_dHd(r);
        dH = dH.slice(0, state, state+2);
        dH = dH.slice(1, state, state+2);
        double * qtemp = new double[intdim];
        double * gtemp = new double[2*2*intdim];
        FL::GT::Cartesian2Internal(r.data_ptr<double>(), dH.data_ptr<double>(),
            qtemp, gtemp, cartdim, intdim, 2, DefID);
        for (size_t i = 0; i < intdim; i++) {
            cd[i] = gtemp[3 * intdim + i] - gtemp[i];
            cd[intdim + i] = gtemp[intdim + i]
        }
        delete [] qtemp;
        delete [] gtemp;
    }

    void search_mex(bool diabatic, std::string opt) {
        // FL::NO::TrustRegion(g, );
    }
} // namespace mex