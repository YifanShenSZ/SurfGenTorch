/*
An evaluation library for SurfGenTorch

Input: Cartesian or internal coordinate
Output: Hd, dHd, ddHd, energy, dHa, ddHa in corresponding coordinate
*/

#include <torch/torch.h>

#include <CppLibrary/TorchSupport.hpp>

#include "SSAIC.hpp"
#include "DimRed.hpp"
#include "observable_net.hpp"
#include "Hd.hpp"

namespace libSGT {

void initialize_libSGT(const std::string & SSAIC_in, const std::string & DimRed_in, const std::string & input_layer_in, const std::string & Hd_in) {
    SSAIC::define_SSAIC(SSAIC_in);
    DimRed::define_DimRed(DimRed_in);
    ON::define_PNR(input_layer_in);
    Hd::define_Hd(Hd_in);
}

// Diabatic Hamiltonian from Cartesian coordinate
at::Tensor compute_Hd(const at::Tensor & r) {
    at::Tensor q = CL::TS::IC::compute_IC(r, SSAIC::DefID);
    std::vector<at::Tensor> SAIgeom = SSAIC::compute_SSAIC(q);
    std::vector<at::Tensor> Redgeom = DimRed::reduce(SAIgeom);
    std::vector<at::Tensor> input_layer = ON::input_layer(Redgeom);
    at::Tensor H = Hd::compute_Hd(input_layer);
    return H;
}

// Diabatic Hamiltonian and gradient in Cartesian coordinate
std::tuple<at::Tensor, at::Tensor> compute_Hd_dHd(const at::Tensor & r) {
    // Cartesian to internal
    at::Tensor q, JT;
    std::tie(q, JT) = CL::TS::IC::compute_IC_J(r, SSAIC::DefID);
    JT = JT.transpose(0, 1);
    q.set_requires_grad(true);
    // Input layer
    std::vector<at::Tensor> SAIgeom = SSAIC::compute_SSAIC(q);
    std::vector<at::Tensor> Redgeom = DimRed::reduce(SAIgeom);
    std::vector<at::Tensor> input_layer = ON::input_layer(Redgeom);
    // Diabatic Hamiltonian and gradient
    at::Tensor  H = Hd::compute_Hd(input_layer);
    at::Tensor dH = H.new_empty({Hd::NStates, Hd::NStates, SSAIC::cartdim});
    for (int i = 0; i < Hd::NStates; i++)
    for (int j = i; j < Hd::NStates; j++) {
        torch::autograd::variable_list g = torch::autograd::grad({H[i][j]}, {q}, {}, true);
        dH[i][j] = JT.mv(g[0]);
    }
    // Stop autograd
    H.detach_();
    return std::make_tuple(H, dH);
}

// Diabatic Hamiltonian from internal coordinate
at::Tensor compute_Hd_int(const at::Tensor & q) {
    std::vector<at::Tensor> SAIgeom = SSAIC::compute_SSAIC(q);
    std::vector<at::Tensor> Redgeom = DimRed::reduce(SAIgeom);
    std::vector<at::Tensor> input_layer = ON::input_layer(Redgeom);
    at::Tensor H = Hd::compute_Hd(input_layer);
    return H;
}

// Diabatic Hamiltonian and gradient in internal coordinate
std::tuple<at::Tensor, at::Tensor> compute_Hd_dHd_int(const at::Tensor & q) {
    assert(("q must require gradient in order to compute gradient", q.requires_grad()));
    // Input layer
    std::vector<at::Tensor> SAIgeom = SSAIC::compute_SSAIC(q);
    std::vector<at::Tensor> Redgeom = DimRed::reduce(SAIgeom);
    std::vector<at::Tensor> input_layer = ON::input_layer(Redgeom);
    // Diabatic Hamiltonian and gradient
    at::Tensor  H = Hd::compute_Hd(input_layer);
    at::Tensor dH = H.new_empty({Hd::NStates, Hd::NStates, SSAIC::intdim});
    for (int i = 0; i < Hd::NStates; i++)
    for (int j = i; j < Hd::NStates; j++) {
        torch::autograd::variable_list g = torch::autograd::grad({H[i][j]}, {q}, {}, true);
        dH[i][j].copy_(g[0]);
    }
    // Stop autograd
    H.detach_();
    return std::make_tuple(H, dH);
}

// Diabatic Hamiltonian and gradient and Hessian in internal coordinate
std::tuple<at::Tensor, at::Tensor, at::Tensor> compute_Hd_dHd_ddHd_int(const at::Tensor & q) {
    assert(("q must require gradient in order to compute gradient and Hessian", q.requires_grad()));
    // Input layer
    std::vector<at::Tensor> SAIgeom = SSAIC::compute_SSAIC(q);
    std::vector<at::Tensor> Redgeom = DimRed::reduce(SAIgeom);
    std::vector<at::Tensor> input_layer = ON::input_layer(Redgeom);
    // Diabatic Hamiltonian and gradient
    at::Tensor  H = Hd::compute_Hd(input_layer);
    at::Tensor dH = H.new_empty({Hd::NStates, Hd::NStates, SSAIC::intdim});
    for (int i = 0; i < Hd::NStates; i++)
    for (int j = i; j < Hd::NStates; j++) {
        torch::autograd::variable_list g = torch::autograd::grad({H[i][j]}, {q}, {}, true, true);
        dH[i][j] = g[0];
    }
    // Compute Hessian
    at::Tensor ddH = H.new_empty({Hd::NStates, Hd::NStates, SSAIC::intdim, SSAIC::intdim});
    for (size_t i = 0; i < Hd::NStates; i++)
    for (size_t j = i; j < Hd::NStates; j++) {
        for (size_t k = 0; k < SSAIC::intdim; k++) {
            torch::autograd::variable_list g = torch::autograd::grad({dH[i][j][k]}, {q}, {}, true);
            ddH[i][j][k].copy_(g[0]);
        }
        for (size_t k = 0    ; k < SSAIC::intdim; k++)
        for (size_t l = k + 1; l < SSAIC::intdim; l++) {
            ddH[i][j][k][l] = (ddH[i][j][k][l] + ddH[i][j][l][k]) / 2.0;
            ddH[i][j][l][k].copy_(ddH[i][j][k][l]);
        }
    }
    // Stop autograd
    H.detach_();
    dH.detach_();
    return std::make_tuple(H, dH, ddH);
}

// Adiabatic energy and gradient and Hessian in internal coordinate
std::tuple<at::Tensor, at::Tensor, at::Tensor> compute_energy_dHa_ddHa_int(const at::Tensor & q) {
    assert(("q must require gradient in order to compute gradient and Hessian", q.requires_grad()));
    // Input layer
    std::vector<at::Tensor> SAIgeom = SSAIC::compute_SSAIC(q);
    std::vector<at::Tensor> Redgeom = DimRed::reduce(SAIgeom);
    std::vector<at::Tensor> input_layer = ON::input_layer(Redgeom);
    // Diabatic Hamiltonian and gradient
    at::Tensor  H = Hd::compute_Hd(input_layer);
    at::Tensor dH = H.new_empty({Hd::NStates, Hd::NStates, SSAIC::intdim});
    for (int i = 0; i < Hd::NStates; i++)
    for (int j = i; j < Hd::NStates; j++) {
        torch::autograd::variable_list g = torch::autograd::grad({H[i][j]}, {q}, {}, true, true);
        dH[i][j] = g[0];
    }
    // Transform to adiabatic representation
    at::Tensor energy, state;
    std::tie(energy, state) = H.symeig(true);
    dH = CL::TS::LA::UT_A3_U(dH, state);
    // Compute Hessian
    at::Tensor ddH = H.new_empty({Hd::NStates, Hd::NStates, SSAIC::intdim, SSAIC::intdim});
    for (size_t i = 0; i < Hd::NStates; i++)
    for (size_t j = i; j < Hd::NStates; j++) {
        for (size_t k = 0; k < SSAIC::intdim; k++) {
            torch::autograd::variable_list g = torch::autograd::grad({dH[i][j][k]}, {q}, {}, true);
            ddH[i][j][k].copy_(g[0]);
        }
        for (size_t k = 0    ; k < SSAIC::intdim; k++)
        for (size_t l = k + 1; l < SSAIC::intdim; l++) {
            ddH[i][j][k][l] = (ddH[i][j][k][l] + ddH[i][j][l][k]) / 2.0;
            ddH[i][j][l][k].copy_(ddH[i][j][k][l]);
        }
    }
    // Stop autograd
    energy.detach_();
    dH.detach_();
    return std::make_tuple(energy, dH, ddH);
}

// Interoperability
void initialize_libSGT() {
    SSAIC::define_SSAIC("SSAIC.in");
    DimRed::define_DimRed("DimRed.in");
    ON::define_PNR("input_layer.in");
    Hd::define_Hd("Hd.in");
}

void compute_Hd(const double * r_, double * H_) {
    at::Tensor r = at::from_blob(const_cast<double *>(r_), SSAIC::cartdim, at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor H = compute_Hd(r);
    std::memcpy(H_, H.data_ptr<double>(), H.numel() * sizeof(double));
}

void compute_Hd_dHd(const double * r_, double * H_, double * dH_) {
    at::Tensor r = at::from_blob(const_cast<double *>(r_), SSAIC::cartdim, at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor H, dH;
    std::tie(H, dH) = compute_Hd_dHd(r);
    std::memcpy( H_,  H.data_ptr<double>(),  H.numel() * sizeof(double));
    std::memcpy(dH_, dH.data_ptr<double>(), dH.numel() * sizeof(double));
}

void compute_Hd_int(const double * q_, double * H_) {
    at::Tensor q = at::from_blob(const_cast<double *>(q_), SSAIC::intdim, at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor H = compute_Hd_int(q);
    std::memcpy(H_, H.data_ptr<double>(), H.numel() * sizeof(double));
}

void compute_Hd_dHd_int(const double * q_, double * H_, double * dH_) {
    at::Tensor q = at::from_blob(const_cast<double *>(q_), SSAIC::intdim, at::TensorOptions().dtype(torch::kFloat64));
    q.set_requires_grad(true);
    at::Tensor H, dH;
    std::tie(H, dH) = compute_Hd_dHd_int(q);
    std::memcpy( H_,  H.data_ptr<double>(),  H.numel() * sizeof(double));
    std::memcpy(dH_, dH.data_ptr<double>(), dH.numel() * sizeof(double));
}

void compute_Hd_dHd_ddHd_int(const double * q_, double * H_, double * dH_, double * ddH_) {
    at::Tensor q = at::from_blob(const_cast<double *>(q_), SSAIC::intdim, at::TensorOptions().dtype(torch::kFloat64));
    q.set_requires_grad(true);
    at::Tensor H, dH, ddH;
    std::tie(H, dH, ddH) = compute_Hd_dHd_ddHd_int(q);
    std::memcpy(  H_,   H.data_ptr<double>(),   H.numel() * sizeof(double));
    std::memcpy( dH_,  dH.data_ptr<double>(),  dH.numel() * sizeof(double));
    std::memcpy(ddH_, ddH.data_ptr<double>(), ddH.numel() * sizeof(double));
}

void compute_energy_dHa_ddHa_int(const double * q_, double * energy_, double * dH_, double * ddH_) {
    at::Tensor q = at::from_blob(const_cast<double *>(q_), SSAIC::intdim, at::TensorOptions().dtype(torch::kFloat64));
    q.set_requires_grad(true);
    at::Tensor energy, dH, ddH;
    std::tie(energy, dH, ddH) = compute_energy_dHa_ddHa_int(q);
    std::memcpy(energy_, energy.data_ptr<double>(), energy.numel() * sizeof(double));
    std::memcpy(    dH_,     dH.data_ptr<double>(),     dH.numel() * sizeof(double));
    std::memcpy(   ddH_,    ddH.data_ptr<double>(),    ddH.numel() * sizeof(double));
}

} // namespace libSGT