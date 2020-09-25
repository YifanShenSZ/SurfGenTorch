/*
An evaluation library for SurfGenTorch

Input: Cartesian coordinate
Output: Hd, dHd, energy, dHa, Hessian
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

std::cout << "SAIgeom\n";
std::cout <<  SAIgeom[0] << '\n'
          <<  SAIgeom[1] << '\n'
          <<  SAIgeom[2] << '\n'
          <<  SAIgeom[3] << '\n';

    std::vector<at::Tensor> Redgeom = DimRed::reduce(SAIgeom);

std::cout << "Redgeom\n";
std::cout <<  Redgeom[0] << '\n'
          <<  Redgeom[1] << '\n'
          <<  Redgeom[2] << '\n'
          <<  Redgeom[3] << '\n';

    std::vector<at::Tensor> input_layer = ON::input_layer(Redgeom);

std::cout << "input layer\n";
std::cout <<  input_layer[0] << '\n'
          <<  input_layer[1] << '\n'
          <<  input_layer[2] << '\n'
          <<  input_layer[3] << '\n';

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
    assert(("q must require gradient in order to compute dHd / dq", q.requires_grad()));
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

// Adiabatic energy and gradient and Hessian in internal coordinate
// Only calculate the Hessian for the state of interest
std::tuple<at::Tensor, at::Tensor, at::Tensor> compute_energy_dHa_hess_int(const at::Tensor & q, const size_t & state_of_interest) {
    assert(("q must require gradient in order to compute dHd / dq", q.requires_grad()));
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
    at::Tensor hess = H.new_empty({SSAIC::intdim, SSAIC::intdim});
    for (size_t i = 0; i < SSAIC::intdim; i++) {
        torch::autograd::variable_list g = torch::autograd::grad({dH[state_of_interest][state_of_interest][i]}, {q}, {}, true);
        hess[i].copy_(g[0]);
    }
    for (size_t i = 0; i < SSAIC::cartdim; i++)
    for (size_t j = i+1; j < SSAIC::cartdim; j++) {
        hess[i][j] = (hess[i][j] + hess[j][i]) / 2.0;
        hess[j][i].copy_(hess[i][j]);
    }
    // Stop autograd
    energy.detach_();
    dH.detach_();
    return std::make_tuple(energy, dH, hess);
}

// Interoperability
void initialize_libSGT() {
    SSAIC::define_SSAIC("SSAIC.in");
    DimRed::define_DimRed("DimRed.in");
    ON::define_PNR("input_layer.in");
    Hd::define_Hd("Hd.in");
}

void compute_Hd(double * r_, double * H_) {
    at::Tensor r = at::from_blob(r_, SSAIC::cartdim, at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor H = compute_Hd(r);
    std::memcpy(H_, H.data_ptr<double>(), H.numel() * sizeof(double));
}

void compute_Hd_dHd(double * r_, double * H_, double * dH_) {
    at::Tensor r = at::from_blob(r_, SSAIC::cartdim, at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor H, dH;
    std::tie(H, dH) = compute_Hd_dHd(r);
    std::memcpy( H_,  H.data_ptr<double>(),  H.numel() * sizeof(double));
    std::memcpy(dH_, dH.data_ptr<double>(), dH.numel() * sizeof(double));
}

void compute_Hd_int(double * q_, double * H_) {
    at::Tensor q = at::from_blob(q_, SSAIC::intdim, at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor H = compute_Hd_int(q);
    std::memcpy(H_, H.data_ptr<double>(), H.numel() * sizeof(double));
}

void compute_Hd_dHd_int(double * q_, double * H_, double * dH_) {
    at::Tensor q = at::from_blob(q_, SSAIC::intdim, at::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
    at::Tensor H, dH;
    std::tie(H, dH) = compute_Hd_dHd(q);
    std::memcpy( H_,  H.data_ptr<double>(),  H.numel() * sizeof(double));
    std::memcpy(dH_, dH.data_ptr<double>(), dH.numel() * sizeof(double));
}

void compute_energy_dHa_hess_int(double * q_, double * energy_, double * dH_, double * hess_, const int & state_of_interest) {
    at::Tensor q = at::from_blob(q_, SSAIC::intdim, at::TensorOptions().dtype(torch::kFloat64).requires_grad(true));
    at::Tensor energy, dH, hess;
    std::tie(energy, dH, hess) = compute_energy_dHa_hess_int(q, state_of_interest);
    std::memcpy(energy_, energy.data_ptr<double>(), energy.numel() * sizeof(double));
    std::memcpy(    dH_,     dH.data_ptr<double>(),     dH.numel() * sizeof(double));
    std::memcpy(  hess_,   hess.data_ptr<double>(),   hess.numel() * sizeof(double));
}

} // namespace libSGT