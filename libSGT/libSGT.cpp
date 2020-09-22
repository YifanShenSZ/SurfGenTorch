/*
An evaluation library for SurfGenTorch

Input: Cartesian coordinate
Output: Hd, dHd, energy, dHa, Hessian
*/

#include <torch/torch.h>

#include <CppLibrary/TorchSupport.hpp>

#include "SSAIC.hpp"
#include "DimRed.hpp"
#include "Hd.hpp"

void initialize_libSGT(const std::string & SSAIC_in, const std::string & DimRed_in, const std::string & Hd_in) {
    SSAIC::define_SSAIC(SSAIC_in);
    DimRed::define_DimRed(DimRed_in);
    Hd::define_Hd(Hd_in);
}

std::vector<at::Tensor> compute_input_layer(const at::Tensor & r) {
    // cart2int
    at::Tensor q = CL::TS::IC::compute_IC(r, SSAIC::DefID);
    // input_layer
    std::vector<at::Tensor> SAIgeom = SSAIC::compute_SSAIC(q);
    std::vector<at::Tensor> Redgeom = DimRed::reduce(SAIgeom);
    std::vector<at::Tensor> input_layer = Hd::input::input_layer(Redgeom);
    return input_layer;
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>>
compute_input_layer_and_JT(const at::Tensor & r) {
    // cart2int
    at::Tensor q, J;
    std::tie(q, J) = CL::TS::IC::compute_IC_J(r, SSAIC::DefID);
    q.set_requires_grad(true);
    // input_layer and J^T
    std::vector<at::Tensor> SAIgeom = SSAIC::compute_SSAIC(q);
    std::vector<at::Tensor> Redgeom = DimRed::reduce(SAIgeom);
    std::vector<at::Tensor> input_layer = Hd::input::input_layer(Redgeom);
    std::vector<at::Tensor> JT(input_layer.size());
    for (size_t irred = 0; irred < input_layer.size(); irred++) {
        at::Tensor J_InpLay_q = at::empty(
            {input_layer[irred].size(0), q.size(0)},
            at::TensorOptions().dtype(torch::kFloat64));
        for (size_t i = 0; i < input_layer[irred].size(0); i++) {
            torch::autograd::variable_list g = torch::autograd::grad({input_layer[irred][i]}, {q}, {}, true);
            J_InpLay_q[i].copy_(g[0]);
        }
        JT[irred] = (J_InpLay_q.mm(J)).transpose(0, 1);
    }
    for (at::Tensor & irred : input_layer) irred.detach_();
    return std::make_tuple(input_layer, JT);
}

at::Tensor compute_Hd(const at::Tensor & r) {
    std::vector<at::Tensor> input_layer = compute_input_layer(r);
    at::Tensor H = Hd::compute_Hd(input_layer);
    return H;
}

std::tuple<at::Tensor, at::Tensor> compute_Hd_dHd(const at::Tensor & r) {
    // Input layer and J^T
    std::vector<at::Tensor> input_layer, JT;
    std::tie(input_layer, JT) = compute_input_layer_and_JT(r);
    for (auto & irred : input_layer) irred.set_requires_grad(true);
    // Compute diabatic quantity
    at::Tensor  H = Hd::compute_Hd(input_layer);
    at::Tensor dH = H.new_empty({Hd::NStates, Hd::NStates, SSAIC::cartdim});
    for (int i = 0; i < Hd::NStates; i++)
    for (int j = i; j < Hd::NStates; j++) {
        auto & irred = Hd::symmetry[i][j];
        torch::autograd::variable_list g = torch::autograd::grad({H[i][j]}, {input_layer[irred]}, {}, true);
        dH[i][j] = JT[irred].mv(g[0]);
    }
    // Stop autograd
    H.detach_();
    return std::make_tuple(H, dH);
}

at::Tensor compute_energy(const at::Tensor & r) {
    std::vector<at::Tensor> input_layer = compute_input_layer(r);
    at::Tensor H = Hd::compute_Hd(input_layer);
    at::Tensor energy, state;
    std::tie(energy, state) = H.symeig();
    return energy;
}

std::tuple<at::Tensor, at::Tensor> compute_energy_dH(const at::Tensor & r) {
    // Input layer and J^T
    std::vector<at::Tensor> input_layer, JT;
    std::tie(input_layer, JT) = compute_input_layer_and_JT(r);
    for (auto & irred : input_layer) irred.set_requires_grad(true);
    // Compute diabatic quantity
    at::Tensor  H = Hd::compute_Hd(input_layer);
    at::Tensor dH = H.new_empty({Hd::NStates, Hd::NStates, SSAIC::cartdim});
    for (int i = 0; i < Hd::NStates; i++)
    for (int j = i; j < Hd::NStates; j++) {
        auto & irred = Hd::symmetry[i][j];
        torch::autograd::variable_list g = torch::autograd::grad({H[i][j]}, {input_layer[irred]}, {}, true);
        dH[i][j] = JT[irred].mv(g[0]);
    }
    // Transform to adiabatic representation
    at::Tensor energy, state;
    std::tie(energy, state) = H.symeig(true);
    CL::TS::LA::UT_A3_U_InPlace(dH, state);
    // Stop autograd
    energy.detach_();
    dH.detach_();
    return std::make_tuple(energy, dH);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> compute_energy_grad_hess(const at::Tensor & r, const size_t & state_of_interest) {
    assert((r.requires_grad(), "The input Cartesian coordinate tensor must require gradient"));
    // Input layer and J^T
    std::vector<at::Tensor> input_layer, JT;
    std::tie(input_layer, JT) = compute_input_layer_and_JT(r);
    for (auto & irred : input_layer) irred.set_requires_grad(true);
    // Compute diabatic quantity
    at::Tensor  H = Hd::compute_Hd(input_layer);
    at::Tensor dH = H.new_empty({Hd::NStates, Hd::NStates, SSAIC::cartdim});
    for (int i = 0; i < Hd::NStates; i++)
    for (int j = i; j < Hd::NStates; j++) {
        auto & irred = Hd::symmetry[i][j];
        torch::autograd::variable_list g = torch::autograd::grad({H[i][j]}, {input_layer[irred]}, {}, true, true);
        dH[i][j] = JT[irred].mv(g[0]);
    }
    // Transform to adiabatic representation
    at::Tensor energy, state;
    std::tie(energy, state) = H.symeig(true);
    CL::TS::LA::UT_A3_U_InPlace(dH, state);
    at::Tensor grad = dH[state_of_interest][state_of_interest];
    // Compute Hessian
    at::Tensor hess = H.new_empty({SSAIC::cartdim, SSAIC::cartdim});
    for (size_t i = 0; i < SSAIC::cartdim; i++) {
        // This may not work for torsion due to the sanity check
        // Maybe have to backward to q & J then manually convert to r
        torch::autograd::variable_list g = torch::autograd::grad({grad[i]}, {r}, {}, true);
        hess[i].copy_(g[0]);
    }
    for (size_t i = 0; i < SSAIC::cartdim; i++)
    for (size_t j = i+1; j < SSAIC::cartdim; j++) {
        hess[i][j] = (hess[i][j] + hess[j][i]) / 2.0;
        hess[j][i] = hess[i][j];
    }
    // Stop autograd
    energy[state_of_interest].detach_();
    grad.detach_();
    return std::make_tuple(energy[state_of_interest], grad, hess);
}

// Interoperability
void compute_Hd(double * r_, double * H_) {
    at::Tensor r = at::from_blob(r_, SSAIC::cartdim, at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor H = compute_Hd(r);
    std::memcpy(H_, H.data_ptr<double>(), H.numel() * sizeof(double));
}
void compute_energy_dH(double * r_, double * energy_, double * dH_) {
    at::Tensor r = at::from_blob(r_, SSAIC::cartdim, at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor energy, dH;
    std::tie(energy, dH) = compute_energy_dH(r);
    std::memcpy(energy_, energy.data_ptr<double>(), energy.numel() * sizeof(double));
    std::memcpy(dH_, dH.data_ptr<double>(), dH.numel() * sizeof(double));
}