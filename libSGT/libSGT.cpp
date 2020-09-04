/*
An evaluation library for SurfGenTorch

Input: Cartesian coordinate
Output: Hd, dHd

The higher order gradients cannot be calculated,
since SurfGenTorch adopts scaled and symmetry adapted internal coordinate
whose Jacobian to Cartesian coordinate is calculated by Fortran-Library rather than pytorch,
which means this Jacobian cannot enjoy pytorch automatic differentiation
*/

#include <torch/torch.h>

#include <CppLibrary/TorchSupport.hpp>
#include <FortranLibrary.hpp>

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
    at::Tensor q = r.new_empty(SSAIC::intdim);
    FL::GT::InternalCoordinate(
        r.data_ptr<double>(), q.data_ptr<double>(),
        SSAIC::cartdim, SSAIC::intdim, SSAIC::DefID);
    // input_layer
    std::vector<at::Tensor> SAIgeom = SSAIC::compute_SSAIC(q);
    std::vector<at::Tensor> Redgeom = DimRed::reduce(SAIgeom);
    std::vector<at::Tensor> input_layer = Hd::input::input_layer(Redgeom);
    return input_layer;
}

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>>
compute_input_layer_and_JT(const at::Tensor & r) {
    // cart2int
    at::Tensor q = r.new_empty(SSAIC::intdim);
    at::Tensor BT = r.new_empty({SSAIC::cartdim, SSAIC::intdim});
    FL::GT::WilsonBMatrixAndInternalCoordinate(
        r.data_ptr<double>(),
        BT.data_ptr<double>(), q.data_ptr<double>(),
        SSAIC::cartdim, SSAIC::intdim, SSAIC::DefID);
    // input_layer
    q.set_requires_grad(true);
    std::vector<at::Tensor> SAIgeom = SSAIC::compute_SSAIC(q);
    std::vector<at::Tensor> Redgeom = DimRed::reduce(SAIgeom);
    std::vector<at::Tensor> InpLay = Hd::input::input_layer(Redgeom);
    std::vector<at::Tensor> input_layer(InpLay.size());
    for (size_t irred = 0; irred < InpLay.size(); irred++) input_layer[irred] = InpLay[irred].detach();
    // J^T
    std::vector<at::Tensor> JT(InpLay.size());
    for (size_t irred = 0; irred < InpLay.size(); irred++) {
        at::Tensor dinput_layer_divide_dintgeom = at::empty(
            {InpLay[irred].size(0), q.size(0)},
            at::TensorOptions().dtype(torch::kFloat64));
        for (size_t i = 0; i < InpLay[irred].size(0); i++) {
            if (q.grad().defined()) {q.grad().detach_(); q.grad().zero_();}
            InpLay[irred][i].backward({}, true);
            dinput_layer_divide_dintgeom[i].copy_(q.grad());
        }
        JT[irred] = BT.mm(dinput_layer_divide_dintgeom.transpose(0, 1));
    }
    return std::make_tuple(input_layer, JT);
}

at::Tensor compute_Hd(const at::Tensor & r) {
    std::vector<at::Tensor> input_layer = compute_input_layer(r);
    at::Tensor H = Hd::compute_Hd(input_layer);
    return H;
}

std::tuple<at::Tensor, at::Tensor> compute_Ha_dHa(const at::Tensor & r) {
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
        auto & g = input_layer[irred].grad();
        if (g.defined()) {g.detach_(); g.zero_();}
        H[i][j].backward({}, true);
        dH[i][j] = JT[irred].mv(g);
    }
    // Transform to adiabatic representation
    at::Tensor energy, state;
    std::tie(energy, state) = H.symeig(true, true);
    dH = CL::TS::LA::UT_A3_U(dH, state);
    return std::make_tuple(energy, dH);
}

// Interoperability
void compute_Hd(double * r_, double * H_) {
    at::Tensor r = at::from_blob(r_, SSAIC::cartdim, at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor H = compute_Hd(r);
    std::memcpy(H_, H.data_ptr<double>(), H.numel() * sizeof(double));
}
void compute_Ha_dHa(double * r_, double * H_, double * dH_) {
    at::Tensor r = at::from_blob(r_, SSAIC::cartdim, at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor H, dH;
    std::tie(H, dH) = compute_Ha_dHa(r);
    std::memcpy(H_, H.data_ptr<double>(), H.numel() * sizeof(double));
    std::memcpy(dH_, dH.data_ptr<double>(), dH.numel() * sizeof(double));
}