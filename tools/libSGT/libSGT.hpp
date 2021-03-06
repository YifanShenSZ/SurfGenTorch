/*
An evaluation library for SurfGenTorch

Input: Cartesian or internal coordinate
Output: Hd, dHd, ddHd, energy, dHa, ddHa in corresponding coordinate
*/

#ifndef libSGT_hpp
#define libSGT_hpp

#include <torch/torch.h>

namespace libSGT {

void initialize_libSGT(const std::string & SSAIC_in, const std::string & DimRed_in, const std::string & input_layer_in, const std::string & Hd_in);

// Diabatic Hamiltonian from Cartesian coordinate
at::Tensor compute_Hd(const at::Tensor & r);

// Diabatic Hamiltonian and gradient in Cartesian coordinate
std::tuple<at::Tensor, at::Tensor> compute_Hd_dHd(const at::Tensor & r);

// Diabatic Hamiltonian from internal coordinate
at::Tensor compute_Hd_int(const at::Tensor & q);

// Adiabatic energy and gradient and Hessian in internal coordinate
// Only calculate the Hessian for the state of interest
std::tuple<at::Tensor, at::Tensor> compute_Hd_dHd_int(const at::Tensor & q);

// Diabatic Hamiltonian and gradient and Hessian in internal coordinate
std::tuple<at::Tensor, at::Tensor, at::Tensor> compute_Hd_dHd_ddHd_int(const at::Tensor & q);

// Adiabatic energy and gradient and Hessian in internal coordinate
std::tuple<at::Tensor, at::Tensor, at::Tensor> compute_energy_dHa_ddHa_int(const at::Tensor & q);

} // namespace libSGT

#endif