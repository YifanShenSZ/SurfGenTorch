#ifndef libSGT_hpp
#define libSGT_hpp

#include <torch/torch.h>

void initialize_libSGT(const std::string & SSAIC_in, const std::string & DimRed_in, const std::string & Hd_in);

at::Tensor compute_Hd(const at::Tensor & r);

std::tuple<at::Tensor, at::Tensor> compute_Hd_dHd(const at::Tensor & r);

at::Tensor compute_energy(const at::Tensor & r);

std::tuple<at::Tensor, at::Tensor> compute_energy_dH(const at::Tensor & r);

#endif