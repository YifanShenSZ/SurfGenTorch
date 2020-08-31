#ifndef libSGT_hpp
#define libSGT_hpp

#include <torch/torch.h>

void initialize_libSGT(const std::string & SSAIC_in, const std::string & DimRed_in, const std::string & Hd_in);

at::Tensor compute_Hd(const at::Tensor & r);

std::tuple<at::Tensor, at::Tensor> compute_Ha_dHa(const at::Tensor & r);

#endif