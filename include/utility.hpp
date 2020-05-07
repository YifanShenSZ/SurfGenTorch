#ifndef utility_hpp
#define utility_hpp

#include <torch/torch.h>

std::tuple<int, at::Tensor> cartdim_origin(const std::string & format, const std::string & origin_file, const int & indim);

std::vector<std::string> verify_data_set(const std::vector<std::string> & original_data_set);

#endif