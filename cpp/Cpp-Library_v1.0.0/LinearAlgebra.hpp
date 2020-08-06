// Vector & matrix & high order tensor operation

#ifndef LinearAlgebra_hpp
#define LinearAlgebra_hpp

#include <torch/torch.h>

namespace CL { namespace LA { // LinearAlgebra

// 2-norm of vector x
double norm2(const double * x, const int & N);
double norm2(const std::vector<double> & x);

} // namespace LA
} // namespace CL

#endif