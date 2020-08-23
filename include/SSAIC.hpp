// Scaled and symmetry adapted internal coordinate (SSAIC)

#ifndef SSAIC_hpp
#define SSAIC_hpp

#include <torch/torch.h>

#include <FortranLibrary.hpp>

namespace SSAIC {

// Internal coordinate dimension, not necessarily = cartdim - 6 or 5
extern int intdim;
// Cartesian coordinate dimension
extern int cartdim;

// Number of irreducible representations
extern size_t NIrred;
// Number of symmetry adapted internal coordinates per irreducible
extern std::vector<size_t> NSAIC_per_irred;

void define_SSAIC(const std::string & SSAIC_in);

std::vector<at::Tensor> compute_SSAIC(const at::Tensor & q);

} // namespace SSAIC

#endif