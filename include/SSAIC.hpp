/*
Scaled and symmetry adapted internal coordinate (SSAIC)

The procedure of this module:
    1. Define internal coordinates
    2. Nondimensionalize the internal coordinates:
       for length, ic = (ic - origin) / origin
       for angle , ic =  ic - origin
    3. Scale the dimensionless internal coordinates:
       if no scaler      : ic = ic
       elif scaler = self: ic = pi * erf(ic)
       else              : ic = ic * exp(-alpha * scaler)
    4. Symmetry adapted linear combinate the scaled dimensionless internal coordinates
*/

#ifndef SSAIC_hpp
#define SSAIC_hpp

#include <torch/torch.h>

#include <FortranLibrary.hpp>

namespace SSAIC {

// Symmetry adapted linear combination
struct SymmAdaptLinComb {
    std::vector<double> coeff;
    std::vector<size_t> IntCoord;

    SymmAdaptLinComb();
    ~SymmAdaptLinComb();
};

// Internal coordinate dimension, not necessarily = cartdim - 6 or 5
extern int intdim;
// Fortran-Library internal coordinate definition
extern std::vector<FL::GT::IntCoordDef> IntCoordDef;
// Cartesian coordinate dimension
extern int cartdim;
// Internal coordinate origin
extern at::Tensor origin;
// Internal coordinates who are scaled by themselves
extern std::vector<size_t> self_scaling;
// other_scaling[i][0] is scaled by [i][1] with alpha = [i][2]
extern std::vector<std::tuple<size_t, size_t, double>> other_scaling;
// Number of irreducible representations
extern size_t NIrred;
// A matrix (2nd order List), as usual product table
extern std::vector<std::vector<size_t>> product_table;
// Number of symmetry adapted internal coordinates per irreducible
extern std::vector<size_t> NSAIC_per_irred;
// symmetry_adaptation[i][j] contains the definition of
// j-th symmetry adapted internal coordinate in i-th irreducible
extern std::vector<std::vector<SymmAdaptLinComb>> symmetry_adaptation;

void define_SSAIC(const std::string & format, const std::string & IntCoordDef_file, const std::string & origin_file, const std::string & ScaleSymm_file);

std::vector<at::Tensor> compute_SSAIC(const at::Tensor & q);

} // namespace SSAIC

#endif