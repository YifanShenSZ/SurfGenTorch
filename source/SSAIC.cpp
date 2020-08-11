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

#include <regex>
#include <torch/torch.h>

#include <FortranLibrary.hpp>
#include <CppLibrary/utility.hpp>
#include <CppLibrary/LinearAlgebra.hpp>
#include <CppLibrary/chemistry.hpp>

#include "SSAIC.hpp"

namespace SSAIC {

SymmAdaptLinComb::SymmAdaptLinComb() {}
SymmAdaptLinComb::~SymmAdaptLinComb() {}

// Internal coordinate dimension, not necessarily = cartdim - 6 or 5
int intdim;
// Fortran-Library internal coordinate definition
std::vector<FL::GT::IntCoordDef> IntCoordDef;
// Cartesian coordinate dimension
int cartdim;
// Internal coordinate origin
at::Tensor origin;
// Internal coordinates who are scaled by themselves
std::vector<size_t> self_scaling;
// other_scaling[i][0] is scaled by [i][1] with alpha = [i][2]
std::vector<std::tuple<size_t, size_t, double>> other_scaling;
// Number of irreducible representations
size_t NIrred;
// A matrix (2nd order List), as usual product table
std::vector<std::vector<size_t>> product_table;
// Number of symmetry adapted internal coordinates per irreducible
std::vector<size_t> NSAIC_per_irred;
// symmetry_adaptation[i][j] contains the definition of
// j-th symmetry adapted internal coordinate in i-th irreducible
std::vector<std::vector<SymmAdaptLinComb>> symmetry_adaptation;

void define_SSAIC(const std::string & format, const std::string & IntCoordDef_file, const std::string & origin_file, const std::string & ScaleSymm_file) {
    // Internal coordinate
    FL::GT::FetchInternalCoordinateDefinition(format, IntCoordDef_file, intdim, IntCoordDef);
    FL::GT::DefineInternalCoordinate(format, IntCoordDef_file);
    std::cout << "Number of internal coordinates: " << intdim << '\n';
    // Origin
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    origin = at::empty(intdim, top);
    if (format == "Columbus7") {
        CL::chemistry::xyz_mass<double> molorigin(origin_file, true);
        cartdim = 3 * molorigin.NAtoms();
        FL::GT::InternalCoordinate(molorigin.geom().data(), origin.data_ptr<double>(), cartdim, intdim);
    }
    else {
        CL::chemistry::xyz<double> molorigin(origin_file, true);
        cartdim = 3 * molorigin.NAtoms();
        FL::GT::InternalCoordinate(molorigin.geom().data(), origin.data_ptr<double>(), cartdim, intdim);
    }
    // Scale and symmetry
    std::ifstream ifs; ifs.open(ScaleSymm_file);
        std::string line;
        std::vector<std::string> strs_vec;
        std::forward_list<std::string> strs_flist;
        // Internal coordinates who are scaled by themselves
        std::getline(ifs, line);
        while (true) {
            std::getline(ifs, line);
            if (! std::regex_match(line, std::regex("\\ *\\d+\\ *"))) break;
            self_scaling.push_back(std::stoul(line)-1);
        }
        // Internal coordinates who are scaled by others
        while (true) {
            std::getline(ifs, line);
            CL::utility::split(line, strs_vec);
            if (! std::regex_match(strs_vec[0], std::regex("\\d+"))) break;
            other_scaling.push_back(std::make_tuple(
                std::stoul(strs_vec[0])-1, std::stoul(strs_vec[1])-1, std::stod(strs_vec[2])));
        }
        // Number of irreducible representations
        std::getline(ifs, line);
        NIrred = std::stoul(line);
        // Product table
        std::getline(ifs, line);
        for (size_t i = 0; i < NIrred; i++) {
            std::getline(ifs, line);
            CL::utility::split(line, strs_vec);
            std::vector<size_t> row(NIrred);
            for (size_t j = 0; j < NIrred; j++) row[j] = std::stoul(strs_vec[j]) - 1;
            product_table.push_back(row);
        }
        // Number of symmetry adapted coordinates per irreducible
        std::getline(ifs, line);
        std::getline(ifs, line);
        CL::utility::split(line, strs_vec);
        NSAIC_per_irred.resize(strs_vec.size());
        for (size_t i = 0; i < NSAIC_per_irred.size(); i++) NSAIC_per_irred[i] = std::stoul(strs_vec[i]);
        // Symmetry adapted linear combinations of each irreducible
        symmetry_adaptation.resize(NIrred);
        std::getline(ifs, line);
        for (std::vector<SymmAdaptLinComb> & SALCs : symmetry_adaptation) {
            int count = -1;
            while (true) {
                std::getline(ifs, line);
                if (! ifs.good()) break;
                CL::utility::split(line, strs_flist);
                if (! std::regex_match(strs_flist.front(), std::regex("-?\\d+\\.?\\d*"))) break;
                if (std::regex_match(strs_flist.front(), std::regex("\\d+"))) {
                    count++;
                    SALCs.push_back(SymmAdaptLinComb());
                    strs_flist.pop_front();
                }
                SALCs[count].coeff.push_back(std::stod(strs_flist.front()));
                strs_flist.pop_front();
                SALCs[count].IntCoord.push_back(std::stoul(strs_flist.front())-1);
            }
            // Normalize linear combination coefficients
            for (SymmAdaptLinComb & SALC : SALCs) {
                double norm = CL::LA::norm2(SALC.coeff);
                for (size_t k = 0; k < SALC.coeff.size(); k++)
                SALC.coeff[k] /= norm;
            }
        }
    ifs.close();
}

std::vector<at::Tensor> compute_SSAIC(const at::Tensor & q) {
    // Nondimensionalize
    at::Tensor work = q.clone();
    work -= origin;
    for (size_t i = 0; i < intdim; i++) {
        if (IntCoordDef[i].motion[0].type == "stretching") work[i] /= origin[i];
    }
    // Scale
    for (std::tuple<size_t, size_t, double> & scaling : other_scaling) work[std::get<0>(scaling)] *= exp(-std::get<2>(scaling) * work[std::get<1>(scaling)]);
    for (size_t & scaling : self_scaling) work[scaling] = M_PI * erf(work[scaling]);
    // Symmetrize
    std::vector<at::Tensor> SSAgeom(NIrred);
    for (size_t irred = 0; irred < NIrred; irred++) {
        std::vector<SymmAdaptLinComb> * SALCs = & symmetry_adaptation[irred];
        at::Tensor irred_geom = work.new_zeros(SALCs->size());
        for (size_t i = 0; i < irred_geom.size(0); i++) {
            SymmAdaptLinComb * SALC = & (*SALCs)[i];
            for (size_t j = 0; j < SALC->coeff.size(); j++) irred_geom[i] += SALC->coeff[j] * work[SALC->IntCoord[j]];
        }
        SSAgeom[irred] = irred_geom;
    }
    return SSAgeom;
}

} // namespace SSAIC